// Motion handling for frame extrapolation: preparing the media engine's input,
// post-processing the field it returns, and warping the newest capture along it.

#include "ShaderCommon.h"

/// Feeds VTMotionEstimation, which takes single-component luma rather than BGRA. Motion is analysed at a
/// bounded size, so the output may be smaller than the input by a whole divisor: each texel is the mean of
/// the block of pixels it covers.
kernel void bgraToLuma(
    texture2d<half, access::read> input [[texture(0)]],
    texture2d<half, access::write> output [[texture(1)]],
    constant uint& divisor [[buffer(0)]],
    uint2 gid [[thread_position_in_grid]]
) {
    uint width = output.get_width();
    uint height = output.get_height();
    if (gid.x >= width || gid.y >= height) return;

    float sum = 0.0f;
    for (uint dy = 0; dy < divisor; dy++) {
        for (uint dx = 0; dx < divisor; dx++) {
            uint2 src = clampCoord(int2(gid * divisor + uint2(dx, dy)), input.get_width(), input.get_height());
            sum += float(rgb2luma(input.read(src).rgb));
        }
    }
    output.write(half4(half(sum / float(divisor * divisor)), 0.0h, 0.0h, 1.0h), gid);
}

/// Copies a motion field into a slot the pipeline owns, resampled to the slot's size, and applies the
/// scale that carries its vectors into the units of the frame the warp runs on: a field measured on an
/// image averaged down by a whole factor is in that image's pixels, and with render scale active the warp
/// runs on a larger restored frame.
///
/// `coverage` is the share of the input the output spans. The media engine pads its fields past the
/// frame, to a multiple of four vectors, and the padding is not the image: spanning it would stretch the
/// field over the frame and misplace every vector by up to a dozen pixels at the far edge. 1 resamples all
/// of the input.
kernel void copyMotionField(
    texture2d<half, access::sample> input [[texture(0)]],
    texture2d<half, access::write> output [[texture(1)]],
    constant float& scale [[buffer(0)]],
    constant float2& coverage [[buffer(1)]],
    uint2 gid [[thread_position_in_grid]]
) {
    uint width = output.get_width();
    uint height = output.get_height();
    if (gid.x >= width || gid.y >= height) return;

    constexpr sampler linearSampler(filter::linear, address::clamp_to_edge, coord::normalized);
    float2 uv = (float2(gid) + 0.5f) / float2(width, height) * coverage;
    float2 mv = float2(input.sample(linearSampler, uv).xy) * scale;
    output.write(half4(half(mv.x), half(mv.y), 0.0h, 1.0h), gid);
}

/// Neighbour disagreement of the motion field, at the field's own resolution.
///
/// Where neighbouring blocks disagree, the field is straddling a motion boundary
/// it cannot represent — an object edge, or ground opening up behind something.
/// Warping across that boundary is what smears the image, so the warp fades out
/// exactly there and the source pixel shows through.
///
/// This is a property of the field alone: every pixel inside a block reads the
/// same five vectors and reaches the same answer. Computing it per output pixel
/// meant five motion fetches and five square roots two million times per
/// generated image to rebuild a few thousand distinct values. The warp samples
/// this map linearly instead, which reproduces the smoothing the per-pixel
/// version got from its own linear fetches.
kernel void motionDisagreement(
    texture2d<half, access::read> motion [[texture(0)]],
    texture2d<half, access::write> output [[texture(1)]],
    constant int& stride [[buffer(0)]],
    uint2 gid [[thread_position_in_grid]]
) {
    uint width = motion.get_width();
    uint height = motion.get_height();
    if (gid.x >= width || gid.y >= height) return;

    // Neighbours are a fixed distance apart *in the frame*, not in the field, so
    // the test measures the same thing whatever resolution the field arrives at.
    //
    // A one-texel step is meaningless on a dense field. Optical flow is smooth
    // by construction — it has no image evidence in a textureless region, so it
    // fills one in from the surrounding motion — and that fill is gradual, so
    // adjacent pixels agree almost exactly and the test passes everything. The
    // boundary it needs to see, between a static overlay and the scene moving
    // behind it, is spread across roughly a block's width; stepping that far
    // finds it.
    int step = max(1, stride);
    int2 p = int2(gid);
    float2 mv = float2(motion.read(gid).xy);
    float2 mL = float2(motion.read(clampCoord(p + int2(-step,  0), width, height)).xy);
    float2 mR = float2(motion.read(clampCoord(p + int2( step,  0), width, height)).xy);
    float2 mU = float2(motion.read(clampCoord(p + int2( 0, -step), width, height)).xy);
    float2 mD = float2(motion.read(clampCoord(p + int2( 0,  step), width, height)).xy);

    float disagreement = max(max(length(mv - mL), length(mv - mR)),
                             max(length(mv - mU), length(mv - mD)));
    output.write(half4(half(disagreement), 0.0h, 0.0h, 1.0h), gid);
}

/// Sums `value` across the whole threadgroup and hands the total to every thread.
/// Each thread adds the per-simdgroup partials itself, which for at most a few
/// simdgroups is cheaper than electing one thread and broadcasting its answer.
/// `scratch` is free for reuse when this returns.
inline float4 threadgroupSum(float4 value, threadgroup float4* scratch,
                             uint lane, uint simdIndex, uint simdCount) {
    float4 partial = simd_sum(value);
    if (lane == 0) scratch[simdIndex] = partial;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float4 total = 0.0f;
    for (uint i = 0; i < simdCount; ++i) total += scratch[i];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    return total;
}

/// Replaces the vectors the block matcher got wildly wrong.
///
/// Where an area has nothing to lock onto — content entering the frame at an edge, a flat sky —
/// the matcher still has to answer, and what it answers is arbitrary: a block that moved 16 pixels
/// comes back as 290. Both consumers take the field at face value, so one such block drags its
/// neighbours' pixels hundreds of pixels the wrong way, and the bilinear expansion to a dense field
/// spreads it further. Measured on a textured pan, 97.6% of blocks were exact and the other 2.4%
/// were off by up to 290 pixels — and those few cost MetalFX's interpolation 15 dB.
///
/// Two tests, in order. A vector far from the median of its own 3x3 neighbourhood is an outlier of
/// it, and takes the median. One that is still absurd next to what the whole frame is doing — an
/// edge block whose whole neighbourhood is garbage — takes the frame's motion instead. Both are
/// deliberately generous, so independent motion survives: only vectors that disagree by more than
/// the neighbourhood's own motion, and by more than a fraction of the frame, are touched.
inline void sortPair(thread float& a, thread float& b) {
    float lower = min(a, b);
    b = max(a, b);
    a = lower;
}

/// Median of nine by a sorting network: no loops, no data-dependent branches.
inline float medianOfNine(thread float (&v)[9]) {
    sortPair(v[1], v[2]); sortPair(v[4], v[5]); sortPair(v[7], v[8]);
    sortPair(v[0], v[1]); sortPair(v[3], v[4]); sortPair(v[6], v[7]);
    sortPair(v[1], v[2]); sortPair(v[4], v[5]); sortPair(v[7], v[8]);
    sortPair(v[0], v[3]); sortPair(v[5], v[8]); sortPair(v[4], v[7]);
    sortPair(v[3], v[6]); sortPair(v[1], v[4]); sortPair(v[2], v[5]);
    sortPair(v[4], v[7]); sortPair(v[4], v[2]); sortPair(v[6], v[4]);
    sortPair(v[4], v[2]);
    return v[4];
}

kernel void despeckleMotion(
    texture2d<half, access::read> input [[texture(0)]],
    texture2d<half, access::write> output [[texture(1)]],
    device const float2& globalMotion [[buffer(0)]],
    constant float& frameWidth [[buffer(1)]],
    uint2 gid [[thread_position_in_grid]]
) {
    uint width = input.get_width();
    uint height = input.get_height();
    if (gid.x >= width || gid.y >= height) return;

    float xs[9];
    float ys[9];
    int n = 0;
    for (int dy = -1; dy <= 1; ++dy) {
        for (int dx = -1; dx <= 1; ++dx) {
            float2 v = float2(input.read(clampCoord(int2(gid) + int2(dx, dy), width, height)).xy);
            xs[n] = v.x;
            ys[n] = v.y;
            ++n;
        }
    }
    const float2 median = float2(medianOfNine(xs), medianOfNine(ys));
    const float2 frame = globalMotion;

    // Distances scale with the frame, so a 4K capture and a 720p one are judged alike.
    const float slack = 0.02f * frameWidth;

    float2 v = float2(input.read(gid).xy);
    if (length(v - median) > max(slack, length(median))) v = median;
    if (length(v - frame) > 2.0f * max(length(frame), slack)) v = frame;
    output.write(half4(half(v.x), half(v.y), 0.0h, 1.0h), gid);
}

/// Collapses the motion field to the one vector that describes the frame.
///
/// A plain mean is the wrong answer: the block matcher reports zero where it has no
/// evidence, and large still regions would drag the estimate toward nothing
/// while the scene sweeps past. So the field is first reduced to a coarse grid of
/// tile means (one thread per tile), then the mean of the tiles is taken, and then
/// taken again over only the tiles that sit within one standard deviation of it.
/// What survives is whatever the bulk of the frame agrees on, which in a
/// first-person view is the camera — and the camera is the motion worth falling
/// back to.
///
/// One threadgroup does all of it, so no single lane walks the grid while the rest of the GPU waits.
///
/// The answer goes into a buffer rather than a texture: the warp reads it as a uniform,
/// once per threadgroup, where a texel would be fetched by every pixel.
[[max_total_threads_per_threadgroup(MG_MOTION_GRID * MG_MOTION_GRID)]]
kernel void globalMotion(
    texture2d<half, access::sample> motion [[texture(0)]],
    device float2& result [[buffer(0)]],
    uint tid [[thread_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],
    uint simdIndex [[simdgroup_index_in_threadgroup]],
    uint simdCount [[simdgroups_per_threadgroup]]
) {
    constexpr sampler linearSampler(filter::linear, address::clamp_to_edge, coord::normalized);
    threadgroup float4 scratch[32];

    const uint2 tile = uint2(tid % MG_MOTION_GRID, tid / MG_MOTION_GRID);
    const int taps = 4;
    float2 sum = 0.0f;
    for (int y = 0; y < taps; ++y) {
        for (int x = 0; x < taps; ++x) {
            float2 uv = (float2(tile) + (float2(x, y) + 0.5f) / float(taps)) / float(MG_MOTION_GRID);
            sum += float2(motion.sample(linearSampler, uv).xy);
        }
    }
    const float2 tileMean = sum / float(taps * taps);

    const float4 total = threadgroupSum(float4(tileMean, 1.0f, 0.0f), scratch, lane, simdIndex, simdCount);
    const float2 mean = total.xy / total.z;

    const float2 offset = tileMean - mean;
    const float variance = threadgroupSum(float4(dot(offset, offset), 0.0f, 0.0f, 0.0f),
                                          scratch, lane, simdIndex, simdCount).x;
    const float spread = sqrt(variance / total.z);

    const bool keep = length(offset) <= spread;
    const float4 trimmed = threadgroupSum(keep ? float4(tileMean, 1.0f, 0.0f) : float4(0.0f),
                                          scratch, lane, simdIndex, simdCount);
    if (tid == 0) {
        result = trimmed.z > 0.0f ? trimmed.xy / trimmed.z : mean;
    }
}

/// 1 where the compositor drew a pixel's whole neighbourhood identically in both
/// captures, 0 elsewhere.
///
/// Static is a property of a region, not of a pixel. Inside moving content
/// individual pixels match by coincidence all the time — a uniform metal
/// surface, a flat wall — and holding those while their neighbours warp is what
/// speckles an image and tears a weapon apart. Requiring the +-2 pixel cross
/// around the pixel to be unchanged makes a coincidental match impossible to
/// mistake for an interface.
///
/// The comparison is exact. Interface elements are redrawn from the same source
/// every frame and arrive bit-identical; the tolerance is for half-precision
/// rounding, not a judgement about how much change counts as motion.
///
/// This depends only on the two captures, not on the warp phase, so it is built
/// once per capture here instead of being rebuilt per output pixel for every
/// phase the warp generates. Each pixel's change flag is evaluated once into
/// threadgroup memory (tile plus a two pixel halo) rather than ten texture reads
/// per pixel. The threadgroup must be exactly MG_MASK_TILE_WIDTH x
/// MG_MASK_TILE_HEIGHT and every threadgroup of the grid must be full, because the
/// halo is loaded cooperatively by all of its threads.
constant int kMaskHalo = 2;
[[max_total_threads_per_threadgroup(MG_MASK_TILE_WIDTH * MG_MASK_TILE_HEIGHT)]]
kernel void staticMask(
    texture2d<half, access::read> now [[texture(0)]],
    texture2d<half, access::read> before [[texture(1)]],
    texture2d<half, access::write> mask [[texture(2)]],
    uint2 gid [[thread_position_in_grid]],
    uint2 lid [[thread_position_in_threadgroup]],
    uint2 tgid [[threadgroup_position_in_grid]]
) {
    constexpr int tileW = MG_MASK_TILE_WIDTH + 2 * kMaskHalo;
    constexpr int tileH = MG_MASK_TILE_HEIGHT + 2 * kMaskHalo;
    threadgroup uchar changed[tileW * tileH];

    const int width = int(now.get_width());
    const int height = int(now.get_height());
    const int2 origin = int2(tgid * uint2(MG_MASK_TILE_WIDTH, MG_MASK_TILE_HEIGHT)) - kMaskHalo;
    const uint threads = MG_MASK_TILE_WIDTH * MG_MASK_TILE_HEIGHT;
    const uint linear = lid.y * MG_MASK_TILE_WIDTH + lid.x;

    for (uint i = linear; i < uint(tileW * tileH); i += threads) {
        int2 p = clamp(origin + int2(int(i) % tileW, int(i) / tileW), int2(0), int2(width - 1, height - 1));
        half3 delta = abs(now.read(uint2(p)).rgb - before.read(uint2(p)).rgb);
        changed[i] = max(max(delta.r, delta.g), delta.b) >= half(1.0f / 255.0f) ? 1 : 0;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (gid.x >= uint(width) || gid.y >= uint(height)) return;

    const int cx = int(lid.x) + kMaskHalo;
    const int cy = int(lid.y) + kMaskHalo;
    const uchar any = changed[cy * tileW + cx]
                    | changed[cy * tileW + cx - kMaskHalo] | changed[cy * tileW + cx + kMaskHalo]
                    | changed[(cy - kMaskHalo) * tileW + cx] | changed[(cy + kMaskHalo) * tileW + cx];
    mask.write(half4(any == 0 ? 1.0h : 0.0h), gid);
}

/// Catmull-Rom resampling in five bilinear fetches: the two taps of each of the four rows and columns around
/// the sample point are folded into one fetch apiece by placing it where the bilinear weights give the
/// cubic's, and the four corners, whose weights are tiny, are dropped. A warp lands between pixels almost
/// everywhere, and bilinear filtering softens whatever it lands on, so a warped image alternated with a
/// captured one — which is sharp — pulses in sharpness at the capture rate. This keeps the detail. Exactly on
/// a pixel centre the weights collapse to that pixel, so static content is untouched.
inline half4 sampleCatmullRom(texture2d<half, access::sample> tex, sampler s, float2 uv, float2 size) {
    const float2 position = uv * size;
    const float2 center = floor(position - 0.5f) + 0.5f;
    const float2 f = position - center;
    const float2 f2 = f * f;
    const float2 f3 = f2 * f;

    const float2 w0 = f2 - 0.5f * (f3 + f);
    const float2 w1 = 1.5f * f3 - 2.5f * f2 + 1.0f;
    const float2 w3 = 0.5f * (f3 - f2);
    const float2 w2 = 1.0f - w0 - w1 - w3;
    const float2 w12 = w1 + w2;

    const float2 near = (center - 1.0f) / size;
    const float2 far = (center + 2.0f) / size;
    const float2 middle = (center + w2 / w12) / size;

    float4 sum = float4(tex.sample(s, float2(middle.x, near.y))) * (w12.x * w0.y)
               + float4(tex.sample(s, float2(near.x, middle.y))) * (w0.x * w12.y)
               + float4(tex.sample(s, float2(middle.x, middle.y))) * (w12.x * w12.y)
               + float4(tex.sample(s, float2(far.x, middle.y))) * (w3.x * w12.y)
               + float4(tex.sample(s, float2(middle.x, far.y))) * (w12.x * w3.y);
    const float total = w12.x * w0.y + w0.x * w12.y + w12.x * w12.y + w3.x * w12.y + w12.x * w3.y;
    return half4(clamp(sum / total, 0.0f, 1.0f));
}

/// Frame extrapolation. VTMotionEstimation returns backward vectors in pixels:
/// `mv` at p says where p's content sat in the previous frame, so the content
/// velocity is -mv per capture interval. Sampling the source at `p + mv * phase`
/// is therefore a backward warp to a future time — it leaves no holes, unlike
/// scattering pixels forward, and needs no second frame so it adds no latency.
///
/// Three questions decide how far each pixel travels, and none of them freezes a
/// pixel while its neighbours move — a frozen pixel tears, and the torn edge is as
/// visible as the smear it replaced. They choose between the pixel's own motion and
/// the frame's, so a mistrusted pixel still travels, just with the crowd. Nothing
/// melts under a global shift: it preserves every spatial relationship in the
/// image and only puts the whole thing slightly in the wrong place, which reads as
/// motion where a dissolving surface reads as a fault. This is what VR
/// reprojection has always done, and why it holds up where per-pixel warping does
/// not.
///
///  1. Is the vector internally consistent? Neighbouring blocks that disagree
///     (relative to the vector's own magnitude, so a coherent pan of any speed
///     passes and a flat wall the block matcher could not lock onto does not)
///     describe a boundary the grid cannot represent. Dividing error by signal
///     also cancels `phase`: predicting further ahead scales the misplacement and
///     the shift by the same factor, so trust does not depend on which phase asked.
///  2. Does the motion where the warp lands match the motion it travelled on? Where
///     a near column passes a far wall both vectors are perfectly measured, they
///     just belong to different surfaces — only the destination shows it.
///  3. Interface protection, both directions. Optical flow has no image evidence
///     inside a thin or flat static element, so it fills one in from the scene
///     moving behind it: the interface gets dragged along, and a copy of it gets
///     dragged into the scene beside it. Neither is visible to any test on the
///     field; the evidence is in the images, which is what `staticMask` holds.
///     Static pixels do not travel at all, and moving pixels do not pull from a
///     static region.
kernel void extrapolateFrame(
    texture2d<half, access::sample> source [[texture(0)]],
    texture2d<half, access::sample> motion [[texture(1)]],
    texture2d<half, access::write> output [[texture(2)]],
    texture2d<half, access::sample> disagreementField [[texture(3)]],
    texture2d<half, access::read> staticMask [[texture(4)]],
    constant float& phase [[buffer(0)]],
    device const float2& globalMotion [[buffer(1)]],
    uint2 gid [[thread_position_in_grid]]
) {
    const uint width = output.get_width();
    const uint height = output.get_height();
    if (gid.x >= width || gid.y >= height) return;

    constexpr sampler linearSampler(filter::linear, address::clamp_to_edge, coord::normalized);
    const float2 size = float2(width, height);
    const float2 uv = (float2(gid) + 0.5f) / size;

    // The field is one vector per block, so a linear fetch smooths it for free.
    const float2 mv = float2(motion.sample(linearSampler, uv).xy);
    const float disagreement = float(disagreementField.sample(linearSampler, uv).x);
    const float magnitude = length(mv);
    const float confidence = magnitude > 0.0f ? saturate(1.0f - disagreement / magnitude) : 0.0f;

    const float staticHere = float(staticMask.read(gid).x);
    const float2 landing = (float2(gid) + 0.5f + mv * phase * confidence) / size;
    const float2 mvLanding = float2(motion.sample(linearSampler, clamp(landing, 0.0f, 1.0f)).xy);
    const float agreement = magnitude > 0.0f ? saturate(1.0f - length(mv - mvLanding) / magnitude) : 1.0f;

    const uint2 landingPixel = uint2(clamp(landing * size, float2(0.0f), size - 1.0f));
    const float staticLanding = float(staticMask.read(landingPixel).x);
    const float pullsFromInterface = staticLanding * (1.0f - staticHere);

    const float trust = saturate(agreement * (1.0f - pullsFromInterface));
    const float2 blended = mix(globalMotion, mv, trust);
    const float2 delta = blended * phase * (1.0f - staticHere);

    const float2 sourceUV = (float2(gid) + 0.5f + delta) / size;
    output.write(sampleCatmullRom(source, linearSampler, clamp(sourceUV, 0.0f, 1.0f), size), gid);
}
