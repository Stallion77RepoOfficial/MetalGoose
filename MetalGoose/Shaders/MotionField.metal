// Motion handling for MetalFX interpolation: preparing the media engine's input and post-processing the field it
// returns.

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
/// scale that carries its vectors into the units of the frame MetalFX runs on: a field measured on an
/// image averaged down by a whole factor is in that image's pixels.
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
/// comes back as 290. MetalFX takes the field at face value, so one such block drags its
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
/// The answer goes into a buffer rather than a texture: `despeckleMotion` reads it as a uniform,
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
