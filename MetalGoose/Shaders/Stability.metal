// Bringing a generated image back toward the captures it was made from, where they barely differ.

#include "ShaderCommon.h"

constant int kStabilityHalo = 2;
constant int kStabilityTileWidth = MG_BLEND_TILE_WIDTH + 2 * kStabilityHalo;
constant int kStabilityTileHeight = MG_BLEND_TILE_HEIGHT + 2 * kStabilityHalo;

/// The largest change between the two captures within two pixels of this one, as a share of the full range.
///
/// The change at a pixel is the largest difference of any channel between the captures, and what counts is the largest
/// in the 5x5 around it: an edge that moved decides its neighbours as well as itself, and a flat patch inside moving
/// content is judged with the content. The difference of each pixel is evaluated once into threadgroup memory (tile plus
/// a two pixel halo); the threadgroup must be exactly MG_BLEND_TILE_WIDTH x MG_BLEND_TILE_HEIGHT and every threadgroup of
/// the grid must be full, because the halo is loaded cooperatively by all of its threads, and this is called by all of them
/// before any returns.
inline half largestChange(texture2d<half, access::read> previous, texture2d<half, access::read> next,
                          threadgroup half* change, uint2 lid, uint2 tgid) {
    const int width = int(previous.get_width());
    const int height = int(previous.get_height());
    const int2 origin = int2(tgid * uint2(MG_BLEND_TILE_WIDTH, MG_BLEND_TILE_HEIGHT)) - kStabilityHalo;
    const uint threads = MG_BLEND_TILE_WIDTH * MG_BLEND_TILE_HEIGHT;
    const uint linear = lid.y * MG_BLEND_TILE_WIDTH + lid.x;

    for (uint i = linear; i < uint(kStabilityTileWidth * kStabilityTileHeight); i += threads) {
        const int2 p = clamp(origin + int2(int(i) % kStabilityTileWidth, int(i) / kStabilityTileWidth),
                             int2(0), int2(width - 1, height - 1));
        const half3 delta = abs(next.read(uint2(p)).rgb - previous.read(uint2(p)).rgb);
        change[i] = max(max(delta.r, delta.g), delta.b);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    const int cx = int(lid.x) + kStabilityHalo;
    const int cy = int(lid.y) + kStabilityHalo;
    half largest = 0.0h;
    for (int dy = -kStabilityHalo; dy <= kStabilityHalo; ++dy) {
        for (int dx = -kStabilityHalo; dx <= kStabilityHalo; ++dx) {
            largest = max(largest, change[(cy + dy) * kStabilityTileWidth + cx + dx]);
        }
    }
    return largest;
}

/// The image shown: the generated one, and the captures' own mix at its phase in the share that `largest` leaves.
///
/// The share of the captures' mix falls linearly from all of it where nothing changed to none where the change reaches
/// `tolerance`. Where nothing changed the capture is taken as it is, not through a mix that could round it.
inline half3 stabilized(half3 image, half3 previous, half3 next, half largest, constant StabilityBlendParams& params) {
    const half weight = saturate(1.0h - largest / half(params.tolerance));
    const half3 captured = mix(previous, next, half(params.phase));
    return clampColor(weight >= 1.0h ? captured : mix(image, captured, weight));
}

/// MetalFX's image, blended with the captures it came from, by how little they changed around each pixel.
///
/// The Neural Engine's images are lossy where nothing moved: its model smooths what it is given, and the 4:2:0 it
/// works in drops the colour of thin coloured text. Scored against the real in-between frame, an interface drawn
/// identically in every frame came out at 21 dB, and a still scene was further from the truth than repeating a
/// capture. Where the captures are the same, the image between them is that same; where they differ by little it lies
/// between them, and the captures' own mix at this phase is closer than a re-drawn image; where they differ by a lot,
/// motion is what the engine is for, and its image stands untouched. MetalFX's images are close to the captures where
/// nothing moved already, and it takes the same rule with a smaller tolerance, for what it gains at an interface.
///
/// On six 720p clips, with and without an interface drawn over them, 1, 2 and 4 captures apart, the Neural Engine's
/// image went from 32.7, 32.0 and 31.2 dB to 40.5, 35.7 and 32.9 without the interface, and from 28.3, 28.0 and 27.7 to
/// 40.6, 35.8 and 32.9 with it; the interface itself from 21 to 42.
[[max_total_threads_per_threadgroup(MG_BLEND_TILE_WIDTH * MG_BLEND_TILE_HEIGHT)]]
kernel void blendTowardCaptures(
    texture2d<half, access::read> generated [[texture(0)]],
    texture2d<half, access::read> previous [[texture(1)]],
    texture2d<half, access::read> next [[texture(2)]],
    texture2d<half, access::write> output [[texture(3)]],
    constant StabilityBlendParams& params [[buffer(0)]],
    uint2 gid [[thread_position_in_grid]],
    uint2 lid [[thread_position_in_threadgroup]],
    uint2 tgid [[threadgroup_position_in_grid]]
) {
    threadgroup half change[kStabilityTileWidth * kStabilityTileHeight];
    const half largest = largestChange(previous, next, change, lid, tgid);
    if (gid.x >= generated.get_width() || gid.y >= generated.get_height()) return;

    output.write(half4(stabilized(generated.read(gid).rgb, previous.read(gid).rgb, next.read(gid).rgb, largest, params), 1.0h), gid);
}

/// Catmull-Rom resampling in five bilinear fetches: the two taps of each of the four rows and columns around
/// the sample point are folded into one fetch apiece by placing it where the bilinear weights give the
/// cubic's, and the four corners, whose weights are tiny, are dropped. Bilinear filtering softens whatever it lands on,
/// and an image enlarged that way next to a captured one, which is sharp, pulses in sharpness at the capture rate. This
/// keeps the detail. Exactly on a pixel centre the weights collapse to that pixel.
inline float4 sampleCatmullRom(texture2d<half, access::sample> tex, sampler s, float2 uv, float2 size) {
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

    const float4 sum = float4(tex.sample(s, float2(middle.x, near.y))) * (w12.x * w0.y)
                     + float4(tex.sample(s, float2(near.x, middle.y))) * (w0.x * w12.y)
                     + float4(tex.sample(s, float2(middle.x, middle.y))) * (w12.x * w12.y)
                     + float4(tex.sample(s, float2(far.x, middle.y))) * (w3.x * w12.y)
                     + float4(tex.sample(s, float2(middle.x, far.y))) * (w12.x * w3.y);
    const float total = w12.x * w0.y + w0.x * w12.y + w12.x * w12.y + w3.x * w12.y + w12.x * w3.y;
    return clamp(sum / total, 0.0f, 1.0f);
}

/// The Neural Engine's image, blended with the captures it came from, straight from the planes it was written in: the
/// luma and the interleaved chroma are turned into colour here, as the image is shown, so that an image that is not shown
/// costs nothing, and one that is does not pass through a texture of its own.
[[max_total_threads_per_threadgroup(MG_BLEND_TILE_WIDTH * MG_BLEND_TILE_HEIGHT)]]
kernel void blendTowardCapturesFromYUV(
    texture2d<half, access::read> luma [[texture(0)]],
    texture2d<half, access::sample> chroma [[texture(1)]],
    texture2d<half, access::read> previous [[texture(2)]],
    texture2d<half, access::read> next [[texture(3)]],
    texture2d<half, access::write> output [[texture(4)]],
    constant StabilityBlendParams& params [[buffer(0)]],
    uint2 gid [[thread_position_in_grid]],
    uint2 lid [[thread_position_in_threadgroup]],
    uint2 tgid [[threadgroup_position_in_grid]]
) {
    threadgroup half change[kStabilityTileWidth * kStabilityTileHeight];
    const half largest = largestChange(previous, next, change, lid, tgid);
    if (gid.x >= luma.get_width() || gid.y >= luma.get_height()) return;

    // The chroma plane is half the size: sampled bilinearly rather than replicated, which would show as blocks of four.
    constexpr sampler linearSampler(filter::linear, address::clamp_to_edge, coord::normalized);
    const float2 uv = (float2(gid) + 0.5f) / float2(luma.get_width(), luma.get_height());
    const half3 image = yuvToRgb(luma.read(gid).r, half2(chroma.sample(linearSampler, uv).rg));
    output.write(half4(stabilized(image, previous.read(gid).rgb, next.read(gid).rgb, largest, params), 1.0h), gid);
}

/// The same for planes of another size than the captures: the Neural Engine works at what it takes, and a capture it does not
/// take whole was shrunk for it. The image is enlarged to the captures' size here, as it is blended — the luma in Catmull-Rom,
/// where the detail is, and the chroma bilinearly, which is as sharp as a plane that was half-size to begin with gets — and
/// where the captures are the same the result is the captures, not the enlarged image.
[[max_total_threads_per_threadgroup(MG_BLEND_TILE_WIDTH * MG_BLEND_TILE_HEIGHT)]]
kernel void blendTowardCapturesFromYUVResampled(
    texture2d<half, access::sample> luma [[texture(0)]],
    texture2d<half, access::sample> chroma [[texture(1)]],
    texture2d<half, access::read> previous [[texture(2)]],
    texture2d<half, access::read> next [[texture(3)]],
    texture2d<half, access::write> output [[texture(4)]],
    constant StabilityBlendParams& params [[buffer(0)]],
    uint2 gid [[thread_position_in_grid]],
    uint2 lid [[thread_position_in_threadgroup]],
    uint2 tgid [[threadgroup_position_in_grid]]
) {
    threadgroup half change[kStabilityTileWidth * kStabilityTileHeight];
    const half largest = largestChange(previous, next, change, lid, tgid);
    if (gid.x >= previous.get_width() || gid.y >= previous.get_height()) return;

    constexpr sampler linearSampler(filter::linear, address::clamp_to_edge, coord::normalized);
    const float2 uv = (float2(gid) + 0.5f) / float2(previous.get_width(), previous.get_height());
    const half y = half(sampleCatmullRom(luma, linearSampler, uv, float2(luma.get_width(), luma.get_height())).x);
    const half3 image = yuvToRgb(y, half2(chroma.sample(linearSampler, uv).rg));
    output.write(half4(stabilized(image, previous.read(gid).rgb, next.read(gid).rgb, largest, params), 1.0h), gid);
}
