// Bringing a generated image back toward the captures it was made from, where the content did not move.

#include "ShaderCommon.h"

/// Two pixels either side for the window the weight is judged over, and one more for the contrast of the pixels at its edge.
constant int kStabilityHalo = 3;
constant int kStabilityTileWidth = MG_BLEND_TILE_WIDTH + 2 * kStabilityHalo;
constant int kStabilityTileHeight = MG_BLEND_TILE_HEIGHT + 2 * kStabilityHalo;
constant int kStabilityWindowWidth = MG_BLEND_TILE_WIDTH + 4;
constant int kStabilityWindowHeight = MG_BLEND_TILE_HEIGHT + 4;

/// How much of the captures' own mix a pixel gets, from how much the two captures differ around it compared with how much
/// there is to differ (`StabilityBlend` has the reasoning and the numbers).
///
/// The captures are read once into packed `uint` threadgroup memory (`uchar4` unorm), tile plus a three pixel halo;
/// the change between them and the contrast in them are worked out once for every pixel of the tile plus two, and
/// reduced over each pixel's 5x5 via separable 1x5 horizontal and 5x1 vertical passes. The threadgroup must be exactly
/// MG_BLEND_TILE_WIDTH x MG_BLEND_TILE_HEIGHT and every threadgroup of the grid must be full, because the halo is
/// loaded cooperatively by all of its threads, and this is called by all of them before any returns.
inline half stabilityWeight(texture2d<half, access::read> previous, texture2d<half, access::read> next,
                            threadgroup uint* tilePrevious, threadgroup uint* tileNext, threadgroup half2* window,
                            uint2 lid, uint2 tgid, constant StabilityBlendParams& params,
                            thread half3& prevColor, thread half3& nextColor) {
    const int width = int(previous.get_width());
    const int height = int(previous.get_height());
    const int2 origin = int2(tgid * uint2(MG_BLEND_TILE_WIDTH, MG_BLEND_TILE_HEIGHT)) - kStabilityHalo;
    const uint threads = MG_BLEND_TILE_WIDTH * MG_BLEND_TILE_HEIGHT;
    const uint linear = lid.y * MG_BLEND_TILE_WIDTH + lid.x;

    for (uint i = linear; i < uint(kStabilityTileWidth * kStabilityTileHeight); i += threads) {
        const int2 p = clamp(origin + int2(int(i) % kStabilityTileWidth, int(i) / kStabilityTileWidth),
                             int2(0), int2(width - 1, height - 1));
        tilePrevious[i] = pack_half_to_unorm4x8(previous.read(uint2(p)));
        tileNext[i] = pack_half_to_unorm4x8(next.read(uint2(p)));
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    const int selfIdx = (int(lid.y) + kStabilityHalo) * kStabilityTileWidth + int(lid.x) + kStabilityHalo;
    prevColor = unpack_unorm4x8_to_half(tilePrevious[selfIdx]).rgb;
    nextColor = unpack_unorm4x8_to_half(tileNext[selfIdx]).rgb;

    // The change between the captures at a pixel is the largest difference of any channel, and its contrast the largest
    // difference of any channel to the pixel beside or above it, in either capture.
    for (uint i = linear; i < uint(kStabilityWindowWidth * kStabilityWindowHeight); i += threads) {
        const int center = (int(i) / kStabilityWindowWidth + 1) * kStabilityTileWidth + int(i) % kStabilityWindowWidth + 1;
        const half3 a = unpack_unorm4x8_to_half(tilePrevious[center]).rgb;
        const half3 b = unpack_unorm4x8_to_half(tileNext[center]).rgb;
        const half3 delta = abs(b - a);
        half contrast = 0.0h;
        const int neighbours[4] = { 1, -1, kStabilityTileWidth, -kStabilityTileWidth };
        for (int k = 0; k < 4; ++k) {
            const half3 da = abs(unpack_unorm4x8_to_half(tilePrevious[center + neighbours[k]]).rgb - a);
            const half3 db = abs(unpack_unorm4x8_to_half(tileNext[center + neighbours[k]]).rgb - b);
            const half3 largest = max(da, db);
            contrast = max(contrast, max(largest.r, max(largest.g, largest.b)));
        }
        window[i] = half2(max(delta.r, max(delta.g, delta.b)), contrast);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Separable 1x5 horizontal pass: reuse `tilePrevious` (532 uint slots >= 384 needed).
    for (uint i = linear; i < uint(MG_BLEND_TILE_WIDTH * kStabilityWindowHeight); i += threads) {
        const int hx = int(i) % MG_BLEND_TILE_WIDTH;
        const int hy = int(i) / MG_BLEND_TILE_WIDTH;
        const int base = hy * kStabilityWindowWidth + hx + 2;
        const half2 wm2 = window[base - 2];
        const half2 wm1 = window[base - 1];
        const half2 w0  = window[base];
        const half2 wp1 = window[base + 1];
        const half2 wp2 = window[base + 2];
        const half nearChange = max(wm1.x, max(w0.x, wp1.x));
        const half maxChange = max(max(wm2.x, wp2.x), nearChange);
        const half maxContrast = max(max(wm2.y, wp2.y), max(wm1.y, max(w0.y, wp1.y)));
        tilePrevious[i] = pack_half_to_unorm4x8(half4(maxChange, nearChange, maxContrast, 0.0h));
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Separable 5x1 vertical pass: 5 packed reads, zero branches.
    const int lx = int(lid.x);
    const int cy = int(lid.y) + 2;
    const half3 vm2 = unpack_unorm4x8_to_half(tilePrevious[(cy - 2) * MG_BLEND_TILE_WIDTH + lx]).xyz;
    const half3 vm1 = unpack_unorm4x8_to_half(tilePrevious[(cy - 1) * MG_BLEND_TILE_WIDTH + lx]).xyz;
    const half3 v0  = unpack_unorm4x8_to_half(tilePrevious[cy * MG_BLEND_TILE_WIDTH + lx]).xyz;
    const half3 vp1 = unpack_unorm4x8_to_half(tilePrevious[(cy + 1) * MG_BLEND_TILE_WIDTH + lx]).xyz;
    const half3 vp2 = unpack_unorm4x8_to_half(tilePrevious[(cy + 2) * MG_BLEND_TILE_WIDTH + lx]).xyz;

    const half nearestChange = max(vm1.y, max(v0.y, vp1.y));
    const half largestChange = max(max(vm2.x, vp2.x), max(vm1.x, max(v0.x, vp1.x)));
    const half largestContrast = max(max(vm2.z, vp2.z), max(vm1.z, max(v0.z, vp1.z)));

    // Where the captures are the same around the pixel they are the picture, whatever the contrast.
    if (nearestChange <= 1.0h / 255.0h) return 1.0h;
    const half remaining = saturate(1.0h - (largestChange / (largestContrast + half(params.noiseFloor))) / half(params.motion));
    return remaining * remaining;
}

/// The image shown: the generated one, and the captures' own mix at its phase in the share that `weight` gives it. Where
/// that is all of it the capture is taken as it is, not through a mix that could round it.
inline half3 stabilized(half3 image, half3 previous, half3 next, half weight, constant StabilityBlendParams& params) {
    const half3 captured = mix(previous, next, half(params.phase));
    return clampColor(weight >= 1.0h ? captured : mix(image, captured, weight));
}

/// MetalFX's image, blended with the captures it came from.
///
/// The Neural Engine's images are lossy where nothing moved: its model smooths what it is given, and the 4:2:0 it
/// works in drops the colour of thin coloured text. Scored against the real in-between frame, an interface drawn
/// identically in every frame came out at 21 dB, and a still scene was further from the truth than repeating a
/// capture. Where the captures are the same, the image between them is that same; where they differ by little it lies
/// between them, and the captures' own mix at this phase is closer than a re-drawn image; where they differ by a lot,
/// motion is what the engine is for, and its image stands untouched. MetalFX's images are close to the captures where
/// nothing moved already, and it takes the same rule with less motion allowed, for what it gains at an interface.
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
    threadgroup uint tilePrevious[kStabilityTileWidth * kStabilityTileHeight];
    threadgroup uint tileNext[kStabilityTileWidth * kStabilityTileHeight];
    threadgroup half2 window[kStabilityWindowWidth * kStabilityWindowHeight];
    half3 prevColor, nextColor;
    const half weight = stabilityWeight(previous, next, tilePrevious, tileNext, window, lid, tgid, params,
                                        prevColor, nextColor);
    if (gid.x >= generated.get_width() || gid.y >= generated.get_height()) return;

    if (weight >= 1.0h) {
        output.write(half4(clampColor(mix(prevColor, nextColor, half(params.phase))), 1.0h), gid);
        return;
    }
    output.write(half4(stabilized(generated.read(gid).rgb, prevColor, nextColor, weight, params), 1.0h), gid);
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
    threadgroup uint tilePrevious[kStabilityTileWidth * kStabilityTileHeight];
    threadgroup uint tileNext[kStabilityTileWidth * kStabilityTileHeight];
    threadgroup half2 window[kStabilityWindowWidth * kStabilityWindowHeight];
    half3 prevColor, nextColor;
    const half weight = stabilityWeight(previous, next, tilePrevious, tileNext, window, lid, tgid, params,
                                        prevColor, nextColor);
    if (gid.x >= luma.get_width() || gid.y >= luma.get_height()) return;

    if (weight >= 1.0h) {
        output.write(half4(clampColor(mix(prevColor, nextColor, half(params.phase))), 1.0h), gid);
        return;
    }

    // The chroma plane is half the size: sampled bilinearly rather than replicated, which would show as blocks of four.
    constexpr sampler linearSampler(filter::linear, address::clamp_to_edge, coord::normalized);
    const float2 uv = (float2(gid) + 0.5f) / float2(luma.get_width(), luma.get_height());
    const half3 image = yuvToRgb(luma.read(gid).r, half2(chroma.sample(linearSampler, uv).rg));
    output.write(half4(stabilized(image, prevColor, nextColor, weight, params), 1.0h), gid);
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
    threadgroup uint tilePrevious[kStabilityTileWidth * kStabilityTileHeight];
    threadgroup uint tileNext[kStabilityTileWidth * kStabilityTileHeight];
    threadgroup half2 window[kStabilityWindowWidth * kStabilityWindowHeight];
    half3 prevColor, nextColor;
    const half weight = stabilityWeight(previous, next, tilePrevious, tileNext, window, lid, tgid, params,
                                        prevColor, nextColor);
    if (gid.x >= previous.get_width() || gid.y >= previous.get_height()) return;

    if (weight >= 1.0h) {
        output.write(half4(clampColor(mix(prevColor, nextColor, half(params.phase))), 1.0h), gid);
        return;
    }

    constexpr sampler linearSampler(filter::linear, address::clamp_to_edge, coord::normalized);
    const float2 uv = (float2(gid) + 0.5f) / float2(previous.get_width(), previous.get_height());
    const half y = half(sampleCatmullRom(luma, linearSampler, uv, float2(luma.get_width(), luma.get_height())).x);
    const half3 image = yuvToRgb(y, half2(chroma.sample(linearSampler, uv).rg));
    output.write(half4(stabilized(image, prevColor, nextColor, weight, params), 1.0h), gid);
}
