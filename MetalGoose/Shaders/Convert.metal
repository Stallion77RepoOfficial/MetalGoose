// BGRA -> 4:2:0 bi-planar conversion for the Neural Engine's frame interpolator, which takes and returns video-range
// YCbCr (BT.709) and no BGRA at all. The way back is part of the blend that shows its images (Stability.metal).

#include "ShaderCommon.h"

// BT.709 luma and chroma weights.
constant half3 kLumaWeights = half3(0.2126h, 0.7152h, 0.0722h);

/// One 2x2 block of BGRA pixels -> its four luma values and the one chroma pair of their average.
///
/// The average is the same box filter the chroma subsampling of any video encoder applies.
inline void writeBlock(const thread half3 (&rgb)[4], texture2d<half, access::write> luma,
                       texture2d<half, access::write> chroma, uint2 gid) {
    const uint2 base = gid * 2;
    half3 average = half3(0.0h);
    for (uint k = 0; k < 4; ++k) {
        // Video range: black is 16/255 and white 235/255.
        luma.write(half4((16.0h + dot(rgb[k], kLumaWeights) * 219.0h) / 255.0h), base + uint2(k & 1, k >> 1));
        average += rgb[k] * 0.25h;
    }

    const half y = dot(average, kLumaWeights);
    const half cb = (average.b - y) / 1.8556h;
    const half cr = (average.r - y) / 1.5748h;
    chroma.write(half4((128.0h + cb * 224.0h) / 255.0h, (128.0h + cr * 224.0h) / 255.0h, 0.0h, 1.0h), gid);
}

/// BGRA -> luma plane (full resolution) and interleaved CbCr plane (half resolution).
///
/// One thread per 2x2 block: it reads the four pixels, writes their four luma values, and averages
/// the four to one chroma pair. The dimensions must be even, which the caller guarantees by not offering this path to a
/// frame that is not.
kernel void bgraTo420(
    texture2d<half, access::read> input [[texture(0)]],
    texture2d<half, access::write> luma [[texture(1)]],
    texture2d<half, access::write> chroma [[texture(2)]],
    uint2 gid [[thread_position_in_grid]]
) {
    if (gid.x >= chroma.get_width() || gid.y >= chroma.get_height()) return;

    const uint2 base = gid * 2;
    half3 rgb[4];
    for (uint k = 0; k < 4; ++k) {
        rgb[k] = input.read(base + uint2(k & 1, k >> 1)).rgb;
    }
    writeBlock(rgb, luma, chroma, gid);
}

/// The mean of the frame over the area one of the planes' pixels covers, `pixel`, from `taps` bilinear fetches a side spread
/// evenly across it. A bilinear fetch between two pixels is their mean, so one fetch at the centre covers the whole area
/// where the frame is up to twice the planes' size on a side. Past that one fetch skips most of the area, and detail finer
/// than the planes aliases (stripes three pixels apart shrunk by three came out as no stripes at all, a black plane), so
/// there is a fetch for every two pixels of the area a side: an exact box at 4x, within a few percent of one in between.
inline half3 footprintMean(texture2d<half, access::sample> input, float2 pixel, float2 ratio, uint2 taps) {
    constexpr sampler linearSampler(filter::linear, address::clamp_to_edge, coord::normalized);
    const float2 inputSize = float2(input.get_width(), input.get_height());
    const float2 spacing = ratio / float2(taps);
    float3 sum = 0.0f;
    for (uint y = 0; y < taps.y; ++y) {
        for (uint x = 0; x < taps.x; ++x) {
            const float2 position = pixel * ratio + (float2(x, y) + 0.5f) * spacing;
            sum += float3(input.sample(linearSampler, position / inputSize).rgb);
        }
    }
    return half3(sum / float(taps.x * taps.y));
}

/// The same from a frame of another size: each of the planes' pixels is the mean of the frame over the area it covers
/// (`footprintMean`). This is how a frame the Neural Engine does not take whole is shrunk to what it takes, in the one pass
/// that converts it.
kernel void bgraTo420Resampled(
    texture2d<half, access::sample> input [[texture(0)]],
    texture2d<half, access::write> luma [[texture(1)]],
    texture2d<half, access::write> chroma [[texture(2)]],
    uint2 gid [[thread_position_in_grid]]
) {
    if (gid.x >= chroma.get_width() || gid.y >= chroma.get_height()) return;

    const float2 ratio = float2(input.get_width(), input.get_height()) / float2(luma.get_width(), luma.get_height());
    const uint2 taps = uint2(max(ceil(ratio * 0.5f), float2(1.0f)));
    const uint2 base = gid * 2;
    half3 rgb[4];
    for (uint k = 0; k < 4; ++k) {
        rgb[k] = footprintMean(input, float2(base + uint2(k & 1, k >> 1)), ratio, taps);
    }
    writeBlock(rgb, luma, chroma, gid);
}

// Content drawn at half a window's pixels and magnified to them, back at its own size, exactly (ContentResolution.swift).
// One thread per content pixel.

/// Nearest magnification made each content pixel a block of two by two.
kernel void demagnifyNearest(texture2d<half, access::read> magnified [[texture(0)]],
                             texture2d<half, access::write> content [[texture(1)]],
                             uint2 gid [[thread_position_in_grid]]) {
    if (gid.x >= content.get_width() || gid.y >= content.get_height()) return;
    content.write(magnified.read(gid * 2), gid);
}

/// Bilinear magnification, samples a quarter and three quarters of the way between content pixels. Across and down, each
/// neighbouring pair gives a content pixel exactly, 2·q[i] = 3·P[2i+1] − P[2i+2] = 3·P[2i] − P[2i−1], and the mean of the
/// two, (3·P[2i] + 3·P[2i+1] − P[2i−1] − P[2i+2]) / 4, carries half the rounding of either (measured: 98.25% of values
/// within 2 with one, largest difference 8). The first and last content pixels are the edge pixels themselves, where the
/// magnification clamps.
inline void bilinearTaps(uint i, uint n, thread uint (&index)[4], thread float (&weight)[4]) {
    if (i == 0 || i + 1 >= n) {
        index[0] = i == 0 ? 0 : 2 * n - 1;
        weight[0] = 1.0;
        for (uint k = 1; k < 4; ++k) { index[k] = index[0]; weight[k] = 0.0; }
        return;
    }
    index[0] = 2 * i - 1; index[1] = 2 * i; index[2] = 2 * i + 1; index[3] = 2 * i + 2;
    weight[0] = -0.25; weight[1] = 0.75; weight[2] = 0.75; weight[3] = -0.25;
}

kernel void demagnifyBilinear(texture2d<float, access::read> magnified [[texture(0)]],
                              texture2d<half, access::write> content [[texture(1)]],
                              uint2 gid [[thread_position_in_grid]]) {
    const uint2 size = uint2(content.get_width(), content.get_height());
    if (gid.x >= size.x || gid.y >= size.y) return;
    uint ix[4], iy[4];
    float wx[4], wy[4];
    bilinearTaps(gid.x, size.x, ix, wx);
    bilinearTaps(gid.y, size.y, iy, wy);
    float4 q = float4(0.0);
    for (uint b = 0; b < 4; ++b) {
        if (wy[b] == 0.0) continue;
        float4 row = float4(0.0);
        for (uint a = 0; a < 4; ++a) {
            if (wx[a] != 0.0) row += wx[a] * magnified.read(uint2(ix[a], iy[b]));
        }
        q += wy[b] * row;
    }
    content.write(half4(saturate(q)), gid);
}
