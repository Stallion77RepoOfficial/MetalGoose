// BGRA <-> 4:2:0 bi-planar conversion for the Neural Engine's frame interpolator, which takes
// and returns video-range YCbCr (BT.709) and no BGRA at all.

#include "ShaderCommon.h"

// BT.709 luma and chroma weights.
constant half3 kLumaWeights = half3(0.2126h, 0.7152h, 0.0722h);

/// BGRA -> luma plane (full resolution) and interleaved CbCr plane (half resolution).
///
/// One thread per 2x2 block: it reads the four pixels, writes their four luma values, and averages
/// the four to one chroma pair — the same box filter the chroma subsampling of any video encoder
/// applies. The dimensions must be even, which the caller guarantees by not offering this path to a
/// window that is not.
kernel void bgraTo420(
    texture2d<half, access::read> input [[texture(0)]],
    texture2d<half, access::write> luma [[texture(1)]],
    texture2d<half, access::write> chroma [[texture(2)]],
    uint2 gid [[thread_position_in_grid]]
) {
    if (gid.x >= chroma.get_width() || gid.y >= chroma.get_height()) return;

    const uint2 base = gid * 2;
    half3 average = half3(0.0h);
    for (uint dy = 0; dy < 2; ++dy) {
        for (uint dx = 0; dx < 2; ++dx) {
            const half3 rgb = input.read(base + uint2(dx, dy)).rgb;
            // Video range: black is 16/255 and white 235/255.
            luma.write(half4((16.0h + dot(rgb, kLumaWeights) * 219.0h) / 255.0h), base + uint2(dx, dy));
            average += rgb * 0.25h;
        }
    }

    const half y = dot(average, kLumaWeights);
    const half cb = (average.b - y) / 1.8556h;
    const half cr = (average.r - y) / 1.5748h;
    chroma.write(half4((128.0h + cb * 224.0h) / 255.0h, (128.0h + cr * 224.0h) / 255.0h, 0.0h, 1.0h), gid);
}

/// Luma plane and CbCr plane -> BGRA. The chroma plane is half the size, so it is sampled
/// bilinearly rather than replicated — a block of four pixels sharing one colour would show.
kernel void yuv420ToBgra(
    texture2d<half, access::read> luma [[texture(0)]],
    texture2d<half, access::sample> chroma [[texture(1)]],
    texture2d<half, access::write> output [[texture(2)]],
    uint2 gid [[thread_position_in_grid]]
) {
    const uint width = output.get_width();
    const uint height = output.get_height();
    if (gid.x >= width || gid.y >= height) return;

    constexpr sampler linearSampler(filter::linear, address::clamp_to_edge, coord::normalized);
    const float2 uv = (float2(gid) + 0.5f) / float2(width, height);

    const half y = (luma.read(gid).r * 255.0h - 16.0h) / 219.0h;
    const half2 cbcr = (half2(chroma.sample(linearSampler, uv).rg) * 255.0h - 128.0h) / 224.0h;
    const half3 rgb = half3(y + 1.5748h * cbcr.y,
                            y - 0.1873h * cbcr.x - 0.4681h * cbcr.y,
                            y + 1.8556h * cbcr.x);
    output.write(half4(clamp(rgb, 0.0h, 1.0h), 1.0h), gid);
}
