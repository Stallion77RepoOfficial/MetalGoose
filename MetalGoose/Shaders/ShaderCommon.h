// Metal-only helpers shared by every kernel (this header is included from .metal files and
// never from Swift, which only sees ShaderTypes.h).

#ifndef MGShaderCommon_h
#define MGShaderCommon_h

#include <metal_stdlib>
using namespace metal;

#include "ShaderTypes.h"

inline half rgb2luma(half3 rgb) {
    return dot(rgb, half3(0.299h, 0.587h, 0.114h));
}

inline half3 clampColor(half3 color) {
    return clamp(color, half3(0.0h), half3(1.0h));
}

/// Video-range BT.709 luma and chroma (CbCr, as the two channels of the interleaved plane) -> RGB: the inverse of what
/// `bgraTo420` writes, and what the Neural Engine's images are in.
inline half3 yuvToRgb(half luma, half2 chroma) {
    const float y = fma(float(luma), 255.0f / 219.0f, -16.0f / 219.0f);
    const float2 cbcr = fma(float2(chroma), 255.0f / 224.0f, -128.0f / 224.0f);
    return clampColor(half3(
        half(fma(1.5748f, cbcr.y, y)),
        half(fma(-0.4681f, cbcr.y, fma(-0.1873f, cbcr.x, y))),
        half(fma(1.8556f, cbcr.x, y))
    ));
}

// Signed clamp is mandatory: `gid` is unsigned, so `gid.x - 1` wraps to 0xFFFFFFFF at
// the left/top edge and `max(0u, ...)` cannot recover it.
inline uint2 clampCoord(int2 p, uint width, uint height) {
    return uint2(clamp(p, int2(0), int2(int(width) - 1, int(height) - 1)));
}

#endif
