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
    const half y = (luma * 255.0h - 16.0h) / 219.0h;
    const half2 cbcr = (chroma * 255.0h - 128.0h) / 224.0h;
    return clampColor(half3(y + 1.5748h * cbcr.y,
                            y - 0.1873h * cbcr.x - 0.4681h * cbcr.y,
                            y + 1.8556h * cbcr.x));
}

// Signed clamp is mandatory: `gid` is unsigned, so `gid.x - 1` wraps to 0xFFFFFFFF at
// the left/top edge and `max(0u, ...)` cannot recover it.
inline uint2 clampCoord(int2 p, uint width, uint height) {
    return uint2(clamp(p, int2(0), int2(int(width) - 1, int(height) - 1)));
}

#endif
