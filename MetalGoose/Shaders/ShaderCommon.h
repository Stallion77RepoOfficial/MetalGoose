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

// Signed clamp is mandatory: `gid` is unsigned, so `gid.x - 1` wraps to 0xFFFFFFFF at
// the left/top edge and `max(0u, ...)` cannot recover it.
inline uint2 clampCoord(int2 p, uint width, uint height) {
    return uint2(clamp(p, int2(0), int2(int(width) - 1, int(height) - 1)));
}

#endif
