#include <metal_stdlib>
using namespace metal;

// Helper: Mitchell-Netravali cubic filter (B=1/3, C=1/3)
inline float mnWeight(float x) {
    x = fabs(x);
    const float B = 1.0f/3.0f;
    const float C = 1.0f/3.0f;
    if (x < 1.0f) {
        return ((12.0f - 9.0f*B - 6.0f*C)*(x*x*x) + (-18.0f + 12.0f*B + 6.0f*C)*(x*x) + (6.0f - 2.0f*B)) / 6.0f;
    } else if (x < 2.0f) {
        return ((-B - 6.0f*C)*(x*x*x) + (6.0f*B + 30.0f*C)*(x*x) + (-12.0f*B - 48.0f*C)*x + (8.0f*B + 24.0f*C)) / 6.0f;
    } else {
        return 0.0f;
    }
}

// Nearest-neighbor integer upscale
kernel void integer_nearest_upscale(
    texture2d<float, access::read>  src [[texture(0)]],
    texture2d<float, access::write> dst [[texture(1)]],
    uint2 gid [[thread_position_in_grid]])
{
    if (gid.x >= dst.get_width() || gid.y >= dst.get_height()) return;
    const uint srcW = src.get_width();
    const uint srcH = src.get_height();
    const uint dstW = dst.get_width();
    const uint dstH = dst.get_height();

    float2 outPix = float2(gid) + 0.5f;
    float2 scale = float2(srcW, srcH) / float2(dstW, dstH);
    float2 srcF = outPix * scale - 0.5f;
    int2  srcI = int2(round(srcF));
    srcI = clamp(srcI, int2(0), int2((int)srcW - 1, (int)srcH - 1));

    float4 c = src.read(uint2(srcI));
    dst.write(c, gid);
}

// Bicubic upscale using Mitchell-Netravali
kernel void bicubic_upscale(
    texture2d<float, access::sample> src [[texture(0)]],
    texture2d<float, access::write>  dst [[texture(1)]],
    uint2 gid [[thread_position_in_grid]])
{
    if (gid.x >= dst.get_width() || gid.y >= dst.get_height()) return;
    const float2 srcSize = float2(src.get_width(), src.get_height());
    const float2 dstSize = float2(dst.get_width(), dst.get_height());

    constexpr sampler s(address::clamp_to_edge, filter::nearest);

    float2 pos = (float2(gid) + 0.5f) * (srcSize / dstSize) - 0.5f;
    float2 base = floor(pos);
    float2 f = pos - base;

    float wx[4];
    float wy[4];
    wx[0] = mnWeight(1.0f + (1.0f - f.x));
    wx[1] = mnWeight(1.0f - f.x);
    wx[2] = mnWeight(f.x);
    wx[3] = mnWeight(1.0f + f.x);
    wy[0] = mnWeight(1.0f + (1.0f - f.y));
    wy[1] = mnWeight(1.0f - f.y);
    wy[2] = mnWeight(f.y);
    wy[3] = mnWeight(1.0f + f.y);

    float4 sum = float4(0.0f);
    float wsum = 0.0f;
    for (int j = -1; j <= 2; ++j) {
        for (int i = -1; i <= 2; ++i) {
            float2 p = base + float2(i, j);
            float2 uv = (p + 0.5f) / srcSize;
            float w = wx[i+1] * wy[j+1];
            sum += w * src.sample(s, uv);
            wsum += w;
        }
    }
    float4 c = (wsum > 0.0f) ? sum / wsum : sum;
    dst.write(c, gid);
}

// Simple bilinear upscale (EASU stub)
kernel void easu_stub(
    texture2d<float, access::sample> src [[texture(0)]],
    texture2d<float, access::write>  dst [[texture(1)]],
    uint2 gid [[thread_position_in_grid]])
{
    if (gid.x >= dst.get_width() || gid.y >= dst.get_height()) return;
    constexpr sampler s(address::clamp_to_edge, filter::linear);
    float2 uv = (float2(gid) + 0.5f) / float2(dst.get_width(), dst.get_height());
    float4 c = src.sample(s, uv);
    dst.write(c, gid);
}

// RCAS-like sharpen with edge awareness
kernel void rcas_stub(
    texture2d<float, access::read>  src [[texture(0)]],
    texture2d<float, access::write> dst [[texture(1)]],
    constant float &amount           [[buffer(0)]],
    uint2 gid                        [[thread_position_in_grid]])
{
    if (gid.x >= dst.get_width() || gid.y >= dst.get_height()) return;
    int2 g = int2(gid);
    int2 size = int2((int)src.get_width(), (int)src.get_height());

    auto clampPix = [&](int2 p)->int2 { return clamp(p, int2(0), size - 1); };

    float4 c  = src.read(uint2(clampPix(g)));
    float4 l  = src.read(uint2(clampPix(g + int2(-1,  0))));
    float4 r  = src.read(uint2(clampPix(g + int2( 1,  0))));
    float4 t  = src.read(uint2(clampPix(g + int2( 0, -1))));
    float4 b  = src.read(uint2(clampPix(g + int2( 0,  1))));

    // 5-tap blur for base detail
    float4 blur = (l + r + t + b + c) / 5.0f;
    float4 detail = c - blur;

    // Edge-aware scaling: use luma gradients to modulate sharpening to reduce ringing
    float3 w = float3(0.299f, 0.587f, 0.114f);
    float lc = dot(c.rgb, w);
    float ll = dot(l.rgb, w);
    float lr = dot(r.rgb, w);
    float lt = dot(t.rgb, w);
    float lb = dot(b.rgb, w);

    float dx = 0.5f * (lr - ll);
    float dy = 0.5f * (lb - lt);
    float edge = sqrt(dx*dx + dy*dy); // edge magnitude

    float a = clamp(amount, 0.0f, 1.0f);
    // Reduce sharpening near strong edges to avoid halos
    float edgeFactor = 1.0f / (1.0f + 2.0f * edge);

    float4 sharp = clamp(c + (a * edgeFactor) * detail, 0.0f, 1.0f);
    sharp.a = c.a;
    dst.write(sharp, gid);
}

// Optical-flow-based warp and blend (t=0.5 interpolation)
// Inputs:
//  src0: previous frame (t=0)
//  src1: current  frame (t=1)
//  flow: RG32F flow from prev->curr in pixels
// Output:
//  dst:  interpolated frame at t=0.5
kernel void flow_warp(
    texture2d<float,  access::sample> src0 [[texture(0)]],
    texture2d<float,  access::sample> src1 [[texture(1)]],
    texture2d<float,  access::read>   flow [[texture(2)]],
    texture2d<float,  access::write>  dst  [[texture(3)]],
    uint2 gid [[thread_position_in_grid]])
{
    if (gid.x >= dst.get_width() || gid.y >= dst.get_height()) return;

    constexpr sampler s(address::clamp_to_edge, filter::linear);
    const float2 outSize = float2(dst.get_width(), dst.get_height());

    // Read flow at this pixel (prev -> curr), in pixels
    float4 flowSample = flow.read(gid);
    float2 f = flowSample.xy;

    // Interpolate at t=0.5 (halfway): sample prev backwards and curr forwards
    float2 uvPrev = (float2(gid) + 0.5f - 0.5f * f) / outSize;
    float2 uvCurr = (float2(gid) + 0.5f + 0.5f * f) / outSize;

    float4 c0 = src0.sample(s, uvPrev);
    float4 c1 = src1.sample(s, uvCurr);
    float4 out = 0.5f * (c0 + c1);
    out.a = 1.0f;
    dst.write(out, gid);
}

// Consistency-aware optical-flow warp (uses forward and backward flows)
kernel void flow_warp_consistent(
    texture2d<float,  access::sample> src0 [[texture(0)]],
    texture2d<float,  access::sample> src1 [[texture(1)]],
    texture2d<float,  access::read>   flowFwd [[texture(2)]],
    texture2d<float,  access::sample>   flowBwd [[texture(3)]],
    texture2d<float,  access::write>  dst    [[texture(4)]],
    constant float &t                         [[buffer(0)]],
    constant float &consistencyThreshold      [[buffer(1)]],
    uint2 gid [[thread_position_in_grid]])
{
    if (gid.x >= dst.get_width() || gid.y >= dst.get_height()) return;

    constexpr sampler s(address::clamp_to_edge, filter::linear);
    const float2 outSize = float2(dst.get_width(), dst.get_height());

    // Forward flow at this pixel (prev->curr)
    float2 f = flowFwd.read(gid).xy;

    // Positions at t
    float2 uvPrev = (float2(gid) + 0.5f - t * f) / outSize;
    float2 uvCurr = (float2(gid) + 0.5f + (1.0f - t) * f) / outSize;

    // Sample backward flow at the current location to test consistency
    float2 b = flowBwd.sample(s, uvCurr).xy;
    float2 consistency = f + b; // should be ~0 for consistent motion
    float incons = length(consistency);

    float4 c0 = src0.sample(s, uvPrev);
    float4 c1 = src1.sample(s, uvCurr);

    float4 interp = mix(c0, c1, t);

    // If inconsistent (likely occlusion), bias toward current frame
    float thr = max(1e-3f, consistencyThreshold);
    float w = smoothstep(thr, 2.0f * thr, incons);
    float4 out = mix(interp, c1, w);
    out.a = 1.0f;
    dst.write(out, gid);
}

// Convert float (e.g., RGBA16F) to unorm (e.g., BGRA8)
kernel void convert_float_to_unorm(
    texture2d<float, access::sample> src [[texture(0)]],
    texture2d<float, access::write>  dst [[texture(1)]],
    uint2 gid [[thread_position_in_grid]])
{
    if (gid.x >= dst.get_width() || gid.y >= dst.get_height()) return;
    constexpr sampler s(address::clamp_to_edge, filter::nearest);
    float2 uv = (float2(gid) + 0.5f) / float2(dst.get_width(), dst.get_height());
    float4 c = src.sample(s, uv);
    c = clamp(c, 0.0f, 1.0f);
    dst.write(c, gid);
}

// Lanczos2 upscale (windowed sinc with a=2)
inline float lanczos(float x, float a) {
    x = fabs(x);
    if (x < 1e-5f) return 1.0f;
    if (x >= a) return 0.0f;
    float pix = M_PI_F * x;
    return (sin(pix) / pix) * (sin(pix / a) / (pix / a));
}

kernel void lanczos2_upscale(
    texture2d<float, access::sample> src [[texture(0)]],
    texture2d<float, access::write>  dst [[texture(1)]],
    uint2 gid [[thread_position_in_grid]])
{
    if (gid.x >= dst.get_width() || gid.y >= dst.get_height()) return;
    constexpr sampler s(address::clamp_to_edge, filter::nearest);

    float2 srcSize = float2(src.get_width(), src.get_height());
    float2 dstSize = float2(dst.get_width(), dst.get_height());
    float2 pos = (float2(gid) + 0.5f) * (srcSize / dstSize) - 0.5f;

    int2 base = int2(floor(pos));
    float2 f = pos - floor(pos);

    const int a = 2; // Lanczos2
    float4 sum = float4(0.0);
    float wsum = 0.0f;
    for (int j = -a*2 + 1; j <= a*2; ++j) {
        for (int i = -a*2 + 1; i <= a*2; ++i) {
            float wx = lanczos((float(i) - f.x), (float)a);
            float wy = lanczos((float(j) - f.y), (float)a);
            float w = wx * wy;
            int2 p = int2(clamp(base + int2(i, j), int2(0), int2(int(srcSize.x)-1, int(srcSize.y)-1)));
            float2 uv = (float2(p) + 0.5f) / srcSize;
            sum += w * src.sample(s, uv);
            wsum += w;
        }
    }
    float4 c = (wsum > 0.0f) ? sum / wsum : sum;
    dst.write(clamp(c, 0.0f, 1.0f), gid);
}

// Clear texture to zero
kernel void clear_texture(
    texture2d<float, access::write> dst [[texture(0)]],
    uint2 gid [[thread_position_in_grid]])
{
    if (gid.x >= dst.get_width() || gid.y >= dst.get_height()) return;
    dst.write(float4(0.0, 0.0, 0.0, 1.0), gid);
}

