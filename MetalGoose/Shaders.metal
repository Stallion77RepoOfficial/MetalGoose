#include <metal_stdlib>
using namespace metal;

// MARK: - Helper Functions
inline float mnWeight(float x) {
    x = fabs(x);
    const float B = 1.0f/3.0f;
    const float C = 1.0f/3.0f;
    if (x < 1.0f) return ((12.0f - 9.0f*B - 6.0f*C)*(x*x*x) + (-18.0f + 12.0f*B + 6.0f*C)*(x*x) + (6.0f - 2.0f*B)) / 6.0f;
    else if (x < 2.0f) return ((-B - 6.0f*C)*(x*x*x) + (6.0f*B + 30.0f*C)*(x*x) + (-12.0f*B - 48.0f*C)*x + (8.0f*B + 24.0f*C)) / 6.0f;
    return 0.0f;
}

// MARK: - 1. Frame Generation (Optical Flow Warp)
kernel void flow_warp(
    texture2d<float, access::sample> src0 [[texture(0)]],
    texture2d<float, access::sample> src1 [[texture(1)]],
    texture2d<float, access::sample> flow [[texture(2)]],
    texture2d<float, access::write>  dst  [[texture(3)]],
    constant float &t                     [[buffer(0)]],
    uint2 gid [[thread_position_in_grid]])
{
    if (gid.x >= dst.get_width() || gid.y >= dst.get_height()) return;

    constexpr sampler s(address::clamp_to_edge, filter::linear);
    float2 size = float2(dst.get_width(), dst.get_height());
    float2 uv = (float2(gid) + 0.5f) / size;

    float4 fVal = flow.sample(s, uv);
    float2 v = fVal.rg;

    // Simple Bidirectional Warp
    float2 uv0 = uv - (v * t) / size;
    float2 uv1 = uv + (v * (1.0 - t)) / size;

    float4 c0 = src0.sample(s, uv0);
    float4 c1 = src1.sample(s, uv1);

    // Mixing
    float4 finalColor = mix(c0, c1, t);
    finalColor.a = 1.0;

    dst.write(finalColor, gid);
}

// MARK: - 2. Integer Scaling (Pixel Art)
kernel void integer_scale(
    texture2d<float, access::sample> src [[texture(0)]],
    texture2d<float, access::write> dst [[texture(1)]],
    constant float &scaleFactor [[buffer(0)]],
    uint2 gid [[thread_position_in_grid]])
{
    if (gid.x >= dst.get_width() || gid.y >= dst.get_height()) return;
    
    // True integer scaling: map destination pixel to source pixel grid using scaleFactor
    constexpr sampler s(address::clamp_to_edge, filter::nearest);
    
    float2 srcSize = float2(src.get_width(), src.get_height());
    float2 dstPx = float2(gid);
    float2 srcPx = floor(dstPx / scaleFactor);
    // Clamp to valid source range to avoid sampling outside
    srcPx = clamp(srcPx, float2(0.0, 0.0), float2(srcSize.x - 1.0, srcSize.y - 1.0));
    float2 uv = (srcPx + 0.5f) / srcSize;
    float4 color = src.sample(s, uv);
    dst.write(color, gid);
}

// MARK: - 3. Bicubic Scaling (Soft)
kernel void bicubic_scale(
    texture2d<float, access::sample> src [[texture(0)]],
    texture2d<float, access::write> dst [[texture(1)]],
    uint2 gid [[thread_position_in_grid]])
{
    if (gid.x >= dst.get_width() || gid.y >= dst.get_height()) return;
    
    float2 srcSize = float2(src.get_width(), src.get_height());
    // Manual filtering
    constexpr sampler s(address::clamp_to_edge, filter::nearest);

    float2 pos = (float2(gid) + 0.5f) * (srcSize / float2(dst.get_width(), dst.get_height())) - 0.5f;
    float2 base = floor(pos);
    float2 f = pos - base;

    float4 wx, wy;
    wx.x = mnWeight(1.0f + f.x); wx.y = mnWeight(f.x); wx.z = mnWeight(1.0f - f.x); wx.w = mnWeight(2.0f - f.x);
    wy.x = mnWeight(1.0f + f.y); wy.y = mnWeight(f.y); wy.z = mnWeight(1.0f - f.y); wy.w = mnWeight(2.0f - f.y);

    float4 sum = float4(0.0f);
    float wsum = 0.0f;
    
    for (int j = -1; j <= 2; ++j) {
        for (int i = -1; i <= 2; ++i) {
            float2 uv = (base + float2(i, j) + 0.5f) / srcSize;
            // 4x4 weight calculation
            float w = ((i==-1)?wx.x:(i==0)?wx.y:(i==1)?wx.z:wx.w) * ((j==-1)?wy.x:(j==0)?wy.y:(j==1)?wy.z:wy.w);
            sum += w * src.sample(s, uv);
            wsum += w;
        }
    }
    dst.write(sum / (wsum > 0.0 ? wsum : 1.0), gid);
}

// MARK: - 4. Render Pipeline (Display)
struct VertexOut {
    float4 position [[position]];
    float2 texCoord;
};

vertex VertexOut texture_vertex(uint vertexID [[vertex_id]]) {
    const float4 positions[4] = { float4(-1, -1, 0, 1), float4( 1, -1, 0, 1), float4(-1,  1, 0, 1), float4( 1,  1, 0, 1) };
    const float2 coords[4] = { float2(0, 1), float2(1, 1), float2(0, 0), float2(1, 0) };
    VertexOut out;
    out.position = positions[vertexID];
    out.texCoord = coords[vertexID];
    return out;
}

fragment float4 texture_fragment(VertexOut in [[stage_in]], texture2d<float> texture [[texture(0)]]) {
    constexpr sampler s(address::clamp_to_edge, filter::linear);
    return texture.sample(s, in.texCoord);
}
