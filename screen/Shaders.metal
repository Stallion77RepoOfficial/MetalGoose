#include <metal_stdlib>
using namespace metal;

// 1. Frame Generation: Optik Akış Warp Çekirdeği
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

    float2 uv0 = uv - (v * t) / size;
    float2 uv1 = uv + (v * (1.0 - t)) / size;

    float4 c0 = src0.sample(s, uv0);
    float4 c1 = src1.sample(s, uv1);

    float4 finalColor = mix(c0, c1, t);
    finalColor.a = 1.0;

    dst.write(finalColor, gid);
}

// 2. Basit Çizim (Passthrough - Vertex/Fragment)
struct VertexOut {
    float4 position [[position]];
    float2 texCoord;
};

vertex VertexOut texture_vertex(uint vertexID [[vertex_id]]) {
    const float4 positions[4] = {
        float4(-1, -1, 0, 1), float4( 1, -1, 0, 1),
        float4(-1,  1, 0, 1), float4( 1,  1, 0, 1)
    };
    const float2 coords[4] = {
        float2(0, 1), float2(1, 1),
        float2(0, 0), float2(1, 0)
    };
    
    VertexOut out;
    out.position = positions[vertexID];
    out.texCoord = coords[vertexID];
    return out;
}

fragment float4 texture_fragment(VertexOut in [[stage_in]],
                                 texture2d<float> texture [[texture(0)]]) {
    constexpr sampler s(address::clamp_to_edge, filter::linear);
    return texture.sample(s, in.texCoord);
}
