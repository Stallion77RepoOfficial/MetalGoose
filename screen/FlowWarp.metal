#include <metal_stdlib>
using namespace metal;

kernel void flowWarpKernel(
    texture2d<float, access::sample> prevTex [[texture(0)]],
    texture2d<float, access::sample> curTex [[texture(1)]],
    texture2d<float, access::sample> flowTex [[texture(2)]], // rg32f: (dx, dy)
    texture2d<float, access::write> outTex [[texture(3)]],
    constant float &t [[buffer(0)]],
    uint2 gid [[thread_position_in_grid]]) {

    if (gid.x >= outTex.get_width() || gid.y >= outTex.get_height()) return;

    constexpr sampler s(address::clamp_to_edge, filter::linear);

    float2 uv = float2(gid) / float2(outTex.get_width(), outTex.get_height());
    float2 flow = flowTex.sample(s, uv).rg; // flow in pixels

    // Convert flow (pixels) to UV offsets
    float2 texSize = float2(outTex.get_width(), outTex.get_height());
    float2 uvOffset = flow / texSize;

    float2 prevUV = uv - uvOffset * t; // move backwards along flow for prev
    float2 curUV  = uv + uvOffset * (1.0 - t); // forward for current

    float4 prevColor = prevTex.sample(s, prevUV);
    float4 curColor  = curTex.sample(s, curUV);

    float4 color = mix(prevColor, curColor, t);
    outTex.write(color, gid);
}
