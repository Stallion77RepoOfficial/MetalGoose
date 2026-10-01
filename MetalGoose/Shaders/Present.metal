// Fullscreen draw used when the image has to be resampled on its way into the
// drawable. A same-size image goes in with a blit instead, which needs no shader.

#include "ShaderCommon.h"

struct PresentVertex {
    float4 position [[position]];
    float2 texCoord;
};

vertex PresentVertex present_vertex(uint vertexID [[vertex_id]]) {
    const float2 positions[4] = { float2(-1.0, -1.0), float2(1.0, -1.0), float2(-1.0, 1.0), float2(1.0, 1.0) };
    const float2 texCoords[4] = { float2(0.0, 1.0), float2(1.0, 1.0), float2(0.0, 0.0), float2(1.0, 0.0) };

    PresentVertex out;
    out.position = float4(positions[vertexID], 0.0, 1.0);
    out.texCoord = texCoords[vertexID];
    return out;
}

fragment half4 present_fragment(PresentVertex in [[stage_in]],
                                texture2d<half> image [[texture(0)]]) {
    constexpr sampler s(filter::linear, address::clamp_to_edge);
    return image.sample(s, in.texCoord);
}
