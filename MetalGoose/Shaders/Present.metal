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

// For content an app drew at half the window's pixels and showed as blocks of two by two: shown the same way again.
fragment half4 present_fragment_nearest(PresentVertex in [[stage_in]],
                                        texture2d<half> image [[texture(0)]]) {
    constexpr sampler s(filter::nearest, address::clamp_to_edge);
    return image.sample(s, in.texCoord);
}

// An image drawn over the picture in the drawable, the HUD or the pointer: `rect` is where it goes, its left, top, right
// and bottom edges as fractions of the target from the top-left corner. The image is premultiplied, and blended over what
// is there.

struct SpriteVertex {
    float4 position [[position]];
    float2 texCoord;
};

vertex SpriteVertex sprite_vertex(uint vertexID [[vertex_id]], constant float4 &rect [[buffer(0)]]) {
    const float2 corners[4] = { float2(0.0, 1.0), float2(1.0, 1.0), float2(0.0, 0.0), float2(1.0, 0.0) };
    const float2 corner = corners[vertexID];
    const float2 place = mix(rect.xy, rect.zw, corner);

    SpriteVertex out;
    out.position = float4(place.x * 2.0 - 1.0, 1.0 - place.y * 2.0, 0.0, 1.0);
    out.texCoord = corner;
    return out;
}

fragment half4 sprite_fragment(SpriteVertex in [[stage_in]],
                               texture2d<half> image [[texture(0)]]) {
    constexpr sampler s(filter::linear, address::clamp_to_edge);
    return image.sample(s, in.texCoord);
}
