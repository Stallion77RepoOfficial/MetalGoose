#include <metal_stdlib>
using namespace metal;

struct VertexOut {
    float4 position [[position]];
    float2 texCoord;
};

inline half rgb2luma(half3 rgb) {
    return dot(rgb, half3(0.299h, 0.587h, 0.114h));
}

inline half3 clampColor(half3 color) {
    return clamp(color, half3(0.0h), half3(1.0h));
}

vertex VertexOut texture_vertex(uint vertexID [[vertex_id]]) {
    const float4 positions[4] = {
        float4(-1.0, -1.0, 0.0, 1.0),
        float4( 1.0, -1.0, 0.0, 1.0),
        float4(-1.0,  1.0, 0.0, 1.0),
        float4( 1.0,  1.0, 0.0, 1.0)
    };
    const float2 texCoords[4] = {
        float2(0.0, 1.0),
        float2(1.0, 1.0),
        float2(0.0, 0.0),
        float2(1.0, 0.0)
    };

    VertexOut out;
    out.position = positions[vertexID];
    out.texCoord = texCoords[vertexID];
    return out;
}

fragment half4 texture_fragment(
    VertexOut in [[stage_in]],
    texture2d<half> texture [[texture(0)]]
) {
    constexpr sampler s(filter::linear, address::clamp_to_edge);
    return texture.sample(s, in.texCoord);
}

struct CursorUniforms {
    float2 center;
    float2 size;
};

vertex VertexOut cursor_vertex(uint vertexID [[vertex_id]],
                                constant CursorUniforms& cursor [[buffer(0)]]) {
    const float2 offsets[4] = {
        float2(0.0, -1.0),
        float2(1.0, -1.0),
        float2(0.0,  0.0),
        float2(1.0,  0.0)
    };
    const float2 texCoords[4] = {
        float2(0.0, 1.0),
        float2(1.0, 1.0),
        float2(0.0, 0.0),
        float2(1.0, 0.0)
    };

    VertexOut out;
    float2 pos = cursor.center + offsets[vertexID] * cursor.size;
    out.position = float4(pos, 0.0, 1.0);
    out.texCoord = texCoords[vertexID];
    return out;
}

fragment half4 cursor_fragment(
    VertexOut in [[stage_in]],
    texture2d<half> texture [[texture(0)]]
) {
    constexpr sampler s(filter::linear, address::clamp_to_edge);
    return texture.sample(s, in.texCoord);
}


kernel void fxaa(
    texture2d<half, access::read> input [[texture(0)]],
    texture2d<half, access::write> output [[texture(1)]],
    constant float& threshold [[buffer(0)]],
    uint2 gid [[thread_position_in_grid]]
) {
    uint width = input.get_width();
    uint height = input.get_height();
    if (gid.x >= width || gid.y >= height) return;

    const half FXAA_REDUCE_MUL = 1.0h / 8.0h;
    const half FXAA_REDUCE_MIN = 1.0h / 128.0h;
    const half FXAA_SPAN_MAX = 8.0h;
    const half FXAA_EDGE_THRESHOLD_MIN = 1.0h / 24.0h;
    const half FXAA_SUBPIX = 0.75h;

    half3 rgbNW = input.read(uint2(max(0u, gid.x - 1), max(0u, gid.y - 1))).rgb;
    half3 rgbNE = input.read(uint2(min(width - 1, gid.x + 1), max(0u, gid.y - 1))).rgb;
    half3 rgbSW = input.read(uint2(max(0u, gid.x - 1), min(height - 1, gid.y + 1))).rgb;
    half3 rgbSE = input.read(uint2(min(width - 1, gid.x + 1), min(height - 1, gid.y + 1))).rgb;
    half3 rgbM = input.read(gid).rgb;

    half lumaNW = rgb2luma(rgbNW);
    half lumaNE = rgb2luma(rgbNE);
    half lumaSW = rgb2luma(rgbSW);
    half lumaSE = rgb2luma(rgbSE);
    half lumaM = rgb2luma(rgbM);

    half lumaMin = min(lumaM, min(min(lumaNW, lumaNE), min(lumaSW, lumaSE)));
    half lumaMax = max(lumaM, max(max(lumaNW, lumaNE), max(lumaSW, lumaSE)));
    half lumaRange = lumaMax - lumaMin;

    half edgeThreshold = max(FXAA_EDGE_THRESHOLD_MIN, lumaMax * half(threshold));
    if (lumaRange < edgeThreshold) {
        output.write(half4(rgbM, 1.0h), gid);
        return;
    }

    half2 dir;
    dir.x = -((lumaNW + lumaNE) - (lumaSW + lumaSE));
    dir.y = ((lumaNW + lumaSW) - (lumaNE + lumaSE));

    half dirReduce = max((lumaNW + lumaNE + lumaSW + lumaSE) * (0.25h * FXAA_REDUCE_MUL), FXAA_REDUCE_MIN);
    half rcpDirMin = 1.0h / (min(abs(dir.x), abs(dir.y)) + dirReduce);
    dir = min(half2(FXAA_SPAN_MAX), max(half2(-FXAA_SPAN_MAX), dir * rcpDirMin));

    int2 pos1 = int2(half2(gid) + dir * (1.0h / 3.0h - 0.5h));
    int2 pos2 = int2(half2(gid) + dir * (2.0h / 3.0h - 0.5h));
    pos1 = clamp(pos1, int2(0), int2(width - 1, height - 1));
    pos2 = clamp(pos2, int2(0), int2(width - 1, height - 1));

    half3 rgbA = (input.read(uint2(pos1)).rgb + input.read(uint2(pos2)).rgb) * 0.5h;

    int2 pos3 = int2(half2(gid) + dir * -0.5h);
    int2 pos4 = int2(half2(gid) + dir * 0.5h);
    pos3 = clamp(pos3, int2(0), int2(width - 1, height - 1));
    pos4 = clamp(pos4, int2(0), int2(width - 1, height - 1));

    half3 rgbB = rgbA * 0.5h + (input.read(uint2(pos3)).rgb + input.read(uint2(pos4)).rgb) * 0.25h;
    half lumaB = rgb2luma(rgbB);

    half3 edgeResult = (lumaB < lumaMin || lumaB > lumaMax) ? rgbA : rgbB;

    half3 lowpass = (rgbNW + rgbNE + rgbSW + rgbSE + rgbM) * 0.2h;
    half lumaLowpass = rgb2luma(lowpass);
    half subpix = clamp(abs(lumaLowpass - lumaM) / max(lumaRange, FXAA_REDUCE_MIN), 0.0h, 1.0h);
    subpix = subpix * subpix * FXAA_SUBPIX;

    half3 result = mix(edgeResult, lowpass, subpix);
    output.write(half4(result, 1.0h), gid);
}






struct SharpenParams {
    float sharpness;
};



struct AntiAliasParams {
    float threshold;
    int maxSearchSteps;
};


kernel void smaaEdgeDetection(
    texture2d<half, access::read> input [[texture(0)]],
    texture2d<half, access::write> edges [[texture(1)]],
    constant AntiAliasParams& params [[buffer(0)]],
    uint2 gid [[thread_position_in_grid]]
) {
    uint width = input.get_width();
    uint height = input.get_height();
    if (gid.x >= width || gid.y >= height) return;

    half threshold = half(params.threshold);

    half lumaC    = rgb2luma(input.read(gid).rgb);
    half lumaLeft = rgb2luma(input.read(uint2(max(0u, gid.x - 1), gid.y)).rgb);
    half lumaTop  = rgb2luma(input.read(uint2(gid.x, max(0u, gid.y - 1))).rgb);

    half2 delta;
    delta.x = abs(lumaC - lumaLeft);
    delta.y = abs(lumaC - lumaTop);

    half2 edge = step(threshold, delta);
    if (edge.x == 0.0h && edge.y == 0.0h) {
        edges.write(half4(0.0h, 0.0h, 0.0h, 1.0h), gid);
        return;
    }

    half lumaRight  = rgb2luma(input.read(uint2(min(width - 1, gid.x + 1), gid.y)).rgb);
    half lumaBottom = rgb2luma(input.read(uint2(gid.x, min(height - 1, gid.y + 1))).rgb);
    half lumaLeftLeft = rgb2luma(input.read(uint2(max(0u, gid.x - 2), gid.y)).rgb);
    half lumaTopTop   = rgb2luma(input.read(uint2(gid.x, max(0u, gid.y - 2))).rgb);

    half2 maxDelta;
    maxDelta.x = max(max(delta.x, abs(lumaC - lumaRight)), abs(lumaLeft - lumaLeftLeft));
    maxDelta.y = max(max(delta.y, abs(lumaC - lumaBottom)), abs(lumaTop - lumaTopTop));

    half finalDelta = max(maxDelta.x, maxDelta.y);
    edge *= step(finalDelta * 0.5h, delta);

    edges.write(half4(edge.x, edge.y, 0.0h, 1.0h), gid);
}


kernel void smaaBlendingWeights(
    texture2d<half, access::read> edges [[texture(0)]],
    texture2d<half, access::write> weights [[texture(1)]],
    constant AntiAliasParams& params [[buffer(0)]],
    uint2 gid [[thread_position_in_grid]]
) {
    uint width = edges.get_width();
    uint height = edges.get_height();
    if (gid.x >= width || gid.y >= height) return;

    half2 e = edges.read(gid).rg;

    if (e.x == 0.0h && e.y == 0.0h) {
        weights.write(half4(0.0h), gid);
        return;
    }


    half4 weight = half4(0.0h);

    if (e.x > 0.0h) {

        int leftDist = 0;
        int rightDist = 0;

        for (int i = 1; i <= params.maxSearchSteps; i++) {
            if (gid.x >= uint(i) && edges.read(uint2(gid.x - i, gid.y)).r > 0.0h)
                leftDist = i;
            else break;
        }

        for (int i = 1; i <= params.maxSearchSteps; i++) {
            if (gid.x + i < width && edges.read(uint2(gid.x + i, gid.y)).r > 0.0h)
                rightDist = i;
            else break;
        }

        half totalDist = half(leftDist + rightDist + 1);
        weight.r = half(leftDist) / totalDist;
        weight.g = half(rightDist) / totalDist;
    }

    if (e.y > 0.0h) {

        int upDist = 0;
        int downDist = 0;

        for (int i = 1; i <= params.maxSearchSteps; i++) {
            if (gid.y >= uint(i) && edges.read(uint2(gid.x, gid.y - i)).g > 0.0h)
                upDist = i;
            else break;
        }

        for (int i = 1; i <= params.maxSearchSteps; i++) {
            if (gid.y + i < height && edges.read(uint2(gid.x, gid.y + i)).g > 0.0h)
                downDist = i;
            else break;
        }

        half totalDist = half(upDist + downDist + 1);
        weight.b = half(upDist) / totalDist;
        weight.a = half(downDist) / totalDist;
    }

    weights.write(weight, gid);
}


kernel void smaaBlend(
    texture2d<half, access::read> input [[texture(0)]],
    texture2d<half, access::read> weights [[texture(1)]],
    texture2d<half, access::write> output [[texture(2)]],
    uint2 gid [[thread_position_in_grid]]
) {
    uint width = input.get_width();
    uint height = input.get_height();
    if (gid.x >= width || gid.y >= height) return;

    half4 w = weights.read(gid);
    half4 c = input.read(gid);


    if (w.r == 0.0h && w.g == 0.0h && w.b == 0.0h && w.a == 0.0h) {
        output.write(c, gid);
        return;
    }


    half4 result = c;

    if (w.r + w.g > 0.0h) {
        half4 left = input.read(uint2(max(0u, gid.x - 1), gid.y));
        half4 right = input.read(uint2(min(width - 1, gid.x + 1), gid.y));
        result = c * (1.0h - w.r - w.g) + left * w.r + right * w.g;
    }

    if (w.b + w.a > 0.0h) {
        half4 up = input.read(uint2(gid.x, max(0u, gid.y - 1)));
        half4 down = input.read(uint2(gid.x, min(height - 1, gid.y + 1)));
        result = result * (1.0h - w.b - w.a) + up * w.b + down * w.a;
    }

    output.write(result, gid);
}








kernel void copyTexture(
    texture2d<half, access::read> input [[texture(0)]],
    texture2d<half, access::write> output [[texture(1)]],
    uint2 gid [[thread_position_in_grid]]
) {
    uint width = output.get_width();
    uint height = output.get_height();
    if (gid.x >= width || gid.y >= height) return;

    output.write(input.read(gid), gid);
}


kernel void blitScaleBilinear(
    texture2d<half, access::read> input [[texture(0)]],
    texture2d<half, access::write> output [[texture(1)]],
    uint2 gid [[thread_position_in_grid]]
) {
    uint outWidth = output.get_width();
    uint outHeight = output.get_height();
    if (gid.x >= outWidth || gid.y >= outHeight) return;

    uint inWidth = input.get_width();
    uint inHeight = input.get_height();


    float2 srcPos = (float2(gid) + 0.5f) * float2(inWidth, inHeight) / float2(outWidth, outHeight) - 0.5f;
    srcPos = max(srcPos, float2(0.0f));

    uint2 srcBase = uint2(srcPos);
    float2 frac = srcPos - float2(srcBase);


    uint2 p00 = min(srcBase, uint2(inWidth - 1, inHeight - 1));
    uint2 p10 = min(srcBase + uint2(1, 0), uint2(inWidth - 1, inHeight - 1));
    uint2 p01 = min(srcBase + uint2(0, 1), uint2(inWidth - 1, inHeight - 1));
    uint2 p11 = min(srcBase + uint2(1, 1), uint2(inWidth - 1, inHeight - 1));


    half4 c00 = input.read(p00);
    half4 c10 = input.read(p10);
    half4 c01 = input.read(p01);
    half4 c11 = input.read(p11);

    half4 top = mix(c00, c10, half(frac.x));
    half4 bot = mix(c01, c11, half(frac.x));
    half4 result = mix(top, bot, half(frac.y));

    output.write(result, gid);
}


kernel void contrastAdaptiveSharpening(
    texture2d<half, access::read> input [[texture(0)]],
    texture2d<half, access::write> output [[texture(1)]],
    constant SharpenParams& params [[buffer(0)]],
    uint2 gid [[thread_position_in_grid]]
) {
    uint width = input.get_width();
    uint height = input.get_height();
    if (gid.x >= width || gid.y >= height) return;
    
    uint2 maxCoord = uint2(width - 1, height - 1);
    
    half3 a = input.read(uint2(max(0u, gid.x - 1), max(0u, gid.y - 1))).rgb;
    half3 b = input.read(uint2(gid.x, max(0u, gid.y - 1))).rgb;
    half3 c = input.read(uint2(min(maxCoord.x, gid.x + 1), max(0u, gid.y - 1))).rgb;
    half3 d = input.read(uint2(max(0u, gid.x - 1), gid.y)).rgb;
    half3 e = input.read(gid).rgb;
    half3 f = input.read(uint2(min(maxCoord.x, gid.x + 1), gid.y)).rgb;
    half3 g = input.read(uint2(max(0u, gid.x - 1), min(maxCoord.y, gid.y + 1))).rgb;
    half3 h = input.read(uint2(gid.x, min(maxCoord.y, gid.y + 1))).rgb;
    half3 i = input.read(uint2(min(maxCoord.x, gid.x + 1), min(maxCoord.y, gid.y + 1))).rgb;
    
    half3 minRGB = min(min(min(d, e), min(f, b)), h);
    half3 maxRGB = max(max(max(d, e), max(f, b)), h);
    
    minRGB = min(min(min(minRGB, a), min(c, g)), i);
    maxRGB = max(max(max(maxRGB, a), max(c, g)), i);
    
    half3 contrast = maxRGB - minRGB;
    half3 ampFactor = saturate(1.0h - contrast * 2.0h);
    
    half sharpness = half(params.sharpness);
    half3 weight = ampFactor * sharpness;
    
    half3 blur = (a + b + c + d + f + g + h + i) / 8.0h;
    half3 sharpened = e + (e - blur) * weight;
    
    sharpened = clamp(sharpened, minRGB, maxRGB);
    
    output.write(half4(clampColor(sharpened), 1.0h), gid);
}
