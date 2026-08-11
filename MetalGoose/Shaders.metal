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

// Signed clamp is mandatory here: gid is uint, so gid.x - 1 wraps to
// 0xFFFFFFFF at the left/top edge and max(0u, ...) cannot recover it.
inline uint2 clampCoord(int2 p, uint width, uint height) {
    return uint2(clamp(p, int2(0), int2(int(width) - 1, int(height) - 1)));
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

    int2 p = int2(gid);
    half3 rgbNW = input.read(clampCoord(p + int2(-1, -1), width, height)).rgb;
    half3 rgbNE = input.read(clampCoord(p + int2( 1, -1), width, height)).rgb;
    half3 rgbSW = input.read(clampCoord(p + int2(-1,  1), width, height)).rgb;
    half3 rgbSE = input.read(clampCoord(p + int2( 1,  1), width, height)).rgb;
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

    uint2 pos1 = clampCoord(int2(half2(gid) + dir * (1.0h / 3.0h - 0.5h)), width, height);
    uint2 pos2 = clampCoord(int2(half2(gid) + dir * (2.0h / 3.0h - 0.5h)), width, height);

    half3 rgbA = (input.read(pos1).rgb + input.read(pos2).rgb) * 0.5h;

    uint2 pos3 = clampCoord(int2(half2(gid) + dir * -0.5h), width, height);
    uint2 pos4 = clampCoord(int2(half2(gid) + dir * 0.5h), width, height);

    half3 rgbB = rgbA * 0.5h + (input.read(pos3).rgb + input.read(pos4).rgb) * 0.25h;
    half lumaB = rgb2luma(rgbB);

    half3 edgeResult = (lumaB < lumaMin || lumaB > lumaMax) ? rgbA : rgbB;

    half3 lowpass = (rgbNW + rgbNE + rgbSW + rgbSE + rgbM) * 0.2h;
    half lumaLowpass = rgb2luma(lowpass);
    half subpix = clamp(abs(lumaLowpass - lumaM) / max(lumaRange, FXAA_REDUCE_MIN), 0.0h, 1.0h);
    subpix = subpix * subpix * FXAA_SUBPIX;

    half3 result = mix(edgeResult, lowpass, subpix);
    output.write(half4(result, 1.0h), gid);
}


// Mirrored in GooseEngine.swift — keep field order and types in sync.
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

    int2 p = int2(gid);
    half lumaC    = rgb2luma(input.read(gid).rgb);
    half lumaLeft = rgb2luma(input.read(clampCoord(p + int2(-1, 0), width, height)).rgb);
    half lumaTop  = rgb2luma(input.read(clampCoord(p + int2(0, -1), width, height)).rgb);

    half2 delta;
    delta.x = abs(lumaC - lumaLeft);
    delta.y = abs(lumaC - lumaTop);

    half2 edge = step(threshold, delta);
    if (edge.x == 0.0h && edge.y == 0.0h) {
        edges.write(half4(0.0h, 0.0h, 0.0h, 1.0h), gid);
        return;
    }

    half lumaRight  = rgb2luma(input.read(clampCoord(p + int2( 1, 0), width, height)).rgb);
    half lumaBottom = rgb2luma(input.read(clampCoord(p + int2(0,  1), width, height)).rgb);
    half lumaLeftLeft = rgb2luma(input.read(clampCoord(p + int2(-2, 0), width, height)).rgb);
    half lumaTopTop   = rgb2luma(input.read(clampCoord(p + int2(0, -2), width, height)).rgb);

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

    half hSum = w.r + w.g;
    if (hSum > 0.5h) { half s = 0.5h / hSum; w.r *= s; w.g *= s; hSum = 0.5h; }
    half vSum = w.b + w.a;
    if (vSum > 0.5h) { half s = 0.5h / vSum; w.b *= s; w.a *= s; vSum = 0.5h; }

    int2 p = int2(gid);

    if (hSum > 0.0h) {
        half4 left = input.read(clampCoord(p + int2(-1, 0), width, height));
        half4 right = input.read(clampCoord(p + int2( 1, 0), width, height));
        result = c * (1.0h - hSum) + left * w.r + right * w.g;
    }

    if (vSum > 0.0h) {
        half4 up = input.read(clampCoord(p + int2(0, -1), width, height));
        half4 down = input.read(clampCoord(p + int2(0,  1), width, height));
        result = result * (1.0h - vSum) + up * w.b + down * w.a;
    }

    output.write(result, gid);
}


/// Feeds VTMotionEstimation, which takes single-component luma rather than BGRA.
kernel void bgraToLuma(
    texture2d<half, access::read> input [[texture(0)]],
    texture2d<half, access::write> output [[texture(1)]],
    uint2 gid [[thread_position_in_grid]]
) {
    uint width = output.get_width();
    uint height = output.get_height();
    if (gid.x >= width || gid.y >= height) return;

    uint2 src = clampCoord(int2(gid), input.get_width(), input.get_height());
    output.write(half4(rgb2luma(input.read(src).rgb), 0.0h, 0.0h, 1.0h), gid);
}

/// Frame extrapolation. VTMotionEstimation returns backward vectors in pixels:
/// `mv` at p says where p's content sat in the previous frame, so the content
/// velocity is -mv per capture interval. Sampling the source at `p + mv * phase`
/// is therefore a backward warp to a future time — it leaves no holes, unlike
/// scattering pixels forward, and needs no second frame so it adds no latency.
kernel void extrapolateFrame(
    texture2d<half, access::sample> source [[texture(0)]],
    texture2d<half, access::sample> motion [[texture(1)]],
    texture2d<half, access::write> output [[texture(2)]],
    constant float& phase [[buffer(0)]],
    constant float& blockSize [[buffer(1)]],
    uint2 gid [[thread_position_in_grid]]
) {
    uint width = output.get_width();
    uint height = output.get_height();
    if (gid.x >= width || gid.y >= height) return;

    constexpr sampler linearSampler(filter::linear, address::clamp_to_edge, coord::normalized);

    float2 size = float2(width, height);
    float2 uv = (float2(gid) + 0.5f) / size;

    // The field is one vector per block, so a linear fetch smooths it for free.
    float2 mv = float2(motion.sample(linearSampler, uv).xy);

    // Where neighbouring blocks disagree, the field is straddling a motion
    // boundary it cannot represent — an object edge, or ground opening up behind
    // something. Warping across that boundary is what smears the image, so the
    // shift is faded out exactly there and the source pixel shows through.
    float2 texel = 1.0f / float2(motion.get_width(), motion.get_height());
    float2 mL = float2(motion.sample(linearSampler, uv - float2(texel.x, 0.0f)).xy);
    float2 mR = float2(motion.sample(linearSampler, uv + float2(texel.x, 0.0f)).xy);
    float2 mU = float2(motion.sample(linearSampler, uv - float2(0.0f, texel.y)).xy);
    float2 mD = float2(motion.sample(linearSampler, uv + float2(0.0f, texel.y)).xy);
    // One block is the field's own resolution, so it is also the scale at which
    // neighbouring vectors are allowed to differ: beyond that they describe a
    // boundary the grid cannot represent, and the warp fades to the source.
    float disagreement = max(max(length(mv - mL), length(mv - mR)),
                             max(length(mv - mU), length(mv - mD)));

    // How far a mistrusted vector actually misplaces the pixel is its
    // disagreement times the phase, so the tolerance is measured in those terms
    // rather than in the field's resolution alone. Fading by disagreement only
    // treats a vector identically at phase 0.25 and 0.75, where the second one
    // lands the pixel three times further from where it belongs — which is why
    // the higher multipliers smear so much more than 2x. Half a block reproduces
    // the previous fade exactly at phase 0.5; below that this is more permissive,
    // above it less.
    float confidence = saturate(1.0f - (disagreement * phase) / (blockSize * 0.5f));

    float2 delta = mv * phase * confidence;

    // The linear fetch above blends across two blocks, which is exactly how far
    // the field can carry a shift before the warp tears between blocks. Past
    // that the result degrades towards the unwarped source instead.
    //
    // Holding the length at the limit — which is what this did — is not that: an
    // over-range pixel keeps moving at full speed while its slower neighbours
    // track their own vectors, and the whole point of the limit was that this
    // pixel's vector is no longer trustworthy. That differential across a
    // neighbourhood is what reads as the image melting, and it shows up worst in
    // fast camera motion, where the most pixels are over range. Fading the shift
    // out instead costs sharpness in those regions and hands back the source,
    // which reads as a held frame rather than a stretched one.
    float maxDisplacement = blockSize * 2.0f;
    float len = length(delta);
    if (len > maxDisplacement && len > 0.0f) {
        float excess = saturate(len / maxDisplacement - 1.0f);
        delta *= (maxDisplacement / len) * (1.0f - excess);
    }

    float2 sourceUV = (float2(gid) + 0.5f + delta) / size;
    output.write(source.sample(linearSampler, clamp(sourceUV, 0.0f, 1.0f)), gid);
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


kernel void contrastAdaptiveSharpening(
    texture2d<half, access::read> input [[texture(0)]],
    texture2d<half, access::write> output [[texture(1)]],
    constant SharpenParams& params [[buffer(0)]],
    uint2 gid [[thread_position_in_grid]]
) {
    uint width = input.get_width();
    uint height = input.get_height();
    if (gid.x >= width || gid.y >= height) return;
    
    int2 p = int2(gid);

    half3 a = input.read(clampCoord(p + int2(-1, -1), width, height)).rgb;
    half3 b = input.read(clampCoord(p + int2( 0, -1), width, height)).rgb;
    half3 c = input.read(clampCoord(p + int2( 1, -1), width, height)).rgb;
    half3 d = input.read(clampCoord(p + int2(-1,  0), width, height)).rgb;
    half3 e = input.read(gid).rgb;
    half3 f = input.read(clampCoord(p + int2( 1,  0), width, height)).rgb;
    half3 g = input.read(clampCoord(p + int2(-1,  1), width, height)).rgb;
    half3 h = input.read(clampCoord(p + int2( 0,  1), width, height)).rgb;
    half3 i = input.read(clampCoord(p + int2( 1,  1), width, height)).rgb;
    
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
