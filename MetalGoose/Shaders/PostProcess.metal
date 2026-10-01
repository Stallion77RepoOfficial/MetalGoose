// Image-space post-processing: contrast-adaptive sharpening, FXAA and SMAA.
// Every kernel works on the finished captured image — no depth buffer or motion
// vectors are needed.

#include "ShaderCommon.h"

// MARK: - FXAA

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

// MARK: - SMAA

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

// MARK: - Contrast-adaptive sharpening

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
