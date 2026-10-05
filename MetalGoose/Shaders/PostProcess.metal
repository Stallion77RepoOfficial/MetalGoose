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

    // Half is sufficient for colour, but loses large pixel addresses and small offsets.
    float2 position = float2(gid);
    float2 direction = float2(dir);
    uint2 pos1 = clampCoord(int2(position + direction * (1.0f / 3.0f - 0.5f)), width, height);
    uint2 pos2 = clampCoord(int2(position + direction * (2.0f / 3.0f - 0.5f)), width, height);

    half3 rgbA = (input.read(pos1).rgb + input.read(pos2).rgb) * 0.5h;

    uint2 pos3 = clampCoord(int2(position + direction * -0.5f), width, height);
    uint2 pos4 = clampCoord(int2(position + direction * 0.5f), width, height);

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

/// A run of edges of one orientation along a border, and how the edge line leaves it at each end.
struct EdgeRun {
    /// Pixels between the one being looked at and the run's first and last pixel.
    int before;
    int after;
    /// Where the estimated edge line meets the run's ends, in pixels from the border the run lies on:
    /// negative towards lower coordinates (above, or left), positive towards higher ones, 0 where the run
    /// simply ends or the end was not found within the search distance.
    float startOffset;
    float endOffset;
};

/// An edge corner whose crossing edge goes on past the next pixel is a real corner of the image, as a
/// window or a glyph has, and keeps most of its sharpness.
constant float cornerKept = 0.25f;

/// One channel of the edge texture; outside the image there are no edges.
inline half edgeAt(texture2d<half, access::read> edges, int2 p, bool green, int2 size) {
    if (any(p < 0) || any(p >= size)) return 0.0h;
    half2 e = edges.read(uint2(p)).rg;
    return green ? e.g : e.r;
}

/// Where the edge line leaves a run at the border position `q`: the edges that cross the run's border
/// there, on either side of it. A crossing on one side puts the line half a pixel to that side; none, or
/// one on both sides, leaves it on the border.
inline float crossingOffset(texture2d<half, access::read> edges, int2 q, int2 across, bool crossIsGreen,
                            int2 size, thread bool& crossed) {
    half negative = edgeAt(edges, q - across, crossIsGreen, size);
    half positive = edgeAt(edges, q, crossIsGreen, size);
    crossed = negative > 0.0h || positive > 0.0h;
    if (!crossed) return 0.0f;

    float offset = 0.5f * float(positive - negative);
    bool continues = (negative > 0.0h && edgeAt(edges, q - 2 * across, crossIsGreen, size) > 0.0h)
                  || (positive > 0.0h && edgeAt(edges, q + across, crossIsGreen, size) > 0.0h);
    return continues ? offset * cornerKept : offset;
}

/// Follows the run of edges through `p` in both directions. A horizontal run is made of top-border edges
/// (green) and is crossed by left-border edges (red); a vertical run is the other way round. The search
/// stops where the run ends or something crosses it.
inline EdgeRun followRun(texture2d<half, access::read> edges, int2 p, bool horizontal, int maxSteps, int2 size) {
    const int2 along = horizontal ? int2(1, 0) : int2(0, 1);
    const int2 across = horizontal ? int2(0, 1) : int2(1, 0);
    const bool runIsGreen = horizontal;

    EdgeRun run = { 0, 0, 0.0f, 0.0f };

    int2 q = p;
    for (int i = 0; i < maxSteps; i++) {
        bool crossed;
        float offset = crossingOffset(edges, q, across, !runIsGreen, size, crossed);
        if (crossed) { run.startOffset = offset; break; }
        if (edgeAt(edges, q - along, runIsGreen, size) == 0.0h) break;
        q -= along;
        run.before++;
    }

    q = p;
    for (int i = 0; i < maxSteps; i++) {
        bool crossed;
        float offset = crossingOffset(edges, q + along, across, !runIsGreen, size, crossed);
        if (crossed) { run.endOffset = offset; break; }
        if (edgeAt(edges, q + along, runIsGreen, size) == 0.0h) break;
        q += along;
        run.after++;
    }
    return run;
}

/// Area between a straight piece of the edge line and its border, split by the side of the border it
/// lies on: x for the negative side, y for the positive.
inline void addArea(thread float2& area, float t0, float h0, float t1, float h1) {
    float width = t1 - t0;
    if (width <= 0.0f) return;
    if (h0 >= 0.0f && h1 >= 0.0f) {
        area.y += 0.5f * (h0 + h1) * width;
    } else if (h0 <= 0.0f && h1 <= 0.0f) {
        area.x -= 0.5f * (h0 + h1) * width;
    } else {
        float meets = width * h0 / (h0 - h1);
        float first = 0.5f * abs(h0) * meets;
        float second = 0.5f * abs(h1) * (width - meets);
        if (h0 > 0.0f) { area.y += first; area.x += second; } else { area.x += first; area.y += second; }
    }
}

/// How much of this pixel's column lies on each side of the border, for the edge line the run implies.
/// Ends on opposite sides are joined by one straight line; otherwise the line comes down to the border
/// in the middle of the run, from the end that crosses (an L) or from both (a U). A run with nothing
/// crossing is a straight edge, which has no stair-steps to smooth.
inline float2 lineAreas(EdgeRun run) {
    float2 area = float2(0.0f);
    float o1 = run.startOffset;
    float o2 = run.endOffset;
    if (o1 == 0.0f && o2 == 0.0f) return area;

    float length = float(run.before + run.after + 1);
    float column = float(run.before);
    float t[3], h[3];
    if (o1 * o2 < 0.0f) {
        t[0] = 0.0f; t[1] = length; t[2] = length;
        h[0] = o1;   h[1] = o2;     h[2] = o2;
    } else {
        t[0] = 0.0f; t[1] = 0.5f * length; t[2] = length;
        h[0] = o1;   h[1] = 0.0f;          h[2] = o2;
    }
    for (int i = 0; i < 2; i++) {
        if (t[i + 1] <= t[i]) continue;
        float a = max(t[i], column);
        float b = min(t[i + 1], column + 1.0f);
        if (b <= a) continue;
        float slope = (h[i + 1] - h[i]) / (t[i + 1] - t[i]);
        addArea(area, a, h[i] + slope * (a - t[i]), b, h[i] + slope * (b - t[i]));
    }
    return area;
}

/// Blend weights, one set per pixel for the two borders it owns (its top and its left):
///   r  the pixel above the top border, towards the pixel below it
///   g  the pixel below the top border, towards the pixel above it
///   b  the pixel left of the left border, towards the pixel right of it
///   a  the pixel right of the left border, towards the pixel left of it
kernel void smaaBlendingWeights(
    texture2d<half, access::read> edges [[texture(0)]],
    texture2d<half, access::write> weights [[texture(1)]],
    constant AntiAliasParams& params [[buffer(0)]],
    uint2 gid [[thread_position_in_grid]]
) {
    int2 size = int2(edges.get_width(), edges.get_height());
    if (int(gid.x) >= size.x || int(gid.y) >= size.y) return;

    half2 e = edges.read(gid).rg;
    half4 result = half4(0.0h);
    int2 p = int2(gid);

    if (e.g > 0.0h) {
        float2 area = lineAreas(followRun(edges, p, true, params.maxSearchSteps, size));
        result.r = half(area.x);
        result.g = half(area.y);
    }
    if (e.r > 0.0h) {
        float2 area = lineAreas(followRun(edges, p, false, params.maxSearchSteps, size));
        result.b = half(area.x);
        result.a = half(area.y);
    }
    weights.write(result, gid);
}

inline half4 weightsAt(texture2d<half, access::read> weights, int2 p, int2 size) {
    if (any(p < 0) || any(p >= size)) return half4(0.0h);
    return weights.read(uint2(p));
}

/// Mixes each pixel with the neighbour the weights name. A pixel can be asked to blend across more than
/// one border; only the axis with the larger weight is used, so a corner is smoothed in one direction
/// instead of being blurred in two.
kernel void smaaBlend(
    texture2d<half, access::read> input [[texture(0)]],
    texture2d<half, access::read> weights [[texture(1)]],
    texture2d<half, access::write> output [[texture(2)]],
    uint2 gid [[thread_position_in_grid]]
) {
    uint width = input.get_width();
    uint height = input.get_height();
    if (gid.x >= width || gid.y >= height) return;

    int2 size = int2(width, height);
    int2 p = int2(gid);
    half4 center = input.read(gid);

    float above = float(weightsAt(weights, p, size).g);
    float below = float(weightsAt(weights, p + int2(0, 1), size).r);
    float left  = float(weightsAt(weights, p, size).a);
    float right = float(weightsAt(weights, p + int2(1, 0), size).b);

    if (above + below + left + right < 1e-5f) {
        output.write(center, gid);
        return;
    }

    bool horizontal = max(left, right) > max(above, below);
    float first = horizontal ? left : above;
    float second = horizontal ? right : below;
    int2 firstStep = horizontal ? int2(-1, 0) : int2(0, -1);
    int2 secondStep = horizontal ? int2(1, 0) : int2(0, 1);

    float4 c = float4(center);
    float4 a = float4(input.read(clampCoord(p + firstStep, width, height)));
    float4 b = float4(input.read(clampCoord(p + secondStep, width, height)));
    float total = first + second;
    float4 result = (first / total) * mix(c, a, first) + (second / total) * mix(c, b, second);
    output.write(half4(result), gid);
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
