// Image-space post-processing: contrast-adaptive sharpening, FXAA and SMAA.
// Every kernel works on the finished captured image — no depth buffer or motion
// vectors are needed.

#include "ShaderCommon.h"

// MARK: - FXAA

// FXAA 3.11, the PC Quality path (Timothy Lottes): the edge's orientation from the 3x3 around the pixel, a search along the
// edge in both directions for where it ends, and the pixel moved across the edge by how far it is from the nearer end, plus
// a sub-pixel term for features thinner than a pixel. It replaced a directional blur that searched for no end and softened
// whatever it touched. Against an 8x8 supersampled truth on seven test scenes (shapes, lines, text, a photo with shapes on it,
// a perspective floor, an interface, the photo alone), at the Balanced threshold: 26.85 dB over all of them against 25.66
// (25.42 with no anti-aliasing), and the photo alone, where nothing should change, 38.1 dB against 29.6. The default
// FXAA 3.11 settings (sub-pixel 0.75, threshold floor half the threshold) scored best; less sub-pixel blur spared the photo
// a little and cost the edges more.
constant float kFxaaSteps[12] = { 1.0f, 1.5f, 2.0f, 2.0f, 2.0f, 2.0f, 2.0f, 2.0f, 2.0f, 2.0f, 4.0f, 8.0f };

inline float lumaAt(texture2d<half, access::sample> t, sampler s, float2 uv) {
    return float(rgb2luma(t.sample(s, uv, level(0)).rgb));
}

kernel void fxaa(
    texture2d<half, access::sample> input [[texture(0)]],
    texture2d<half, access::write> output [[texture(1)]],
    constant float& threshold [[buffer(0)]],
    uint2 gid [[thread_position_in_grid]]
) {
    const uint width = input.get_width(), height = input.get_height();
    if (gid.x >= width || gid.y >= height) return;
    constexpr sampler s(filter::linear, address::clamp_to_edge, coord::normalized);
    const float2 rcp = 1.0f / float2(width, height);
    float2 pos = (float2(gid) + 0.5f) * rcp;

    const float edgeThreshold = threshold;
    const float edgeThresholdMin = threshold * 0.5f;
    const float subpixQuality = 0.75f;

    const half4 centre = input.read(gid);
    const float lumaM = float(rgb2luma(centre.rgb));
    float lumaS = lumaAt(input, s, pos + float2(0, 1) * rcp);
    float lumaE = lumaAt(input, s, pos + float2(1, 0) * rcp);
    float lumaN = lumaAt(input, s, pos + float2(0, -1) * rcp);
    float lumaW = lumaAt(input, s, pos + float2(-1, 0) * rcp);

    const float rangeMax = max(max(max(lumaS, lumaM), max(lumaE, lumaN)), lumaW);
    const float rangeMin = min(min(min(lumaS, lumaM), min(lumaE, lumaN)), lumaW);
    const float range = rangeMax - rangeMin;
    if (range < max(edgeThresholdMin, rangeMax * edgeThreshold)) {
        output.write(centre, gid);
        return;
    }

    const float lumaNW = lumaAt(input, s, pos + float2(-1, -1) * rcp);
    const float lumaSE = lumaAt(input, s, pos + float2(1, 1) * rcp);
    const float lumaNE = lumaAt(input, s, pos + float2(1, -1) * rcp);
    const float lumaSW = lumaAt(input, s, pos + float2(-1, 1) * rcp);

    const float lumaNS = lumaN + lumaS, lumaWE = lumaW + lumaE;
    const float edgeHorz1 = -2.0f * lumaM + lumaNS, edgeVert1 = -2.0f * lumaM + lumaWE;
    const float lumaNESE = lumaNE + lumaSE, lumaNWNE = lumaNW + lumaNE;
    const float edgeHorz2 = -2.0f * lumaE + lumaNESE, edgeVert2 = -2.0f * lumaN + lumaNWNE;
    const float lumaNWSW = lumaNW + lumaSW, lumaSWSE = lumaSW + lumaSE;
    const float edgeHorz4 = abs(edgeHorz1) * 2.0f + abs(edgeHorz2), edgeVert4 = abs(edgeVert1) * 2.0f + abs(edgeVert2);
    const float edgeHorz3 = -2.0f * lumaW + lumaNWSW, edgeVert3 = -2.0f * lumaS + lumaSWSE;
    const float edgeHorz = abs(edgeHorz3) + edgeHorz4, edgeVert = abs(edgeVert3) + edgeVert4;

    const bool horzSpan = edgeHorz >= edgeVert;
    float lengthSign = horzSpan ? rcp.y : rcp.x;
    if (!horzSpan) { lumaN = lumaW; lumaS = lumaE; }

    const float subpixA = (lumaNS + lumaWE) * 2.0f + lumaNWSW + lumaNESE;
    const float subpixB = subpixA * (1.0f / 12.0f) - lumaM;
    const float gradientN = lumaN - lumaM, gradientS = lumaS - lumaM;
    float lumaNN = lumaN + lumaM;
    const float lumaSS = lumaS + lumaM;
    const bool pairN = abs(gradientN) >= abs(gradientS);
    const float gradient = max(abs(gradientN), abs(gradientS));
    if (pairN) lengthSign = -lengthSign;
    const float subpixC = saturate(abs(subpixB) / range);

    float2 posB = pos;
    const float2 offNP = horzSpan ? float2(rcp.x, 0) : float2(0, rcp.y);
    if (horzSpan) posB.y += lengthSign * 0.5f; else posB.x += lengthSign * 0.5f;

    float2 posN = posB - offNP * kFxaaSteps[0];
    float2 posP = posB + offNP * kFxaaSteps[0];
    const float subpixD = -2.0f * subpixC + 3.0f;
    const float subpixE = subpixC * subpixC;
    if (!pairN) lumaNN = lumaSS;
    const float gradientScaled = gradient * 0.25f;
    const float lumaMM = lumaM - lumaNN * 0.5f;
    const float subpixF = subpixD * subpixE;
    const bool lumaMLTZero = lumaMM < 0.0f;

    float lumaEndN = lumaAt(input, s, posN) - lumaNN * 0.5f;
    float lumaEndP = lumaAt(input, s, posP) - lumaNN * 0.5f;
    bool doneN = abs(lumaEndN) >= gradientScaled;
    bool doneP = abs(lumaEndP) >= gradientScaled;
    for (int i = 1; i < 12 && !(doneN && doneP); ++i) {
        if (!doneN) { posN -= offNP * kFxaaSteps[i]; lumaEndN = lumaAt(input, s, posN) - lumaNN * 0.5f; doneN = abs(lumaEndN) >= gradientScaled; }
        if (!doneP) { posP += offNP * kFxaaSteps[i]; lumaEndP = lumaAt(input, s, posP) - lumaNN * 0.5f; doneP = abs(lumaEndP) >= gradientScaled; }
    }

    const float dstN = horzSpan ? pos.x - posN.x : pos.y - posN.y;
    const float dstP = horzSpan ? posP.x - pos.x : posP.y - pos.y;
    const bool goodSpanN = (lumaEndN < 0.0f) != lumaMLTZero;
    const bool goodSpanP = (lumaEndP < 0.0f) != lumaMLTZero;
    const float spanLength = dstP + dstN;
    const bool directionN = dstN < dstP;
    const float dst = min(dstN, dstP);
    const bool goodSpan = directionN ? goodSpanN : goodSpanP;
    const float subpixG = subpixF * subpixF;
    const float pixelOffset = dst * (-1.0f / spanLength) + 0.5f;
    const float subpixH = subpixG * subpixQuality;
    const float pixelOffsetSubpix = max(goodSpan ? pixelOffset : 0.0f, subpixH);
    if (horzSpan) pos.y += pixelOffsetSubpix * lengthSign; else pos.x += pixelOffsetSubpix * lengthSign;
    output.write(half4(input.sample(s, pos, level(0)).rgb, 1.0h), gid);
}

// MARK: - SMAA

/// Edges where neighbouring pixels differ by more than the threshold in any channel, as SMAA's colour edge detection has it:
/// an edge between two colours of one brightness is an edge. One that is less than two thirds of the largest difference
/// around it is dropped, which keeps the edges of a texture's detail out (SMAA's local contrast adaptation). Measured on the
/// same scenes as FXAA, against luma and half the largest difference: 26.72 dB over all against 26.65, the scenes with edges
/// 28.70 against 28.52, and the photo alone unchanged (44.8 against 44.9). Blending across both axes of a staircase as well
/// gained another 0.1 dB and took 1.5 dB more from the photo, and was left out.
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
    half3 cC = input.read(gid).rgb;
    half3 cLeft = input.read(clampCoord(p + int2(-1, 0), width, height)).rgb;
    half3 cTop  = input.read(clampCoord(p + int2(0, -1), width, height)).rgb;
    auto diff = [](half3 a, half3 b) { half3 d = abs(a - b); return max(d.r, max(d.g, d.b)); };

    half2 delta;
    delta.x = diff(cC, cLeft);
    delta.y = diff(cC, cTop);

    half2 edge = step(threshold, delta);
    if (edge.x == 0.0h && edge.y == 0.0h) {
        edges.write(half4(0.0h, 0.0h, 0.0h, 1.0h), gid);
        return;
    }

    half3 cRight  = input.read(clampCoord(p + int2( 1, 0), width, height)).rgb;
    half3 cBottom = input.read(clampCoord(p + int2(0,  1), width, height)).rgb;
    half3 cLeftLeft = input.read(clampCoord(p + int2(-2, 0), width, height)).rgb;
    half3 cTopTop   = input.read(clampCoord(p + int2(0, -2), width, height)).rgb;

    half2 maxDelta;
    maxDelta.x = max(max(delta.x, diff(cC, cRight)), diff(cLeft, cLeftLeft));
    maxDelta.y = max(max(delta.y, diff(cC, cBottom)), diff(cTop, cTopTop));

    half finalDelta = max(maxDelta.x, maxDelta.y);
    edge *= step(finalDelta * 0.66h, delta);

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
