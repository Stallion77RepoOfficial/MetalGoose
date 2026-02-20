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

// =============================================================================
// Optical Flow / Frame Generation (GPU-only)
// =============================================================================
struct FlowInitParams {
    uint searchRadius;
};

struct FlowRefineParams {
    uint searchRadius;
};

struct FlowWarpParams {
    float scale;
};

struct FlowComposeParams {
    float t;
    float errorThreshold;
    float flowThreshold;
};

struct FlowOcclusionParams {
    float threshold;
};

struct TemporalParams {
    float blendFactor;
};

kernel void lumaFromColor(
    texture2d<half, access::read> input [[texture(0)]],
    texture2d<half, access::write> output [[texture(1)]],
    uint2 gid [[thread_position_in_grid]]
) {
    uint width = output.get_width();
    uint height = output.get_height();
    if (gid.x >= width || gid.y >= height) return;
    half3 rgb = input.read(gid).rgb;
    half l = rgb2luma(rgb);
    output.write(half4(l, 0.0h, 0.0h, 0.0h), gid);
}

kernel void lumaDownsample2x(
    texture2d<half, access::read> input [[texture(0)]],
    texture2d<half, access::write> output [[texture(1)]],
    uint2 gid [[thread_position_in_grid]]
) {
    uint outW = output.get_width();
    uint outH = output.get_height();
    if (gid.x >= outW || gid.y >= outH) return;
    uint inW = input.get_width();
    uint inH = input.get_height();
    uint2 src = gid * 2;
    uint2 p00 = uint2(min(src.x, inW - 1), min(src.y, inH - 1));
    uint2 p10 = uint2(min(src.x + 1, inW - 1), min(src.y, inH - 1));
    uint2 p01 = uint2(min(src.x, inW - 1), min(src.y + 1, inH - 1));
    uint2 p11 = uint2(min(src.x + 1, inW - 1), min(src.y + 1, inH - 1));
    half l = (input.read(p00).r + input.read(p10).r + input.read(p01).r + input.read(p11).r) * 0.25h;
    output.write(half4(l, 0.0h, 0.0h, 0.0h), gid);
}

kernel void flowInit(
    texture2d<half, access::read> prevLuma [[texture(0)]],
    texture2d<half, access::read> nextLuma [[texture(1)]],
    texture2d<half, access::write> flowOut [[texture(2)]],
    constant FlowInitParams& params [[buffer(0)]],
    uint2 gid [[thread_position_in_grid]]
) {
    uint width = prevLuma.get_width();
    uint height = prevLuma.get_height();
    if (gid.x >= width || gid.y >= height) return;
    
    int radius = int(params.searchRadius);
    half bestError = half(1e9);
    int2 bestOffset = int2(0, 0);
    
    // 3x3 SAD error function for a given offset
    auto computeSAD = [&](int2 offset) -> half {
        half err = 0.0h;
        for (int py = -1; py <= 1; py++) {
            for (int px = -1; px <= 1; px++) {
                int2 p = int2(gid) + int2(px, py);
                p.x = clamp(p.x, 0, int(width) - 1);
                p.y = clamp(p.y, 0, int(height) - 1);
                int2 q = int2(gid) + offset + int2(px, py);
                q.x = clamp(q.x, 0, int(width) - 1);
                q.y = clamp(q.y, 0, int(height) - 1);
                err += abs(prevLuma.read(uint2(p)).r - nextLuma.read(uint2(q)).r);
            }
        }
        return err;
    };
    
    for (int dy = -radius; dy <= radius; dy++) {
        for (int dx = -radius; dx <= radius; dx++) {
            half err = computeSAD(int2(dx, dy));
            if (err < bestError) {
                bestError = err;
                bestOffset = int2(dx, dy);
            }
        }
    }
    
    // Sub-pixel quadratic refinement
    half2 subPixel = half2(bestOffset);
    
    // X-axis parabolic fit
    half eL = computeSAD(bestOffset + int2(-1, 0));
    half eR = computeSAD(bestOffset + int2(1, 0));
    half denomX = eL + eR - 2.0h * bestError;
    if (abs(denomX) > 1e-6h) {
        half dx = 0.5h * (eL - eR) / denomX;
        subPixel.x += clamp(dx, -0.5h, 0.5h);
    }
    
    // Y-axis parabolic fit
    half eU = computeSAD(bestOffset + int2(0, -1));
    half eD = computeSAD(bestOffset + int2(0, 1));
    half denomY = eU + eD - 2.0h * bestError;
    if (abs(denomY) > 1e-6h) {
        half dy = 0.5h * (eU - eD) / denomY;
        subPixel.y += clamp(dy, -0.5h, 0.5h);
    }
    
    flowOut.write(half4(subPixel.x, subPixel.y, 0.0h, 0.0h), gid);
}

kernel void flowUpsample2x(
    texture2d<half, access::read> flowIn [[texture(0)]],
    texture2d<half, access::write> flowOut [[texture(1)]],
    uint2 gid [[thread_position_in_grid]]
) {
    uint outW = flowOut.get_width();
    uint outH = flowOut.get_height();
    if (gid.x >= outW || gid.y >= outH) return;
    uint2 src = uint2(gid.x / 2, gid.y / 2);
    half2 f = flowIn.read(src).rg * 2.0h;
    flowOut.write(half4(f.x, f.y, 0.0h, 0.0h), gid);
}

kernel void flowRefine(
    texture2d<half, access::read> prevLuma [[texture(0)]],
    texture2d<half, access::read> nextLuma [[texture(1)]],
    texture2d<half, access::read> flowPred [[texture(2)]],
    texture2d<half, access::write> flowOut [[texture(3)]],
    constant FlowRefineParams& params [[buffer(0)]],
    uint2 gid [[thread_position_in_grid]]
) {
    uint width = prevLuma.get_width();
    uint height = prevLuma.get_height();
    if (gid.x >= width || gid.y >= height) return;
    
    half2 pred = flowPred.read(gid).rg;
    int2 predInt = int2(round(pred));
    int radius = int(params.searchRadius);
    half bestError = half(1e9);
    int2 bestOffset = predInt;
    
    // 3x3 SAD error function
    auto computeSAD = [&](int2 offset) -> half {
        half err = 0.0h;
        for (int py = -1; py <= 1; py++) {
            for (int px = -1; px <= 1; px++) {
                int2 p = int2(gid) + int2(px, py);
                p.x = clamp(p.x, 0, int(width) - 1);
                p.y = clamp(p.y, 0, int(height) - 1);
                int2 q = int2(gid) + offset + int2(px, py);
                q.x = clamp(q.x, 0, int(width) - 1);
                q.y = clamp(q.y, 0, int(height) - 1);
                err += abs(prevLuma.read(uint2(p)).r - nextLuma.read(uint2(q)).r);
            }
        }
        return err;
    };
    
    for (int dy = -radius; dy <= radius; dy++) {
        for (int dx = -radius; dx <= radius; dx++) {
            int2 offset = predInt + int2(dx, dy);
            half err = computeSAD(offset);
            if (err < bestError) {
                bestError = err;
                bestOffset = offset;
            }
        }
    }
    
    // Sub-pixel quadratic refinement
    half2 subPixel = half2(bestOffset);
    
    // X-axis parabolic fit
    half eL = computeSAD(bestOffset + int2(-1, 0));
    half eR = computeSAD(bestOffset + int2(1, 0));
    half denomX = eL + eR - 2.0h * bestError;
    if (abs(denomX) > 1e-6h) {
        half dx = 0.5h * (eL - eR) / denomX;
        subPixel.x += clamp(dx, -0.5h, 0.5h);
    }
    
    // Y-axis parabolic fit
    half eU = computeSAD(bestOffset + int2(0, -1));
    half eD = computeSAD(bestOffset + int2(0, 1));
    half denomY = eU + eD - 2.0h * bestError;
    if (abs(denomY) > 1e-6h) {
        half dy = 0.5h * (eU - eD) / denomY;
        subPixel.y += clamp(dy, -0.5h, 0.5h);
    }
    
    flowOut.write(half4(subPixel.x, subPixel.y, 0.0h, 0.0h), gid);
}

kernel void flowWarp(
    texture2d<half, access::sample> input [[texture(0)]],
    texture2d<half, access::read> flowTex [[texture(1)]],
    texture2d<half, access::write> output [[texture(2)]],
    constant FlowWarpParams& params [[buffer(0)]],
    uint2 gid [[thread_position_in_grid]]
) {
    uint width = output.get_width();
    uint height = output.get_height();
    if (gid.x >= width || gid.y >= height) return;
    float2 size = float2(width, height);
    float2 uv = (float2(gid) + 0.5f) / size;
    half2 flow = flowTex.read(gid).rg;
    float2 offset = float2(flow) * params.scale / size;
    float2 sampleUV = clamp(uv - offset, float2(0.0f), float2(1.0f));
    constexpr sampler s(filter::linear, address::clamp_to_edge);
    half4 color = input.sample(s, sampleUV);
    output.write(color, gid);
}

kernel void flowCompose(
    texture2d<half, access::read> warpPrev [[texture(0)]],
    texture2d<half, access::read> warpNext [[texture(1)]],
    texture2d<half, access::read> occlusion [[texture(2)]],
    texture2d<half, access::write> output [[texture(3)]],
    constant FlowComposeParams& params [[buffer(0)]],
    uint2 gid [[thread_position_in_grid]]
) {
    uint width = output.get_width();
    uint height = output.get_height();
    if (gid.x >= width || gid.y >= height) return;
    
    half4 a = warpPrev.read(gid);
    half4 b = warpNext.read(gid);
    half t = half(params.t);
    
    // 1. Color consistency error (difference between forward-warped and backward-warped)
    half colorErr = length(a.rgb - b.rgb);
    
    // 2. Flow consistency error (passed from flowOcclusion)
    half flowErr = occlusion.read(gid).r;
    
    // 3. Calculate confidence (0.0 = bad, 1.0 = good)
    // Smoothstep creates a soft transition instead of a hard cutoff
    // params.flowThreshold is used as the center of the transition for flow error (pixels)
    half flowConf = 1.0h - smoothstep(half(params.flowThreshold) * 0.5h, half(params.flowThreshold) * 1.5h, flowErr);
    half colorConf = 1.0h - smoothstep(0.1h, 0.3h, colorErr);
    half confidence = min(flowConf, colorConf);
    
    // 4. Interpolate
    half4 interpolated = mix(a, b, t);
    
    // 5. Fallback: nearest neighbor (prev if t < 0.5, else next)
    // We blend smoothly from interpolated to nearest based on confidence
    half4 nearest = t < 0.5h ? a : b;
    
    output.write(mix(nearest, interpolated, confidence), gid);
}

kernel void flowOcclusion(
    texture2d<half, access::read> flowForward [[texture(0)]],
    texture2d<half, access::sample> flowBackward [[texture(1)]],
    texture2d<half, access::write> occlusion [[texture(2)]],
    constant FlowOcclusionParams& params [[buffer(0)]],
    uint2 gid [[thread_position_in_grid]]
) {
    uint width = occlusion.get_width();
    uint height = occlusion.get_height();
    if (gid.x >= width || gid.y >= height) return;
    
    float2 size = float2(width, height);
    float2 uv = (float2(gid) + 0.5f) / size;
    half2 f = flowForward.read(gid).rg;
    
    // Check flow consistency: F(p) + B(p + F(p)) should be close to 0
    float2 uvNext = clamp(uv + float2(f) / size, float2(0.0f), float2(1.0f));
    constexpr sampler s(filter::linear, address::clamp_to_edge);
    half2 b = flowBackward.sample(s, uvNext).rg;
    
    // Output raw error magnitude instead of binary threshold
    float2 sum = float2(f + b);
    half err = half(length(sum));
    
    // Store error in R channel
    occlusion.write(half4(err, 0.0h, 0.0h, 0.0h), gid);
}

// 3x3 median filter for flow field smoothing (edge-preserving outlier removal)
kernel void flowMedianFilter(
    texture2d<half, access::read> flowIn [[texture(0)]],
    texture2d<half, access::write> flowOut [[texture(1)]],
    uint2 gid [[thread_position_in_grid]]
) {
    uint width = flowIn.get_width();
    uint height = flowIn.get_height();
    if (gid.x >= width || gid.y >= height) return;
    
    half flowX[9];
    half flowY[9];
    int idx = 0;
    
    for (int dy = -1; dy <= 1; dy++) {
        for (int dx = -1; dx <= 1; dx++) {
            int2 pos = int2(gid) + int2(dx, dy);
            pos.x = clamp(pos.x, 0, int(width) - 1);
            pos.y = clamp(pos.y, 0, int(height) - 1);
            half2 f = flowIn.read(uint2(pos)).rg;
            flowX[idx] = f.x;
            flowY[idx] = f.y;
            idx++;
        }
    }
    
    // Simple insertion sort for 9 elements (fast for small N)
    for (int i = 1; i < 9; i++) {
        half kx = flowX[i]; half ky = flowY[i];
        int j = i - 1;
        while (j >= 0 && flowX[j] > kx) { flowX[j+1] = flowX[j]; j--; }
        flowX[j+1] = kx;
        j = i - 1;
        while (j >= 0 && flowY[j] > ky) { flowY[j+1] = flowY[j]; j--; }
        flowY[j+1] = ky;
    }
    
    flowOut.write(half4(flowX[4], flowY[4], 0.0h, 0.0h), gid);
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

    if (lumaRange < half(threshold)) {
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

    half3 result = (lumaB < lumaMin || lumaB > lumaMax) ? rgbA : rgbB;
    output.write(half4(result, 1.0h), gid);
}






struct SharpenParams {
    float sharpness;
    float radius;
};



struct AntiAliasParams {
    float threshold;
    float depthThreshold;
    int maxSearchSteps;
    float subpixelBlend;
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


    half3 c = input.read(gid).rgb;
    half3 cLeft = input.read(uint2(max(0u, gid.x - 1), gid.y)).rgb;
    half3 cTop = input.read(uint2(gid.x, max(0u, gid.y - 1))).rgb;

    half lumaC = rgb2luma(c);
    half lumaLeft = rgb2luma(cLeft);
    half lumaTop = rgb2luma(cTop);


    half2 delta;
    delta.x = abs(lumaC - lumaLeft);
    delta.y = abs(lumaC - lumaTop);

    half2 edge = step(threshold, delta);

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






kernel void temporalAccumulation(
    texture2d<half, access::read> current [[texture(0)]],
    texture2d<half, access::read> history [[texture(1)]],
    texture2d<half, access::write> output [[texture(2)]],
    constant float& blendFactor [[buffer(0)]],
    uint2 gid [[thread_position_in_grid]]
) {
    uint width = current.get_width();
    uint height = current.get_height();
    if (gid.x >= width || gid.y >= height) return;

    half4 curr = current.read(gid);
    half4 hist = history.read(gid);


    half4 minNeighbor = curr;
    half4 maxNeighbor = curr;

    for (int dy = -1; dy <= 1; dy++) {
        for (int dx = -1; dx <= 1; dx++) {
            if (dx == 0 && dy == 0) continue;
            int2 pos = int2(gid) + int2(dx, dy);
            pos = clamp(pos, int2(0), int2(width - 1, height - 1));
            half4 neighbor = current.read(uint2(pos));
            minNeighbor = min(minNeighbor, neighbor);
            maxNeighbor = max(maxNeighbor, neighbor);
        }
    }


    half4 clampedHist = clamp(hist, minNeighbor, maxNeighbor);


    half4 result = mix(clampedHist, curr, half(blendFactor));
    output.write(result, gid);
}

kernel void temporalReproject(
    texture2d<half, access::sample> current [[texture(0)]],
    texture2d<half, access::sample> history [[texture(1)]],
    texture2d<half, access::read> flow [[texture(2)]],
    texture2d<half, access::write> output [[texture(3)]],
    constant TemporalParams& params [[buffer(0)]],
    uint2 gid [[thread_position_in_grid]]
) {
    uint width = output.get_width();
    uint height = output.get_height();
    if (gid.x >= width || gid.y >= height) return;
    
    float2 size = float2(width, height);
    float2 uv = (float2(gid) + 0.5f) / size;
    half2 flowVec = flow.read(gid).rg;
    float2 histUV = clamp(uv + float2(flowVec) / size, float2(0.0f), float2(1.0f));
    constexpr sampler s(filter::linear, address::clamp_to_edge);
    half4 curr = current.sample(s, uv);
    half4 hist = history.sample(s, histUV);
    
    half blend = half(params.blendFactor);
    half4 result = mix(hist, curr, blend);
    output.write(result, gid);
}


kernel void msaa(
    texture2d<half, access::read> input [[texture(0)]],
    texture2d<half, access::write> output [[texture(1)]],
    constant AntiAliasParams& params [[buffer(0)]],
    uint2 gid [[thread_position_in_grid]]
) {
    uint width = input.get_width();
    uint height = input.get_height();
    if (gid.x >= width || gid.y >= height) return;


    half4 center = input.read(gid);
    half centerLuma = rgb2luma(center.rgb);

    half4 sum = center;
    half weightSum = 1.0h;

    for (int dy = -1; dy <= 1; dy++) {
        for (int dx = -1; dx <= 1; dx++) {
            if (dx == 0 && dy == 0) continue;

            int2 pos = int2(gid) + int2(dx, dy);
            if (pos.x < 0 || pos.x >= int(width) ||
                pos.y < 0 || pos.y >= int(height)) continue;

            half4 neighbor = input.read(uint2(pos));
            half neighborLuma = rgb2luma(neighbor.rgb);


            half diff = abs(centerLuma - neighborLuma);
            half weight = exp(-diff * 10.0h) * 0.5h;

            sum += neighbor * weight;
            weightSum += weight;
        }
    }

    output.write(sum / weightSum, gid);
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

// ============================================================================
// CONTRAST ADAPTIVE SHARPENING (CAS)
// Based on AMD FidelityFX CAS - enhances details without over-sharpening
// Uses existing SharpenParams struct defined above
// ============================================================================

kernel void contrastAdaptiveSharpening(
    texture2d<half, access::read> input [[texture(0)]],
    texture2d<half, access::write> output [[texture(1)]],
    constant SharpenParams& params [[buffer(0)]],
    uint2 gid [[thread_position_in_grid]]
) {
    uint width = input.get_width();
    uint height = input.get_height();
    if (gid.x >= width || gid.y >= height) return;
    
    // Read 3x3 neighborhood
    uint2 maxCoord = uint2(width - 1, height - 1);
    
    half3 a = input.read(uint2(max(0u, gid.x - 1), max(0u, gid.y - 1))).rgb;
    half3 b = input.read(uint2(gid.x, max(0u, gid.y - 1))).rgb;
    half3 c = input.read(uint2(min(maxCoord.x, gid.x + 1), max(0u, gid.y - 1))).rgb;
    half3 d = input.read(uint2(max(0u, gid.x - 1), gid.y)).rgb;
    half3 e = input.read(gid).rgb;  // Center pixel
    half3 f = input.read(uint2(min(maxCoord.x, gid.x + 1), gid.y)).rgb;
    half3 g = input.read(uint2(max(0u, gid.x - 1), min(maxCoord.y, gid.y + 1))).rgb;
    half3 h = input.read(uint2(gid.x, min(maxCoord.y, gid.y + 1))).rgb;
    half3 i = input.read(uint2(min(maxCoord.x, gid.x + 1), min(maxCoord.y, gid.y + 1))).rgb;
    
    // Calculate local min and max (contrast range)
    half3 minRGB = min(min(min(d, e), min(f, b)), h);
    half3 maxRGB = max(max(max(d, e), max(f, b)), h);
    
    // Expand to include corners for better edge handling
    minRGB = min(min(min(minRGB, a), min(c, g)), i);
    maxRGB = max(max(max(maxRGB, a), max(c, g)), i);
    
    // Calculate the sharpening weight based on local contrast
    // Higher contrast = less sharpening to avoid artifacts
    half3 contrast = maxRGB - minRGB;
    half3 ampFactor = saturate(1.0h - contrast * 2.0h);
    
    // Apply sharpening
    half sharpness = half(params.sharpness);
    half3 weight = ampFactor * sharpness;
    
    // Unsharp mask style sharpening
    half3 blur = (a + b + c + d + f + g + h + i) / 8.0h;
    half3 sharpened = e + (e - blur) * weight;
    
    // Clamp to local range to prevent artifacts
    sharpened = clamp(sharpened, minRGB, maxRGB);
    
    output.write(half4(clampColor(sharpened), 1.0h), gid);
}
