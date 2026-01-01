#include <metal_stdlib>
using namespace metal;

struct VertexOut {
    float4 position [[position]];
    float2 texCoord;
};

struct UpscaleParams {
    float sharpness;
    uint2 inputSize;
    uint2 outputSize;
};

struct FrameBlendParams {
    float t;
    uint2 textureSize;
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

kernel void mgup1_upscale(
    texture2d<half, access::read> input [[texture(0)]],
    texture2d<half, access::write> output [[texture(1)]],
    constant UpscaleParams& params [[buffer(0)]],
    uint2 gid [[thread_position_in_grid]]
) {
    if (gid.x >= params.outputSize.x || gid.y >= params.outputSize.y) {
        return;
    }
    
    float2 srcPos = float2(gid) * float2(params.inputSize) / float2(params.outputSize);
    uint2 srcCoord = uint2(clamp(srcPos, float2(0.0), float2(params.inputSize) - 1.0));
    
    half4 center = input.read(srcCoord);
    
    if (params.sharpness < 0.01h) {
        output.write(center, gid);
        return;
    }
    
    uint2 inputMax = params.inputSize - 1;
    
    half4 n = input.read(uint2(srcCoord.x, max(0u, srcCoord.y - 1)));
    half4 s = input.read(uint2(srcCoord.x, min(inputMax.y, srcCoord.y + 1)));
    half4 w = input.read(uint2(max(0u, srcCoord.x - 1), srcCoord.y));
    half4 e = input.read(uint2(min(inputMax.x, srcCoord.x + 1), srcCoord.y));
    
    half4 minNeighbor = min(min(n, s), min(w, e));
    half4 maxNeighbor = max(max(n, s), max(w, e));
    
    half4 contrast = maxNeighbor - minNeighbor;
    half4 sharpWeight = half4(params.sharpness) * saturate(1.0h - contrast * 2.0h);
    
    half4 neighbors = (n + s + w + e) * 0.25h;
    half4 sharpened = center + (center - neighbors) * sharpWeight;
    
    output.write(half4(clampColor(sharpened.rgb), 1.0h), gid);
}

kernel void mgfg1_blend(
    texture2d<half, access::read> prevFrame [[texture(0)]],
    texture2d<half, access::read> currFrame [[texture(1)]],
    texture2d<half, access::write> output [[texture(2)]],
    constant FrameBlendParams& params [[buffer(0)]],
    uint2 gid [[thread_position_in_grid]]
) {
    if (gid.x >= params.textureSize.x || gid.y >= params.textureSize.y) {
        return;
    }
    
    half4 prev = prevFrame.read(gid);
    half4 curr = currFrame.read(gid);
    
    half t = half(params.t);
    t = t * t * (3.0h - 2.0h * t);
    
    half4 result = mix(prev, curr, t);
    
    output.write(result, gid);
}


// NOTE: passthrough_copy was removed as an unused duplicate of copyTexture (line 1144)
// The copyTexture kernel is used by Engine.mm::copyPipeline_


// NOTE: fxaa_simple was removed as an unused duplicate of fxaa
// The fxaa kernel with threshold=0 provides equivalent behavior

// FXAA - Fast approximate anti-aliasing with configurable threshold
// When threshold=0, this behaves identically to the removed fxaa_simple
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

// ============================================================================
// MGFG-1 Frame Generation Shaders
// ============================================================================

struct MGFG1Params {
    float t;
    float motionScale;
    float occlusionThreshold;
    float temporalWeight;
    uint2 textureSize;
    uint qualityMode;
    uint padding;
};

struct BalancedParams {
    float t;
    uint textureWidth;
    uint textureHeight;
    float gradientThreshold;
    float padding;
};

// Performance mode: Simple smoothstep blend (fastest)
kernel void mgfg1Performance(
    texture2d<half, access::read> prevFrame [[texture(0)]],
    texture2d<half, access::read> currFrame [[texture(1)]],
    texture2d<half, access::write> output [[texture(2)]],
    constant float& t [[buffer(0)]],
    uint2 gid [[thread_position_in_grid]]
) {
    uint width = prevFrame.get_width();
    uint height = prevFrame.get_height();
    if (gid.x >= width || gid.y >= height) return;
    
    half4 prev = prevFrame.read(gid);
    half4 curr = currFrame.read(gid);
    
    // Smoothstep interpolation for smoother transitions
    half tSmooth = half(t);
    tSmooth = tSmooth * tSmooth * (3.0h - 2.0h * tSmooth);
    
    half4 result = mix(prev, curr, tSmooth);
    output.write(result, gid);
}

// Balanced mode: Gradient-aware blending
kernel void mgfg1Balanced(
    texture2d<half, access::read> prevFrame [[texture(0)]],
    texture2d<half, access::read> currFrame [[texture(1)]],
    texture2d<half, access::write> output [[texture(2)]],
    constant BalancedParams& params [[buffer(0)]],
    uint2 gid [[thread_position_in_grid]]
) {
    if (gid.x >= params.textureWidth || gid.y >= params.textureHeight) return;
    
    half4 prev = prevFrame.read(gid);
    half4 curr = currFrame.read(gid);
    
    // Calculate local gradient for motion detection
    half prevLuma = rgb2luma(prev.rgb);
    half currLuma = rgb2luma(curr.rgb);
    half lumaDiff = abs(prevLuma - currLuma);
    
    // Sample neighbors for edge detection
    uint2 size = uint2(params.textureWidth, params.textureHeight);
    half4 prevN = prevFrame.read(uint2(gid.x, max(0u, gid.y - 1)));
    half4 prevS = prevFrame.read(uint2(gid.x, min(size.y - 1, gid.y + 1)));
    half4 currN = currFrame.read(uint2(gid.x, max(0u, gid.y - 1)));
    half4 currS = currFrame.read(uint2(gid.x, min(size.y - 1, gid.y + 1)));
    
    half gradPrev = abs(rgb2luma(prevN.rgb) - rgb2luma(prevS.rgb));
    half gradCurr = abs(rgb2luma(currN.rgb) - rgb2luma(currS.rgb));
    half gradientStrength = max(gradPrev, gradCurr);
    
    // Adaptive blend factor based on motion and gradients
    half tBase = half(params.t);
    half motionWeight = saturate(lumaDiff * 5.0h);
    half edgeWeight = saturate(gradientStrength * 3.0h);
    
    // Use linear blend in high-motion/edge areas, smoothstep in smooth areas
    half tSmooth = tBase * tBase * (3.0h - 2.0h * tBase);
    half tAdaptive = mix(tSmooth, tBase, max(motionWeight, edgeWeight));
    
    half4 result = mix(prev, curr, tAdaptive);
    output.write(result, gid);
}

// Quality mode: Motion-compensated interpolation with occlusion handling
kernel void mgfg1Quality(
    texture2d<half, access::read> prevFrame [[texture(0)]],
    texture2d<half, access::read> currFrame [[texture(1)]],
    texture2d<half, access::read> motionVectors [[texture(2)]],
    texture2d<half, access::read> confidence [[texture(3)]],
    texture2d<half, access::write> output [[texture(4)]],
    constant MGFG1Params& params [[buffer(0)]],
    uint2 gid [[thread_position_in_grid]]
) {
    if (gid.x >= params.textureSize.x || gid.y >= params.textureSize.y) return;
    
    half4 mv = motionVectors.read(gid);
    half conf = confidence.read(gid).r;
    
    float2 motion = float2(mv.x, mv.y) * params.motionScale;
    
    // Calculate warped positions
    float2 pos = float2(gid);
    float2 prevPos = pos - motion * (1.0f - params.t);
    float2 currPos = pos + motion * params.t;
    
    // Clamp to valid coordinates
    float2 maxCoord = float2(params.textureSize) - 1.0f;
    prevPos = clamp(prevPos, float2(0), maxCoord);
    currPos = clamp(currPos, float2(0), maxCoord);
    
    // Bilinear sample (manual implementation for read textures)
    uint2 prevCoord = uint2(prevPos);
    uint2 currCoord = uint2(currPos);
    
    half4 prevSample = prevFrame.read(prevCoord);
    half4 currSample = currFrame.read(currCoord);
    
    // Occlusion detection
    half4 prevDirect = prevFrame.read(gid);
    half4 currDirect = currFrame.read(gid);
    half occlusionPrev = abs(rgb2luma(prevSample.rgb) - rgb2luma(prevDirect.rgb));
    half occlusionCurr = abs(rgb2luma(currSample.rgb) - rgb2luma(currDirect.rgb));
    
    half occlusionWeight = saturate((occlusionPrev + occlusionCurr) / half(params.occlusionThreshold));
    
    // Blend based on confidence and occlusion
    half t = half(params.t);
    half motionBlend = mix(prevSample, currSample, t).r; // Placeholder for full blend
    half4 motionResult = mix(prevSample, currSample, t);
    half4 directResult = mix(prevDirect, currDirect, t * t * (3.0h - 2.0h * t));
    
    // Use motion-compensated result when confident, fallback otherwise
    half blendWeight = conf * (1.0h - occlusionWeight);
    half4 result = mix(directResult, motionResult, blendWeight);
    
    output.write(result, gid);
}

// Adaptive interpolation: Combines multiple strategies based on content
kernel void mgfg1AdaptiveInterpolation(
    texture2d<half, access::read> prevFrame [[texture(0)]],
    texture2d<half, access::read> currFrame [[texture(1)]],
    texture2d<half, access::read> motionVectors [[texture(2)]],
    texture2d<half, access::read> confidence [[texture(3)]],
    texture2d<half, access::write> output [[texture(4)]],
    constant MGFG1Params& params [[buffer(0)]],
    uint2 gid [[thread_position_in_grid]]
) {
    if (gid.x >= params.textureSize.x || gid.y >= params.textureSize.y) return;
    
    half4 prev = prevFrame.read(gid);
    half4 curr = currFrame.read(gid);
    half4 mv = motionVectors.read(gid);
    half conf = confidence.read(gid).r;
    
    // Motion magnitude
    half motionMag = length(half2(mv.x, mv.y));
    
    // Decide strategy based on motion and confidence
    half t = half(params.t);
    half4 result;
    
    if (motionMag < 0.5h || conf < 0.3h) {
        // Static or low-confidence: use smoothstep blend
        half tSmooth = t * t * (3.0h - 2.0h * t);
        result = mix(prev, curr, tSmooth);
    } else if (conf > 0.7h) {
        // High confidence: motion-compensated
        float2 motion = float2(mv.x, mv.y) * params.motionScale;
        float2 pos = float2(gid);
        float2 prevPos = clamp(pos - motion * (1.0f - params.t), float2(0), float2(params.textureSize) - 1.0f);
        float2 currPos = clamp(pos + motion * params.t, float2(0), float2(params.textureSize) - 1.0f);
        
        half4 prevWarp = prevFrame.read(uint2(prevPos));
        half4 currWarp = currFrame.read(uint2(currPos));
        result = mix(prevWarp, currWarp, t);
    } else {
        // Medium confidence: weighted blend
        half tSmooth = t * t * (3.0h - 2.0h * t);
        half4 simpleBlend = mix(prev, curr, tSmooth);
        
        float2 motion = float2(mv.x, mv.y) * params.motionScale * 0.5f;
        float2 pos = float2(gid);
        float2 prevPos = clamp(pos - motion * (1.0f - params.t), float2(0), float2(params.textureSize) - 1.0f);
        float2 currPos = clamp(pos + motion * params.t, float2(0), float2(params.textureSize) - 1.0f);
        
        half4 prevWarp = prevFrame.read(uint2(prevPos));
        half4 currWarp = currFrame.read(uint2(currPos));
        half4 motionBlend = mix(prevWarp, currWarp, t);
        
        result = mix(simpleBlend, motionBlend, conf);
    }
    
    output.write(result, gid);
}

// ============================================================================
// Motion Estimation Shaders
// ============================================================================

// Block-matching motion estimation
kernel void mgfg1MotionEstimation(
    texture2d<half, access::read> prevFrame [[texture(0)]],
    texture2d<half, access::read> currFrame [[texture(1)]],
    texture2d<half, access::write> motionVectors [[texture(2)]],
    texture2d<half, access::write> confidence [[texture(3)]],
    constant MGFG1Params& params [[buffer(0)]],
    uint2 gid [[thread_position_in_grid]]
) {
    uint width = currFrame.get_width();
    uint height = currFrame.get_height();
    if (gid.x >= width || gid.y >= height) return;
    
    const int SEARCH_RADIUS = 4;
    const int BLOCK_SIZE = 4;
    
    half currLuma = rgb2luma(currFrame.read(gid).rgb);
    
    half bestSAD = HALF_MAX;
    int2 bestMotion = int2(0, 0);
    
    // Diamond search pattern
    for (int dy = -SEARCH_RADIUS; dy <= SEARCH_RADIUS; dy++) {
        for (int dx = -SEARCH_RADIUS; dx <= SEARCH_RADIUS; dx++) {
            int2 searchPos = int2(gid) + int2(dx, dy);
            
            if (searchPos.x < 0 || searchPos.x >= int(width) ||
                searchPos.y < 0 || searchPos.y >= int(height)) continue;
            
            half sad = 0.0h;
            int validSamples = 0;
            
            // Block matching
            for (int by = -BLOCK_SIZE/2; by <= BLOCK_SIZE/2; by++) {
                for (int bx = -BLOCK_SIZE/2; bx <= BLOCK_SIZE/2; bx++) {
                    int2 currBlockPos = int2(gid) + int2(bx, by);
                    int2 prevBlockPos = searchPos + int2(bx, by);
                    
                    if (currBlockPos.x >= 0 && currBlockPos.x < int(width) &&
                        currBlockPos.y >= 0 && currBlockPos.y < int(height) &&
                        prevBlockPos.x >= 0 && prevBlockPos.x < int(width) &&
                        prevBlockPos.y >= 0 && prevBlockPos.y < int(height)) {
                        
                        half currL = rgb2luma(currFrame.read(uint2(currBlockPos)).rgb);
                        half prevL = rgb2luma(prevFrame.read(uint2(prevBlockPos)).rgb);
                        sad += abs(currL - prevL);
                        validSamples++;
                    }
                }
            }
            
            if (validSamples > 0) {
                sad /= half(validSamples);
                if (sad < bestSAD) {
                    bestSAD = sad;
                    bestMotion = int2(dx, dy);
                }
            }
        }
    }
    
    // Output motion vector (normalized)
    half2 mv = half2(bestMotion) / half(SEARCH_RADIUS);
    motionVectors.write(half4(mv.x, mv.y, 0.0h, 1.0h), gid);
    
    // Confidence based on SAD
    half conf = saturate(1.0h - bestSAD * 4.0h);
    confidence.write(half4(conf, conf, conf, 1.0h), gid);
}

// Optimized motion estimation with early termination
kernel void mgfg1MotionEstimationOptimized(
    texture2d<half, access::read> prevFrame [[texture(0)]],
    texture2d<half, access::read> currFrame [[texture(1)]],
    texture2d<half, access::write> motionVectors [[texture(2)]],
    texture2d<half, access::write> confidence [[texture(3)]],
    constant MGFG1Params& params [[buffer(0)]],
    uint2 gid [[thread_position_in_grid]]
) {
    uint width = currFrame.get_width();
    uint height = currFrame.get_height();
    if (gid.x >= width || gid.y >= height) return;
    
    half4 curr = currFrame.read(gid);
    half4 prev = prevFrame.read(gid);
    half currLuma = rgb2luma(curr.rgb);
    half prevLuma = rgb2luma(prev.rgb);
    
    // Early exit for static pixels
    if (abs(currLuma - prevLuma) < 0.01h) {
        motionVectors.write(half4(0.0h, 0.0h, 0.0h, 1.0h), gid);
        confidence.write(half4(1.0h, 1.0h, 1.0h, 1.0h), gid);
        return;
    }
    
    // Three-step search (faster than full search)
    const int steps[3] = {4, 2, 1};
    int2 bestMotion = int2(0, 0);
    half bestSAD = HALF_MAX;
    
    for (int step = 0; step < 3; step++) {
        int s = steps[step];
        int2 center = bestMotion;
        
        // Check 9 positions in a 3x3 grid
        for (int dy = -1; dy <= 1; dy++) {
            for (int dx = -1; dx <= 1; dx++) {
                int2 offset = center + int2(dx * s, dy * s);
                int2 searchPos = int2(gid) + offset;
                
                if (searchPos.x < 0 || searchPos.x >= int(width) ||
                    searchPos.y < 0 || searchPos.y >= int(height)) continue;
                
                half searchLuma = rgb2luma(prevFrame.read(uint2(searchPos)).rgb);
                half sad = abs(currLuma - searchLuma);
                
                if (sad < bestSAD) {
                    bestSAD = sad;
                    bestMotion = offset;
                }
            }
        }
    }
    
    half2 mv = half2(bestMotion);
    motionVectors.write(half4(mv.x, mv.y, 0.0h, 1.0h), gid);
    
    half conf = saturate(1.0h - bestSAD * 5.0h);
    confidence.write(half4(conf, conf, conf, 1.0h), gid);
}

// Motion refinement pass
kernel void mgfg1MotionRefinement(
    texture2d<half, access::read> motionIn [[texture(0)]],
    texture2d<half, access::read> confidenceIn [[texture(1)]],
    texture2d<half, access::write> motionOut [[texture(2)]],
    uint2 gid [[thread_position_in_grid]]
) {
    uint width = motionIn.get_width();
    uint height = motionIn.get_height();
    if (gid.x >= width || gid.y >= height) return;
    
    half4 centerMV = motionIn.read(gid);
    half centerConf = confidenceIn.read(gid).r;
    
    // Weighted median filter for motion vectors
    half2 sumMV = half2(0.0h);
    half sumWeight = 0.0h;
    
    for (int dy = -1; dy <= 1; dy++) {
        for (int dx = -1; dx <= 1; dx++) {
            int2 pos = int2(gid) + int2(dx, dy);
            if (pos.x < 0 || pos.x >= int(width) ||
                pos.y < 0 || pos.y >= int(height)) continue;
            
            half4 mv = motionIn.read(uint2(pos));
            half conf = confidenceIn.read(uint2(pos)).r;
            
            // Spatial weight
            half spatialWeight = (dx == 0 && dy == 0) ? 2.0h : 1.0h;
            half weight = conf * spatialWeight;
            
            sumMV += half2(mv.x, mv.y) * weight;
            sumWeight += weight;
        }
    }
    
    half2 refinedMV = (sumWeight > 0.0h) ? (sumMV / sumWeight) : half2(centerMV.x, centerMV.y);
    
    // Blend with original based on confidence
    half2 finalMV = mix(half2(centerMV.x, centerMV.y), refinedMV, 0.5h);
    
    motionOut.write(half4(finalMV.x, finalMV.y, 0.0h, 1.0h), gid);
}

// ============================================================================
// Pyramid Processing Shaders
// ============================================================================

// Downsample by 2x with Gaussian blur
kernel void pyramidDownsample2x(
    texture2d<half, access::read> input [[texture(0)]],
    texture2d<half, access::write> output [[texture(1)]],
    uint2 gid [[thread_position_in_grid]]
) {
    uint outWidth = output.get_width();
    uint outHeight = output.get_height();
    if (gid.x >= outWidth || gid.y >= outHeight) return;
    
    uint2 srcBase = gid * 2;
    uint inWidth = input.get_width();
    uint inHeight = input.get_height();
    
    // 4-tap box filter
    half4 sum = half4(0.0h);
    for (int dy = 0; dy < 2; dy++) {
        for (int dx = 0; dx < 2; dx++) {
            uint2 srcPos = srcBase + uint2(dx, dy);
            srcPos = min(srcPos, uint2(inWidth - 1, inHeight - 1));
            sum += input.read(srcPos);
        }
    }
    
    output.write(sum * 0.25h, gid);
}

// Pyramid motion estimation (coarse level)
kernel void mgfg1PyramidMotionEstimation(
    texture2d<half, access::read> prevPyramid [[texture(0)]],
    texture2d<half, access::read> currPyramid [[texture(1)]],
    texture2d<half, access::read> prevLevelMotion [[texture(2)]],
    texture2d<half, access::write> motionOut [[texture(3)]],
    constant uint& pyramidLevel [[buffer(0)]],
    uint2 gid [[thread_position_in_grid]]
) {
    uint width = currPyramid.get_width();
    uint height = currPyramid.get_height();
    if (gid.x >= width || gid.y >= height) return;
    
    // Get motion prediction from coarser level (if available)
    half2 motionPred = half2(0.0h);
    if (pyramidLevel > 0) {
        uint2 coarsePos = gid / 2;
        half4 coarseMotion = prevLevelMotion.read(coarsePos);
        motionPred = half2(coarseMotion.x, coarseMotion.y) * 2.0h;
    }
    
    half currLuma = rgb2luma(currPyramid.read(gid).rgb);
    
    // Local search around prediction
    const int SEARCH_RADIUS = 2;
    half bestSAD = HALF_MAX;
    half2 bestMotion = motionPred;
    
    for (int dy = -SEARCH_RADIUS; dy <= SEARCH_RADIUS; dy++) {
        for (int dx = -SEARCH_RADIUS; dx <= SEARCH_RADIUS; dx++) {
            int2 searchPos = int2(gid) + int2(motionPred) + int2(dx, dy);
            
            if (searchPos.x < 0 || searchPos.x >= int(width) ||
                searchPos.y < 0 || searchPos.y >= int(height)) continue;
            
            half prevLuma = rgb2luma(prevPyramid.read(uint2(searchPos)).rgb);
            half sad = abs(currLuma - prevLuma);
            
            if (sad < bestSAD) {
                bestSAD = sad;
                bestMotion = half2(searchPos) - half2(gid);
            }
        }
    }
    
    motionOut.write(half4(bestMotion.x, bestMotion.y, 0.0h, 1.0h), gid);
}

// Upsample motion vectors
kernel void mgfg1UpsampleMotion(
    texture2d<half, access::read> coarseMotion [[texture(0)]],
    texture2d<half, access::write> fineMotion [[texture(1)]],
    uint2 gid [[thread_position_in_grid]]
) {
    uint width = fineMotion.get_width();
    uint height = fineMotion.get_height();
    if (gid.x >= width || gid.y >= height) return;
    
    // Bilinear interpolation from coarse level
    float2 coarsePos = float2(gid) * 0.5f;
    uint2 coarseBase = uint2(coarsePos);
    float2 frac = coarsePos - float2(coarseBase);
    
    uint coarseWidth = coarseMotion.get_width();
    uint coarseHeight = coarseMotion.get_height();
    
    uint2 p00 = min(coarseBase, uint2(coarseWidth - 1, coarseHeight - 1));
    uint2 p10 = min(coarseBase + uint2(1, 0), uint2(coarseWidth - 1, coarseHeight - 1));
    uint2 p01 = min(coarseBase + uint2(0, 1), uint2(coarseWidth - 1, coarseHeight - 1));
    uint2 p11 = min(coarseBase + uint2(1, 1), uint2(coarseWidth - 1, coarseHeight - 1));
    
    half4 mv00 = coarseMotion.read(p00);
    half4 mv10 = coarseMotion.read(p10);
    half4 mv01 = coarseMotion.read(p01);
    half4 mv11 = coarseMotion.read(p11);
    
    half4 mvTop = mix(mv00, mv10, half(frac.x));
    half4 mvBot = mix(mv01, mv11, half(frac.x));
    half4 mv = mix(mvTop, mvBot, half(frac.y));
    
    // Scale motion by 2 for finer level
    fineMotion.write(half4(mv.x * 2.0h, mv.y * 2.0h, mv.z, mv.w), gid);
}

// ============================================================================
// MGUP-1 Sharpening Shaders
// ============================================================================

struct SharpenParams {
    float sharpness;
    float radius;
};

// AMD CAS-style contrast adaptive sharpening
kernel void mgup1ContrastAdaptiveSharpening(
    texture2d<half, access::read> input [[texture(0)]],
    texture2d<half, access::write> output [[texture(1)]],
    constant SharpenParams& params [[buffer(0)]],
    uint2 gid [[thread_position_in_grid]]
) {
    uint width = input.get_width();
    uint height = input.get_height();
    if (gid.x >= width || gid.y >= height) return;
    
    // Sample 3x3 neighborhood
    half3 a = input.read(uint2(max(0u, gid.x - 1), max(0u, gid.y - 1))).rgb;
    half3 b = input.read(uint2(gid.x, max(0u, gid.y - 1))).rgb;
    half3 c = input.read(uint2(min(width - 1, gid.x + 1), max(0u, gid.y - 1))).rgb;
    half3 d = input.read(uint2(max(0u, gid.x - 1), gid.y)).rgb;
    half3 e = input.read(gid).rgb;
    half3 f = input.read(uint2(min(width - 1, gid.x + 1), gid.y)).rgb;
    half3 g = input.read(uint2(max(0u, gid.x - 1), min(height - 1, gid.y + 1))).rgb;
    half3 h = input.read(uint2(gid.x, min(height - 1, gid.y + 1))).rgb;
    half3 i = input.read(uint2(min(width - 1, gid.x + 1), min(height - 1, gid.y + 1))).rgb;
    
    // Soft min/max for contrast detection
    half3 minRGB = min(min(min(d, e), min(f, b)), h);
    half3 maxRGB = max(max(max(d, e), max(f, b)), h);
    
    // Wider min/max with corners
    half3 minRGB2 = min(min(minRGB, min(a, c)), min(g, i));
    half3 maxRGB2 = max(max(maxRGB, max(a, c)), max(g, i));
    
    // Smooth min/max
    half3 ampRGB = saturate(min(minRGB, 2.0h - maxRGB) / maxRGB);
    
    // Sharpening kernel
    half peak = half(8.0 - 3.0 * params.sharpness);
    half3 w = ampRGB / peak;
    
    half3 rcpW = 1.0h / (1.0h + 4.0h * w);
    half3 sharpened = saturate((b * w + d * w + f * w + h * w + e) * rcpW);
    
    output.write(half4(sharpened, 1.0h), gid);
}

// Unsharp mask sharpening
kernel void mgup1UnsharpMask(
    texture2d<half, access::read> input [[texture(0)]],
    texture2d<half, access::write> output [[texture(1)]],
    constant SharpenParams& params [[buffer(0)]],
    uint2 gid [[thread_position_in_grid]]
) {
    uint width = input.get_width();
    uint height = input.get_height();
    if (gid.x >= width || gid.y >= height) return;
    
    half3 center = input.read(gid).rgb;
    
    // Gaussian blur (3x3)
    half3 blur = half3(0.0h);
    half totalWeight = 0.0h;
    
    const half weights[3][3] = {
        {1.0h, 2.0h, 1.0h},
        {2.0h, 4.0h, 2.0h},
        {1.0h, 2.0h, 1.0h}
    };
    
    for (int dy = -1; dy <= 1; dy++) {
        for (int dx = -1; dx <= 1; dx++) {
            int2 pos = int2(gid) + int2(dx, dy);
            pos = clamp(pos, int2(0), int2(width - 1, height - 1));
            
            half w = weights[dy + 1][dx + 1];
            blur += input.read(uint2(pos)).rgb * w;
            totalWeight += w;
        }
    }
    blur /= totalWeight;
    
    // Unsharp mask: center + (center - blur) * strength
    half3 sharpened = center + (center - blur) * half(params.sharpness);
    sharpened = clampColor(sharpened);
    
    output.write(half4(sharpened, 1.0h), gid);
}

// ============================================================================
// SMAA (Subpixel Morphological Anti-Aliasing)
// ============================================================================

struct AntiAliasParams {
    float threshold;
    float depthThreshold;
    int maxSearchSteps;
    float subpixelBlend;
};

// SMAA Edge Detection Pass
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
    
    // Sample current pixel and neighbors
    half3 c = input.read(gid).rgb;
    half3 cLeft = input.read(uint2(max(0u, gid.x - 1), gid.y)).rgb;
    half3 cTop = input.read(uint2(gid.x, max(0u, gid.y - 1))).rgb;
    
    half lumaC = rgb2luma(c);
    half lumaLeft = rgb2luma(cLeft);
    half lumaTop = rgb2luma(cTop);
    
    // Edge detection
    half2 delta;
    delta.x = abs(lumaC - lumaLeft);
    delta.y = abs(lumaC - lumaTop);
    
    half2 edge = step(threshold, delta);
    
    edges.write(half4(edge.x, edge.y, 0.0h, 1.0h), gid);
}

// SMAA Blending Weight Calculation
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
    
    // Simplified weight calculation (full SMAA uses lookup textures)
    half4 weight = half4(0.0h);
    
    if (e.x > 0.0h) {
        // Horizontal edge - search left and right
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
        // Vertical edge - search up and down
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

// SMAA Final Blend Pass
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
    
    // No blending needed
    if (w.r == 0.0h && w.g == 0.0h && w.b == 0.0h && w.a == 0.0h) {
        output.write(c, gid);
        return;
    }
    
    // Blend with neighbors based on weights
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

// ============================================================================
// TAA (Temporal Anti-Aliasing)
// ============================================================================

// Temporal accumulation with history rejection
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
    
    // Neighborhood clamping to reduce ghosting
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
    
    // Clamp history to neighborhood
    half4 clampedHist = clamp(hist, minNeighbor, maxNeighbor);
    
    // Blend
    half4 result = mix(clampedHist, curr, half(blendFactor));
    output.write(result, gid);
}

// TAA main pass
kernel void taa(
    texture2d<half, access::read> input [[texture(0)]],
    texture2d<half, access::read> history [[texture(1)]],
    texture2d<half, access::write> output [[texture(2)]],
    constant AntiAliasParams& params [[buffer(0)]],
    uint2 gid [[thread_position_in_grid]]
) {
    uint width = input.get_width();
    uint height = input.get_height();
    if (gid.x >= width || gid.y >= height) return;
    
    half4 curr = input.read(gid);
    half4 hist = history.read(gid);
    
    // Color space conversion to YCoCg for better clamping
    half Y = rgb2luma(curr.rgb);
    half histY = rgb2luma(hist.rgb);
    
    // Variance clipping
    half4 m1 = half4(0.0h);
    half4 m2 = half4(0.0h);
    
    for (int dy = -1; dy <= 1; dy++) {
        for (int dx = -1; dx <= 1; dx++) {
            int2 pos = int2(gid) + int2(dx, dy);
            pos = clamp(pos, int2(0), int2(width - 1, height - 1));
            half4 s = input.read(uint2(pos));
            m1 += s;
            m2 += s * s;
        }
    }
    
    m1 /= 9.0h;
    m2 /= 9.0h;
    
    half4 sigma = sqrt(max(half4(0.0h), m2 - m1 * m1));
    half4 minC = m1 - sigma * 1.25h;
    half4 maxC = m1 + sigma * 1.25h;
    
    // Clamp history
    half4 clampedHist = clamp(hist, minC, maxC);
    
    // Adaptive blend based on luma difference
    half lumaDiff = abs(Y - histY);
    half blend = saturate(half(params.subpixelBlend) + lumaDiff * 0.5h);
    blend = clamp(blend, 0.05h, 0.5h);
    
    half4 result = mix(clampedHist, curr, blend);
    output.write(result, gid);
}

// MSAA resolve (edge-aware downsampling)
kernel void msaa(
    texture2d<half, access::read> input [[texture(0)]],
    texture2d<half, access::write> output [[texture(1)]],
    constant AntiAliasParams& params [[buffer(0)]],
    uint2 gid [[thread_position_in_grid]]
) {
    uint width = input.get_width();
    uint height = input.get_height();
    if (gid.x >= width || gid.y >= height) return;
    
    // Multi-sample approximation using edge detection
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
            
            // Weight by similarity
            half diff = abs(centerLuma - neighborLuma);
            half weight = exp(-diff * 10.0h) * 0.5h;
            
            sum += neighbor * weight;
            weightSum += weight;
        }
    }
    
    output.write(sum / weightSum, gid);
}

// ============================================================================
// Utility Shaders
// ============================================================================

// Texture copy (alias for passthrough_copy)
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

// Bilinear scaling
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
    
    // Calculate source position
    float2 srcPos = (float2(gid) + 0.5f) * float2(inWidth, inHeight) / float2(outWidth, outHeight) - 0.5f;
    srcPos = max(srcPos, float2(0.0f));
    
    uint2 srcBase = uint2(srcPos);
    float2 frac = srcPos - float2(srcBase);
    
    // Clamp coordinates
    uint2 p00 = min(srcBase, uint2(inWidth - 1, inHeight - 1));
    uint2 p10 = min(srcBase + uint2(1, 0), uint2(inWidth - 1, inHeight - 1));
    uint2 p01 = min(srcBase + uint2(0, 1), uint2(inWidth - 1, inHeight - 1));
    uint2 p11 = min(srcBase + uint2(1, 1), uint2(inWidth - 1, inHeight - 1));
    
    // Bilinear interpolation
    half4 c00 = input.read(p00);
    half4 c10 = input.read(p10);
    half4 c01 = input.read(p01);
    half4 c11 = input.read(p11);
    
    half4 top = mix(c00, c10, half(frac.x));
    half4 bot = mix(c01, c11, half(frac.x));
    half4 result = mix(top, bot, half(frac.y));
    
    output.write(result, gid);
}

