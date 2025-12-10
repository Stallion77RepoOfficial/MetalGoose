#include <metal_stdlib>
#include <metal_integer>
#include <metal_math>
using namespace metal;

struct VertexOut {
    float4 position [[position]];
    float2 texCoord;
};

struct MGFG1Params {
    float t;
    float motionScale;
    float occlusionThreshold;
    float temporalWeight;
    uint2 textureSize;
    uint qualityMode;
};

struct FrameTimingParams {
    float prevFrameTimestamp;
    float currFrameTimestamp;
    float targetFrameTime;
    float interpolationOffset;
};

struct AntiAliasParams {
    float threshold;
    float depthThreshold;
    int maxSearchSteps;
    float subpixelBlend;
};

struct SharpenParams {
    float sharpness;
    float radius;
};

inline float rgb2luma(float3 rgb) {
    return dot(rgb, float3(0.299, 0.587, 0.114));
}

inline float3 srgbToLinear(float3 srgb) {
    return pow(srgb, float3(2.2));
}

inline float3 linearToSrgb(float3 linear) {
    return pow(linear, float3(1.0 / 2.2));
}

inline half rgb2lumaH(half3 rgb) {
    return dot(rgb, half3(0.299h, 0.587h, 0.114h));
}

inline float4 catmullRom(float4 p0, float4 p1, float4 p2, float4 p3, float t) {
    float t2 = t * t;
    float t3 = t2 * t;
    return 0.5 * ((2.0 * p1) +
                  (-p0 + p2) * t +
                  (2.0 * p0 - 5.0 * p1 + 4.0 * p2 - p3) * t2 +
                  (-p0 + 3.0 * p1 - 3.0 * p2 + p3) * t3);
}

inline float smootherStep(float t) {
    return t * t * t * (t * (t * 6.0 - 15.0) + 10.0);
}

vertex VertexOut texture_vertex(uint vertexID [[vertex_id]]) {
    const float4 positions[4] = { 
        float4(-1, -1, 0, 1),  
        float4( 1, -1, 0, 1),  
        float4(-1,  1, 0, 1),  
        float4( 1,  1, 0, 1)   
    };
    const float2 coords[4] = { 
        float2(0, 1),  
        float2(1, 1),  
        float2(0, 0),  
        float2(1, 0)   
    };
    
    VertexOut out;
    out.position = positions[vertexID];
    out.texCoord = coords[vertexID];
    return out;
}

fragment float4 texture_fragment(
    VertexOut in [[stage_in]], 
    texture2d<float> texture [[texture(0)]]
) {
    constexpr sampler s(address::clamp_to_edge, filter::linear);
    return texture.sample(s, in.texCoord);
}

kernel void pyramidDownsample2x(
    texture2d<half, access::read> input [[texture(0)]],
    texture2d<half, access::write> output [[texture(1)]],
    uint2 gid [[thread_position_in_grid]]
) {
    uint2 outSize = uint2(output.get_width(), output.get_height());
    if (gid.x >= outSize.x || gid.y >= outSize.y) return;
    
    uint2 srcPos = gid * 2;
    
    half4 s00 = input.read(srcPos);
    half4 s10 = input.read(srcPos + uint2(1, 0));
    half4 s01 = input.read(srcPos + uint2(0, 1));
    half4 s11 = input.read(srcPos + uint2(1, 1));
    
    half4 result = (s00 + s10 + s01 + s11) * 0.25h;
    
    output.write(result, gid);
}

kernel void pyramidDownsample4x(
    texture2d<half, access::read> input [[texture(0)]],
    texture2d<half, access::write> output [[texture(1)]],
    uint2 gid [[thread_position_in_grid]]
) {
    uint2 outSize = uint2(output.get_width(), output.get_height());
    if (gid.x >= outSize.x || gid.y >= outSize.y) return;
    
    uint2 srcPos = gid * 4;
    
    half4 sum = half4(0.0h);
    for (uint dy = 0; dy < 4; dy++) {
        for (uint dx = 0; dx < 4; dx++) {
            sum += input.read(srcPos + uint2(dx, dy));
        }
    }
    
    output.write(sum * (1.0h / 16.0h), gid);
}

kernel void calculateInterpolationT(
    constant FrameTimingParams& timing [[buffer(0)]],
    device float* outputT [[buffer(1)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid != 0) return; 
    
    float frameInterval = timing.currFrameTimestamp - timing.prevFrameTimestamp;
    
    float t = timing.interpolationOffset / frameInterval;
    t = clamp(t, 0.0f, 1.0f);
    
    t = t * t * (3.0f - 2.0f * t);
    
    *outputT = t;
}

constant uint TILE_SIZE = 8;
constant uint SEARCH_PADDING = 16;  
constant uint TILE_WITH_PADDING = TILE_SIZE + SEARCH_PADDING * 2;

kernel void mgfg1MotionEstimationOptimized(
    texture2d<half, access::read> prevFrame [[texture(0)]],
    texture2d<half, access::read> currFrame [[texture(1)]],
    texture2d<half, access::write> motionVectors [[texture(2)]],  
    texture2d<half, access::write> confidence [[texture(3)]],
    constant MGFG1Params& params [[buffer(0)]],
    uint2 gid [[thread_position_in_grid]],
    uint2 tid [[thread_position_in_threadgroup]],
    uint2 tgid [[threadgroup_position_in_grid]]
) {
    threadgroup half prevTile[TILE_WITH_PADDING][TILE_WITH_PADDING];
    threadgroup half currTile[TILE_SIZE][TILE_SIZE];
    
    uint2 texSize = uint2(motionVectors.get_width(), motionVectors.get_height());
    
    int2 tileBase = int2(tgid) * int2(TILE_SIZE) - int2(SEARCH_PADDING);
    
    uint loadSteps = (TILE_WITH_PADDING * TILE_WITH_PADDING + TILE_SIZE * TILE_SIZE - 1) / (TILE_SIZE * TILE_SIZE);
    uint flatTid = tid.y * TILE_SIZE + tid.x;
    
    for (uint step = 0; step < loadSteps; step++) {
        uint flatIdx = flatTid + step * TILE_SIZE * TILE_SIZE;
        if (flatIdx < TILE_WITH_PADDING * TILE_WITH_PADDING) {
            uint loadY = flatIdx / TILE_WITH_PADDING;
            uint loadX = flatIdx % TILE_WITH_PADDING;
            int2 loadPos = tileBase + int2(loadX, loadY);
            loadPos = clamp(loadPos, int2(0), int2(texSize) - 1);
            prevTile[loadY][loadX] = rgb2lumaH(prevFrame.read(uint2(loadPos)).rgb);
        }
    }
    
    if (gid.x < texSize.x && gid.y < texSize.y) {
        currTile[tid.y][tid.x] = rgb2lumaH(currFrame.read(gid).rgb);
    }
    
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    if (gid.x >= texSize.x || gid.y >= texSize.y) return;
    
    half currLuma = currTile[tid.y][tid.x];
    
    const int searchRadius[3] = {16, 8, 2};
    
    half2 bestMotion = half2(0.0h);
    half minSAD = half(1e10);
    
    half2 searchCenter = half2(0.0h);
    
    int2 sharedPos = int2(tid) + int2(SEARCH_PADDING);
    
    for (int level = 0; level < 3; level++) {
        int radius = searchRadius[level];
        int step = max(1, radius / 4);
        
        for (int dy = -radius; dy <= radius; dy += step) {
            for (int dx = -radius; dx <= radius; dx += step) {
                int2 offset = int2(searchCenter) + int2(dx, dy);
                int2 samplePos = sharedPos + offset;
                
                if (samplePos.x >= 0 && samplePos.x < int(TILE_WITH_PADDING) &&
                    samplePos.y >= 0 && samplePos.y < int(TILE_WITH_PADDING)) {
                    
                    half prevLuma = prevTile[samplePos.y][samplePos.x];
                    half sad = abs(currLuma - prevLuma);
                    
                    if (sad < minSAD) {
                        minSAD = sad;
                        bestMotion = half2(offset);
                    }
                }
            }
        }
        searchCenter = bestMotion;
    }
    
    if (length(float2(bestMotion)) > 0.5f) {
        int2 basePos = sharedPos + int2(bestMotion);
        if (basePos.x > 0 && basePos.x < int(TILE_WITH_PADDING) - 1 &&
            basePos.y > 0 && basePos.y < int(TILE_WITH_PADDING) - 1) {
            
            half lumaN = prevTile[basePos.y - 1][basePos.x];
            half lumaS = prevTile[basePos.y + 1][basePos.x];
            half lumaW = prevTile[basePos.y][basePos.x - 1];
            half lumaE = prevTile[basePos.y][basePos.x + 1];
            half lumaC = prevTile[basePos.y][basePos.x];
            
            half2 gradient = half2(lumaE - lumaW, lumaS - lumaN) * 0.5h;
            half diff = currLuma - lumaC;
            
            half gradLen = length(gradient);
            if (gradLen > 0.001h) {
                half2 subpixel = gradient * diff / (gradLen * gradLen);
                subpixel = clamp(subpixel, half2(-0.5h), half2(0.5h));
                bestMotion += subpixel;
            }
        }
    }
    
    half conf = saturate(1.0h - minSAD * 5.0h);
    
    bestMotion *= half(params.motionScale);
    
    motionVectors.write(half4(bestMotion, minSAD, 1.0h), gid);
    confidence.write(half4(conf, conf, conf, 1.0h), gid);
}

kernel void mgfg1MotionEstimation(
    texture2d<float, access::read> prevFrame [[texture(0)]],
    texture2d<float, access::read> currFrame [[texture(1)]],
    texture2d<float, access::write> motionVectors [[texture(2)]],
    texture2d<float, access::write> confidence [[texture(3)]],
    constant MGFG1Params& params [[buffer(0)]],
    uint2 gid [[thread_position_in_grid]]
) {
    uint2 texSize = uint2(motionVectors.get_width(), motionVectors.get_height());
    if (gid.x >= texSize.x || gid.y >= texSize.y) return;
    
    const int searchRadius[3] = {16, 8, 2};  
    
    float2 bestMotion = float2(0.0);
    float minSAD = FLT_MAX;
    
    float3 currBlock = currFrame.read(gid).rgb;
    float currLuma = rgb2luma(currBlock);
    
    float2 searchCenter = float2(0.0);
    
    for (int level = 0; level < 3; level++) {
        int radius = searchRadius[level];
        int step = max(1, radius / 4);
        
        for (int dy = -radius; dy <= radius; dy += step) {
            for (int dx = -radius; dx <= radius; dx += step) {
                float2 offset = searchCenter + float2(dx, dy);
                int2 samplePos = int2(gid) + int2(offset);
                
                if (samplePos.x < 0 || samplePos.x >= int(texSize.x) ||
                    samplePos.y < 0 || samplePos.y >= int(texSize.y)) continue;
                
                float3 prevBlock = prevFrame.read(uint2(samplePos)).rgb;
                float prevLuma = rgb2luma(prevBlock);
                
                float sad = abs(currLuma - prevLuma);
                sad += length(currBlock - prevBlock) * 0.3;
                
                if (sad < minSAD) {
                    minSAD = sad;
                    bestMotion = offset;
                }
            }
        }
        searchCenter = bestMotion;
    }
    
    if (length(bestMotion) > 0.5) {
        int2 basePos = int2(gid) + int2(bestMotion);
        if (basePos.x > 0 && basePos.x < int(texSize.x) - 1 &&
            basePos.y > 0 && basePos.y < int(texSize.y) - 1) {
            
            float lumaN = rgb2luma(prevFrame.read(uint2(basePos.x, basePos.y - 1)).rgb);
            float lumaS = rgb2luma(prevFrame.read(uint2(basePos.x, basePos.y + 1)).rgb);
            float lumaW = rgb2luma(prevFrame.read(uint2(basePos.x - 1, basePos.y)).rgb);
            float lumaE = rgb2luma(prevFrame.read(uint2(basePos.x + 1, basePos.y)).rgb);
            float lumaC = rgb2luma(prevFrame.read(uint2(basePos)).rgb);
            
            float2 gradient = float2(lumaE - lumaW, lumaS - lumaN) * 0.5;
            float diff = currLuma - lumaC;
            
            if (length(gradient) > 0.001) {
                float2 subpixel = gradient * diff / dot(gradient, gradient);
                subpixel = clamp(subpixel, float2(-0.5), float2(0.5));
                bestMotion += subpixel;
            }
        }
    }
    
    float conf = saturate(1.0 - minSAD * 5.0);
    
    bestMotion *= params.motionScale;
    
    motionVectors.write(float4(bestMotion, minSAD, 1.0), gid);
    confidence.write(float4(conf, conf, conf, 1.0), gid);
}

kernel void mgfg1MotionRefinement(
    texture2d<float, access::read> coarseMotion [[texture(0)]],
    texture2d<float, access::read> confidence [[texture(1)]],
    texture2d<float, access::write> refinedMotion [[texture(2)]],
    uint2 gid [[thread_position_in_grid]]
) {
    uint2 texSize = uint2(refinedMotion.get_width(), refinedMotion.get_height());
    if (gid.x >= texSize.x || gid.y >= texSize.y) return;
    
    float4 centerMV = coarseMotion.read(gid);
    float centerConf = confidence.read(gid).r;
    
    float2 motionSum = float2(0.0);
    float weightSum = 0.0;
    
    const int kernelRadius = 2;
    
    for (int dy = -kernelRadius; dy <= kernelRadius; dy++) {
        for (int dx = -kernelRadius; dx <= kernelRadius; dx++) {
            int2 samplePos = int2(gid) + int2(dx, dy);
            
            if (samplePos.x < 0 || samplePos.x >= int(texSize.x) ||
                samplePos.y < 0 || samplePos.y >= int(texSize.y)) continue;
            
            float4 neighborMV = coarseMotion.read(uint2(samplePos));
            float neighborConf = confidence.read(uint2(samplePos)).r;
            
            float dist = length(float2(dx, dy));
            float spatialWeight = exp(-dist * 0.5);
            float weight = neighborConf * spatialWeight;
            
            float mvDiff = length(neighborMV.xy - centerMV.xy);
            if (mvDiff < 8.0) {
                motionSum += neighborMV.xy * weight;
                weightSum += weight;
            }
        }
    }
    
    float2 refinedMV = (weightSum > 0.001) ? motionSum / weightSum : centerMV.xy;
    
    refinedMV = mix(refinedMV, centerMV.xy, centerConf * 0.5);
    
    refinedMotion.write(float4(refinedMV, centerMV.z, 1.0), gid);
}

kernel void mgfg1PyramidMotionEstimation(
    texture2d<half, access::read> prevPyramid [[texture(0)]],   
    texture2d<half, access::read> currPyramid [[texture(1)]],   
    texture2d<half, access::read> coarserMotion [[texture(2)]], 
    texture2d<half, access::write> motionOut [[texture(3)]],    
    constant uint& pyramidLevel [[buffer(0)]],                  
    uint2 gid [[thread_position_in_grid]]
) {
    uint2 texSize = uint2(motionOut.get_width(), motionOut.get_height());
    if (gid.x >= texSize.x || gid.y >= texSize.y) return;
    
    half2 initialMotion = half2(0.0h);
    if (pyramidLevel > 0) {
        uint2 coarserPos = gid / 2;
        coarserPos = clamp(coarserPos, uint2(0), uint2(coarserMotion.get_width() - 1, coarserMotion.get_height() - 1));
        initialMotion = coarserMotion.read(coarserPos).xy * 2.0h;
    }
    
    half currLuma = rgb2lumaH(currPyramid.read(gid).rgb);
    
    int searchRadius = (pyramidLevel == 0) ? 4 : 8;
    
    half2 bestMotion = initialMotion;
    half minSAD = half(1e10);
    
    for (int dy = -searchRadius; dy <= searchRadius; dy += 2) {
        for (int dx = -searchRadius; dx <= searchRadius; dx += 2) {
            half2 offset = initialMotion + half2(dx, dy);
            int2 samplePos = int2(gid) + int2(offset);
            
            if (samplePos.x >= 0 && samplePos.x < int(texSize.x) &&
                samplePos.y >= 0 && samplePos.y < int(texSize.y)) {
                
                half prevLuma = rgb2lumaH(prevPyramid.read(uint2(samplePos)).rgb);
                half sad = abs(currLuma - prevLuma);
                
                if (sad < minSAD) {
                    minSAD = sad;
                    bestMotion = offset;
                }
            }
        }
    }
    
    half2 fineCenter = bestMotion;
    for (int dy = -2; dy <= 2; dy++) {
        for (int dx = -2; dx <= 2; dx++) {
            half2 offset = fineCenter + half2(dx, dy);
            int2 samplePos = int2(gid) + int2(offset);
            
            if (samplePos.x >= 0 && samplePos.x < int(texSize.x) &&
                samplePos.y >= 0 && samplePos.y < int(texSize.y)) {
                
                half prevLuma = rgb2lumaH(prevPyramid.read(uint2(samplePos)).rgb);
                half sad = abs(currLuma - prevLuma);
                
                if (sad < minSAD) {
                    minSAD = sad;
                    bestMotion = offset;
                }
            }
        }
    }
    
    half conf = saturate(1.0h - minSAD * 5.0h);
    motionOut.write(half4(bestMotion, minSAD, conf), gid);
}

kernel void mgfg1UpsampleMotion(
    texture2d<half, access::read> coarseMotion [[texture(0)]],
    texture2d<half, access::write> fineMotion [[texture(1)]],
    uint2 gid [[thread_position_in_grid]]
) {
    uint2 texSize = uint2(fineMotion.get_width(), fineMotion.get_height());
    if (gid.x >= texSize.x || gid.y >= texSize.y) return;
    
    float2 srcPos = float2(gid) * 0.5f;
    uint2 srcPosInt = uint2(srcPos);
    float2 frac = fract(srcPos);
    
    srcPosInt = clamp(srcPosInt, uint2(0), uint2(coarseMotion.get_width() - 1, coarseMotion.get_height() - 1));
    uint2 srcPos1 = clamp(srcPosInt + uint2(1, 0), uint2(0), uint2(coarseMotion.get_width() - 1, coarseMotion.get_height() - 1));
    uint2 srcPos2 = clamp(srcPosInt + uint2(0, 1), uint2(0), uint2(coarseMotion.get_width() - 1, coarseMotion.get_height() - 1));
    uint2 srcPos3 = clamp(srcPosInt + uint2(1, 1), uint2(0), uint2(coarseMotion.get_width() - 1, coarseMotion.get_height() - 1));
    
    half4 m00 = coarseMotion.read(srcPosInt);
    half4 m10 = coarseMotion.read(srcPos1);
    half4 m01 = coarseMotion.read(srcPos2);
    half4 m11 = coarseMotion.read(srcPos3);
    
    half4 m0 = mix(m00, m10, half(frac.x));
    half4 m1 = mix(m01, m11, half(frac.x));
    half4 motion = mix(m0, m1, half(frac.y));
    
    motion.xy *= 2.0h;
    
    fineMotion.write(motion, gid);
}

kernel void mgfg1Performance(
    texture2d<float, access::read> prevFrame [[texture(0)]],
    texture2d<float, access::read> currFrame [[texture(1)]],
    texture2d<float, access::write> output [[texture(2)]],
    constant float& t [[buffer(0)]],
    uint2 gid [[thread_position_in_grid]]
) {
    if (gid.x >= output.get_width() || gid.y >= output.get_height()) return;
    
    float4 result;
    
    if (t < 0.35) {
        result = prevFrame.read(gid);
    } else if (t > 0.65) {
        result = currFrame.read(gid);
    } else {
        float blendT = (t - 0.35) / 0.30;
        blendT = smootherStep(blendT);
        result = mix(prevFrame.read(gid), currFrame.read(gid), blendT);
    }
    
    output.write(result, gid);
}

struct BalancedParams {
    float t;
    uint textureWidth;    
    uint textureHeight;
    float gradientThreshold;
    float padding;  
};

kernel void mgfg1Balanced(
    texture2d<float, access::read> prevFrame [[texture(0)]],
    texture2d<float, access::read> currFrame [[texture(1)]],
    texture2d<float, access::write> output [[texture(2)]],
    constant BalancedParams& params [[buffer(0)]],
    uint2 gid [[thread_position_in_grid]]
) {
    uint2 texSize = uint2(params.textureWidth, params.textureHeight);
    if (texSize.x == 0 || texSize.y == 0) {
        texSize = uint2(output.get_width(), output.get_height());
    }
    if (gid.x >= texSize.x || gid.y >= texSize.y) return;
    
    float4 prev = prevFrame.read(gid);
    float4 curr = currFrame.read(gid);
    
    float3 diff = abs(curr.rgb - prev.rgb);
    float motion = saturate(dot(diff, float3(0.299, 0.587, 0.114)) * 3.0);
    
    float2 gradient = float2(0.0);
    if (gid.x > 0 && gid.x < texSize.x - 1) {
        float left = rgb2luma(currFrame.read(uint2(gid.x - 1, gid.y)).rgb);
        float right = rgb2luma(currFrame.read(uint2(gid.x + 1, gid.y)).rgb);
        float prevLeft = rgb2luma(prevFrame.read(uint2(gid.x - 1, gid.y)).rgb);
        float prevRight = rgb2luma(prevFrame.read(uint2(gid.x + 1, gid.y)).rgb);
        gradient.x = (right - left) * 0.5 + (prevRight - prevLeft) * 0.25;
    }
    if (gid.y > 0 && gid.y < texSize.y - 1) {
        float up = rgb2luma(currFrame.read(uint2(gid.x, gid.y - 1)).rgb);
        float down = rgb2luma(currFrame.read(uint2(gid.x, gid.y + 1)).rgb);
        float prevUp = rgb2luma(prevFrame.read(uint2(gid.x, gid.y - 1)).rgb);
        float prevDown = rgb2luma(prevFrame.read(uint2(gid.x, gid.y + 1)).rgb);
        gradient.y = (down - up) * 0.5 + (prevDown - prevUp) * 0.25;
    }
    
    float gradMag = length(gradient);
    if (gradMag < params.gradientThreshold) {
        gradient = float2(0.0);
    }
    
    float2 offset = gradient * motion * 1.0 * (0.5 - params.t);
    offset = clamp(offset, float2(-8.0), float2(8.0));
    
    int2 warpPosPrev = int2(float2(gid) + offset);
    int2 warpPosCurr = int2(float2(gid) - offset * (1.0 - params.t) / max(params.t, 0.01));
    
    warpPosPrev = clamp(warpPosPrev, int2(0), int2(texSize) - 1);
    warpPosCurr = clamp(warpPosCurr, int2(0), int2(texSize) - 1);
    
    float4 warpedPrev = prevFrame.read(uint2(warpPosPrev));
    float4 warpedCurr = currFrame.read(uint2(warpPosCurr));
    
    float warpedDiff = length(warpedPrev.rgb - warpedCurr.rgb);
    float occlusion = saturate(warpedDiff * 2.0);
    
    float smoothT = smootherStep(params.t);
    
    float4 motionBlend = mix(warpedPrev, warpedCurr, smoothT);
    float4 simpleBlend = mix(prev, curr, smoothT);
    
    float blendWeight = motion * (1.0 - occlusion * 0.7) * 0.5;
    float4 result = mix(simpleBlend, motionBlend, blendWeight);
    
    output.write(result, gid);
}

kernel void mgfg1Quality(
    texture2d<float, access::read> prevFrame [[texture(0)]],
    texture2d<float, access::read> currFrame [[texture(1)]],
    texture2d<float, access::read> motionVectors [[texture(2)]],
    texture2d<float, access::read> confidence [[texture(3)]],
    texture2d<float, access::write> output [[texture(4)]],
    constant MGFG1Params& params [[buffer(0)]],
    uint2 gid [[thread_position_in_grid]]
) {
    uint2 texSize = params.textureSize;
    if (texSize.x == 0 || texSize.y == 0) {
        texSize = uint2(output.get_width(), output.get_height());
    }
    if (gid.x >= texSize.x || gid.y >= texSize.y) return;
    
    float2 uv = float2(gid) / float2(texSize);
    float edgeMargin = 0.03f;
    bool isNearEdge = uv.x < edgeMargin || uv.x > (1.0f - edgeMargin) || 
                      uv.y < edgeMargin || uv.y > (1.0f - edgeMargin);
    float edgeFactor = isNearEdge ? 0.3f : 1.0f;
    
    float4 mvData = motionVectors.read(gid);
    float2 motion = mvData.xy;
    float conf = confidence.read(gid).r;
    
    motion *= params.motionScale * edgeFactor;
    conf *= edgeFactor;
    
    float motionMag = length(motion);
    const float maxMotion = 48.0;  
    if (motionMag > maxMotion) {
        float scale = maxMotion / motionMag;
        motion *= scale;
        conf *= scale;  
    }
    
    float t = clamp(params.t, 0.0f, 1.0f);
    float2 pos1 = float2(gid) - motion * t;
    float2 pos2 = float2(gid) + motion * (1.0 - t);
    
    pos1 = clamp(pos1, float2(0.0), float2(texSize) - 1.0);
    pos2 = clamp(pos2, float2(0.0), float2(texSize) - 1.0);
    
    float4 color1 = prevFrame.read(uint2(clamp(pos1, float2(0), float2(texSize) - 1)));
    float4 color2 = currFrame.read(uint2(clamp(pos2, float2(0), float2(texSize) - 1)));
    
    float colorDiff = length(color1.rgb - color2.rgb);
    float occlusionWeight = saturate(colorDiff / params.occlusionThreshold);
    
    float4 directPrev = prevFrame.read(gid);
    float4 directCurr = currFrame.read(gid);
    
    float smoothT = smootherStep(params.t);
    float4 motionBlend = mix(color1, color2, smoothT);
    float4 simpleBlend = mix(directPrev, directCurr, smoothT);
    
    float blendWeight = conf * (1.0 - occlusionWeight);
    if (isNearEdge) blendWeight = min(blendWeight, 0.3f);
    
    float4 result = mix(simpleBlend, motionBlend, blendWeight);
    
    output.write(result, gid);
}

kernel void mgfg1AdaptiveInterpolation(
    texture2d<float, access::read> prevFrame [[texture(0)]],
    texture2d<float, access::read> currFrame [[texture(1)]],
    texture2d<float, access::read> motionVectors [[texture(2)]],
    texture2d<float, access::read> confidence [[texture(3)]],
    texture2d<float, access::write> output [[texture(4)]],
    constant MGFG1Params& params [[buffer(0)]],
    uint2 gid [[thread_position_in_grid]]
) {
    uint2 texSize = uint2(output.get_width(), output.get_height());
    if (gid.x >= texSize.x || gid.y >= texSize.y) return;
    
    float2 uv = float2(gid) / float2(texSize);
    float edgeMargin = 0.03f;
    bool isNearEdge = uv.x < edgeMargin || uv.x > (1.0f - edgeMargin) || 
                      uv.y < edgeMargin || uv.y > (1.0f - edgeMargin);
    
    float4 prev = prevFrame.read(gid);
    float4 curr = currFrame.read(gid);
    float4 mvData = motionVectors.read(gid);
    float conf = confidence.read(gid).r;
    
    if (isNearEdge) conf *= 0.3f;
    
    float2 motion = mvData.xy;
    float motionMag = length(motion);
    
    if (isNearEdge) motion *= 0.3f;
    
    float4 result;
    float smoothT = smootherStep(params.t);
    
    if (motionMag < 2.0 || conf < 0.2 || isNearEdge) {
        result = mix(prev, curr, smoothT);
    } else if (motionMag < 16.0) {
        float2 pos1 = clamp(float2(gid) - motion * params.t, float2(0), float2(texSize) - 1.0);
        float2 pos2 = clamp(float2(gid) + motion * (1.0 - params.t), float2(0), float2(texSize) - 1.0);
        
        float4 warped1 = prevFrame.read(uint2(pos1));
        float4 warped2 = currFrame.read(uint2(pos2));
        
        float4 motionBlend = mix(warped1, warped2, smoothT);
        float4 simpleBlend = mix(prev, curr, smoothT);
        
        result = mix(simpleBlend, motionBlend, conf);
    } else {
        if (params.t < 0.4) {
            result = prev;
        } else if (params.t > 0.6) {
            result = curr;
        } else {
            float narrowT = (params.t - 0.4) / 0.2;
            result = mix(prev, curr, narrowT);
        }
    }
    
    output.write(result, gid);
}

kernel void fxaa(
    texture2d<float, access::read> input [[texture(0)]],
    texture2d<float, access::write> output [[texture(1)]],
    constant AntiAliasParams& params [[buffer(0)]],
    uint2 gid [[thread_position_in_grid]]
) {
    uint width = input.get_width();
    uint height = input.get_height();
    if (gid.x >= width || gid.y >= height) return;
    
    const float FXAA_REDUCE_MUL = 1.0 / 8.0;
    const float FXAA_REDUCE_MIN = 1.0 / 128.0;
    const float FXAA_SPAN_MAX = 8.0;
    
    float3 rgbNW = input.read(uint2(max(0u, gid.x - 1), max(0u, gid.y - 1))).rgb;
    float3 rgbNE = input.read(uint2(min(width - 1, gid.x + 1), max(0u, gid.y - 1))).rgb;
    float3 rgbSW = input.read(uint2(max(0u, gid.x - 1), min(height - 1, gid.y + 1))).rgb;
    float3 rgbSE = input.read(uint2(min(width - 1, gid.x + 1), min(height - 1, gid.y + 1))).rgb;
    float3 rgbM = input.read(gid).rgb;
    
    float lumaNW = rgb2luma(rgbNW);
    float lumaNE = rgb2luma(rgbNE);
    float lumaSW = rgb2luma(rgbSW);
    float lumaSE = rgb2luma(rgbSE);
    float lumaM = rgb2luma(rgbM);
    
    float lumaMin = min(lumaM, min(min(lumaNW, lumaNE), min(lumaSW, lumaSE)));
    float lumaMax = max(lumaM, max(max(lumaNW, lumaNE), max(lumaSW, lumaSE)));
    
    float2 dir;
    dir.x = -((lumaNW + lumaNE) - (lumaSW + lumaSE));
    dir.y = ((lumaNW + lumaSW) - (lumaNE + lumaSE));
    
    float dirReduce = max((lumaNW + lumaNE + lumaSW + lumaSE) * (0.25 * FXAA_REDUCE_MUL), FXAA_REDUCE_MIN);
    float rcpDirMin = 1.0 / (min(abs(dir.x), abs(dir.y)) + dirReduce);
    
    dir = min(float2(FXAA_SPAN_MAX), max(float2(-FXAA_SPAN_MAX), dir * rcpDirMin));
    
    float2 texOffset1 = dir * (1.0 / 3.0 - 0.5);
    float2 texOffset2 = dir * (2.0 / 3.0 - 0.5);
    
    int2 pos1 = int2(float2(gid) + texOffset1);
    int2 pos2 = int2(float2(gid) + texOffset2);
    
    pos1 = clamp(pos1, int2(0), int2(width - 1, height - 1));
    pos2 = clamp(pos2, int2(0), int2(width - 1, height - 1));
    
    float3 rgbA = (input.read(uint2(pos1)).rgb + input.read(uint2(pos2)).rgb) * 0.5;
    
    float2 texOffset3 = dir * -0.5;
    float2 texOffset4 = dir * 0.5;
    
    int2 pos3 = int2(float2(gid) + texOffset3);
    int2 pos4 = int2(float2(gid) + texOffset4);
    
    pos3 = clamp(pos3, int2(0), int2(width - 1, height - 1));
    pos4 = clamp(pos4, int2(0), int2(width - 1, height - 1));
    
    float3 rgbB = rgbA * 0.5 + (input.read(uint2(pos3)).rgb + input.read(uint2(pos4)).rgb) * 0.25;
    
    float lumaB = rgb2luma(rgbB);
    
    float3 result;
    if (lumaB < lumaMin || lumaB > lumaMax) {
        result = rgbA;
    } else {
        result = rgbB;
    }
    
    output.write(float4(result, 1.0), gid);
}

kernel void smaaEdgeDetection(
    texture2d<float, access::read> input [[texture(0)]],
    texture2d<float, access::write> edges [[texture(1)]],
    constant AntiAliasParams& params [[buffer(0)]],
    uint2 gid [[thread_position_in_grid]]
) {
    uint width = input.get_width();
    uint height = input.get_height();
    if (gid.x >= width || gid.y >= height) return;
    
    float threshold = params.threshold;
    
    float L = rgb2luma(input.read(gid).rgb);
    float Lleft = rgb2luma(input.read(uint2(max(0u, gid.x - 1), gid.y)).rgb);
    float Ltop = rgb2luma(input.read(uint2(gid.x, max(0u, gid.y - 1))).rgb);
    float Lright = rgb2luma(input.read(uint2(min(width - 1, gid.x + 1), gid.y)).rgb);
    float Lbottom = rgb2luma(input.read(uint2(gid.x, min(height - 1, gid.y + 1))).rgb);
    
    float4 delta = abs(float4(L) - float4(Lleft, Ltop, Lright, Lbottom));
    
    float2 edges_val = step(float2(threshold), delta.xy);
    
    if (dot(edges_val, float2(1.0)) == 0.0) {
        edges.write(float4(0.0), gid);
        return;
    }
    
    float Lleftleft = rgb2luma(input.read(uint2(max(0u, gid.x - 2), gid.y)).rgb);
    float Ltoptop = rgb2luma(input.read(uint2(gid.x, max(0u, gid.y - 2))).rgb);
    
    float4 delta2 = abs(float4(Lleft, Ltop, Lleft, Ltop) - float4(Lleftleft, Ltoptop, L, L));
    float2 maxDelta = max(delta.xy, delta2.xy);
    
    edges_val = step(float2(threshold), maxDelta);
    
    edges.write(float4(edges_val, 0.0, 1.0), gid);
}

kernel void smaaBlendingWeights(
    texture2d<float, access::read> edges [[texture(0)]],
    texture2d<float, access::write> weights [[texture(1)]],
    constant AntiAliasParams& params [[buffer(0)]],
    uint2 gid [[thread_position_in_grid]]
) {
    uint width = edges.get_width();
    uint height = edges.get_height();
    if (gid.x >= width || gid.y >= height) return;
    
    float4 e = edges.read(gid);
    
    if (e.x == 0.0 && e.y == 0.0) {
        weights.write(float4(0.0), gid);
        return;
    }
    
    float4 blendWeights = float4(0.0);
    int maxSearchSteps = params.maxSearchSteps;
    
    if (e.x > 0.0) {
        int distance = 0;
        for (int i = 1; i <= maxSearchSteps; i++) {
            int2 pos = int2(gid) + int2(-i, 0);
            if (pos.x < 0) break;
            float4 edgeSample = edges.read(uint2(pos));
            if (edgeSample.x == 0.0) break;
            distance = i;
        }
        blendWeights.x = float(distance) / float(maxSearchSteps);
        
        distance = 0;
        for (int i = 1; i <= maxSearchSteps; i++) {
            int2 pos = int2(gid) + int2(i, 0);
            if (pos.x >= int(width)) break;
            float4 edgeSample = edges.read(uint2(pos));
            if (edgeSample.x == 0.0) break;
            distance = i;
        }
        blendWeights.y = float(distance) / float(maxSearchSteps);
    }
    
    if (e.y > 0.0) {
        int distance = 0;
        for (int i = 1; i <= maxSearchSteps; i++) {
            int2 pos = int2(gid) + int2(0, -i);
            if (pos.y < 0) break;
            float4 edgeSample = edges.read(uint2(pos));
            if (edgeSample.y == 0.0) break;
            distance = i;
        }
        blendWeights.z = float(distance) / float(maxSearchSteps);
        
        distance = 0;
        for (int i = 1; i <= maxSearchSteps; i++) {
            int2 pos = int2(gid) + int2(0, i);
            if (pos.y >= int(height)) break;
            float4 edgeSample = edges.read(uint2(pos));
            if (edgeSample.y == 0.0) break;
            distance = i;
        }
        blendWeights.w = float(distance) / float(maxSearchSteps);
    }
    
    weights.write(blendWeights, gid);
}

kernel void smaaBlend(
    texture2d<float, access::read> input [[texture(0)]],
    texture2d<float, access::read> weights [[texture(1)]],
    texture2d<float, access::write> output [[texture(2)]],
    uint2 gid [[thread_position_in_grid]]
) {
    uint width = input.get_width();
    uint height = input.get_height();
    if (gid.x >= width || gid.y >= height) return;
    
    float4 w = weights.read(gid);
    
    if (dot(w, float4(1.0)) == 0.0) {
        output.write(input.read(gid), gid);
        return;
    }
    
    float4 color = input.read(gid);
    
    if (w.x + w.y > 0.0) {
        float blend = (w.x + w.y) * 0.5;
        float4 left = input.read(uint2(max(0u, gid.x - 1), gid.y));
        float4 right = input.read(uint2(min(width - 1, gid.x + 1), gid.y));
        color = mix(color, (left + right) * 0.5, blend * 0.5);
    }
    
    if (w.z + w.w > 0.0) {
        float blend = (w.z + w.w) * 0.5;
        float4 top = input.read(uint2(gid.x, max(0u, gid.y - 1)));
        float4 bottom = input.read(uint2(gid.x, min(height - 1, gid.y + 1)));
        color = mix(color, (top + bottom) * 0.5, blend * 0.5);
    }
    
    output.write(color, gid);
}

kernel void msaa(
    texture2d<float, access::read> input [[texture(0)]],
    texture2d<float, access::write> output [[texture(1)]],
    constant AntiAliasParams& params [[buffer(0)]],
    uint2 gid [[thread_position_in_grid]]
) {
    uint width = input.get_width();
    uint height = input.get_height();
    if (gid.x >= width || gid.y >= height) return;
    
    float threshold = params.threshold;
    float subpixelBlend = params.subpixelBlend;
    
    float4 center = input.read(gid);
    float centerLuma = rgb2luma(center.rgb);
    
    float4 samples[8];
    float lumas[8];
    
    const int2 offsets[8] = {
        int2(-1, -1), int2(0, -1), int2(1, -1),
        int2(-1,  0),              int2(1,  0),
        int2(-1,  1), int2(0,  1), int2(1,  1)
    };
    
    float edgeWeight = 0.0;
    
    for (int i = 0; i < 8; i++) {
        int2 pos = int2(gid) + offsets[i];
        pos = clamp(pos, int2(0), int2(width - 1, height - 1));
        samples[i] = input.read(uint2(pos));
        lumas[i] = rgb2luma(samples[i].rgb);
        
        float diff = abs(centerLuma - lumas[i]);
        if (diff > threshold) {
            edgeWeight += 1.0;
        }
    }
    
    if (edgeWeight == 0.0) {
        output.write(center, gid);
        return;
    }
    
    edgeWeight = saturate(edgeWeight / 8.0);
    
    float4 blended = float4(0.0);
    float totalWeight = 0.0;
    
    for (int i = 0; i < 8; i++) {
        float lumaDiff = abs(centerLuma - lumas[i]);
        float weight = 1.0 - saturate(lumaDiff / threshold);
        weight = weight * weight;
        blended += samples[i] * weight;
        totalWeight += weight;
    }
    
    if (totalWeight > 0.0) {
        blended /= totalWeight;
    } else {
        blended = center;
    }
    
    float blendFactor = edgeWeight * subpixelBlend;
    float4 result = mix(center, blended, blendFactor);
    
    output.write(result, gid);
}

kernel void taa(
    texture2d<float, access::read> currentFrame [[texture(0)]],
    texture2d<float, access::read> historyFrame [[texture(1)]],
    texture2d<float, access::read> motionVectors [[texture(2)]],
    texture2d<float, access::write> output [[texture(3)]],
    constant float& blendFactor [[buffer(0)]],
    uint2 gid [[thread_position_in_grid]]
) {
    uint2 texSize = uint2(output.get_width(), output.get_height());
    if (gid.x >= texSize.x || gid.y >= texSize.y) return;
    
    float4 current = currentFrame.read(gid);
    
    float4 mvData = motionVectors.read(gid);
    float2 motion = mvData.xy;
    float mvConfidence = mvData.w;  
    
    float2 historyPos = float2(gid) - motion;
    historyPos = clamp(historyPos, float2(0.0), float2(texSize) - 1.0);
    
    float4 history = historyFrame.read(uint2(historyPos));
    
    float3 m1 = float3(0.0);
    float3 m2 = float3(0.0);
    float3 neighborMin = current.rgb;
    float3 neighborMax = current.rgb;
    
    for (int dy = -1; dy <= 1; dy++) {
        for (int dx = -1; dx <= 1; dx++) {
            int2 pos = int2(gid) + int2(dx, dy);
            pos = clamp(pos, int2(0), int2(texSize) - 1);
            
            float3 s = currentFrame.read(uint2(pos)).rgb;
            neighborMin = min(neighborMin, s);
            neighborMax = max(neighborMax, s);
            m1 += s;
            m2 += s * s;
        }
    }
    
    m1 /= 9.0;
    m2 /= 9.0;
    float3 sigma = sqrt(max(m2 - m1 * m1, float3(0.0)));
    float3 clipMin = m1 - 1.25 * sigma;
    float3 clipMax = m1 + 1.25 * sigma;
    
    clipMin = max(clipMin, neighborMin);
    clipMax = min(clipMax, neighborMax);
    
    float3 clampedHistory = clamp(history.rgb, clipMin, clipMax);
    
    float motionMag = length(motion);
    float clampDist = length(history.rgb - clampedHistory);
    
    float confidenceWeight = saturate(mvConfidence);
    
    float adaptiveBlend = blendFactor;
    adaptiveBlend = mix(adaptiveBlend, adaptiveBlend * 3.0, saturate(motionMag / 8.0));
    adaptiveBlend = mix(adaptiveBlend, adaptiveBlend * 2.0, saturate(clampDist * 5.0));
    adaptiveBlend = mix(adaptiveBlend, adaptiveBlend * 0.5, confidenceWeight * 0.5);
    adaptiveBlend = clamp(adaptiveBlend, 0.04, 0.5);
    
    float3 result = mix(clampedHistory, current.rgb, adaptiveBlend);
    
    output.write(float4(result, 1.0), gid);
}

kernel void mgup1ContrastAdaptiveSharpening(
    texture2d<float, access::read> input [[texture(0)]],
    texture2d<float, access::write> output [[texture(1)]],
    constant SharpenParams& params [[buffer(0)]],
    uint2 gid [[thread_position_in_grid]]
) {
    uint2 texSize = uint2(output.get_width(), output.get_height());
    if (gid.x >= texSize.x || gid.y >= texSize.y) return;
    
    float3 a = input.read(uint2(max(0u, gid.x - 1), max(0u, gid.y - 1))).rgb;
    float3 b = input.read(uint2(gid.x, max(0u, gid.y - 1))).rgb;
    float3 c = input.read(uint2(min(texSize.x - 1, gid.x + 1), max(0u, gid.y - 1))).rgb;
    float3 d = input.read(uint2(max(0u, gid.x - 1), gid.y)).rgb;
    float3 e = input.read(gid).rgb;
    float3 f = input.read(uint2(min(texSize.x - 1, gid.x + 1), gid.y)).rgb;
    float3 g = input.read(uint2(max(0u, gid.x - 1), min(texSize.y - 1, gid.y + 1))).rgb;
    float3 h = input.read(uint2(gid.x, min(texSize.y - 1, gid.y + 1))).rgb;
    float3 i = input.read(uint2(min(texSize.x - 1, gid.x + 1), min(texSize.y - 1, gid.y + 1))).rgb;
    
    float3 mnRGB = min(min(min(d, e), min(f, b)), h);
    float3 mnRGB2 = min(min(min(mnRGB, a), min(c, g)), i);
    mnRGB += mnRGB2;
    
    float3 mxRGB = max(max(max(d, e), max(f, b)), h);
    float3 mxRGB2 = max(max(max(mxRGB, a), max(c, g)), i);
    mxRGB += mxRGB2;
    
    float3 ampRGB = saturate(min(mnRGB, 2.0 - mxRGB) / mxRGB);
    ampRGB = sqrt(ampRGB);
    
    float peak = 8.0 - 3.0 * params.sharpness;
    float3 wRGB = -1.0 / (ampRGB * peak);
    float3 rcpWeightRGB = 1.0 / (1.0 + 4.0 * wRGB);
    
    float3 window = b + d + f + h;
    float3 result = saturate((window * wRGB + e) * rcpWeightRGB);
    
    output.write(float4(result, 1.0), gid);
}

kernel void mgup1UnsharpMask(
    texture2d<float, access::read> input [[texture(0)]],
    texture2d<float, access::write> output [[texture(1)]],
    constant float& strength [[buffer(0)]],
    uint2 gid [[thread_position_in_grid]]
) {
    uint2 texSize = uint2(output.get_width(), output.get_height());
    if (gid.x >= texSize.x || gid.y >= texSize.y) return;
    
    float4 center = input.read(gid);
    
    float4 blur = float4(0.0);
    int samples = 0;
    
    for (int dy = -1; dy <= 1; dy++) {
        for (int dx = -1; dx <= 1; dx++) {
            if (dx == 0 && dy == 0) continue;
            
            int2 pos = int2(gid) + int2(dx, dy);
            pos = clamp(pos, int2(0), int2(texSize) - 1);
            blur += input.read(uint2(pos));
            samples++;
        }
    }
    blur /= float(samples);
    
    float4 result = center + (center - blur) * strength;
    result = clamp(result, float4(0.0), float4(1.0));
    
    output.write(result, gid);
}

kernel void temporalAccumulation(
    texture2d<float, access::read> currentFrame [[texture(0)]],
    texture2d<float, access::read> historyFrame [[texture(1)]],
    texture2d<float, access::write> output [[texture(2)]],
    constant float& blendFactor [[buffer(0)]],
    uint2 gid [[thread_position_in_grid]]
) {
    uint2 texSize = uint2(output.get_width(), output.get_height());
    if (gid.x >= texSize.x || gid.y >= texSize.y) return;
    
    float4 current = currentFrame.read(gid);
    float4 history = historyFrame.read(gid);
    
    float3 neighborMin = current.rgb;
    float3 neighborMax = current.rgb;
    
    for (int dy = -1; dy <= 1; dy++) {
        for (int dx = -1; dx <= 1; dx++) {
            int2 pos = int2(gid) + int2(dx, dy);
            pos = clamp(pos, int2(0), int2(texSize) - 1);
            
            float3 sample = currentFrame.read(uint2(pos)).rgb;
            neighborMin = min(neighborMin, sample);
            neighborMax = max(neighborMax, sample);
        }
    }
    
    float3 clampedHistory = clamp(history.rgb, neighborMin, neighborMax);
    
    float3 result = mix(current.rgb, clampedHistory, blendFactor);
    
    output.write(float4(result, 1.0), gid);
}

kernel void blitScaleBilinear(
    texture2d<float, access::read> input [[texture(0)]],
    texture2d<float, access::write> output [[texture(1)]],
    uint2 gid [[thread_position_in_grid]]
) {
    uint2 outSize = uint2(output.get_width(), output.get_height());
    if (gid.x >= outSize.x || gid.y >= outSize.y) return;

    uint2 inSize = uint2(input.get_width(), input.get_height());
    
    float2 srcPos = (float2(gid) + 0.5f) * float2(inSize) / float2(outSize) - 0.5f;
    
    srcPos = clamp(srcPos, float2(0.0f), float2(inSize) - 1.0f);
    
    int2 pos0 = int2(floor(srcPos));
    int2 pos1 = min(pos0 + 1, int2(inSize) - 1);
    float2 frac = srcPos - float2(pos0);
    
    float4 c00 = input.read(uint2(pos0.x, pos0.y));
    float4 c10 = input.read(uint2(pos1.x, pos0.y));
    float4 c01 = input.read(uint2(pos0.x, pos1.y));
    float4 c11 = input.read(uint2(pos1.x, pos1.y));
    
    float4 c0 = mix(c00, c10, frac.x);
    float4 c1 = mix(c01, c11, frac.x);
    float4 c = mix(c0, c1, frac.y);
    
    c.a = 1.0f;
    
    output.write(c, gid);
}

kernel void copyTexture(
    texture2d<float, access::read> input [[texture(0)]],
    texture2d<float, access::write> output [[texture(1)]],
    uint2 gid [[thread_position_in_grid]]
) {
    if (gid.x >= output.get_width() || gid.y >= output.get_height()) return;
    output.write(input.read(gid), gid);
}

kernel void convertColorSpace(
    texture2d<float, access::read> input [[texture(0)]],
    texture2d<float, access::write> output [[texture(1)]],
    constant bool& toLinear [[buffer(0)]],
    uint2 gid [[thread_position_in_grid]]
) {
    if (gid.x >= output.get_width() || gid.y >= output.get_height()) return;
    
    float4 color = input.read(gid);
    
    if (toLinear) {
        color.rgb = srgbToLinear(color.rgb);
    } else {
        color.rgb = linearToSrgb(color.rgb);
    }
    
    output.write(color, gid);
}

kernel void clearTexture(
    texture2d<float, access::write> output [[texture(0)]],
    constant float4& clearColor [[buffer(0)]],
    uint2 gid [[thread_position_in_grid]]
) {
    if (gid.x >= output.get_width() || gid.y >= output.get_height()) return;
    output.write(clearColor, gid);
}

kernel void convertVisionFlow(
    texture2d<float, access::read> visionFlow [[texture(0)]],
    texture2d<float, access::write> motionVectors [[texture(1)]],
    uint2 gid [[thread_position_in_grid]]
) {
    if (gid.x >= motionVectors.get_width() || gid.y >= motionVectors.get_height()) return;
    
    float4 flow = visionFlow.read(gid);
    motionVectors.write(float4(flow.xy, 0.0, 1.0), gid);
}

kernel void opticalFlowCompute(
    texture2d<half, access::read> prevFrame [[texture(0)]],
    texture2d<half, access::read> currFrame [[texture(1)]],
    texture2d<half, access::write> flowTexture [[texture(2)]],  
    constant MGFG1Params& params [[buffer(0)]],
    uint2 gid [[thread_position_in_grid]],
    uint2 tid [[thread_position_in_threadgroup]],
    uint2 tgid [[threadgroup_position_in_grid]]
) {
    (void)tid;   
    (void)tgid;  
    
    uint2 texSize = uint2(flowTexture.get_width(), flowTexture.get_height());
    if (gid.x >= texSize.x || gid.y >= texSize.y) return;
    
    const int windowRadius = 4;
    
    half Ixx = half(0.0);
    half Iyy = half(0.0);
    half Ixy = half(0.0);
    half Ixt = half(0.0);
    half Iyt = half(0.0);
    
    for (int dy = -windowRadius; dy <= windowRadius; dy++) {
        for (int dx = -windowRadius; dx <= windowRadius; dx++) {
            int2 pos = int2(gid) + int2(dx, dy);
            pos = clamp(pos, int2(0), int2(texSize) - 1);
            
            int2 posL = clamp(pos + int2(-1, 0), int2(0), int2(texSize) - 1);
            int2 posR = clamp(pos + int2(1, 0), int2(0), int2(texSize) - 1);
            int2 posU = clamp(pos + int2(0, -1), int2(0), int2(texSize) - 1);
            int2 posD = clamp(pos + int2(0, 1), int2(0), int2(texSize) - 1);
            
            half currLuma = rgb2lumaH(currFrame.read(uint2(pos)).rgb);
            half prevLuma = rgb2lumaH(prevFrame.read(uint2(pos)).rgb);
            
            half gx = (rgb2lumaH(currFrame.read(uint2(posR)).rgb) - 
                       rgb2lumaH(currFrame.read(uint2(posL)).rgb)) * 0.5h;
            half gy = (rgb2lumaH(currFrame.read(uint2(posD)).rgb) - 
                       rgb2lumaH(currFrame.read(uint2(posU)).rgb)) * 0.5h;
            
            half gt = currLuma - prevLuma;
            
            half weight = exp(-half(dx * dx + dy * dy) / half(windowRadius * windowRadius * 2));
            
            Ixx += gx * gx * weight;
            Iyy += gy * gy * weight;
            Ixy += gx * gy * weight;
            Ixt += gx * gt * weight;
            Iyt += gy * gt * weight;
        }
    }
    
    half det = Ixx * Iyy - Ixy * Ixy;
    half2 flow = half2(0.0h);
    
    if (abs(det) > half(1e-6)) {
        half invDet = half(1.0) / det;
        flow.x = -(Iyy * Ixt - Ixy * Iyt) * invDet;
        flow.y = -(Ixx * Iyt - Ixy * Ixt) * invDet;
    }
    
    flow *= half(params.motionScale);
    
    flow = clamp(flow, half2(-32.0h), half2(32.0h));
    
    flowTexture.write(half4(flow.x, flow.y, 0.0h, 1.0h), gid);
}

kernel void frameGenCompute(
    texture2d<half, access::sample> prevFrame [[texture(0)]],
    texture2d<half, access::sample> currFrame [[texture(1)]],
    texture2d<half, access::read> flowTexture [[texture(2)]],  
    texture2d<half, access::write> interpolatedTexture [[texture(3)]],  
    constant MGFG1Params& params [[buffer(0)]],
    uint2 gid [[thread_position_in_grid]]
) {
    uint2 texSize = uint2(interpolatedTexture.get_width(), interpolatedTexture.get_height());
    if (gid.x >= texSize.x || gid.y >= texSize.y) return;
    
    half2 flow = flowTexture.read(gid).xy;
    
    half t = half(params.t);
    half tInv = half(1.0) - t;
    
    float2 uv = (float2(gid) + 0.5f) / float2(texSize);
    float2 flowNorm = float2(flow) / float2(texSize);
    
    float2 prevUV = clamp(uv - flowNorm * float(tInv), float2(0.002f), float2(0.998f));
    float2 currUV = clamp(uv + flowNorm * float(t), float2(0.002f), float2(0.998f));
    
    float edgeMargin = 0.03f;
    bool isNearEdge = uv.x < edgeMargin || uv.x > (1.0f - edgeMargin) || 
                      uv.y < edgeMargin || uv.y > (1.0f - edgeMargin);
    half edgeFactor = isNearEdge ? half(0.3) : half(1.0);
    
    constexpr sampler linearSampler(address::clamp_to_edge, filter::linear);
    
    half4 prevSample = prevFrame.sample(linearSampler, prevUV);
    half4 currSample = currFrame.sample(linearSampler, currUV);
    
    uint2 flowReadPos = uint2(clamp(prevUV * float2(texSize), float2(0), float2(texSize) - 1));
    half2 flowAtPrev = flowTexture.read(flowReadPos).xy;
    
    half2 flowDiff = flow + flowAtPrev;
    half consistency = exp(-length(flowDiff) * half(params.occlusionThreshold));
    
    half motionMag = length(flow);
    
    half colorDiff = length(prevSample.rgb - currSample.rgb);
    half colorConfidence = exp(-colorDiff * 5.0h);
    
    half confidence = consistency * colorConfidence * edgeFactor;
    
    half4 result;
    
    if (params.qualityMode == 2) {
        half prevWeight = tInv * confidence;
        half currWeight = t * confidence;
        half totalWeight = prevWeight + currWeight;
        
        half motionFactor = saturate(motionMag * 0.05h);
        
        if (totalWeight > 0.001h) {
            result = (prevSample * prevWeight + currSample * currWeight) / totalWeight;
        } else {
            result = mix(prevSample, currSample, t);
        }
        
        half4 simpleBlend = mix(prevSample, currSample, t);
        half blendFactor = half(params.temporalWeight) * (half(1.0) - confidence) + motionFactor * 0.2h;
        if (isNearEdge) blendFactor = max(blendFactor, half(0.5));
        result = mix(result, simpleBlend, blendFactor);
    } else if (params.qualityMode == 1) {
        result = mix(prevSample, currSample, t);
        
        half4 compensated = (prevSample + currSample) * 0.5h;
        result = mix(result, compensated, confidence * 0.5h);
    } else {
        result = mix(prevSample, currSample, t);
    }
    
    result.a = half(1.0);
    interpolatedTexture.write(result, gid);
}

struct FullscreenVertexOut {
    float4 position [[position]];
    float2 texCoord;
};

vertex FullscreenVertexOut fullscreenQuadVertex(uint vertexID [[vertex_id]]) {
    FullscreenVertexOut out;
    const float2 positions[4] = {
        float2(-1.0, -1.0), 
        float2( 1.0, -1.0), 
        float2(-1.0,  1.0), 
        float2( 1.0,  1.0)  
    };
    const float2 uvs[4] = {
        float2(0.0, 1.0),
        float2(1.0, 1.0),
        float2(0.0, 0.0),
        float2(1.0, 0.0)
    };
    out.position = float4(positions[vertexID], 0.0, 1.0);
    out.texCoord = uvs[vertexID];
    return out;
}

fragment float4 fullscreenQuadFragment(
    FullscreenVertexOut in [[stage_in]],
    texture2d<float, access::sample> inputTexture [[texture(0)]]
) {
    float2 uv = clamp(in.texCoord, float2(0.0), float2(1.0));
    
    constexpr sampler s(address::clamp_to_edge, filter::linear, mip_filter::none);
    float4 color = inputTexture.sample(s, uv);
    
    color.a = 1.0;
    
    return color;
}

fragment float4 fullscreenQuadFragmentManual(
    FullscreenVertexOut in [[stage_in]],
    texture2d<float, access::read> inputTexture [[texture(0)]]
) {
    float2 uv = clamp(in.texCoord, float2(0.0), float2(1.0));
    
    uint2 texSize = uint2(inputTexture.get_width(), inputTexture.get_height());
    
    float2 srcPos = uv * float2(texSize) - 0.5f;
    srcPos = clamp(srcPos, float2(0.0f), float2(texSize) - 1.0f);
    
    int2 pos0 = int2(floor(srcPos));
    int2 pos1 = min(pos0 + 1, int2(texSize) - 1);
    float2 frac = srcPos - float2(pos0);
    
    float4 c00 = inputTexture.read(uint2(pos0.x, pos0.y));
    float4 c10 = inputTexture.read(uint2(pos1.x, pos0.y));
    float4 c01 = inputTexture.read(uint2(pos0.x, pos1.y));
    float4 c11 = inputTexture.read(uint2(pos1.x, pos1.y));
    
    float4 c0 = mix(c00, c10, frac.x);
    float4 c1 = mix(c01, c11, frac.x);
    float4 color = mix(c0, c1, frac.y);
    
    color.a = 1.0;
    return color;
}
