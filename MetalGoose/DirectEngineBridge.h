#ifndef DirectEngineBridge_h
#define DirectEngineBridge_h

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#import <MetalFX/MetalFX.h>
#import <IOSurface/IOSurface.h>
#import <CoreGraphics/CoreGraphics.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct DirectFrameEngineOpaque* DirectEngineRef;


typedef NS_ENUM(NSInteger, DirectUpscaleMode) {
    DirectUpscaleModeOff = 0,
    DirectUpscaleModeMGUP1 = 1,
    DirectUpscaleModeMGUP1Fast = 2,
    DirectUpscaleModeMGUP1Quality = 3
};

typedef NS_ENUM(NSInteger, DirectRenderScale) {
    DirectRenderScaleNative = 0,
    DirectRenderScale75 = 1,
    DirectRenderScale67 = 2,
    DirectRenderScale50 = 3,
    DirectRenderScale33 = 4
};

typedef NS_ENUM(NSInteger, DirectFrameGen) {
    DirectFrameGenOff = 0,
    DirectFrameGenMGFG1 = 1
};

typedef NS_ENUM(NSInteger, DirectFrameGenType) {
    DirectFrameGenTypeAdaptive = 0,
    DirectFrameGenTypeFixed = 1
};

typedef NS_ENUM(NSInteger, DirectFrameGenQuality) {
    DirectFrameGenQualityPerformance = 0,
    DirectFrameGenQualityBalanced = 1,
    DirectFrameGenQualityQuality = 2
};

typedef NS_ENUM(NSInteger, DirectAAMode) {
    DirectAAModeOff = 0,
    DirectAAModeFXAA = 1,
    DirectAAModeSMAA = 2,
    DirectAAModeMSAA = 3,
    DirectAAModeTAA = 4
};

typedef NS_ENUM(NSInteger, DirectEngineState) {
    DirectEngineStateIdle = 0,
    DirectEngineStateCapturing = 1,
    DirectEngineStateProcessing = 2,
    DirectEngineStatePresenting = 3,
    DirectEngineStateError = 4
};


typedef struct {
    DirectUpscaleMode upscaleMode;
    DirectRenderScale renderScale;
    float scaleFactor;
    
    DirectFrameGen frameGenMode;
    DirectFrameGenType frameGenType;
    DirectFrameGenQuality frameGenQuality;
    int frameGenMultiplier;
    int adaptiveTargetFPS;
    
    DirectAAMode aaMode;
    float aaThreshold;
    
    int baseWidth;
    int baseHeight;
    int outputWidth;
    int outputHeight;
    
    int targetFPS;
    
    bool useMotionVectors;
    bool vsyncEnabled;
    bool reduceLatency;
    bool adaptiveSync;
    bool captureMouseCursor;
    
    float sharpness;
    float temporalBlend;
    float motionScale;
} DirectEngineConfig;

typedef struct {
    float fps;
    float interpolatedFPS;
    float captureFPS;
    
    float frameTime;
    float gpuTime;
    float captureLatency;
    float presentLatency;
    
    uint64_t frameCount;
    uint64_t interpolatedFrameCount;
    uint64_t droppedFrames;
    
    uint64_t gpuMemoryUsed;
    uint64_t gpuMemoryTotal;
    uint64_t textureMemoryUsed;
    
    uint32_t renderEncoders;
    uint32_t computeEncoders;
    uint32_t blitEncoders;
    uint32_t commandBuffers;
    uint32_t drawCalls;
    
    uint32_t upscaleMode;
    uint32_t frameGenMode;
    uint32_t aaMode;
} DirectEngineStats;

DirectEngineRef _Nullable DirectEngine_Create(id<MTLDevice> _Nonnull device, 
                                               id<MTLCommandQueue> _Nonnull queue);

void DirectEngine_Destroy(DirectEngineRef _Nullable engine);


void DirectEngine_SetConfig(DirectEngineRef _Nonnull engine, DirectEngineConfig config);

DirectEngineConfig DirectEngine_GetConfig(DirectEngineRef _Nonnull engine);


bool DirectEngine_SetTargetWindow(DirectEngineRef _Nonnull engine, CGWindowID windowID);

bool DirectEngine_SetTargetDisplay(DirectEngineRef _Nonnull engine, CGDirectDisplayID displayID);

bool DirectEngine_SetOutputDisplay(DirectEngineRef _Nonnull engine, CGDirectDisplayID displayID);


bool DirectEngine_StartCapture(DirectEngineRef _Nonnull engine);

void DirectEngine_StopCapture(DirectEngineRef _Nonnull engine);

void DirectEngine_PauseCapture(DirectEngineRef _Nonnull engine);

void DirectEngine_ResumeCapture(DirectEngineRef _Nonnull engine);


DirectEngineState DirectEngine_GetState(DirectEngineRef _Nonnull engine);

bool DirectEngine_HasNewFrame(DirectEngineRef _Nonnull engine);

float DirectEngine_GetCurrentFPS(DirectEngineRef _Nonnull engine);

float DirectEngine_GetInterpolatedFPS(DirectEngineRef _Nonnull engine);

float DirectEngine_GetCaptureFPS(DirectEngineRef _Nonnull engine);


IOSurfaceRef _Nullable DirectEngine_GetCurrentSurface(DirectEngineRef _Nonnull engine);

id<MTLTexture> _Nullable DirectEngine_GetFrameTexture(DirectEngineRef _Nonnull engine, 
                                                       id<MTLDevice> _Nonnull device);


id<MTLTexture> _Nullable DirectEngine_ProcessFrame(DirectEngineRef _Nonnull engine, 
                                                    id<MTLDevice> _Nonnull device, 
                                                    id<MTLCommandBuffer> _Nonnull commandBuffer);

id<MTLTexture> _Nullable DirectEngine_GetInterpolatedFrame(DirectEngineRef _Nonnull engine, 
                                                            id<MTLDevice> _Nonnull device, 
                                                            id<MTLCommandBuffer> _Nonnull commandBuffer, 
                                                            bool* _Nonnull isInterpolated);

id<MTLTexture> _Nullable DirectEngine_GetInterpolatedFrameWithT(DirectEngineRef _Nonnull engine, 
                                                                 id<MTLDevice> _Nonnull device, 
                                                                 id<MTLCommandBuffer> _Nonnull commandBuffer, 
                                                                 float t,
                                                                 bool* _Nonnull isInterpolated);

id<MTLTexture> _Nullable DirectEngine_InterpolateTextures(DirectEngineRef _Nonnull engine, 
                                                           id<MTLDevice> _Nonnull device, 
                                                           id<MTLCommandBuffer> _Nonnull commandBuffer, 
                                                           id<MTLTexture> _Nonnull prevTexture, 
                                                           id<MTLTexture> _Nonnull currTexture, 
                                                           float t);


bool DirectEngine_ShouldShowInterpolatedFrame(DirectEngineRef _Nonnull engine);

int DirectEngine_GetFrameGenMode(DirectEngineRef _Nonnull engine);

int DirectEngine_GetFrameGenMultiplier(DirectEngineRef _Nonnull engine);


bool DirectEngine_IsAsyncFrameGenEnabled(DirectEngineRef _Nonnull engine);


bool DirectEngine_HasPendingAsyncFrame(DirectEngineRef _Nonnull engine);


id<MTLTexture> _Nullable DirectEngine_GetNextAsyncFrame(DirectEngineRef _Nonnull engine);


void DirectEngine_SetTargetFrameMultiplier(DirectEngineRef _Nonnull engine, int multiplier);


bool DirectEngine_HasValidCPUMotion(DirectEngineRef _Nonnull engine);


DirectEngineStats DirectEngine_GetStats(DirectEngineRef _Nonnull engine);

void DirectEngine_ResetStats(DirectEngineRef _Nonnull engine);


NSString* _Nullable DirectEngine_GetLastError(DirectEngineRef _Nonnull engine);

void DirectEngine_ClearError(DirectEngineRef _Nonnull engine);


void DirectEngine_SetDebugMode(DirectEngineRef _Nonnull engine, bool enabled);

NSString* _Nullable DirectEngine_GetDebugInfo(DirectEngineRef _Nonnull engine);

#ifdef __cplusplus
}
#endif

#endif
