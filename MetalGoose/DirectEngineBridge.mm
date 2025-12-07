#import "DirectEngineBridge.h"
#import <CoreVideo/CoreVideo.h>
#import <MetalPerformanceShaders/MetalPerformanceShaders.h>
#import <ScreenCaptureKit/ScreenCaptureKit.h>
#import <algorithm>
#import <atomic>
#import <chrono>
#import <cmath>
#import <deque>
#import <mutex>

class Engine;
extern "C" Engine *Engine_Create(void *device, void *queue);
extern "C" void Engine_Destroy(Engine *engine);
extern "C" void Engine_SetConfig(Engine *engine, void *config);
extern "C" void *Engine_ProcessFrame(Engine *engine, void *inputTexture,
                                     void *cmdBuf);
extern "C" void *Engine_GenerateInterpolatedFrame(Engine *engine, void *prevTex,
                                                  void *currTex, float t,
                                                  void *cmdBuf);
extern "C" void Engine_PushFrame(Engine *engine, void *texture,
                                 double timestamp);
extern "C" uint64_t Engine_GetFrameIndex(Engine *engine);

struct FrameBufferEntry {
  IOSurfaceRef surface;
  std::chrono::high_resolution_clock::time_point timestamp;
  uint64_t frameNumber;
  double captureTimestamp;
};

struct DirectFrameEngineOpaque {
  Engine *cppEngine;
  id<MTLDevice> device;
  id<MTLCommandQueue> commandQueue;

  CVMetalTextureCacheRef textureCache;

  SCStream *captureStream;
  SCContentFilter *contentFilter;
  SCStreamConfiguration *streamConfig;
  dispatch_queue_t captureQueue;

  std::mutex frameMutex;
  std::deque<FrameBufferEntry> frameBuffer;
  static constexpr size_t kMaxFrameBufferNormal = 4;
  static constexpr size_t kMaxFrameBufferLowLatency = 2;
  IOSurfaceRef currentSurface;
  IOSurfaceRef previousSurface;
  CVPixelBufferRef currentPixelBuffer;
  CVPixelBufferRef previousPixelBuffer;
  std::atomic<bool> hasNewFrame;
  std::atomic<uint64_t> frameNumber;

  size_t getMaxFrameBuffer() const {
    return config.reduceLatency ? kMaxFrameBufferLowLatency
                                : kMaxFrameBufferNormal;
  }

  double currentFrameTimestamp;
  double previousFrameTimestamp;
  float deltaTime;

  std::atomic<float> currentFPS;
  std::atomic<float> captureFPS;
  std::atomic<float> interpolatedFPS;
  std::atomic<float> processingTime;
  std::atomic<float> gpuTime;
  std::atomic<float> captureLatency;
  std::atomic<float> presentLatency;
  std::atomic<uint64_t> droppedFrames;
  std::atomic<uint64_t> interpolatedFrameCount;
  std::atomic<uint32_t> renderEncoderCount;
  std::atomic<uint32_t> computeEncoderCount;
  std::atomic<uint32_t> blitEncoderCount;
  std::atomic<uint32_t> commandBufferCount;
  std::atomic<uint32_t> drawCallCount;

  std::chrono::high_resolution_clock::time_point lastFrameTime;
  std::chrono::high_resolution_clock::time_point fpsCounterStart;
  int fpsFrameCount;
  double lastCaptureTimestamp;

  DirectEngineConfig config;
  std::atomic<DirectEngineState> state;
  std::string lastError;
  bool debugMode;

  CGWindowID targetWindowID;
  CGDirectDisplayID targetDisplayID;
  CGDirectDisplayID outputDisplayID;
  bool useWindowCapture;
  bool isPaused;

  uint64_t lastFrameHash;
  uint64_t duplicateFrameCount;
  std::chrono::high_resolution_clock::time_point lastRealFrameTime;
  std::atomic<float> realGameFPS;

  static uint64_t computeFrameHash(const uint8_t *data, size_t width,
                                   size_t height, size_t bytesPerRow) {
    if (!data || width == 0 || height == 0)
      return 0;

    uint64_t hash = 0;
    const int sampleCount = 64;

    size_t stepX = std::max<size_t>(1, width / 8);
    size_t stepY = std::max<size_t>(1, height / 8);

    int samples = 0;
    for (size_t y = stepY / 2; y < height && samples < sampleCount;
         y += stepY) {
      for (size_t x = stepX / 2; x < width && samples < sampleCount;
           x += stepX) {
        size_t offset = y * bytesPerRow + x * 4;

        uint32_t pixel = *reinterpret_cast<const uint32_t *>(data + offset);

        hash ^= pixel + 0x9e3779b97f4a7c15ULL + (hash << 6) + (hash >> 2);
        samples++;
      }
    }

    return hash;
  }

  std::atomic<bool> shouldInterpolate;
  std::atomic<float> interpolationT;
  std::atomic<int> frameGenCycle;

  dispatch_queue_t asyncFrameGenQueue;
  std::atomic<bool> asyncFrameGenEnabled;
  std::atomic<int> pendingInterpolatedFrames;
  std::atomic<int> generatedFramesThisCycle;

  id<MTLTexture> asyncOutputTextures[3];
  std::atomic<int> asyncWriteIndex;
  std::atomic<int> asyncReadIndex;
  std::mutex asyncTextureMutex;

  struct CPUMotionData {
    std::vector<float> motionX;
    std::vector<float> motionY;
    std::vector<float> confidence;
    size_t width;
    size_t height;
    bool valid;
  };
  CPUMotionData cpuMotionCache;
  std::mutex cpuMotionMutex;

  std::atomic<double> frameInterval;
  std::atomic<double> lastInterpolationTime;
  std::atomic<int> targetFrameMultiplier;

  id<MTLTexture> cachedOutputTex;
  id<MTLTexture> cachedPrevTex;
  id<MTLTexture> cachedInterpTex;
  id<MTLTexture> cachedInputTexture;
  id<MTLComputePipelineState> cachedPerformancePipeline;
  id<MTLComputePipelineState> cachedBalancedPipeline;
  id<MTLComputePipelineState> cachedQualityPipeline;
  id<MTLComputePipelineState> cachedMotionEstPipeline;
  id<MTLComputePipelineState> cachedMotionEstOptPipeline;
  id<MTLComputePipelineState> cachedMotionRefinePipeline;
  id<MTLComputePipelineState> cachedAdaptiveInterpPipeline;
  id<MTLTexture> cachedMotionVectorTex;
  id<MTLTexture> cachedConfidenceTex;
  size_t cachedWidth;
  size_t cachedHeight;

  id<MTLComputePipelineState> opticalFlowPipeline;
  id<MTLTexture> flowTexture;

  id<MTLComputePipelineState> frameGenPipeline;
  id<MTLTexture> interpolatedTexture;

  id<MTLComputePipelineState> scalePipeline;
  id<MTLTexture> scaledTexture;

  id<MTLTexture> previousFrameTexture;
  id<MTLTexture> currentFrameTexture;

  id<MTLTexture> finalOutputTexture;

  DirectFrameEngineOpaque()
      : cppEngine(nullptr), device(nil), commandQueue(nil),
        textureCache(nullptr), captureStream(nil), contentFilter(nil),
        streamConfig(nil), captureQueue(nullptr), currentSurface(nullptr),
        previousSurface(nullptr), currentPixelBuffer(nullptr),
        previousPixelBuffer(nullptr), hasNewFrame(false), frameNumber(0),
        currentFrameTimestamp(0.0), previousFrameTimestamp(0.0),
        deltaTime(0.0f), currentFPS(0.0f), captureFPS(0.0f),
        interpolatedFPS(0.0f), processingTime(0.0f), gpuTime(0.0f),
        captureLatency(0.0f), presentLatency(0.0f), droppedFrames(0),
        interpolatedFrameCount(0), renderEncoderCount(0),
        computeEncoderCount(0), blitEncoderCount(0), commandBufferCount(0),
        drawCallCount(0), fpsFrameCount(0), lastCaptureTimestamp(0.0),
        state(DirectEngineStateIdle), debugMode(false),
        targetWindowID(kCGNullWindowID), targetDisplayID(0), outputDisplayID(0),
        useWindowCapture(false), isPaused(false), lastFrameHash(0),
        duplicateFrameCount(0), realGameFPS(0.0f), shouldInterpolate(false),
        interpolationT(0.0f), frameGenCycle(0), asyncFrameGenQueue(nullptr),
        asyncFrameGenEnabled(false), pendingInterpolatedFrames(0),
        generatedFramesThisCycle(0), asyncWriteIndex(0), asyncReadIndex(0),
        frameInterval(16.67), lastInterpolationTime(0.0),
        targetFrameMultiplier(2), cachedOutputTex(nil), cachedPrevTex(nil),
        cachedInterpTex(nil), cachedInputTexture(nil),
        cachedPerformancePipeline(nil), cachedBalancedPipeline(nil),
        cachedQualityPipeline(nil), cachedMotionEstPipeline(nil),
        cachedMotionEstOptPipeline(nil), cachedMotionRefinePipeline(nil),
        cachedAdaptiveInterpPipeline(nil), cachedMotionVectorTex(nil),
        cachedConfidenceTex(nil), cachedWidth(0), cachedHeight(0),
        opticalFlowPipeline(nil), flowTexture(nil), frameGenPipeline(nil),
        interpolatedTexture(nil), scalePipeline(nil), scaledTexture(nil),
        previousFrameTexture(nil), currentFrameTexture(nil),
        finalOutputTexture(nil) {
    lastFrameTime = std::chrono::high_resolution_clock::now();
    lastRealFrameTime = lastFrameTime;
    fpsCounterStart = lastFrameTime;

    config.upscaleMode = DirectUpscaleModeOff;
    config.renderScale = DirectRenderScaleNative;
    config.scaleFactor = 1.0f;
    config.frameGenMode = DirectFrameGenOff;
    config.frameGenType = DirectFrameGenTypeFixed;
    config.frameGenQuality = DirectFrameGenQualityBalanced;
    config.frameGenMultiplier = 2;
    config.adaptiveTargetFPS = 120;
    config.aaMode = DirectAAModeOff;
    config.aaThreshold = 0.166f;
    config.baseWidth = 0;
    config.baseHeight = 0;
    config.outputWidth = 0;
    config.outputHeight = 0;
    config.targetFPS = 120;
    config.useMotionVectors = true;
    config.vsyncEnabled = true;
    config.reduceLatency = true;
    config.adaptiveSync = true;
    config.captureMouseCursor = true;
    config.sharpness = 0.5f;
    config.temporalBlend = 0.1f;
    config.motionScale = 1.0f;
  }

  ~DirectFrameEngineOpaque() {
    cleanupFrameBuffer();
    if (currentSurface)
      CFRelease(currentSurface);
    if (previousSurface)
      CFRelease(previousSurface);
    if (currentPixelBuffer)
      CVPixelBufferRelease(currentPixelBuffer);
    if (previousPixelBuffer)
      CVPixelBufferRelease(previousPixelBuffer);
    if (textureCache)
      CFRelease(textureCache);
  }

  void cleanupFrameBuffer() {
    std::lock_guard<std::mutex> lock(frameMutex);
    for (auto &entry : frameBuffer) {
      if (entry.surface)
        CFRelease(entry.surface);
    }
    frameBuffer.clear();
  }

  void pushFrame(IOSurfaceRef surface, double captureTimestamp = 0.0) {
    std::lock_guard<std::mutex> lock(frameMutex);

    size_t maxBuffer = getMaxFrameBuffer();
    while (frameBuffer.size() >= maxBuffer) {
      auto &oldest = frameBuffer.front();
      if (oldest.surface)
        CFRelease(oldest.surface);
      frameBuffer.pop_front();
      droppedFrames.fetch_add(1, std::memory_order_relaxed);
    }

    FrameBufferEntry entry;
    entry.surface = (IOSurfaceRef)CFRetain(surface);
    entry.timestamp = std::chrono::high_resolution_clock::now();
    entry.frameNumber = frameNumber.fetch_add(1, std::memory_order_relaxed);
    entry.captureTimestamp = captureTimestamp;

    frameBuffer.push_back(entry);
    hasNewFrame.store(true, std::memory_order_release);

    if (captureTimestamp > 0.0) {
      double now = CACurrentMediaTime();
      captureLatency.store((float)((now - captureTimestamp) * 1000.0),
                           std::memory_order_relaxed);
    }
  }

  FrameBufferEntry popFrame() {
    std::lock_guard<std::mutex> lock(frameMutex);

    if (frameBuffer.empty()) {
      hasNewFrame.store(false, std::memory_order_release);
      return {nullptr, {}, 0};
    }

    auto entry = frameBuffer.front();
    frameBuffer.pop_front();

    if (frameBuffer.empty()) {
      hasNewFrame.store(false, std::memory_order_release);
    }

    return entry;
  }

  void updateFPS() {
    fpsFrameCount++;
    auto now = std::chrono::high_resolution_clock::now();
    float elapsed = std::chrono::duration<float>(now - fpsCounterStart).count();

    if (elapsed >= 0.1f) {
      float measuredFPS = static_cast<float>(fpsFrameCount) / elapsed;
      float currentOutputFPS = currentFPS.load(std::memory_order_relaxed);

      if (currentOutputFPS < 1.0f) {
        currentFPS.store(measuredFPS, std::memory_order_relaxed);
      } else {
        float smoothedFPS = currentOutputFPS * 0.7f + measuredFPS * 0.3f;
        currentFPS.store(smoothedFPS, std::memory_order_relaxed);
      }

      fpsFrameCount = 0;
      fpsCounterStart = now;
    } else if (fpsFrameCount == 1 && elapsed > 0.001f) {
      float instantFPS = 1.0f / elapsed;
      currentFPS.store(instantFPS, std::memory_order_relaxed);
    }

    if (config.frameGenMode != DirectFrameGenOff) {
      float capFPS = captureFPS.load(std::memory_order_relaxed);
      if (capFPS < 1.0f)
        capFPS = 30.0f;

      float interpFPS;
      if (config.frameGenType == DirectFrameGenTypeAdaptive) {
        interpFPS = static_cast<float>(config.adaptiveTargetFPS);
      } else {
        interpFPS = capFPS * static_cast<float>(config.frameGenMultiplier);
      }
      interpolatedFPS.store(interpFPS, std::memory_order_relaxed);
    } else {
      interpolatedFPS.store(currentFPS.load(std::memory_order_relaxed),
                            std::memory_order_relaxed);
    }
  }

  void initAsyncFrameGen() {
    if (asyncFrameGenQueue)
      return;
    if (!device) {
      return;
    }

    dispatch_queue_attr_t attr = dispatch_queue_attr_make_with_qos_class(
        DISPATCH_QUEUE_SERIAL, QOS_CLASS_USER_INTERACTIVE, 0);
    asyncFrameGenQueue =
        dispatch_queue_create("com.metalgoose.asyncframegen", attr);

    if (!asyncFrameGenQueue) {
      return;
    }

    asyncFrameGenEnabled.store(true, std::memory_order_release);

    ensureAsyncBuffers(cachedWidth > 0 ? cachedWidth : 1920,
                       cachedHeight > 0 ? cachedHeight : 1080);
  }

  void ensureAsyncBuffers(size_t width, size_t height) {
    if (!device)
      return;

    MTLTextureDescriptor *desc = [MTLTextureDescriptor
        texture2DDescriptorWithPixelFormat:MTLPixelFormatBGRA8Unorm
                                     width:width
                                    height:height
                                 mipmapped:NO];
    desc.usage = MTLTextureUsageShaderRead | MTLTextureUsageShaderWrite |
                 MTLTextureUsageRenderTarget;
    desc.storageMode = MTLStorageModePrivate;

    for (int i = 0; i < 3; i++) {
      if (!asyncOutputTextures[i] || asyncOutputTextures[i].width != width ||
          asyncOutputTextures[i].height != height) {
        asyncOutputTextures[i] = [device newTextureWithDescriptor:desc];
        asyncOutputTextures[i].label =
            [NSString stringWithFormat:@"AsyncOutput%d", i];
      }
    }

    {
      std::lock_guard<std::mutex> lock(cpuMotionMutex);
      size_t mvWidth = (width + 7) / 8;
      size_t mvHeight = (height + 7) / 8;
      size_t mvSize = mvWidth * mvHeight;

      if (cpuMotionCache.width != mvWidth ||
          cpuMotionCache.height != mvHeight) {
        cpuMotionCache.motionX.resize(mvSize, 0.0f);
        cpuMotionCache.motionY.resize(mvSize, 0.0f);
        cpuMotionCache.confidence.resize(mvSize, 1.0f);
        cpuMotionCache.width = mvWidth;
        cpuMotionCache.height = mvHeight;
        cpuMotionCache.valid = false;
      }
    }
  }

  void computeCPUMotionVectors(const uint8_t *prevData, const uint8_t *currData,
                               size_t width, size_t height,
                               size_t bytesPerRow) {
    std::lock_guard<std::mutex> lock(cpuMotionMutex);

    size_t blockSize = 8;
    size_t mvWidth = (width + blockSize - 1) / blockSize;
    size_t mvHeight = (height + blockSize - 1) / blockSize;

    dispatch_apply(
        mvHeight, dispatch_get_global_queue(QOS_CLASS_USER_INITIATED, 0),
        ^(size_t by) {
          for (size_t bx = 0; bx < mvWidth; bx++) {
            size_t mvIdx = by * mvWidth + bx;

            size_t cx = bx * blockSize + blockSize / 2;
            size_t cy = by * blockSize + blockSize / 2;

            if (cx >= width || cy >= height) {
              cpuMotionCache.motionX[mvIdx] = 0.0f;
              cpuMotionCache.motionY[mvIdx] = 0.0f;
              cpuMotionCache.confidence[mvIdx] = 0.0f;
              continue;
            }

            float bestDx = 0.0f, bestDy = 0.0f;
            float bestSAD = INFINITY;

            const int searchRange = 4;
            const int diamondPattern[9][2] = {{0, 0},  {-2, 0}, {2, 0},
                                              {0, -2}, {0, 2},  {-1, -1},
                                              {1, -1}, {-1, 1}, {1, 1}};

            for (int step = 0; step < 3; step++) {
              int baseDx = (int)bestDx;
              int baseDy = (int)bestDy;

              for (int p = 0; p < 9; p++) {
                int dx = baseDx + diamondPattern[p][0] * (1 << (2 - step));
                int dy = baseDy + diamondPattern[p][1] * (1 << (2 - step));

                if (abs(dx) > searchRange || abs(dy) > searchRange)
                  continue;

                float sad = 0.0f;
                int validPixels = 0;

                for (int py = -2; py <= 2; py++) {
                  for (int px = -2; px <= 2; px++) {
                    int sx = (int)cx + px;
                    int sy = (int)cy + py;
                    int tx = sx + dx;
                    int ty = sy + dy;

                    if (sx >= 0 && sx < (int)width && sy >= 0 &&
                        sy < (int)height && tx >= 0 && tx < (int)width &&
                        ty >= 0 && ty < (int)height) {

                      size_t prevOffset = sy * bytesPerRow + sx * 4;
                      size_t currOffset = ty * bytesPerRow + tx * 4;

                      float prevLum = prevData[prevOffset] * 0.114f +
                                      prevData[prevOffset + 1] * 0.587f +
                                      prevData[prevOffset + 2] * 0.299f;
                      float currLum = currData[currOffset] * 0.114f +
                                      currData[currOffset + 1] * 0.587f +
                                      currData[currOffset + 2] * 0.299f;

                      sad += fabsf(prevLum - currLum);
                      validPixels++;
                    }
                  }
                }

                if (validPixels > 0) {
                  sad /= (float)validPixels;
                  if (sad < bestSAD) {
                    bestSAD = sad;
                    bestDx = (float)dx;
                    bestDy = (float)dy;
                  }
                }
              }
            }

            cpuMotionCache.motionX[mvIdx] = bestDx;
            cpuMotionCache.motionY[mvIdx] = bestDy;
            cpuMotionCache.confidence[mvIdx] =
                fmaxf(0.0f, 1.0f - bestSAD / 128.0f);
          }
        });

    cpuMotionCache.valid = true;
  }

  void uploadMotionToGPU(id<MTLCommandBuffer> commandBuffer) {
    std::lock_guard<std::mutex> lock(cpuMotionMutex);

    if (!cpuMotionCache.valid || !cachedMotionVectorTex)
      return;

    size_t dataSize =
        cpuMotionCache.width * cpuMotionCache.height * 4 * sizeof(float);
    id<MTLBuffer> stagingBuffer =
        [device newBufferWithLength:dataSize
                            options:MTLResourceStorageModeShared];

    float *bufferData = (float *)stagingBuffer.contents;
    for (size_t i = 0; i < cpuMotionCache.width * cpuMotionCache.height; i++) {
      bufferData[i * 4 + 0] = cpuMotionCache.motionX[i];
      bufferData[i * 4 + 1] = cpuMotionCache.motionY[i];
      bufferData[i * 4 + 2] = cpuMotionCache.confidence[i];
      bufferData[i * 4 + 3] = 1.0f;
    }

    id<MTLBlitCommandEncoder> blit = [commandBuffer blitCommandEncoder];
    if (blit) {
      [blit copyFromBuffer:stagingBuffer
                 sourceOffset:0
            sourceBytesPerRow:cpuMotionCache.width * 4 * sizeof(float)
          sourceBytesPerImage:dataSize
                   sourceSize:MTLSizeMake(cpuMotionCache.width,
                                          cpuMotionCache.height, 1)
                    toTexture:cachedMotionVectorTex
             destinationSlice:0
             destinationLevel:0
            destinationOrigin:MTLOriginMake(0, 0, 0)];
      [blit endEncoding];
    }
  }

  void scheduleAsyncInterpolation(id<MTLTexture> prevTex,
                                  id<MTLTexture> currTex, int frameMultiplier) {
    if (!asyncFrameGenEnabled.load(std::memory_order_acquire))
      return;

    targetFrameMultiplier.store(frameMultiplier, std::memory_order_release);

    double currentTime = CACurrentMediaTime();
    double prevTime =
        lastInterpolationTime.exchange(currentTime, std::memory_order_relaxed);
    if (prevTime > 0) {
      double interval = (currentTime - prevTime) * 1000.0;
      frameInterval.store(interval, std::memory_order_relaxed);
    }

    for (int i = 1; i < frameMultiplier; i++) {
      pendingInterpolatedFrames.fetch_add(1, std::memory_order_relaxed);
    }

    generatedFramesThisCycle.store(0, std::memory_order_release);
  }

  bool hasReadyInterpolatedFrame() {
    return generatedFramesThisCycle.load(std::memory_order_acquire) > 0;
  }

  id<MTLTexture> getNextAsyncOutputTexture() {
    std::lock_guard<std::mutex> lock(asyncTextureMutex);
    int readIdx = asyncReadIndex.load(std::memory_order_acquire);
    int writeIdx = asyncWriteIndex.load(std::memory_order_acquire);

    if (readIdx == writeIdx)
      return nil;

    id<MTLTexture> result = asyncOutputTextures[readIdx];
    asyncReadIndex.store((readIdx + 1) % 3, std::memory_order_release);
    generatedFramesThisCycle.fetch_sub(1, std::memory_order_relaxed);

    return result;
  }

  void setupPipelines() {
    if (!device)
      return;

    id<MTLLibrary> library = [device newDefaultLibrary];
    if (!library) {
      lastError = "Failed to load Metal shader library";
      return;
    }

    NSError *error = nil;

    auto createPipeline = [&](NSString *name) -> id<MTLComputePipelineState> {
      id<MTLFunction> func = [library newFunctionWithName:name];
      if (!func) {
        return nil;
      }

      id<MTLComputePipelineState> pipeline =
          [device newComputePipelineStateWithFunction:func error:&error];
      if (error) {
        return nil;
      }
      return pipeline;
    };

    cachedPerformancePipeline = createPipeline(@"mgfg1Performance");
    cachedBalancedPipeline = createPipeline(@"mgfg1Balanced");
    cachedQualityPipeline = createPipeline(@"mgfg1Quality");
    cachedAdaptiveInterpPipeline =
        createPipeline(@"mgfg1AdaptiveInterpolation");

    cachedMotionEstPipeline = createPipeline(@"mgfg1MotionEstimation");
    cachedMotionEstOptPipeline =
        createPipeline(@"mgfg1MotionEstimationOptimized");
    cachedMotionRefinePipeline = createPipeline(@"mgfg1MotionRefinement");

    opticalFlowPipeline = createPipeline(@"opticalFlowCompute");
    if (!opticalFlowPipeline) {
      opticalFlowPipeline = cachedMotionEstOptPipeline;
    }

    frameGenPipeline = createPipeline(@"frameGenCompute");
    if (!frameGenPipeline) {
      frameGenPipeline = cachedQualityPipeline;
    }

    scalePipeline = createPipeline(@"blitScaleBilinear");
  }

  void ensureTextures(size_t width, size_t height) {
    if (width == cachedWidth && height == cachedHeight &&
        cachedOutputTex != nil) {
      return;
    }

    cachedWidth = width;
    cachedHeight = height;

    MTLTextureDescriptor *outputDesc = [MTLTextureDescriptor
        texture2DDescriptorWithPixelFormat:MTLPixelFormatBGRA8Unorm
                                     width:width
                                    height:height
                                 mipmapped:NO];
    outputDesc.usage = MTLTextureUsageShaderRead | MTLTextureUsageShaderWrite |
                       MTLTextureUsageRenderTarget;
    outputDesc.storageMode = MTLStorageModePrivate;

    cachedOutputTex = [device newTextureWithDescriptor:outputDesc];
    cachedOutputTex.label = @"MetalGoose Output";

    finalOutputTexture = [device newTextureWithDescriptor:outputDesc];
    finalOutputTexture.label = @"MetalGoose Final Output (bgra8Unorm)";

    MTLTextureDescriptor *frameDesc = [MTLTextureDescriptor
        texture2DDescriptorWithPixelFormat:MTLPixelFormatRGBA16Float
                                     width:width
                                    height:height
                                 mipmapped:NO];
    frameDesc.usage = MTLTextureUsageShaderRead | MTLTextureUsageShaderWrite;
    frameDesc.storageMode = MTLStorageModePrivate;

    previousFrameTexture = [device newTextureWithDescriptor:frameDesc];
    previousFrameTexture.label = @"MetalGoose Previous Frame (rgba16Float)";

    currentFrameTexture = [device newTextureWithDescriptor:frameDesc];
    currentFrameTexture.label = @"MetalGoose Current Frame (rgba16Float)";

    cachedPrevTex = [device newTextureWithDescriptor:outputDesc];
    cachedPrevTex.label = @"MetalGoose Cached Previous";

    cachedInterpTex = [device newTextureWithDescriptor:outputDesc];
    cachedInterpTex.label = @"MetalGoose Interpolated Frame";

    MTLTextureDescriptor *flowDesc = [MTLTextureDescriptor
        texture2DDescriptorWithPixelFormat:MTLPixelFormatRG16Float
                                     width:width
                                    height:height
                                 mipmapped:NO];
    flowDesc.usage = MTLTextureUsageShaderRead | MTLTextureUsageShaderWrite;
    flowDesc.storageMode = MTLStorageModePrivate;

    flowTexture = [device newTextureWithDescriptor:flowDesc];
    flowTexture.label = @"MetalGoose Optical Flow (rg16Float)";

    MTLTextureDescriptor *mvDesc = [MTLTextureDescriptor
        texture2DDescriptorWithPixelFormat:MTLPixelFormatRGBA16Float
                                     width:width
                                    height:height
                                 mipmapped:NO];
    mvDesc.usage = MTLTextureUsageShaderRead | MTLTextureUsageShaderWrite;
    mvDesc.storageMode = MTLStorageModePrivate;

    cachedMotionVectorTex = [device newTextureWithDescriptor:mvDesc];
    cachedMotionVectorTex.label = @"MetalGoose Motion Vectors";

    interpolatedTexture = [device newTextureWithDescriptor:frameDesc];
    interpolatedTexture.label = @"MetalGoose Interpolated (rgba16Float)";

    scaledTexture = [device newTextureWithDescriptor:frameDesc];
    scaledTexture.label = @"MetalGoose Scaled";

    MTLTextureDescriptor *confDesc = [MTLTextureDescriptor
        texture2DDescriptorWithPixelFormat:MTLPixelFormatR16Float
                                     width:width
                                    height:height
                                 mipmapped:NO];
    confDesc.usage = MTLTextureUsageShaderRead | MTLTextureUsageShaderWrite;
    confDesc.storageMode = MTLStorageModePrivate;

    cachedConfidenceTex = [device newTextureWithDescriptor:confDesc];
    cachedConfidenceTex.label = @"MetalGoose Confidence";
  }

  id<MTLComputePipelineState> getInterpolationPipeline() {
    switch (config.frameGenQuality) {
    case DirectFrameGenQualityPerformance:
      return cachedPerformancePipeline;
    case DirectFrameGenQualityBalanced:
      return cachedBalancedPipeline;
    case DirectFrameGenQualityQuality:
      return cachedQualityPipeline ? cachedQualityPipeline
                                   : cachedAdaptiveInterpPipeline;
    default:
      return cachedBalancedPipeline;
    }
  }
};

@interface DirectEngineStreamOutput
    : NSObject <SCStreamOutput, SCStreamDelegate>
@property(nonatomic, assign) DirectFrameEngineOpaque *engine;
@end

@implementation DirectEngineStreamOutput

- (void)stream:(SCStream *)stream
    didOutputSampleBuffer:(CMSampleBufferRef)sampleBuffer
                   ofType:(SCStreamOutputType)type {
  if (type != SCStreamOutputTypeScreen)
    return;
  if (!self.engine)
    return;
  if (self.engine->isPaused)
    return;

  CFArrayRef attachments =
      CMSampleBufferGetSampleAttachmentsArray(sampleBuffer, false);
  if (attachments && CFArrayGetCount(attachments) > 0) {
    CFDictionaryRef attachment =
        (CFDictionaryRef)CFArrayGetValueAtIndex(attachments, 0);

    CFTypeRef statusRef = CFDictionaryGetValue(
        attachment, (__bridge CFStringRef)SCStreamFrameInfoStatus);
    if (statusRef) {
      NSInteger status = [(NSNumber *)(__bridge id)statusRef integerValue];
      if (status != 0) {
        return;
      }
    }

    CFTypeRef contentRectRef = CFDictionaryGetValue(
        attachment, (__bridge CFStringRef)SCStreamFrameInfoContentRect);
    if (contentRectRef) {
      (void)contentRectRef;
    }
  }

  CVPixelBufferRef pixelBuffer = CMSampleBufferGetImageBuffer(sampleBuffer);
  if (!pixelBuffer)
    return;

  DirectFrameEngineOpaque *engine = self.engine;

  CMTime presentationTime =
      CMSampleBufferGetPresentationTimeStamp(sampleBuffer);
  double timestamp = CMTimeGetSeconds(presentationTime);

  size_t width = CVPixelBufferGetWidth(pixelBuffer);
  size_t height = CVPixelBufferGetHeight(pixelBuffer);

  if (width == 0 || height == 0)
    return;

  CVReturn lockResult =
      CVPixelBufferLockBaseAddress(pixelBuffer, kCVPixelBufferLock_ReadOnly);
  if (lockResult != kCVReturnSuccess)
    return;

  void *baseAddress = CVPixelBufferGetBaseAddress(pixelBuffer);
  size_t bytesPerRow = CVPixelBufferGetBytesPerRow(pixelBuffer);
  size_t dataSize = bytesPerRow * height;

  uint64_t currentHash = DirectFrameEngineOpaque::computeFrameHash(
      (const uint8_t *)baseAddress, width, height, bytesPerRow);

  if (currentHash == engine->lastFrameHash && engine->lastFrameHash != 0) {
    engine->duplicateFrameCount++;
    CVPixelBufferUnlockBaseAddress(pixelBuffer, kCVPixelBufferLock_ReadOnly);
    return;
  }

  engine->lastFrameHash = currentHash;

  auto now = std::chrono::high_resolution_clock::now();
  float realFrameDelta =
      std::chrono::duration<float>(now - engine->lastRealFrameTime).count();
  engine->lastRealFrameTime = now;

  if (realFrameDelta > 0.001f && realFrameDelta < 1.0f) {
    float instantGameFPS = 1.0f / realFrameDelta;
    float currentRealFPS = engine->realGameFPS.load(std::memory_order_relaxed);
    if (currentRealFPS < 1.0f) {
      engine->realGameFPS.store(instantGameFPS, std::memory_order_relaxed);
    } else {
      engine->realGameFPS.store(currentRealFPS * 0.85f + instantGameFPS * 0.15f,
                                std::memory_order_relaxed);
    }
  }

  void *pixelDataCopy = malloc(dataSize);
  if (pixelDataCopy) {
    memcpy(pixelDataCopy, baseAddress, dataSize);
  }

  CVPixelBufferUnlockBaseAddress(pixelBuffer, kCVPixelBufferLock_ReadOnly);

  if (!pixelDataCopy)
    return;

  CVPixelBufferRef copiedBuffer = nullptr;
  NSDictionary *pixelBufferAttributes = @{
    (NSString *)kCVPixelBufferIOSurfacePropertiesKey : @{},
    (NSString *)kCVPixelBufferMetalCompatibilityKey : @YES
  };

  CVReturn createResult = CVPixelBufferCreateWithBytes(
      kCFAllocatorDefault, width, height, kCVPixelFormatType_32BGRA,
      pixelDataCopy, bytesPerRow,
      [](void *releaseRefCon, const void *baseAddress) {
        free((void *)baseAddress);
      },
      nullptr, (__bridge CFDictionaryRef)pixelBufferAttributes, &copiedBuffer);

  if (createResult != kCVReturnSuccess || !copiedBuffer) {
    free(pixelDataCopy);
    return;
  }

  {
    std::lock_guard<std::mutex> lock(engine->frameMutex);

    engine->previousFrameTimestamp = engine->currentFrameTimestamp;
    engine->currentFrameTimestamp = timestamp;
    engine->deltaTime = (float)(timestamp - engine->previousFrameTimestamp);

    if (engine->previousPixelBuffer) {
      CVPixelBufferRelease(engine->previousPixelBuffer);
    }
    engine->previousPixelBuffer = engine->currentPixelBuffer;
    engine->currentPixelBuffer = copiedBuffer;
  }

  IOSurfaceRef copiedSurface = CVPixelBufferGetIOSurface(copiedBuffer);
  if (copiedSurface) {
    engine->pushFrame(copiedSurface, timestamp);
  }

  engine->hasNewFrame.store(true, std::memory_order_release);
  engine->frameNumber.fetch_add(1, std::memory_order_relaxed);

  CMTime outputPresentationTime =
      CMSampleBufferGetOutputPresentationTimeStamp(sampleBuffer);
  if (CMTIME_IS_VALID(outputPresentationTime)) {
    double outputTimestamp = CMTimeGetSeconds(outputPresentationTime);
    if (outputTimestamp > 0 && timestamp > 0) {
      float latencyMs = (float)((outputTimestamp - timestamp) * 1000.0);
      engine->captureLatency.store(latencyMs, std::memory_order_relaxed);
    }
  }

  auto fpsNow = std::chrono::high_resolution_clock::now();
  float elapsedTime =
      std::chrono::duration<float>(fpsNow - engine->lastFrameTime).count();
  engine->lastFrameTime = fpsNow;

  if (elapsedTime > 0.001f && elapsedTime < 1.0f) {
    float instantFPS = 1.0f / elapsedTime;
    float currentCaptureFPS =
        engine->captureFPS.load(std::memory_order_relaxed);

    if (currentCaptureFPS < 1.0f) {
      engine->captureFPS.store(instantFPS, std::memory_order_relaxed);
    } else {
      float alpha = 0.15f;
      engine->captureFPS.store(currentCaptureFPS * (1.0f - alpha) +
                                   instantFPS * alpha,
                               std::memory_order_relaxed);
    }

    if (engine->config.frameGenMode != DirectFrameGenOff) {
      float capFPS = engine->captureFPS.load();
      float interpFPS;

      if (engine->config.frameGenType == DirectFrameGenTypeAdaptive) {
        interpFPS = static_cast<float>(engine->config.adaptiveTargetFPS);
      } else {
        interpFPS = capFPS * engine->config.frameGenMultiplier;
      }
      engine->interpolatedFPS.store(interpFPS, std::memory_order_relaxed);
    } else {
      engine->interpolatedFPS.store(engine->captureFPS.load(),
                                    std::memory_order_relaxed);
    }
  }

  engine->state.store(DirectEngineStateCapturing, std::memory_order_relaxed);
}

- (void)stream:(SCStream *)stream didStopWithError:(NSError *)error {
  if (error && self.engine) {
    self.engine->lastError = error.localizedDescription.UTF8String;
    self.engine->state.store(DirectEngineStateError, std::memory_order_relaxed);
  }
}

@end

static DirectEngineStreamOutput *gStreamOutput = nil;

DirectEngineRef DirectEngine_Create(id<MTLDevice> device,
                                    id<MTLCommandQueue> queue) {
  if (!device || !queue) {
    return nullptr;
  }

  DirectFrameEngineOpaque *engine = new DirectFrameEngineOpaque();
  engine->device = device;
  engine->commandQueue = queue;

  CVReturn status = CVMetalTextureCacheCreate(kCFAllocatorDefault, nil, device,
                                              nil, &engine->textureCache);

  if (status != kCVReturnSuccess || !engine->textureCache) {
    delete engine;
    return nullptr;
  }

  void *cppDevice = (__bridge void *)device;
  void *cppQueue = (__bridge void *)queue;
  engine->cppEngine = Engine_Create(cppDevice, cppQueue);

  if (!engine->cppEngine) {
    CFRelease(engine->textureCache);
    delete engine;
    return nullptr;
  }

  engine->setupPipelines();

  engine->initAsyncFrameGen();

  dispatch_queue_attr_t attr = dispatch_queue_attr_make_with_qos_class(
      DISPATCH_QUEUE_SERIAL, QOS_CLASS_USER_INTERACTIVE, -1);
  engine->captureQueue = dispatch_queue_create("com.metalgoose.capture", attr);

  engine->state.store(DirectEngineStateIdle, std::memory_order_relaxed);

  return engine;
}

void DirectEngine_Destroy(DirectEngineRef engine) {
  if (!engine)
    return;

  DirectEngine_StopCapture(engine);

  if (engine->cppEngine) {
    Engine_Destroy(engine->cppEngine);
    engine->cppEngine = nullptr;
  }

  delete engine;
}

void DirectEngine_SetConfig(DirectEngineRef engine, DirectEngineConfig config) {
  if (!engine || !engine->cppEngine)
    return;

  engine->config = config;
  Engine_SetConfig(engine->cppEngine, &config);

  if (engine->captureStream && engine->streamConfig) {
    SCStreamConfiguration *streamConfig = engine->streamConfig;
    streamConfig.minimumFrameInterval = CMTimeMake(1, config.targetFPS);

    [engine->captureStream updateConfiguration:streamConfig
                             completionHandler:^(NSError *error) {
                               if (error) {
                                 engine->lastError =
                                     [error.localizedDescription UTF8String];
                               }
                             }];
  }
}

bool DirectEngine_SetTargetWindow(DirectEngineRef engine, CGWindowID windowID) {
  if (!engine)
    return false;

  engine->targetWindowID = windowID;
  engine->useWindowCapture = true;
  engine->contentFilter = nil;

  return true;
}

bool DirectEngine_SetTargetDisplay(DirectEngineRef engine,
                                   CGDirectDisplayID displayID) {
  if (!engine)
    return false;

  engine->targetDisplayID = displayID;
  engine->useWindowCapture = false;
  engine->contentFilter = nil;

  return true;
}

bool DirectEngine_StartCapture(DirectEngineRef engine) {
  if (!engine)
    return false;
  if (engine->state.load(std::memory_order_relaxed) ==
      DirectEngineStateCapturing) {
    return true;
  }

  SCStreamConfiguration *config = [[SCStreamConfiguration alloc] init];

  config.pixelFormat = kCVPixelFormatType_32BGRA;

  config.minimumFrameInterval = CMTimeMake(1, engine->config.targetFPS);

  config.queueDepth = 4;

  config.captureResolution = SCCaptureResolutionAutomatic;
  config.scalesToFit = NO;
  config.showsCursor = engine->config.captureMouseCursor;
  config.colorSpaceName = kCGColorSpaceDisplayP3;
  config.backgroundColor = CGColorGetConstantColor(kCGColorBlack);
  config.preservesAspectRatio = YES;

  engine->streamConfig = config;

  dispatch_semaphore_t semaphore = dispatch_semaphore_create(0);
  __block bool success = false;

  CGWindowID targetWID = engine->targetWindowID;
  bool useWindow = engine->useWindowCapture;
  float renderScaleFactor = 1.0f;
  switch (engine->config.renderScale) {
  case DirectRenderScale75:
    renderScaleFactor = 0.75f;
    break;
  case DirectRenderScale67:
    renderScaleFactor = 0.67f;
    break;
  case DirectRenderScale50:
    renderScaleFactor = 0.50f;
    break;
  case DirectRenderScale33:
    renderScaleFactor = 0.33f;
    break;
  default:
    renderScaleFactor = 1.0f;
    break;
  }

  [SCShareableContent getShareableContentWithCompletionHandler:^(
                          SCShareableContent *content, NSError *error) {
    if (error) {
      engine->lastError = [error.localizedDescription UTF8String];
      dispatch_semaphore_signal(semaphore);
      return;
    }

    SCContentFilter *filter = nil;

    if (useWindow && targetWID != kCGNullWindowID) {
      SCWindow *targetWindow = nil;
      for (SCWindow *window in content.windows) {
        if (window.windowID == targetWID) {
          targetWindow = window;
          break;
        }
      }

      if (targetWindow) {
        size_t baseW = engine->config.baseWidth > 0
                           ? (size_t)engine->config.baseWidth
                           : (size_t)targetWindow.frame.size.width;
        size_t baseH = engine->config.baseHeight > 0
                           ? (size_t)engine->config.baseHeight
                           : (size_t)targetWindow.frame.size.height;
        config.width = std::max<size_t>(
            1, (size_t)std::llround((double)baseW * (double)renderScaleFactor));
        config.height = std::max<size_t>(
            1, (size_t)std::llround((double)baseH * (double)renderScaleFactor));
        filter = [[SCContentFilter alloc]
            initWithDesktopIndependentWindow:targetWindow];
      } else {
        engine->lastError = "Target window not found";
        dispatch_semaphore_signal(semaphore);
        return;
      }
    } else {
      NSArray<SCDisplay *> *displays = content.displays;
      if (displays.count == 0) {
        engine->lastError = "No displays available";
        dispatch_semaphore_signal(semaphore);
        return;
      }

      SCDisplay *selected = displays[0];
      if (engine->targetDisplayID != 0) {
        for (SCDisplay *disp in displays) {
          if (disp.displayID == engine->targetDisplayID) {
            selected = disp;
            break;
          }
        }
      }

      size_t baseW = engine->config.baseWidth > 0
                         ? (size_t)engine->config.baseWidth
                         : (size_t)selected.width;
      size_t baseH = engine->config.baseHeight > 0
                         ? (size_t)engine->config.baseHeight
                         : (size_t)selected.height;
      config.width = std::max<size_t>(
          1, (size_t)std::llround((double)baseW * (double)renderScaleFactor));
      config.height = std::max<size_t>(
          1, (size_t)std::llround((double)baseH * (double)renderScaleFactor));
      filter = [[SCContentFilter alloc] initWithDisplay:selected
                                       excludingWindows:@[]];
    }

    engine->contentFilter = filter;

    engine->captureStream = [[SCStream alloc] initWithFilter:filter
                                               configuration:config
                                                    delegate:nil];

    if (!gStreamOutput) {
      gStreamOutput = [[DirectEngineStreamOutput alloc] init];
    }
    gStreamOutput.engine = engine;

    NSError *streamError = nil;
    [engine->captureStream addStreamOutput:gStreamOutput
                                      type:SCStreamOutputTypeScreen
                        sampleHandlerQueue:engine->captureQueue
                                     error:&streamError];

    if (streamError) {
      engine->lastError = [streamError.localizedDescription UTF8String];
      engine->captureStream = nil;
      dispatch_semaphore_signal(semaphore);
      return;
    }

    [engine->captureStream startCaptureWithCompletionHandler:^(NSError *error) {
      if (error) {
        engine->lastError = [error.localizedDescription UTF8String];
        engine->state.store(DirectEngineStateError, std::memory_order_relaxed);
        success = false;
      } else {
        auto now = std::chrono::high_resolution_clock::now();
        engine->lastFrameTime = now;
        engine->fpsCounterStart = now;
        engine->fpsFrameCount = 0;
        engine->captureFPS.store(0.0f, std::memory_order_relaxed);
        engine->currentFPS.store(0.0f, std::memory_order_relaxed);

        engine->state.store(DirectEngineStateCapturing,
                            std::memory_order_relaxed);
        success = true;
      }
      dispatch_semaphore_signal(semaphore);
    }];
  }];

  dispatch_semaphore_wait(semaphore, DISPATCH_TIME_FOREVER);

  return success;
}

void DirectEngine_StopCapture(DirectEngineRef engine) {
  if (!engine)
    return;
  if (!engine->captureStream)
    return;

  dispatch_semaphore_t semaphore = dispatch_semaphore_create(0);

  [engine->captureStream stopCaptureWithCompletionHandler:^(NSError *error) {
    if (error) {
      engine->lastError = [error.localizedDescription UTF8String];
    }
    dispatch_semaphore_signal(semaphore);
  }];

  dispatch_semaphore_wait(semaphore, DISPATCH_TIME_FOREVER);

  engine->captureStream = nil;
  engine->contentFilter = nil;
  engine->streamConfig = nil;
  engine->cleanupFrameBuffer();
  engine->state.store(DirectEngineStateIdle, std::memory_order_relaxed);
}

DirectEngineState DirectEngine_GetState(DirectEngineRef engine) {
  return engine ? engine->state.load(std::memory_order_relaxed)
                : DirectEngineStateError;
}

float DirectEngine_GetCurrentFPS(DirectEngineRef engine) {
  return engine ? engine->currentFPS.load(std::memory_order_relaxed) : 0.0f;
}

bool DirectEngine_HasNewFrame(DirectEngineRef engine) {
  if (!engine)
    return false;
  return engine->hasNewFrame.load(std::memory_order_acquire);
}

IOSurfaceRef DirectEngine_GetCurrentSurface(DirectEngineRef engine) {
  if (!engine)
    return nullptr;

  std::lock_guard<std::mutex> lock(engine->frameMutex);
  if (engine->currentSurface) {
    return (IOSurfaceRef)CFRetain(engine->currentSurface);
  }
  return nullptr;
}

id<MTLTexture> DirectEngine_GetFrameTexture(DirectEngineRef engine,
                                            id<MTLDevice> device) {
  if (!engine || !device)
    return nil;

  IOSurfaceRef surface = DirectEngine_GetCurrentSurface(engine);
  if (!surface)
    return nil;

  MTLTextureDescriptor *desc = [MTLTextureDescriptor
      texture2DDescriptorWithPixelFormat:MTLPixelFormatBGRA8Unorm
                                   width:IOSurfaceGetWidth(surface)
                                  height:IOSurfaceGetHeight(surface)
                               mipmapped:NO];
  desc.usage = MTLTextureUsageShaderRead | MTLTextureUsageShaderWrite;
  desc.storageMode = MTLStorageModeShared;

  id<MTLTexture> texture = [device newTextureWithDescriptor:desc
                                                  iosurface:surface
                                                      plane:0];
  CFRelease(surface);

  return texture;
}

id<MTLTexture> DirectEngine_ProcessFrame(DirectEngineRef engine,
                                         id<MTLDevice> device,
                                         id<MTLCommandBuffer> commandBuffer) {
  if (!engine || !engine->cppEngine || !device || !commandBuffer)
    return nil;

  auto startTime = std::chrono::high_resolution_clock::now();

  auto frameEntry = engine->popFrame();
  if (!frameEntry.surface) {
    if (engine->cachedInputTexture) {
      return engine->cachedInputTexture;
    }
    return nil;
  }

  size_t width = IOSurfaceGetWidth(frameEntry.surface);
  size_t height = IOSurfaceGetHeight(frameEntry.surface);

  engine->ensureTextures(width, height);

  MTLTextureDescriptor *desc = [MTLTextureDescriptor
      texture2DDescriptorWithPixelFormat:MTLPixelFormatBGRA8Unorm
                                   width:width
                                  height:height
                               mipmapped:NO];
  desc.usage = MTLTextureUsageShaderRead | MTLTextureUsageShaderWrite |
               MTLTextureUsageRenderTarget;
  desc.storageMode = MTLStorageModeShared;

  id<MTLTexture> inputTexture =
      [device newTextureWithDescriptor:desc
                             iosurface:frameEntry.surface
                                 plane:0];

  if (engine->previousSurface) {
    CFRelease(engine->previousSurface);
  }
  engine->previousSurface = engine->currentSurface;
  engine->currentSurface = (IOSurfaceRef)CFRetain(frameEntry.surface);

  CFRelease(frameEntry.surface);

  if (!inputTexture)
    return nil;

  engine->cachedInputTexture = inputTexture;

  if (engine->config.frameGenMode != DirectFrameGenOff &&
      engine->asyncFrameGenEnabled.load(std::memory_order_acquire) &&
      engine->previousSurface && engine->currentSurface) {

    IOSurfaceRef prevSurf = engine->previousSurface;
    IOSurfaceRef currSurf = engine->currentSurface;

    dispatch_async(engine->asyncFrameGenQueue, ^{
      IOSurfaceLock(prevSurf, kIOSurfaceLockReadOnly, nullptr);
      IOSurfaceLock(currSurf, kIOSurfaceLockReadOnly, nullptr);

      const uint8_t *prevData =
          (const uint8_t *)IOSurfaceGetBaseAddress(prevSurf);
      const uint8_t *currData =
          (const uint8_t *)IOSurfaceGetBaseAddress(currSurf);
      size_t bytesPerRow = IOSurfaceGetBytesPerRow(currSurf);
      size_t w = IOSurfaceGetWidth(currSurf);
      size_t h = IOSurfaceGetHeight(currSurf);

      engine->computeCPUMotionVectors(prevData, currData, w, h, bytesPerRow);

      IOSurfaceUnlock(prevSurf, kIOSurfaceLockReadOnly, nullptr);
      IOSurfaceUnlock(currSurf, kIOSurfaceLockReadOnly, nullptr);
    });
  }

  engine->state.store(DirectEngineStateProcessing, std::memory_order_relaxed);
  engine->commandBufferCount.fetch_add(1, std::memory_order_relaxed);

  void *cppInput = (__bridge void *)inputTexture;
  void *cppCmdBuf = (__bridge void *)commandBuffer;

  void *cppOutput = Engine_ProcessFrame(engine->cppEngine, cppInput, cppCmdBuf);

  engine->state.store(DirectEngineStatePresenting, std::memory_order_relaxed);

  auto endTime = std::chrono::high_resolution_clock::now();
  float processingMs =
      std::chrono::duration<float, std::milli>(endTime - startTime).count();
  engine->processingTime.store(processingMs, std::memory_order_relaxed);

  float currentGpuTime = engine->gpuTime.load(std::memory_order_relaxed);
  if (currentGpuTime < 0.1f) {
    engine->gpuTime.store(processingMs * 0.8f, std::memory_order_relaxed);
  } else {
    engine->gpuTime.store(currentGpuTime * 0.9f + processingMs * 0.8f * 0.1f,
                          std::memory_order_relaxed);
  }

  engine->updateFPS();

  engine->computeEncoderCount.fetch_add(1, std::memory_order_relaxed);

  if (cppOutput) {
    id<MTLTexture> outputTex = (__bridge id<MTLTexture>)cppOutput;
    engine->cachedOutputTex = outputTex;
    return outputTex;
  }

  return inputTexture;
}

id<MTLTexture>
DirectEngine_GetInterpolatedFrame(DirectEngineRef engine, id<MTLDevice> device,
                                  id<MTLCommandBuffer> commandBuffer,
                                  bool *isInterpolated) {
  return DirectEngine_GetInterpolatedFrameWithT(engine, device, commandBuffer,
                                                0.5f, isInterpolated);
}

id<MTLTexture> DirectEngine_GetInterpolatedFrameWithT(
    DirectEngineRef engine, id<MTLDevice> device,
    id<MTLCommandBuffer> commandBuffer, float t, bool *isInterpolated) {
  if (!engine || !device || !commandBuffer) {
    if (isInterpolated)
      *isInterpolated = false;
    return nil;
  }

  *isInterpolated = false;

  if (engine->config.frameGenMode == DirectFrameGenOff) {
    return nil;
  }

  if (!engine->currentSurface) {
    return nil;
  }

  IOSurfaceRef prevSurface = engine->previousSurface ? engine->previousSurface
                                                     : engine->currentSurface;
  IOSurfaceRef currSurface = engine->currentSurface;

  size_t width = IOSurfaceGetWidth(currSurface);
  size_t height = IOSurfaceGetHeight(currSurface);

  if (width == 0 || height == 0) {
    return nil;
  }

  MTLTextureDescriptor *surfaceDesc = [MTLTextureDescriptor
      texture2DDescriptorWithPixelFormat:MTLPixelFormatBGRA8Unorm
                                   width:width
                                  height:height
                               mipmapped:NO];
  surfaceDesc.usage = MTLTextureUsageShaderRead | MTLTextureUsageShaderWrite;
  surfaceDesc.storageMode = MTLStorageModeShared;

  id<MTLTexture> currentTex = [device newTextureWithDescriptor:surfaceDesc
                                                     iosurface:currSurface
                                                         plane:0];
  id<MTLTexture> previousTex = [device newTextureWithDescriptor:surfaceDesc
                                                      iosurface:prevSurface
                                                          plane:0];

  if (!currentTex || !previousTex) {
    return nil;
  }

  if (engine->cppEngine) {
    void *cppResult = Engine_GenerateInterpolatedFrame(
        engine->cppEngine, (__bridge void *)previousTex,
        (__bridge void *)currentTex, t, (__bridge void *)commandBuffer);

    if (cppResult) {
      *isInterpolated = true;
      engine->interpolatedFrameCount.fetch_add(1, std::memory_order_relaxed);
      return (__bridge id<MTLTexture>)cppResult;
    }
  }

  engine->ensureTextures(width, height);

  if (!engine->cachedOutputTex) {
    return nil;
  }

  bool useExternalMotionEstimation =
      (engine->config.frameGenQuality == DirectFrameGenQualityQuality);

  bool useCPUMotionCache = false;
  {
    std::lock_guard<std::mutex> lock(engine->cpuMotionMutex);
    useCPUMotionCache = engine->cpuMotionCache.valid &&
                        engine->cpuMotionCache.width > 0 &&
                        useExternalMotionEstimation;
  }

  if (useCPUMotionCache && engine->cachedMotionVectorTex) {
    engine->uploadMotionToGPU(commandBuffer);
  } else if (useExternalMotionEstimation &&
             engine->cachedMotionEstOptPipeline &&
             engine->cachedMotionVectorTex) {
    id<MTLComputeCommandEncoder> motionEncoder =
        [commandBuffer computeCommandEncoder];
    if (motionEncoder) {
      motionEncoder.label = @"MGFG-1 Quality Motion Estimation";

      struct MGFG1ParamsBridge {
        float t;
        float motionScale;
        float occlusionThreshold;
        float temporalWeight;
        simd_uint2 textureSize;
        uint32_t qualityMode;
        uint32_t padding;
      };

      MGFG1ParamsBridge mvParams;
      mvParams.t = t;
      mvParams.motionScale = engine->config.motionScale;
      mvParams.occlusionThreshold = 0.15f;
      mvParams.temporalWeight = engine->config.temporalBlend;
      mvParams.textureSize = simd_make_uint2((uint32_t)width, (uint32_t)height);
      mvParams.qualityMode = (uint32_t)engine->config.frameGenQuality;

      [motionEncoder
          setComputePipelineState:engine->cachedMotionEstOptPipeline];
      [motionEncoder setTexture:previousTex atIndex:0];
      [motionEncoder setTexture:currentTex atIndex:1];
      [motionEncoder setTexture:engine->cachedMotionVectorTex atIndex:2];
      [motionEncoder setTexture:engine->cachedConfidenceTex atIndex:3];
      [motionEncoder setBytes:&mvParams
                       length:sizeof(MGFG1ParamsBridge)
                      atIndex:0];

      MTLSize motionThreadGroupSize = MTLSizeMake(8, 8, 1);
      MTLSize motionGridSize =
          MTLSizeMake((width + 7) / 8, (height + 7) / 8, 1);
      [motionEncoder dispatchThreadgroups:motionGridSize
                    threadsPerThreadgroup:motionThreadGroupSize];
      [motionEncoder endEncoding];

      if (engine->config.frameGenQuality == DirectFrameGenQualityQuality &&
          engine->cachedMotionRefinePipeline) {
        id<MTLComputeCommandEncoder> refineEncoder =
            [commandBuffer computeCommandEncoder];
        if (refineEncoder) {
          refineEncoder.label = @"MetalGoose Motion Refinement";
          [refineEncoder
              setComputePipelineState:engine->cachedMotionRefinePipeline];
          [refineEncoder setTexture:engine->cachedMotionVectorTex atIndex:0];
          [refineEncoder setTexture:engine->cachedConfidenceTex atIndex:1];
          [refineEncoder setTexture:engine->cachedMotionVectorTex atIndex:2];
          [refineEncoder dispatchThreadgroups:motionGridSize
                        threadsPerThreadgroup:motionThreadGroupSize];
          [refineEncoder endEncoding];
        }
      }
    }
  }

  id<MTLComputeCommandEncoder> encoder = [commandBuffer computeCommandEncoder];
  if (!encoder) {
    return nil;
  }
  encoder.label = @"MetalGoose Frame Interpolation";

  id<MTLComputePipelineState> pipeline = nil;
  switch (engine->config.frameGenQuality) {
  case DirectFrameGenQualityPerformance:
    pipeline = engine->cachedPerformancePipeline;
    break;
  case DirectFrameGenQualityBalanced:
    pipeline = engine->cachedBalancedPipeline;
    break;
  case DirectFrameGenQualityQuality:
    pipeline = engine->cachedQualityPipeline;
    break;
  default:
    pipeline = engine->cachedBalancedPipeline;
  }

  if (!pipeline) {
    pipeline = engine->cachedPerformancePipeline;
  }

  if (!pipeline) {
    [encoder endEncoding];
    return nil;
  }

  [encoder setComputePipelineState:pipeline];
  [encoder setTexture:previousTex atIndex:0];
  [encoder setTexture:currentTex atIndex:1];

  if (engine->config.frameGenQuality == DirectFrameGenQualityQuality &&
      engine->cachedMotionVectorTex && engine->cachedConfidenceTex) {
    [encoder setTexture:engine->cachedMotionVectorTex atIndex:2];
    [encoder setTexture:engine->cachedConfidenceTex atIndex:3];
    [encoder setTexture:engine->cachedOutputTex atIndex:4];

    struct MGFG1ParamsBridge {
      float t;
      float motionScale;
      float occlusionThreshold;
      float temporalWeight;
      simd_uint2 textureSize;
      uint32_t qualityMode;
      uint32_t padding;
    };

    MGFG1ParamsBridge params;
    params.t = t;
    params.motionScale = engine->config.motionScale;
    params.occlusionThreshold = 0.15f;
    params.temporalWeight = engine->config.temporalBlend;
    params.textureSize = simd_make_uint2((uint32_t)width, (uint32_t)height);
    params.qualityMode = 2;

    [encoder setBytes:&params length:sizeof(MGFG1ParamsBridge) atIndex:0];
  } else if (engine->config.frameGenQuality == DirectFrameGenQualityBalanced) {
    [encoder setTexture:engine->cachedOutputTex atIndex:2];

    struct BalancedParamsGPU {
      float t;
      uint32_t textureWidth;
      uint32_t textureHeight;
      float gradientThreshold;
      float padding;
    };

    BalancedParamsGPU params;
    params.t = t;
    params.textureWidth = (uint32_t)width;
    params.textureHeight = (uint32_t)height;
    params.gradientThreshold = 0.05f;
    params.padding = 0.0f;

    [encoder setBytes:&params length:sizeof(BalancedParamsGPU) atIndex:0];
  } else {
    [encoder setTexture:engine->cachedOutputTex atIndex:2];
    [encoder setBytes:&t length:sizeof(float) atIndex:0];
  }

  MTLSize threadGroupSize = MTLSizeMake(16, 16, 1);
  MTLSize gridSize = MTLSizeMake((width + 15) / 16, (height + 15) / 16, 1);
  [encoder dispatchThreadgroups:gridSize threadsPerThreadgroup:threadGroupSize];
  [encoder endEncoding];

  engine->computeEncoderCount.fetch_add(1, std::memory_order_relaxed);

  *isInterpolated = true;
  engine->interpolatedFrameCount.fetch_add(1, std::memory_order_relaxed);

  return engine->cachedOutputTex;
}

id<MTLTexture>
DirectEngine_InterpolateTextures(DirectEngineRef engine, id<MTLDevice> device,
                                 id<MTLCommandBuffer> commandBuffer,
                                 id<MTLTexture> prevTexture,
                                 id<MTLTexture> currTexture, float t) {
  if (!engine || !device || !commandBuffer || !prevTexture || !currTexture) {
    return nil;
  }

  if (engine->config.frameGenMode == DirectFrameGenOff) {
    return currTexture;
  }

  size_t width = currTexture.width;
  size_t height = currTexture.height;

  if (width == 0 || height == 0) {
    return currTexture;
  }

  engine->ensureTextures(width, height);

  if (!engine->cachedOutputTex) {
    return currTexture;
  }

  bool useMotionCompensation =
      (engine->config.frameGenQuality == DirectFrameGenQualityQuality ||
       engine->config.frameGenQuality == DirectFrameGenQualityBalanced);

  if (useMotionCompensation && engine->cachedMotionEstOptPipeline &&
      engine->cachedMotionVectorTex) {
    id<MTLComputeCommandEncoder> motionEncoder =
        [commandBuffer computeCommandEncoder];
    if (motionEncoder) {
      motionEncoder.label = @"MetalGoose Motion Est (InterpolateTextures)";

      struct MGFG1ParamsBridge {
        float t;
        float motionScale;
        float occlusionThreshold;
        float temporalWeight;
        simd_uint2 textureSize;
        uint32_t qualityMode;
        uint32_t padding;
      };

      MGFG1ParamsBridge mvParams;
      mvParams.t = t;
      mvParams.motionScale = engine->config.motionScale;
      mvParams.occlusionThreshold = 0.15f;
      mvParams.temporalWeight = engine->config.temporalBlend;
      mvParams.textureSize = simd_make_uint2((uint32_t)width, (uint32_t)height);
      mvParams.qualityMode = (uint32_t)engine->config.frameGenQuality;

      [motionEncoder
          setComputePipelineState:engine->cachedMotionEstOptPipeline];
      [motionEncoder setTexture:prevTexture atIndex:0];
      [motionEncoder setTexture:currTexture atIndex:1];
      [motionEncoder setTexture:engine->cachedMotionVectorTex atIndex:2];
      [motionEncoder setTexture:engine->cachedConfidenceTex atIndex:3];
      [motionEncoder setBytes:&mvParams
                       length:sizeof(MGFG1ParamsBridge)
                      atIndex:0];

      MTLSize motionThreadGroupSize = MTLSizeMake(8, 8, 1);
      MTLSize motionGridSize =
          MTLSizeMake((width + 7) / 8, (height + 7) / 8, 1);
      [motionEncoder dispatchThreadgroups:motionGridSize
                    threadsPerThreadgroup:motionThreadGroupSize];
      [motionEncoder endEncoding];
    }
  }

  id<MTLComputeCommandEncoder> encoder = [commandBuffer computeCommandEncoder];
  if (!encoder) {
    return currTexture;
  }
  encoder.label = @"MetalGoose Interpolation (InterpolateTextures)";

  id<MTLComputePipelineState> pipeline = nil;
  switch (engine->config.frameGenQuality) {
  case DirectFrameGenQualityPerformance:
    pipeline = engine->cachedPerformancePipeline;
    break;
  case DirectFrameGenQualityBalanced:
    pipeline = engine->cachedBalancedPipeline;
    break;
  case DirectFrameGenQualityQuality:
    pipeline = engine->cachedQualityPipeline;
    break;
  default:
    pipeline = engine->cachedBalancedPipeline;
  }

  if (!pipeline) {
    pipeline = engine->cachedPerformancePipeline;
  }

  if (!pipeline) {
    [encoder endEncoding];
    return currTexture;
  }

  [encoder setComputePipelineState:pipeline];
  [encoder setTexture:prevTexture atIndex:0];
  [encoder setTexture:currTexture atIndex:1];

  if (engine->config.frameGenQuality == DirectFrameGenQualityQuality &&
      engine->cachedMotionVectorTex && engine->cachedConfidenceTex) {
    [encoder setTexture:engine->cachedMotionVectorTex atIndex:2];
    [encoder setTexture:engine->cachedConfidenceTex atIndex:3];
    [encoder setTexture:engine->cachedOutputTex atIndex:4];

    struct MGFG1ParamsBridge {
      float t;
      float motionScale;
      float occlusionThreshold;
      float temporalWeight;
      simd_uint2 textureSize;
      uint32_t qualityMode;
      uint32_t padding;
    };

    MGFG1ParamsBridge params;
    params.t = t;
    params.motionScale = engine->config.motionScale;
    params.occlusionThreshold = 0.15f;
    params.temporalWeight = engine->config.temporalBlend;
    params.textureSize = simd_make_uint2((uint32_t)width, (uint32_t)height);
    params.qualityMode = 2;

    [encoder setBytes:&params length:sizeof(MGFG1ParamsBridge) atIndex:0];
  } else {
    [encoder setTexture:engine->cachedOutputTex atIndex:2];
    [encoder setBytes:&t length:sizeof(float) atIndex:0];
  }

  MTLSize threadGroupSize = MTLSizeMake(16, 16, 1);
  MTLSize gridSize = MTLSizeMake((width + 15) / 16, (height + 15) / 16, 1);
  [encoder dispatchThreadgroups:gridSize threadsPerThreadgroup:threadGroupSize];
  [encoder endEncoding];

  return engine->cachedOutputTex;
}

bool DirectEngine_ShouldShowInterpolatedFrame(DirectEngineRef engine) {
  if (!engine)
    return false;
  return engine->shouldInterpolate.load(std::memory_order_relaxed);
}

int DirectEngine_GetEffectiveMultiplier(DirectEngineRef engine) {
  if (!engine)
    return 1;
  if (engine->config.frameGenMode == DirectFrameGenOff)
    return 1;

  if (engine->config.frameGenType == DirectFrameGenTypeAdaptive) {
    float captureFPS = engine->captureFPS.load(std::memory_order_relaxed);
    if (captureFPS < 10.0f)
      captureFPS = 30.0f;

    float targetFPS = static_cast<float>(engine->config.adaptiveTargetFPS);
    int neededMultiplier = static_cast<int>(std::ceil(targetFPS / captureFPS));

    return std::max(1, std::min(4, neededMultiplier));
  } else {
    return engine->config.frameGenMultiplier;
  }
}

float DirectEngine_GetInterpolatedFPS(DirectEngineRef engine) {
  if (!engine)
    return 0.0f;

  float baseFPS = engine->captureFPS.load(std::memory_order_relaxed);
  if (baseFPS <= 0.0f)
    baseFPS = 60.0f;

  if (engine->config.frameGenMode != DirectFrameGenOff) {
    if (engine->config.frameGenType == DirectFrameGenTypeAdaptive) {
      return static_cast<float>(engine->config.adaptiveTargetFPS);
    } else {
      return baseFPS * engine->config.frameGenMultiplier;
    }
  }

  return baseFPS;
}

int DirectEngine_GetFrameGenMode(DirectEngineRef engine) {
  if (!engine)
    return 0;
  return static_cast<int>(engine->config.frameGenMode);
}

int DirectEngine_GetFrameGenType(DirectEngineRef engine) {
  if (!engine)
    return 0;
  return static_cast<int>(engine->config.frameGenType);
}

int DirectEngine_GetFrameGenMultiplier(DirectEngineRef engine) {
  if (!engine)
    return 1;
  return DirectEngine_GetEffectiveMultiplier(engine);
}

bool DirectEngine_IsAsyncFrameGenEnabled(DirectEngineRef engine) {
  if (!engine)
    return false;
  return engine->asyncFrameGenEnabled.load(std::memory_order_acquire);
}

bool DirectEngine_HasPendingAsyncFrame(DirectEngineRef engine) {
  if (!engine)
    return false;
  return engine->hasReadyInterpolatedFrame();
}

id<MTLTexture> DirectEngine_GetNextAsyncFrame(DirectEngineRef engine) {
  if (!engine)
    return nil;
  return engine->getNextAsyncOutputTexture();
}

void DirectEngine_SetTargetFrameMultiplier(DirectEngineRef engine,
                                           int multiplier) {
  if (!engine)
    return;
  multiplier = std::max(1, std::min(multiplier, 4));
  engine->targetFrameMultiplier.store(multiplier, std::memory_order_release);
  engine->config.frameGenMultiplier = multiplier;
}

bool DirectEngine_HasValidCPUMotion(DirectEngineRef engine) {
  if (!engine)
    return false;
  std::lock_guard<std::mutex> lock(engine->cpuMotionMutex);
  return engine->cpuMotionCache.valid;
}

static uint64_t calculateTextureMemory(DirectFrameEngineOpaque *engine) {
  if (!engine || engine->cachedWidth == 0 || engine->cachedHeight == 0)
    return 0;

  size_t w = engine->cachedWidth;
  size_t h = engine->cachedHeight;
  uint64_t totalBytes = 0;

  size_t bgra8Size = w * h * 4;
  int bgra8Count = 0;
  if (engine->cachedOutputTex)
    bgra8Count++;
  if (engine->cachedPrevTex)
    bgra8Count++;
  if (engine->cachedInterpTex)
    bgra8Count++;
  if (engine->cachedInputTexture)
    bgra8Count++;
  if (engine->finalOutputTexture)
    bgra8Count++;
  totalBytes += bgra8Size * bgra8Count;

  size_t rgba16fSize = w * h * 8;
  int rgba16fCount = 0;
  if (engine->previousFrameTexture)
    rgba16fCount++;
  if (engine->currentFrameTexture)
    rgba16fCount++;
  if (engine->interpolatedTexture)
    rgba16fCount++;
  if (engine->scaledTexture)
    rgba16fCount++;
  if (engine->cachedMotionVectorTex)
    rgba16fCount++;
  totalBytes += rgba16fSize * rgba16fCount;

  size_t rg16fSize = w * h * 4;
  if (engine->flowTexture)
    totalBytes += rg16fSize;

  size_t r16fSize = w * h * 2;
  if (engine->cachedConfidenceTex)
    totalBytes += r16fSize;

  return totalBytes;
}

DirectEngineStats DirectEngine_GetStats(DirectEngineRef engine) {
  DirectEngineStats stats = {};
  if (!engine)
    return stats;

  float currentFPS = engine->currentFPS.load(std::memory_order_relaxed);
  float captureFPS = engine->captureFPS.load(std::memory_order_relaxed);

  stats.fps = (currentFPS > 0.0f) ? currentFPS : captureFPS;
  stats.captureFPS = captureFPS;

  stats.interpolatedFPS =
      engine->interpolatedFPS.load(std::memory_order_relaxed);
  if (stats.interpolatedFPS <= 0.0f &&
      engine->config.frameGenMode != DirectFrameGenOff) {
    float baseFPS = (captureFPS > 0.0f) ? captureFPS : 30.0f;
    if (engine->config.frameGenType == DirectFrameGenTypeAdaptive) {
      stats.interpolatedFPS =
          static_cast<float>(engine->config.adaptiveTargetFPS);
    } else {
      stats.interpolatedFPS =
          baseFPS * static_cast<float>(engine->config.frameGenMultiplier);
    }
  } else if (stats.interpolatedFPS <= 0.0f) {
    stats.interpolatedFPS = stats.fps;
  }

  stats.frameTime = engine->processingTime.load(std::memory_order_relaxed);
  stats.gpuTime = engine->gpuTime.load(std::memory_order_relaxed);
  stats.captureLatency = engine->captureLatency.load(std::memory_order_relaxed);
  stats.presentLatency = engine->presentLatency.load(std::memory_order_relaxed);

  stats.frameCount = engine->frameNumber.load(std::memory_order_relaxed);
  stats.interpolatedFrameCount =
      engine->interpolatedFrameCount.load(std::memory_order_relaxed);
  stats.droppedFrames = engine->droppedFrames.load(std::memory_order_relaxed);

  if (engine->device) {
    uint64_t textureMemory = calculateTextureMemory(engine);

    uint64_t deviceAllocated = engine->device.currentAllocatedSize;
    uint64_t recommendedMax = engine->device.recommendedMaxWorkingSetSize;

    if (deviceAllocated > 0) {
      stats.gpuMemoryUsed = deviceAllocated;
    } else {
      stats.gpuMemoryUsed = textureMemory;
    }

    if (recommendedMax > 0) {
      stats.gpuMemoryTotal = recommendedMax;
    } else {
      stats.gpuMemoryTotal = textureMemory * 8;
    }

    stats.textureMemoryUsed = textureMemory;
  }

  stats.renderEncoders =
      engine->renderEncoderCount.load(std::memory_order_relaxed);
  stats.computeEncoders =
      engine->computeEncoderCount.load(std::memory_order_relaxed);
  stats.blitEncoders = engine->blitEncoderCount.load(std::memory_order_relaxed);
  stats.commandBuffers =
      engine->commandBufferCount.load(std::memory_order_relaxed);
  stats.drawCalls = engine->drawCallCount.load(std::memory_order_relaxed);

  stats.upscaleMode = static_cast<uint32_t>(engine->config.upscaleMode);
  stats.frameGenMode = static_cast<uint32_t>(engine->config.frameGenMode);
  stats.aaMode = static_cast<uint32_t>(engine->config.aaMode);

  return stats;
}

NSString *DirectEngine_GetLastError(DirectEngineRef engine) {
  if (!engine || engine->lastError.empty())
    return nil;
  return [NSString stringWithUTF8String:engine->lastError.c_str()];
}

void DirectEngine_ClearError(DirectEngineRef engine) {
  if (!engine)
    return;
  engine->lastError.clear();
}

DirectEngineConfig DirectEngine_GetConfig(DirectEngineRef engine) {
  if (!engine) {
    DirectEngineConfig empty = {};
    return empty;
  }
  return engine->config;
}

bool DirectEngine_SetOutputDisplay(DirectEngineRef engine,
                                   CGDirectDisplayID displayID) {
  if (!engine)
    return false;
  engine->outputDisplayID = displayID;
  return true;
}

void DirectEngine_PauseCapture(DirectEngineRef engine) {
  if (!engine)
    return;
  engine->isPaused = true;
}

void DirectEngine_ResumeCapture(DirectEngineRef engine) {
  if (!engine)
    return;
  engine->isPaused = false;
}

float DirectEngine_GetCaptureFPS(DirectEngineRef engine) {
  return engine ? engine->captureFPS.load(std::memory_order_relaxed) : 0.0f;
}

void DirectEngine_ResetStats(DirectEngineRef engine) {
  if (!engine)
    return;

  engine->frameNumber.store(0, std::memory_order_relaxed);
  engine->droppedFrames.store(0, std::memory_order_relaxed);
  engine->interpolatedFrameCount.store(0, std::memory_order_relaxed);
  engine->renderEncoderCount.store(0, std::memory_order_relaxed);
  engine->computeEncoderCount.store(0, std::memory_order_relaxed);
  engine->blitEncoderCount.store(0, std::memory_order_relaxed);
  engine->commandBufferCount.store(0, std::memory_order_relaxed);
  engine->drawCallCount.store(0, std::memory_order_relaxed);
  engine->fpsFrameCount = 0;
  engine->fpsCounterStart = std::chrono::high_resolution_clock::now();
}

void DirectEngine_SetDebugMode(DirectEngineRef engine, bool enabled) {
  if (!engine)
    return;
  engine->debugMode = enabled;
}

NSString *DirectEngine_GetDebugInfo(DirectEngineRef engine) {
  if (!engine)
    return nil;

  DirectEngineStats stats = DirectEngine_GetStats(engine);

  NSString *stateStr;
  switch (engine->state.load()) {
  case DirectEngineStateIdle:
    stateStr = @"Idle";
    break;
  case DirectEngineStateCapturing:
    stateStr = @"Capturing";
    break;
  case DirectEngineStateProcessing:
    stateStr = @"Processing";
    break;
  case DirectEngineStatePresenting:
    stateStr = @"Presenting";
    break;
  case DirectEngineStateError:
    stateStr = @"Error";
    break;
  default:
    stateStr = @"Unknown";
  }

  return [NSString
      stringWithFormat:@"MetalGoose Engine Debug Info\n"
                       @"============================\n"
                       @"State: %@\n"
                       @"Device: %@\n"
                       @"Capture FPS: %.1f\n"
                       @"Output FPS: %.1f\n"
                       @"Interpolated FPS: %.1f\n"
                       @"Frame Time: %.2f ms\n"
                       @"GPU Time: %.2f ms\n"
                       @"Capture Latency: %.2f ms\n"
                       @"Frames Processed: %llu\n"
                       @"Frames Dropped: %llu\n"
                       @"Frames Interpolated: %llu\n"
                       @"GPU Memory: %.1f / %.1f MB\n"
                       @"Upscale Mode: %ld\n"
                       @"Frame Gen Mode: %ld (x%d)\n"
                       @"AA Mode: %ld\n",
                       stateStr, engine->device.name, stats.captureFPS,
                       stats.fps, stats.interpolatedFPS, stats.frameTime,
                       stats.gpuTime, engine->captureLatency.load(),
                       stats.frameCount, stats.droppedFrames,
                       engine->interpolatedFrameCount.load(),
                       stats.gpuMemoryUsed / (1024.0 * 1024.0),
                       stats.gpuMemoryTotal / (1024.0 * 1024.0),
                       (long)engine->config.upscaleMode,
                       (long)engine->config.frameGenMode,
                       engine->config.frameGenMultiplier,
                       (long)engine->config.aaMode];
}
