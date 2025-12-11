#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#if __has_include(<MetalFX/MetalFX.h>)
#import <MetalFX/MetalFX.h>
#define HAS_METALFX 1
#else
#define HAS_METALFX 0
#endif
#import <simd/simd.h>

#include <array>
#include <atomic>
#include <chrono>

enum class FrameGenQuality : uint32_t {
  Performance = 0,
  Balanced = 1,
  Quality = 2
};

enum class AAMode : uint32_t { Off = 0, FXAA = 1, SMAA = 2, MSAA = 3, TAA = 4 };

enum class UpscaleMode : uint32_t {
  Off = 0,
  Bilinear = 1,
  CAS = 2,
  MetalFX = 3
};

struct EngineConfig {
  bool frameGenEnabled = false;
  FrameGenQuality frameGenQuality = FrameGenQuality::Balanced;
  uint32_t frameGenMultiplier = 2;
  AAMode aaMode = AAMode::Off;
  float aaThreshold = 0.166f;
  UpscaleMode upscaleMode = UpscaleMode::Off;
  float sharpness = 0.5f;
  float upscaleFactor = 1.0f;
  float renderScaleFactor = 1.0f;
  uint32_t baseWidth = 0;
  uint32_t baseHeight = 0;
  uint32_t outputWidth = 0;
  uint32_t outputHeight = 0;
  uint32_t targetFPS = 120;
  bool adaptiveQuality = true;
  float temporalBlend = 0.1f;
  bool enableTemporalAccumulation = true;
  bool useMotionVectors = true;
  float motionScale = 1.0f;
};

struct FrameData {
  id<MTLTexture> texture = nil;
  double timestamp = 0.0;
  uint64_t frameIndex = 0;
  float deltaTime = 0.0f;
};

template <typename T, size_t N> class CircularBuffer {
public:
  void push(const T &item) {
    buffer_[writeIdx_] = item;
    writeIdx_ = (writeIdx_ + 1) % N;
    if (count_ < N)
      count_++;
  }

  T *get(size_t offset) {
    if (offset >= count_)
      return nullptr;
    size_t idx = (writeIdx_ + N - 1 - offset) % N;
    return &buffer_[idx];
  }

  T *newest() { return get(0); }
  T *previous() { return get(1); }
  size_t size() const { return count_; }
  void clear() {
    count_ = 0;
    writeIdx_ = 0;
  }

private:
  std::array<T, N> buffer_;
  size_t writeIdx_ = 0;
  size_t count_ = 0;
};

struct MGFG1Params {
  float t;
  float motionScale;
  float occlusionThreshold;
  float temporalWeight;
  simd_uint2 textureSize;
  uint32_t qualityMode;
  uint32_t padding;
};

struct AntiAliasParams {
  float threshold;
  float depthThreshold;
  int32_t maxSearchSteps;
  float subpixelBlend;
};

struct SharpenParams {
  float sharpness;
  float radius;
};

class Engine {
public:
  Engine(id<MTLDevice> device, id<MTLCommandQueue> queue);
  ~Engine();

  void setConfig(const EngineConfig &config) { config_ = config; }
  EngineConfig getConfig() const { return config_; }

  void pushFrame(id<MTLTexture> texture, double timestamp);
  uint64_t getFrameIndex() const {
    return frameIndex_.load(std::memory_order_relaxed);
  }

  id<MTLTexture> processFrame(id<MTLTexture> inputTexture,
                              id<MTLCommandBuffer> commandBuffer);
  id<MTLTexture> generateInterpolatedFrame(id<MTLTexture> prevTexture,
                                           id<MTLTexture> currTexture, float t,
                                           id<MTLCommandBuffer> commandBuffer);

  id<MTLTexture> applyAntiAliasing(id<MTLTexture> inputTexture,
                                   id<MTLCommandBuffer> commandBuffer);
  id<MTLTexture> applySharpening(id<MTLTexture> inputTexture,
                                 id<MTLCommandBuffer> commandBuffer);
  id<MTLTexture> applyTAA(id<MTLTexture> inputTexture,
                          id<MTLCommandBuffer> commandBuffer);

  float getProcessingTime() const {
    return processingTime_.load(std::memory_order_relaxed);
  }
  float getGPUTime() const { return gpuTime_.load(std::memory_order_relaxed); }

private:
  void setupPipelines();
  void ensureTextures(size_t width, size_t height);
  void ensureMotionTextures(size_t width, size_t height);
  void ensurePyramidTextures(size_t width, size_t height);
  void ensureScaledOutputTexture(size_t outputWidth, size_t outputHeight);
  void ensureScaler(size_t inputWidth, size_t inputHeight, size_t outputWidth,
                    size_t outputHeight);

  void computeMotionVectors(id<MTLTexture> prevTex, id<MTLTexture> currTex,
                            id<MTLCommandBuffer> commandBuffer);
  void computeMotionVectorsPyramid(id<MTLTexture> prevTex,
                                   id<MTLTexture> currTex,
                                   id<MTLCommandBuffer> commandBuffer);
  id<MTLTexture> interpolateWithMotion(id<MTLTexture> prevTex,
                                       id<MTLTexture> currTex, float t,
                                       id<MTLCommandBuffer> commandBuffer);
  id<MTLTexture> interpolateSimple(id<MTLTexture> prevTex,
                                   id<MTLTexture> currTex, float t,
                                   id<MTLCommandBuffer> commandBuffer);
  id<MTLTexture> interpolateBalanced(id<MTLTexture> prevTex,
                                     id<MTLTexture> currTex, float t,
                                     id<MTLCommandBuffer> commandBuffer);
  void copyTextureToHistory(id<MTLTexture> source,
                            id<MTLCommandBuffer> commandBuffer);

  id<MTLDevice> device_;
  id<MTLCommandQueue> queue_;

  id<MTLComputePipelineState> motionEstimationPipeline_;
  id<MTLComputePipelineState> motionEstimationOptimizedPipeline_;
  id<MTLComputePipelineState> motionRefinementPipeline_;
  id<MTLComputePipelineState> pyramidDownsample2xPipeline_;
  id<MTLComputePipelineState> pyramidDownsample4xPipeline_;
  id<MTLComputePipelineState> pyramidMotionPipeline_;
  id<MTLComputePipelineState> upsampleMotionPipeline_;
  id<MTLComputePipelineState> performancePipeline_;
  id<MTLComputePipelineState> balancedPipeline_;
  id<MTLComputePipelineState> qualityPipeline_;
  id<MTLComputePipelineState> adaptivePipeline_;
  id<MTLComputePipelineState> fxaaPipeline_;
  id<MTLComputePipelineState> smaaEdgePipeline_;
  id<MTLComputePipelineState> smaaWeightPipeline_;
  id<MTLComputePipelineState> smaaBlendPipeline_;
  id<MTLComputePipelineState> msaaPipeline_;
  id<MTLComputePipelineState> taaPipeline_;
  id<MTLComputePipelineState> casPipeline_;
  id<MTLComputePipelineState> usmPipeline_;
  id<MTLComputePipelineState> temporalPipeline_;
  id<MTLComputePipelineState> copyPipeline_;
  id<MTLComputePipelineState> scalePipeline_;
#if HAS_METALFX
  id<MTLFXSpatialScaler> spatialScaler_;
#else
  id<NSObject> spatialScaler_;
#endif
  id<MTLTexture> scaledOutputTexture_;
  size_t scalerInputWidth_ = 0;
  size_t scalerInputHeight_ = 0;
  size_t scalerOutputWidth_ = 0;
  size_t scalerOutputHeight_ = 0;

  id<MTLTexture> outputTexture_;
  id<MTLTexture> tempTexture_;
  id<MTLTexture> motionVectorTexture_;
  id<MTLTexture> confidenceTexture_;
  id<MTLTexture> historyTexture_;
  id<MTLTexture> previousFrameTexture_;
  id<MTLTexture> taaOutputTexture_;
  id<MTLTexture> smaaEdgeTexture_;
  id<MTLTexture> smaaWeightTexture_;

  id<MTLTexture> pyramidPrevLevel1_;
  id<MTLTexture> pyramidPrevLevel2_;
  id<MTLTexture> pyramidCurrLevel1_;
  id<MTLTexture> pyramidCurrLevel2_;
  id<MTLTexture> motionLevel1_;
  id<MTLTexture> motionLevel2_;

  size_t cachedWidth_ = 0;
  size_t cachedHeight_ = 0;

  bool hasValidHistory_ = false;

  CircularBuffer<FrameData, 4> frameHistory_;
  std::atomic<uint64_t> frameIndex_{0};

  EngineConfig config_;

  std::atomic<float> processingTime_{0.0f};
  std::atomic<float> gpuTime_{0.0f};
  std::string lastError_;

public:
  const char *getLastError() const { return lastError_.c_str(); }
};

Engine::Engine(id<MTLDevice> device, id<MTLCommandQueue> queue)
    : device_(device), queue_(queue), motionEstimationPipeline_(nil),
      motionEstimationOptimizedPipeline_(nil), motionRefinementPipeline_(nil),
      pyramidDownsample2xPipeline_(nil), pyramidDownsample4xPipeline_(nil),
      pyramidMotionPipeline_(nil), upsampleMotionPipeline_(nil),
      performancePipeline_(nil), balancedPipeline_(nil), qualityPipeline_(nil),
      adaptivePipeline_(nil), fxaaPipeline_(nil), casPipeline_(nil),
      usmPipeline_(nil), temporalPipeline_(nil), copyPipeline_(nil),
      scalePipeline_(nil), outputTexture_(nil), tempTexture_(nil),
      motionVectorTexture_(nil), confidenceTexture_(nil), historyTexture_(nil),
      previousFrameTexture_(nil), taaOutputTexture_(nil),
      pyramidPrevLevel1_(nil), pyramidPrevLevel2_(nil), pyramidCurrLevel1_(nil),
      pyramidCurrLevel2_(nil), motionLevel1_(nil), motionLevel2_(nil),
      hasValidHistory_(false) {
  if (device_ && queue_) {
    setupPipelines();
  }
}

Engine::~Engine() {}

void Engine::setupPipelines() {
  if (!device_)
    return;

  id<MTLLibrary> library = [device_ newDefaultLibrary];
  if (!library) {
    NSBundle *mainBundle = [NSBundle mainBundle];
    NSURL *url = [mainBundle URLForResource:@"default"
                              withExtension:@"metallib"];
    if (url) {
      library = [device_ newLibraryWithURL:url error:nil];
    }
  }
  if (!library) {
    NSBundle *mainBundle = [NSBundle mainBundle];
    NSURL *url = [mainBundle URLForResource:@"Shaders"
                              withExtension:@"metallib"];
    if (url) {
      library = [device_ newLibraryWithURL:url error:nil];
    }
  }

  if (!library) {
    lastError_ = "Failed to load Metal shader library (default.metallib or "
                 "Shaders.metallib absent)";
    return;
  }

  NSError *error = nil;

  auto createPipeline = [&](NSString *name) -> id<MTLComputePipelineState> {
    id<MTLFunction> func = [library newFunctionWithName:name];
    if (!func) {
      lastError_ = "Missing shader function: " + std::string([name UTF8String]);
      return nil;
    }

    id<MTLComputePipelineState> pipeline =
        [device_ newComputePipelineStateWithFunction:func error:&error];
    if (error) {
      lastError_ =
          "Failed to create pipeline: " + std::string([name UTF8String]) +
          " - " + std::string([error.localizedDescription UTF8String]);
      return nil;
    }
    return pipeline;
  };

  motionEstimationPipeline_ = createPipeline(@"mgfg1MotionEstimation");
  motionEstimationOptimizedPipeline_ =
      createPipeline(@"mgfg1MotionEstimationOptimized");
  motionRefinementPipeline_ = createPipeline(@"mgfg1MotionRefinement");
  pyramidDownsample2xPipeline_ = createPipeline(@"pyramidDownsample2x");
  pyramidDownsample4xPipeline_ = createPipeline(@"pyramidDownsample4x");
  pyramidMotionPipeline_ = createPipeline(@"mgfg1PyramidMotionEstimation");
  upsampleMotionPipeline_ = createPipeline(@"mgfg1UpsampleMotion");
  performancePipeline_ = createPipeline(@"mgfg1Performance");
  balancedPipeline_ = createPipeline(@"mgfg1Balanced");
  qualityPipeline_ = createPipeline(@"mgfg1Quality");
  adaptivePipeline_ = createPipeline(@"mgfg1AdaptiveInterpolation");

  fxaaPipeline_ = createPipeline(@"fxaa");
  smaaEdgePipeline_ = createPipeline(@"smaaEdgeDetection");
  smaaWeightPipeline_ = createPipeline(@"smaaBlendingWeights");
  smaaBlendPipeline_ = createPipeline(@"smaaBlend");
  msaaPipeline_ = createPipeline(@"msaa");
  taaPipeline_ = createPipeline(@"taa");

  casPipeline_ = createPipeline(@"mgup1ContrastAdaptiveSharpening");
  usmPipeline_ = createPipeline(@"mgup1UnsharpMask");

  temporalPipeline_ = createPipeline(@"temporalAccumulation");
  copyPipeline_ = createPipeline(@"copyTexture");
  scalePipeline_ = createPipeline(@"blitScaleBilinear");
}

void Engine::ensureTextures(size_t width, size_t height) {
  if (width == cachedWidth_ && height == cachedHeight_ &&
      outputTexture_ != nil) {
    return;
  }

  cachedWidth_ = width;
  cachedHeight_ = height;

  hasValidHistory_ = false;

  MTLTextureDescriptor *desc = [MTLTextureDescriptor
      texture2DDescriptorWithPixelFormat:MTLPixelFormatBGRA8Unorm
                                   width:width
                                  height:height
                               mipmapped:NO];
  desc.usage = MTLTextureUsageShaderRead | MTLTextureUsageShaderWrite |
               MTLTextureUsageRenderTarget;
  desc.storageMode = MTLStorageModePrivate;

  outputTexture_ = [device_ newTextureWithDescriptor:desc];
  tempTexture_ = [device_ newTextureWithDescriptor:desc];
  historyTexture_ = [device_ newTextureWithDescriptor:desc];
  taaOutputTexture_ = [device_ newTextureWithDescriptor:desc];

  smaaEdgeTexture_ = [device_ newTextureWithDescriptor:desc];
  smaaWeightTexture_ = [device_ newTextureWithDescriptor:desc];
}

void Engine::ensureMotionTextures(size_t width, size_t height) {
  if (motionVectorTexture_ != nil && cachedWidth_ == width &&
      cachedHeight_ == height) {
    return;
  }

  MTLTextureDescriptor *mvDesc = [MTLTextureDescriptor
      texture2DDescriptorWithPixelFormat:MTLPixelFormatRGBA16Float
                                   width:width
                                  height:height
                               mipmapped:NO];
  mvDesc.usage = MTLTextureUsageShaderRead | MTLTextureUsageShaderWrite;
  mvDesc.storageMode = MTLStorageModePrivate;

  motionVectorTexture_ = [device_ newTextureWithDescriptor:mvDesc];

  MTLTextureDescriptor *confDesc = [MTLTextureDescriptor
      texture2DDescriptorWithPixelFormat:MTLPixelFormatR16Float
                                   width:width
                                  height:height
                               mipmapped:NO];
  confDesc.usage = MTLTextureUsageShaderRead | MTLTextureUsageShaderWrite;
  confDesc.storageMode = MTLStorageModePrivate;

  confidenceTexture_ = [device_ newTextureWithDescriptor:confDesc];
}

void Engine::ensurePyramidTextures(size_t width, size_t height) {
  size_t level1Width = width / 2;
  size_t level1Height = height / 2;
  size_t level2Width = width / 4;
  size_t level2Height = height / 4;

  if (pyramidPrevLevel1_ != nil && pyramidPrevLevel1_.width == level1Width &&
      pyramidPrevLevel1_.height == level1Height) {
    return;
  }

  MTLTextureDescriptor *pyramidDesc1 = [MTLTextureDescriptor
      texture2DDescriptorWithPixelFormat:MTLPixelFormatRGBA16Float
                                   width:level1Width
                                  height:level1Height
                               mipmapped:NO];
  pyramidDesc1.usage = MTLTextureUsageShaderRead | MTLTextureUsageShaderWrite;
  pyramidDesc1.storageMode = MTLStorageModePrivate;

  pyramidPrevLevel1_ = [device_ newTextureWithDescriptor:pyramidDesc1];
  pyramidCurrLevel1_ = [device_ newTextureWithDescriptor:pyramidDesc1];

  MTLTextureDescriptor *pyramidDesc2 = [MTLTextureDescriptor
      texture2DDescriptorWithPixelFormat:MTLPixelFormatRGBA16Float
                                   width:level2Width
                                  height:level2Height
                               mipmapped:NO];
  pyramidDesc2.usage = MTLTextureUsageShaderRead | MTLTextureUsageShaderWrite;
  pyramidDesc2.storageMode = MTLStorageModePrivate;

  pyramidPrevLevel2_ = [device_ newTextureWithDescriptor:pyramidDesc2];
  pyramidCurrLevel2_ = [device_ newTextureWithDescriptor:pyramidDesc2];

  MTLTextureDescriptor *mvDesc1 = [MTLTextureDescriptor
      texture2DDescriptorWithPixelFormat:MTLPixelFormatRGBA16Float
                                   width:level1Width
                                  height:level1Height
                               mipmapped:NO];
  mvDesc1.usage = MTLTextureUsageShaderRead | MTLTextureUsageShaderWrite;
  mvDesc1.storageMode = MTLStorageModePrivate;

  motionLevel1_ = [device_ newTextureWithDescriptor:mvDesc1];

  MTLTextureDescriptor *mvDesc2 = [MTLTextureDescriptor
      texture2DDescriptorWithPixelFormat:MTLPixelFormatRGBA16Float
                                   width:level2Width
                                  height:level2Height
                               mipmapped:NO];
  mvDesc2.usage = MTLTextureUsageShaderRead | MTLTextureUsageShaderWrite;
  mvDesc2.storageMode = MTLStorageModePrivate;

  motionLevel2_ = [device_ newTextureWithDescriptor:mvDesc2];
}

void Engine::ensureScaledOutputTexture(size_t outputWidth,
                                       size_t outputHeight) {
  if (scaledOutputTexture_ && scalerOutputWidth_ == outputWidth &&
      scalerOutputHeight_ == outputHeight) {
    return;
  }

  scalerOutputWidth_ = outputWidth;
  scalerOutputHeight_ = outputHeight;

  MTLTextureDescriptor *outDesc = [MTLTextureDescriptor
      texture2DDescriptorWithPixelFormat:MTLPixelFormatBGRA8Unorm
                                   width:outputWidth
                                  height:outputHeight
                               mipmapped:NO];
  outDesc.usage = MTLTextureUsageShaderRead | MTLTextureUsageShaderWrite |
                  MTLTextureUsageRenderTarget;
  outDesc.storageMode = MTLStorageModePrivate;
  scaledOutputTexture_ = [device_ newTextureWithDescriptor:outDesc];
}

#if HAS_METALFX
void Engine::ensureScaler(size_t inputWidth, size_t inputHeight,
                          size_t outputWidth, size_t outputHeight) {
  if (@available(macOS 26.0, *)) {
  } else {
    spatialScaler_ = nil;
    scaledOutputTexture_ = nil;
    return;
  }

  if (spatialScaler_ && scalerInputWidth_ == inputWidth &&
      scalerInputHeight_ == inputHeight && scalerOutputWidth_ == outputWidth &&
      scalerOutputHeight_ == outputHeight && scaledOutputTexture_) {
    return;
  }

  scalerInputWidth_ = inputWidth;
  scalerInputHeight_ = inputHeight;
  ensureScaledOutputTexture(outputWidth, outputHeight);

  MTLFXSpatialScalerDescriptor *desc =
      [[MTLFXSpatialScalerDescriptor alloc] init];
  desc.inputWidth = inputWidth;
  desc.inputHeight = inputHeight;
  desc.outputWidth = outputWidth;
  desc.outputHeight = outputHeight;
  desc.colorTextureFormat = MTLPixelFormatBGRA8Unorm;
  desc.outputTextureFormat = MTLPixelFormatBGRA8Unorm;

  desc.colorProcessingMode = MTLFXSpatialScalerColorProcessingModePerceptual;

  if (![MTLFXSpatialScalerDescriptor supportsDevice:device_]) {
    spatialScaler_ = nil;
    return;
  }

  spatialScaler_ = [desc newSpatialScalerWithDevice:device_];
}
#else
void Engine::ensureScaler(size_t inputWidth, size_t inputHeight,
                          size_t outputWidth, size_t outputHeight) {
  (void)inputWidth;
  (void)inputHeight;
  (void)outputWidth;
  (void)outputHeight;
  spatialScaler_ = nil;
  scaledOutputTexture_ = nil;
}
#endif

void Engine::pushFrame(id<MTLTexture> texture, double timestamp) {
  FrameData data;
  data.texture = texture;
  data.timestamp = timestamp;
  data.frameIndex = frameIndex_.fetch_add(1, std::memory_order_relaxed);

  FrameData *prev = frameHistory_.newest();
  if (prev) {
    data.deltaTime = static_cast<float>(timestamp - prev->timestamp);
  }

  frameHistory_.push(data);
}

id<MTLTexture> Engine::processFrame(id<MTLTexture> inputTexture,
                                    id<MTLCommandBuffer> commandBuffer) {
  auto startTime = std::chrono::high_resolution_clock::now();

  if (!inputTexture)
    return nil;

  size_t inputWidth = inputTexture.width;
  size_t inputHeight = inputTexture.height;
  size_t baseWidth = config_.baseWidth > 0 ? config_.baseWidth : inputWidth;
  size_t baseHeight = config_.baseHeight > 0 ? config_.baseHeight : inputHeight;
  if (config_.baseWidth == 0 && config_.renderScaleFactor > 0.0f &&
      config_.renderScaleFactor != 1.0f) {
    baseWidth = static_cast<size_t>(static_cast<float>(inputWidth) /
                                    config_.renderScaleFactor);
  }
  if (config_.baseHeight == 0 && config_.renderScaleFactor > 0.0f &&
      config_.renderScaleFactor != 1.0f) {
    baseHeight = static_cast<size_t>(static_cast<float>(inputHeight) /
                                     config_.renderScaleFactor);
  }
  size_t outputWidth =
      config_.outputWidth > 0 ? config_.outputWidth : baseWidth;
  size_t outputHeight =
      config_.outputHeight > 0 ? config_.outputHeight : baseHeight;

  bool needsResize = (outputWidth != inputWidth || outputHeight != inputHeight);
  if (!needsResize && config_.upscaleFactor > 1.0f &&
      config_.upscaleMode != UpscaleMode::Off) {
    outputWidth = static_cast<size_t>(static_cast<float>(inputWidth) *
                                      config_.upscaleFactor);
    outputHeight = static_cast<size_t>(static_cast<float>(inputHeight) *
                                       config_.upscaleFactor);
    needsResize = true;
  }

  ensureTextures(inputWidth, inputHeight);

  id<MTLTexture> result = inputTexture;

  if (needsResize) {
    ensureScaledOutputTexture(outputWidth, outputHeight);

    bool useMetalFX = (config_.upscaleMode == UpscaleMode::MetalFX ||
                       config_.upscaleMode == UpscaleMode::CAS);
#if HAS_METALFX
    if (@available(macOS 26.0, *)) {
      if (useMetalFX && scaledOutputTexture_) {
        ensureScaler(result.width, result.height, outputWidth, outputHeight);
        if (spatialScaler_) {
          [spatialScaler_ setColorTexture:result];
          [spatialScaler_ setOutputTexture:scaledOutputTexture_];
          [spatialScaler_ encodeToCommandBuffer:commandBuffer];
          result = scaledOutputTexture_;

          bool applySharpening = (config_.upscaleMode == UpscaleMode::CAS ||
                                  config_.sharpness > 0.35f);
          if (applySharpening && casPipeline_) {
            MTLTextureDescriptor *sharpDesc = [MTLTextureDescriptor
                texture2DDescriptorWithPixelFormat:MTLPixelFormatBGRA8Unorm
                                             width:outputWidth
                                            height:outputHeight
                                         mipmapped:NO];
            sharpDesc.usage =
                MTLTextureUsageShaderRead | MTLTextureUsageShaderWrite;
            sharpDesc.storageMode = MTLStorageModePrivate;
            id<MTLTexture> sharpTex =
                [device_ newTextureWithDescriptor:sharpDesc];

            if (sharpTex) {
              id<MTLComputeCommandEncoder> sharpEncoder =
                  [commandBuffer computeCommandEncoder];
              if (sharpEncoder) {
                sharpEncoder.label = config_.upscaleMode == UpscaleMode::CAS
                                         ? @"MGUP-1 Quality CAS"
                                         : @"MGUP-1 Standard CAS";

                SharpenParams sharpParams;
                sharpParams.sharpness = config_.sharpness;
                sharpParams.radius =
                    config_.upscaleMode == UpscaleMode::CAS ? 1.2f : 1.0f;

                [sharpEncoder setComputePipelineState:casPipeline_];
                [sharpEncoder setTexture:result atIndex:0];
                [sharpEncoder setTexture:sharpTex atIndex:1];
                [sharpEncoder setBytes:&sharpParams
                                length:sizeof(SharpenParams)
                               atIndex:0];

                MTLSize threadGroupSize = MTLSizeMake(16, 16, 1);
                MTLSize gridSize = MTLSizeMake((outputWidth + 15) / 16,
                                               (outputHeight + 15) / 16, 1);
                [sharpEncoder dispatchThreadgroups:gridSize
                             threadsPerThreadgroup:threadGroupSize];
                [sharpEncoder endEncoding];
                result = sharpTex;
              }
            }
          }
        }
      }
    }
#endif

    if (config_.upscaleMode == UpscaleMode::Bilinear ||
        (useMetalFX && result.width != outputWidth)) {
      if (scalePipeline_ && scaledOutputTexture_) {
        id<MTLComputeCommandEncoder> encoder =
            [commandBuffer computeCommandEncoder];
        if (encoder) {
          encoder.label = @"MGUP-1 Fast Bilinear Scale";
          [encoder setComputePipelineState:scalePipeline_];
          [encoder setTexture:result atIndex:0];
          [encoder setTexture:scaledOutputTexture_ atIndex:1];
          MTLSize threadGroupSize = MTLSizeMake(16, 16, 1);
          MTLSize gridSize =
              MTLSizeMake((outputWidth + 15) / 16, (outputHeight + 15) / 16, 1);
          [encoder dispatchThreadgroups:gridSize
                  threadsPerThreadgroup:threadGroupSize];
          [encoder endEncoding];
          result = scaledOutputTexture_;
        }
      }
    }
  }

  if (config_.aaMode != AAMode::Off && fxaaPipeline_) {
    id<MTLTexture> aaResult = applyAntiAliasing(result, commandBuffer);
    if (aaResult) {
      result = aaResult;
    }
  }

  if (config_.upscaleMode == UpscaleMode::Bilinear &&
      config_.sharpness > 0.2f && casPipeline_) {
    id<MTLTexture> sharpResult = applySharpening(result, commandBuffer);
    if (sharpResult) {
      result = sharpResult;
    }
  }

  auto endTime = std::chrono::high_resolution_clock::now();
  float processingMs =
      std::chrono::duration<float, std::milli>(endTime - startTime).count();
  processingTime_.store(processingMs, std::memory_order_relaxed);

  return result;
}

id<MTLTexture>
Engine::generateInterpolatedFrame(id<MTLTexture> prevTexture,
                                  id<MTLTexture> currTexture, float t,
                                  id<MTLCommandBuffer> commandBuffer) {
  if (!prevTexture || !currTexture)
    return nil;

  size_t width = currTexture.width;
  size_t height = currTexture.height;

  if (width == 0 || height == 0)
    return nil;

  ensureTextures(width, height);

  if (!outputTexture_) {
    MTLTextureDescriptor *desc = [MTLTextureDescriptor
        texture2DDescriptorWithPixelFormat:MTLPixelFormatBGRA8Unorm
                                     width:width
                                    height:height
                                 mipmapped:NO];
    desc.usage = MTLTextureUsageShaderRead | MTLTextureUsageShaderWrite |
                 MTLTextureUsageRenderTarget;
    desc.storageMode = MTLStorageModePrivate;
    outputTexture_ = [device_ newTextureWithDescriptor:desc];
  }

  if (!outputTexture_)
    return nil;

  id<MTLComputePipelineState> pipeline = nil;
  switch (config_.frameGenQuality) {
  case FrameGenQuality::Performance:
    pipeline = performancePipeline_;
    break;
  case FrameGenQuality::Balanced:
    pipeline = balancedPipeline_;
    break;
  case FrameGenQuality::Quality:
    pipeline = config_.useMotionVectors ? qualityPipeline_ : balancedPipeline_;
    break;
  }

  if (!pipeline)
    pipeline = balancedPipeline_;
  if (!pipeline)
    pipeline = performancePipeline_;
  if (!pipeline)
    return nil;

  id<MTLComputeCommandEncoder> encoder = [commandBuffer computeCommandEncoder];
  if (!encoder)
    return nil;

  if (config_.frameGenQuality == FrameGenQuality::Performance) {
    encoder.label = @"MGFG-1 Performance";
    [encoder setComputePipelineState:pipeline];
    [encoder setTexture:prevTexture atIndex:0];
    [encoder setTexture:currTexture atIndex:1];
    [encoder setTexture:outputTexture_ atIndex:2];
    [encoder setBytes:&t length:sizeof(float) atIndex:0];
  } else {
    encoder.label = @"MGFG-1 Balanced";
    [encoder setComputePipelineState:pipeline];
    [encoder setTexture:prevTexture atIndex:0];
    [encoder setTexture:currTexture atIndex:1];
    [encoder setTexture:outputTexture_ atIndex:2];

    struct BalancedParams {
      float t;
      uint32_t textureWidth;
      uint32_t textureHeight;
      float gradientThreshold;
      float padding;
    };

    BalancedParams params;
    params.t = t;
    params.textureWidth = static_cast<uint32_t>(width);
    params.textureHeight = static_cast<uint32_t>(height);
    params.gradientThreshold = 0.05f;
    params.padding = 0.0f;
    [encoder setBytes:&params length:sizeof(BalancedParams) atIndex:0];
  }

  MTLSize threadGroupSize = MTLSizeMake(16, 16, 1);
  MTLSize gridSize = MTLSizeMake((width + 15) / 16, (height + 15) / 16, 1);
  [encoder dispatchThreadgroups:gridSize threadsPerThreadgroup:threadGroupSize];
  [encoder endEncoding];

  return outputTexture_;
}

void Engine::computeMotionVectors(id<MTLTexture> prevTex,
                                  id<MTLTexture> currTex,
                                  id<MTLCommandBuffer> commandBuffer) {
  if (!motionEstimationPipeline_ || !motionVectorTexture_ ||
      !confidenceTexture_)
    return;
  if (!config_.useMotionVectors)
    return;

  id<MTLComputeCommandEncoder> encoder = [commandBuffer computeCommandEncoder];
  if (!encoder)
    return;

  MGFG1Params params;
  params.t = 0.5f;
  params.motionScale = config_.motionScale;
  params.occlusionThreshold = 0.15f;
  params.temporalWeight = config_.temporalBlend;
  params.textureSize =
      simd_make_uint2((uint32_t)currTex.width, (uint32_t)currTex.height);
  params.qualityMode = static_cast<uint32_t>(config_.frameGenQuality);

  [encoder setComputePipelineState:motionEstimationPipeline_];
  [encoder setTexture:prevTex atIndex:0];
  [encoder setTexture:currTex atIndex:1];
  [encoder setTexture:motionVectorTexture_ atIndex:2];
  [encoder setTexture:confidenceTexture_ atIndex:3];
  [encoder setBytes:&params length:sizeof(MGFG1Params) atIndex:0];

  MTLSize threadGroupSize = MTLSizeMake(16, 16, 1);
  MTLSize gridSize =
      MTLSizeMake((currTex.width + 15) / 16, (currTex.height + 15) / 16, 1);
  [encoder dispatchThreadgroups:gridSize threadsPerThreadgroup:threadGroupSize];

  [encoder endEncoding];

  if (motionRefinementPipeline_) {
    id<MTLComputeCommandEncoder> refineEncoder =
        [commandBuffer computeCommandEncoder];
    if (refineEncoder) {
      [refineEncoder setComputePipelineState:motionRefinementPipeline_];
      [refineEncoder setTexture:motionVectorTexture_ atIndex:0];
      [refineEncoder setTexture:confidenceTexture_ atIndex:1];
      [refineEncoder setTexture:motionVectorTexture_ atIndex:2];

      [refineEncoder dispatchThreadgroups:gridSize
                    threadsPerThreadgroup:threadGroupSize];
      [refineEncoder endEncoding];
    }
  }
}

void Engine::computeMotionVectorsPyramid(id<MTLTexture> prevTex,
                                         id<MTLTexture> currTex,
                                         id<MTLCommandBuffer> commandBuffer) {
  if (!config_.useMotionVectors) {
    return;
  }
  if (!pyramidDownsample2xPipeline_ || !pyramidMotionPipeline_ ||
      !upsampleMotionPipeline_) {
    computeMotionVectors(prevTex, currTex, commandBuffer);
    return;
  }

  ensurePyramidTextures(currTex.width, currTex.height);
  ensureMotionTextures(currTex.width, currTex.height);

  MTLSize threadGroupSize = MTLSizeMake(8, 8, 1);

  id<MTLComputeCommandEncoder> encoder = [commandBuffer computeCommandEncoder];
  if (!encoder)
    return;

  encoder.label = @"Pyramid Motion Estimation";

  [encoder setComputePipelineState:pyramidDownsample2xPipeline_];
  [encoder setTexture:prevTex atIndex:0];
  [encoder setTexture:pyramidPrevLevel1_ atIndex:1];
  MTLSize gridL1 = MTLSizeMake((pyramidPrevLevel1_.width + 7) / 8,
                               (pyramidPrevLevel1_.height + 7) / 8, 1);
  [encoder dispatchThreadgroups:gridL1 threadsPerThreadgroup:threadGroupSize];

  [encoder setTexture:pyramidPrevLevel1_ atIndex:0];
  [encoder setTexture:pyramidPrevLevel2_ atIndex:1];
  MTLSize gridL2 = MTLSizeMake((pyramidPrevLevel2_.width + 7) / 8,
                               (pyramidPrevLevel2_.height + 7) / 8, 1);
  [encoder dispatchThreadgroups:gridL2 threadsPerThreadgroup:threadGroupSize];

  [encoder setTexture:currTex atIndex:0];
  [encoder setTexture:pyramidCurrLevel1_ atIndex:1];
  [encoder dispatchThreadgroups:gridL1 threadsPerThreadgroup:threadGroupSize];

  [encoder setTexture:pyramidCurrLevel1_ atIndex:0];
  [encoder setTexture:pyramidCurrLevel2_ atIndex:1];
  [encoder dispatchThreadgroups:gridL2 threadsPerThreadgroup:threadGroupSize];

  [encoder setComputePipelineState:pyramidMotionPipeline_];
  uint32_t pyramidLevel = 2;
  [encoder setTexture:pyramidPrevLevel2_ atIndex:0];
  [encoder setTexture:pyramidCurrLevel2_ atIndex:1];
  [encoder setTexture:nil atIndex:2];
  [encoder setTexture:motionLevel2_ atIndex:3];
  [encoder setBytes:&pyramidLevel length:sizeof(uint32_t) atIndex:0];
  [encoder dispatchThreadgroups:gridL2 threadsPerThreadgroup:threadGroupSize];

  pyramidLevel = 1;
  [encoder setTexture:pyramidPrevLevel1_ atIndex:0];
  [encoder setTexture:pyramidCurrLevel1_ atIndex:1];
  [encoder setTexture:motionLevel2_ atIndex:2];
  [encoder setTexture:motionLevel1_ atIndex:3];
  [encoder setBytes:&pyramidLevel length:sizeof(uint32_t) atIndex:0];
  [encoder dispatchThreadgroups:gridL1 threadsPerThreadgroup:threadGroupSize];

  [encoder endEncoding];

  id<MTLComputeCommandEncoder> upsampleEncoder =
      [commandBuffer computeCommandEncoder];
  if (upsampleEncoder) {
    upsampleEncoder.label = @"Motion Upsample";
    [upsampleEncoder setComputePipelineState:upsampleMotionPipeline_];
    [upsampleEncoder setTexture:motionLevel1_ atIndex:0];
    [upsampleEncoder setTexture:motionVectorTexture_ atIndex:1];

    MTLSize gridFull =
        MTLSizeMake((currTex.width + 7) / 8, (currTex.height + 7) / 8, 1);
    [upsampleEncoder dispatchThreadgroups:gridFull
                    threadsPerThreadgroup:threadGroupSize];
    [upsampleEncoder endEncoding];
  }

  if (motionRefinementPipeline_) {
    id<MTLComputeCommandEncoder> refineEncoder =
        [commandBuffer computeCommandEncoder];
    if (refineEncoder) {
      refineEncoder.label = @"Motion Refinement";
      [refineEncoder setComputePipelineState:motionRefinementPipeline_];
      [refineEncoder setTexture:motionVectorTexture_ atIndex:0];
      [refineEncoder setTexture:confidenceTexture_ atIndex:1];
      [refineEncoder setTexture:motionVectorTexture_ atIndex:2];

      MTLSize gridFull =
          MTLSizeMake((currTex.width + 7) / 8, (currTex.height + 7) / 8, 1);
      [refineEncoder dispatchThreadgroups:gridFull
                    threadsPerThreadgroup:threadGroupSize];
      [refineEncoder endEncoding];
    }
  }
}

id<MTLTexture>
Engine::interpolateWithMotion(id<MTLTexture> prevTex, id<MTLTexture> currTex,
                              float t, id<MTLCommandBuffer> commandBuffer) {
  if (!qualityPipeline_ || !outputTexture_)
    return nil;

  id<MTLComputeCommandEncoder> encoder = [commandBuffer computeCommandEncoder];
  if (!encoder)
    return nil;

  MGFG1Params params;
  params.t = t;
  params.motionScale = config_.motionScale;
  params.occlusionThreshold = 0.15f;
  params.temporalWeight = config_.temporalBlend;
  params.textureSize =
      simd_make_uint2((uint32_t)currTex.width, (uint32_t)currTex.height);
  params.qualityMode = static_cast<uint32_t>(config_.frameGenQuality);

  [encoder setComputePipelineState:qualityPipeline_];
  [encoder setTexture:prevTex atIndex:0];
  [encoder setTexture:currTex atIndex:1];
  [encoder setTexture:motionVectorTexture_ atIndex:2];
  [encoder setTexture:confidenceTexture_ atIndex:3];
  [encoder setTexture:outputTexture_ atIndex:4];
  [encoder setBytes:&params length:sizeof(MGFG1Params) atIndex:0];

  MTLSize threadGroupSize = MTLSizeMake(16, 16, 1);
  MTLSize gridSize =
      MTLSizeMake((currTex.width + 15) / 16, (currTex.height + 15) / 16, 1);
  [encoder dispatchThreadgroups:gridSize threadsPerThreadgroup:threadGroupSize];
  [encoder endEncoding];

  return outputTexture_;
}

id<MTLTexture> Engine::interpolateSimple(id<MTLTexture> prevTex,
                                         id<MTLTexture> currTex, float t,
                                         id<MTLCommandBuffer> commandBuffer) {
  if (!performancePipeline_ || !outputTexture_)
    return nil;

  id<MTLComputeCommandEncoder> encoder = [commandBuffer computeCommandEncoder];
  if (!encoder)
    return nil;

  encoder.label = @"MGFG-1 Performance Interpolation";

  [encoder setComputePipelineState:performancePipeline_];
  [encoder setTexture:prevTex atIndex:0];
  [encoder setTexture:currTex atIndex:1];
  [encoder setTexture:outputTexture_ atIndex:2];
  [encoder setBytes:&t length:sizeof(float) atIndex:0];

  MTLSize threadGroupSize = MTLSizeMake(16, 16, 1);
  MTLSize gridSize =
      MTLSizeMake((currTex.width + 15) / 16, (currTex.height + 15) / 16, 1);
  [encoder dispatchThreadgroups:gridSize threadsPerThreadgroup:threadGroupSize];
  [encoder endEncoding];

  return outputTexture_;
}

id<MTLTexture> Engine::interpolateBalanced(id<MTLTexture> prevTex,
                                           id<MTLTexture> currTex, float t,
                                           id<MTLCommandBuffer> commandBuffer) {
  if (!balancedPipeline_ || !outputTexture_)
    return nil;

  id<MTLComputeCommandEncoder> encoder = [commandBuffer computeCommandEncoder];
  if (!encoder)
    return nil;

  encoder.label = @"MGFG-1 Balanced Interpolation";

  [encoder setComputePipelineState:balancedPipeline_];
  [encoder setTexture:prevTex atIndex:0];
  [encoder setTexture:currTex atIndex:1];
  [encoder setTexture:outputTexture_ atIndex:2];

  struct BalancedParamsHost {
    float t;
    uint32_t textureWidth;
    uint32_t textureHeight;
    float gradientThreshold;
    float padding;
  };

  BalancedParamsHost params;
  params.t = t;
  params.textureWidth = static_cast<uint32_t>(currTex.width);
  params.textureHeight = static_cast<uint32_t>(currTex.height);
  params.gradientThreshold = 0.05f;
  params.padding = 0.0f;

  [encoder setBytes:&params length:sizeof(BalancedParamsHost) atIndex:0];

  MTLSize threadGroupSize = MTLSizeMake(16, 16, 1);
  MTLSize gridSize =
      MTLSizeMake((currTex.width + 15) / 16, (currTex.height + 15) / 16, 1);
  [encoder dispatchThreadgroups:gridSize threadsPerThreadgroup:threadGroupSize];
  [encoder endEncoding];

  return outputTexture_;
}

id<MTLTexture> Engine::applyAntiAliasing(id<MTLTexture> inputTexture,
                                         id<MTLCommandBuffer> commandBuffer) {
  MTLSize threadGroupSize = MTLSizeMake(16, 16, 1);
  MTLSize gridSize = MTLSizeMake((inputTexture.width + 15) / 16,
                                 (inputTexture.height + 15) / 16, 1);

  AntiAliasParams params;
  params.threshold = config_.aaThreshold;
  params.depthThreshold = 0.1f;
  params.maxSearchSteps = 16;
  params.subpixelBlend = 0.75f;

  switch (config_.aaMode) {
  case AAMode::Off:
    return inputTexture;

  case AAMode::FXAA: {
    if (!fxaaPipeline_ || !tempTexture_)
      return inputTexture;

    id<MTLComputeCommandEncoder> encoder =
        [commandBuffer computeCommandEncoder];
    if (!encoder)
      return inputTexture;

    [encoder setComputePipelineState:fxaaPipeline_];
    [encoder setTexture:inputTexture atIndex:0];
    [encoder setTexture:tempTexture_ atIndex:1];
    [encoder setBytes:&params length:sizeof(AntiAliasParams) atIndex:0];
    [encoder dispatchThreadgroups:gridSize
            threadsPerThreadgroup:threadGroupSize];
    [encoder endEncoding];

    return tempTexture_;
  }

  case AAMode::SMAA: {
    if (!smaaEdgePipeline_ || !smaaWeightPipeline_ || !smaaBlendPipeline_)
      return inputTexture;
    if (!smaaEdgeTexture_ || !smaaWeightTexture_ || !outputTexture_)
      return inputTexture;

    {
      id<MTLComputeCommandEncoder> encoder =
          [commandBuffer computeCommandEncoder];
      if (!encoder)
        return inputTexture;

      [encoder setComputePipelineState:smaaEdgePipeline_];
      [encoder setTexture:inputTexture atIndex:0];
      [encoder setTexture:smaaEdgeTexture_ atIndex:1];
      [encoder setBytes:&params length:sizeof(AntiAliasParams) atIndex:0];
      [encoder dispatchThreadgroups:gridSize
              threadsPerThreadgroup:threadGroupSize];
      [encoder endEncoding];
    }

    {
      id<MTLComputeCommandEncoder> encoder =
          [commandBuffer computeCommandEncoder];
      if (!encoder)
        return inputTexture;

      [encoder setComputePipelineState:smaaWeightPipeline_];
      [encoder setTexture:smaaEdgeTexture_ atIndex:0];
      [encoder setTexture:smaaWeightTexture_ atIndex:1];
      [encoder setBytes:&params length:sizeof(AntiAliasParams) atIndex:0];
      [encoder dispatchThreadgroups:gridSize
              threadsPerThreadgroup:threadGroupSize];
      [encoder endEncoding];
    }

    {
      id<MTLComputeCommandEncoder> encoder =
          [commandBuffer computeCommandEncoder];
      if (!encoder)
        return inputTexture;

      [encoder setComputePipelineState:smaaBlendPipeline_];
      [encoder setTexture:inputTexture atIndex:0];
      [encoder setTexture:smaaWeightTexture_ atIndex:1];
      [encoder setTexture:outputTexture_ atIndex:2];
      [encoder dispatchThreadgroups:gridSize
              threadsPerThreadgroup:threadGroupSize];
      [encoder endEncoding];
    }

    return outputTexture_;
  }

  case AAMode::TAA: {
    return applyTAA(inputTexture, commandBuffer);
  }

  case AAMode::MSAA: {
    if (!msaaPipeline_ || !tempTexture_)
      return inputTexture;

    id<MTLComputeCommandEncoder> encoder =
        [commandBuffer computeCommandEncoder];
    if (!encoder)
      return inputTexture;

    [encoder setComputePipelineState:msaaPipeline_];
    [encoder setTexture:inputTexture atIndex:0];
    [encoder setTexture:tempTexture_ atIndex:1];
    [encoder setBytes:&params length:sizeof(AntiAliasParams) atIndex:0];
    [encoder dispatchThreadgroups:gridSize
            threadsPerThreadgroup:threadGroupSize];
    [encoder endEncoding];

    return tempTexture_;
  }

  default:
    return inputTexture;
  }
}

id<MTLTexture> Engine::applyTAA(id<MTLTexture> inputTexture,
                                id<MTLCommandBuffer> commandBuffer) {
  if (!temporalPipeline_ || !historyTexture_ || !taaOutputTexture_)
    return inputTexture;

  if (!hasValidHistory_) {
    copyTextureToHistory(inputTexture, commandBuffer);
    hasValidHistory_ = true;
    return inputTexture;
  }

  if (previousFrameTexture_ && motionEstimationPipeline_ &&
      config_.useMotionVectors) {
    ensureMotionTextures(inputTexture.width, inputTexture.height);
    computeMotionVectors(previousFrameTexture_, inputTexture, commandBuffer);
  }

  id<MTLComputeCommandEncoder> encoder = [commandBuffer computeCommandEncoder];
  if (!encoder)
    return inputTexture;

  float blendFactor = config_.temporalBlend;

  [encoder setComputePipelineState:temporalPipeline_];
  [encoder setTexture:inputTexture atIndex:0];
  [encoder setTexture:historyTexture_ atIndex:1];
  [encoder setTexture:taaOutputTexture_ atIndex:2];
  [encoder setBytes:&blendFactor length:sizeof(float) atIndex:0];

  MTLSize threadGroupSize = MTLSizeMake(16, 16, 1);
  MTLSize gridSize = MTLSizeMake((inputTexture.width + 15) / 16,
                                 (inputTexture.height + 15) / 16, 1);
  [encoder dispatchThreadgroups:gridSize threadsPerThreadgroup:threadGroupSize];
  [encoder endEncoding];

  copyTextureToHistory(taaOutputTexture_, commandBuffer);

  previousFrameTexture_ = inputTexture;

  return taaOutputTexture_;
}

void Engine::copyTextureToHistory(id<MTLTexture> source,
                                  id<MTLCommandBuffer> commandBuffer) {
  if (!copyPipeline_ || !historyTexture_)
    return;

  id<MTLComputeCommandEncoder> encoder = [commandBuffer computeCommandEncoder];
  if (!encoder)
    return;

  [encoder setComputePipelineState:copyPipeline_];
  [encoder setTexture:source atIndex:0];
  [encoder setTexture:historyTexture_ atIndex:1];

  MTLSize threadGroupSize = MTLSizeMake(16, 16, 1);
  MTLSize gridSize =
      MTLSizeMake((source.width + 15) / 16, (source.height + 15) / 16, 1);
  [encoder dispatchThreadgroups:gridSize threadsPerThreadgroup:threadGroupSize];
  [encoder endEncoding];
}

id<MTLTexture> Engine::applySharpening(id<MTLTexture> inputTexture,
                                       id<MTLCommandBuffer> commandBuffer) {
  if (!casPipeline_ || !outputTexture_)
    return nil;

  id<MTLComputeCommandEncoder> encoder = [commandBuffer computeCommandEncoder];
  if (!encoder)
    return nil;

  SharpenParams params;
  params.sharpness = config_.sharpness;
  params.radius = 1.0f;

  [encoder setComputePipelineState:casPipeline_];
  [encoder setTexture:inputTexture atIndex:0];
  [encoder setTexture:outputTexture_ atIndex:1];
  [encoder setBytes:&params length:sizeof(SharpenParams) atIndex:0];

  MTLSize threadGroupSize = MTLSizeMake(16, 16, 1);
  MTLSize gridSize = MTLSizeMake((inputTexture.width + 15) / 16,
                                 (inputTexture.height + 15) / 16, 1);
  [encoder dispatchThreadgroups:gridSize threadsPerThreadgroup:threadGroupSize];
  [encoder endEncoding];

  return outputTexture_;
}

extern "C" {

Engine *Engine_Create(void *device, void *queue) {
  if (!device || !queue) {
    return nullptr;
  }
  id<MTLDevice> mtlDevice = (__bridge id<MTLDevice>)device;
  id<MTLCommandQueue> mtlQueue = (__bridge id<MTLCommandQueue>)queue;
  if (!mtlDevice || !mtlQueue) {
    return nullptr;
  }
  return new Engine(mtlDevice, mtlQueue);
}

void Engine_Destroy(Engine *engine) {
  if (engine) {
    delete engine;
  }
}

void Engine_SetConfig(Engine *engine, void *configPtr) {
  if (!engine || !configPtr)
    return;

  struct DirectEngineConfigInternal {
    int upscaleMode;
    int renderScale;
    float scaleFactor;

    int frameGenMode;
    int frameGenType;
    int frameGenQuality;
    int frameGenMultiplier;
    int adaptiveTargetFPS;

    int aaMode;
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
  };

  DirectEngineConfigInternal *cfg =
      static_cast<DirectEngineConfigInternal *>(configPtr);

  EngineConfig engineConfig;
  engineConfig.targetFPS = static_cast<uint32_t>(cfg->targetFPS);
  engineConfig.frameGenEnabled = (cfg->frameGenMode > 0);
  engineConfig.frameGenMultiplier =
      static_cast<uint32_t>(cfg->frameGenMultiplier);
  engineConfig.upscaleFactor = cfg->scaleFactor;
  engineConfig.sharpness = cfg->sharpness;
  engineConfig.temporalBlend = cfg->temporalBlend;
  engineConfig.adaptiveQuality = cfg->adaptiveSync;
  engineConfig.baseWidth =
      cfg->baseWidth > 0 ? static_cast<uint32_t>(cfg->baseWidth) : 0;
  engineConfig.baseHeight =
      cfg->baseHeight > 0 ? static_cast<uint32_t>(cfg->baseHeight) : 0;
  engineConfig.outputWidth =
      cfg->outputWidth > 0 ? static_cast<uint32_t>(cfg->outputWidth) : 0;
  engineConfig.outputHeight =
      cfg->outputHeight > 0 ? static_cast<uint32_t>(cfg->outputHeight) : 0;
  engineConfig.useMotionVectors = cfg->useMotionVectors;
  engineConfig.motionScale = cfg->motionScale;

  switch (cfg->frameGenQuality) {
  case 0:
    engineConfig.frameGenQuality = FrameGenQuality::Performance;
    break;
  case 1:
    engineConfig.frameGenQuality = FrameGenQuality::Balanced;
    break;
  case 2:
    engineConfig.frameGenQuality = FrameGenQuality::Quality;
    break;
  default:
    engineConfig.frameGenQuality = FrameGenQuality::Balanced;
  }

  switch (cfg->aaMode) {
  case 0:
    engineConfig.aaMode = AAMode::Off;
    break;
  case 1:
    engineConfig.aaMode = AAMode::FXAA;
    break;
  case 2:
    engineConfig.aaMode = AAMode::SMAA;
    break;
  case 3:
    engineConfig.aaMode = AAMode::MSAA;
    break;
  case 4:
    engineConfig.aaMode = AAMode::TAA;
    break;
  default:
    engineConfig.aaMode = AAMode::Off;
  }

  switch (cfg->upscaleMode) {
  case 0:
    engineConfig.upscaleMode = UpscaleMode::Off;
    engineConfig.sharpness = cfg->sharpness;
    break;
  case 1:
    engineConfig.upscaleMode = UpscaleMode::MetalFX;
    engineConfig.sharpness = std::max(cfg->sharpness, 0.4f);
    break;
  case 2:
    engineConfig.upscaleMode = UpscaleMode::Bilinear;
    engineConfig.sharpness = std::max(cfg->sharpness, 0.25f);
    break;
  case 3:
    engineConfig.upscaleMode = UpscaleMode::CAS;
    engineConfig.sharpness = std::max(cfg->sharpness, 0.65f);
    break;
  default:
    engineConfig.upscaleMode = UpscaleMode::Off;
    engineConfig.sharpness = cfg->sharpness;
  }

  switch (cfg->renderScale) {
  case 0:
    engineConfig.renderScaleFactor = 1.0f;
    break;
  case 1:
    engineConfig.renderScaleFactor = 0.75f;
    break;
  case 2:
    engineConfig.renderScaleFactor = 0.67f;
    break;
  case 3:
    engineConfig.renderScaleFactor = 0.50f;
    break;
  case 4:
    engineConfig.renderScaleFactor = 0.33f;
    break;
  default:
    engineConfig.renderScaleFactor = 1.0f;
    break;
  }

  engine->setConfig(engineConfig);
}

void *Engine_ProcessFrame(Engine *engine, void *inputTexture, void *cmdBuf) {
  if (!engine)
    return nullptr;
  id<MTLTexture> result =
      engine->processFrame((__bridge id<MTLTexture>)inputTexture,
                           (__bridge id<MTLCommandBuffer>)cmdBuf);
  return (__bridge void *)result;
}

void *Engine_GenerateInterpolatedFrame(Engine *engine, void *prevTex,
                                       void *currTex, float t, void *cmdBuf) {
  if (!engine)
    return nullptr;
  id<MTLTexture> result = engine->generateInterpolatedFrame(
      (__bridge id<MTLTexture>)prevTex, (__bridge id<MTLTexture>)currTex, t,
      (__bridge id<MTLCommandBuffer>)cmdBuf);
  return (__bridge void *)result;
}

void Engine_PushFrame(Engine *engine, void *texture, double timestamp) {
  if (!engine)
    return;
  engine->pushFrame((__bridge id<MTLTexture>)texture, timestamp);
}

uint64_t Engine_GetFrameIndex(Engine *engine) {
  if (!engine)
    return 0;
  return engine->getFrameIndex();
}
}
