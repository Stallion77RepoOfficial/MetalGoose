import Foundation
import os
@preconcurrency import Metal
@preconcurrency import MetalFX

/// The device, its Metal 4 queues, and every pipeline the engine runs. Built once; a pipeline that
/// cannot be built is a startup failure with the function's name on it, not an error that
/// surfaces later from whichever frame first needed it.
final class GPUContext: @unchecked Sendable {

    struct Pipelines {
        let sharpen: any MTLComputePipelineState
        let fxaa: any MTLComputePipelineState
        let smaaEdges: any MTLComputePipelineState
        let smaaWeights: any MTLComputePipelineState
        let smaaBlend: any MTLComputePipelineState
        let luma: any MTLComputePipelineState
        let copyMotion: any MTLComputePipelineState
        let despeckle: any MTLComputePipelineState
        let globalMotion: any MTLComputePipelineState
        let stabilityBlend: any MTLComputePipelineState
        let stabilityBlendFromYUV: any MTLComputePipelineState
        let stabilityBlendFromYUVResampled: any MTLComputePipelineState
        let convertTo420: any MTLComputePipelineState
        let convertTo420Resampled: any MTLComputePipelineState
        let present: any MTLRenderPipelineState
    }

    let device: any MTLDevice
    /// Builds the pipelines and the MetalFX effects, which Metal 4 makes for its own command buffers.
    let compiler: any MTL4Compiler
    let pipelines: Pipelines
    let reports: GPUReports
    var errors: ErrorLog { reports.errors }

    /// The capture path: conversions, sharpening and anti-aliasing, the motion field, the Neural Engine's input.
    let capture: GPULane
    /// MetalFX interpolation, which can take a large part of a capture interval and holds up nothing else.
    let interpolation: GPULane
    /// Presentation: the blend of a generated image with its captures, the upscale, the drawable.
    let render: GPULane

    var failureGeneration: Int { reports.failureGeneration }

    /// Seconds of GPU time the pipeline has used since it started.
    var busyTime: Double { reports.busyTime }

    private init(device: any MTLDevice, compiler: any MTL4Compiler, pipelines: Pipelines, reports: GPUReports,
                 capture: GPULane, interpolation: GPULane, render: GPULane) {
        self.device = device
        self.compiler = compiler
        self.pipelines = pipelines
        self.reports = reports
        self.capture = capture
        self.interpolation = interpolation
        self.render = render
    }

    static func make() -> Result<GPUContext, MGError> {
        guard let device = MTLCreateSystemDefaultDevice() else { return .failure(.metalDeviceUnavailable) }
        let reports = GPUReports()
        guard let capture = GPULane(device: device, label: "capture", reports: reports),
              let interpolation = GPULane(device: device, label: "interpolation", reports: reports),
              let render = GPULane(device: device, label: "render", reports: reports) else {
            return .failure(.commandQueueUnavailable)
        }
        guard let library = device.makeDefaultLibrary() else { return .failure(.pipelineSetupFailed()) }

        do {
            let compiler = try device.makeCompiler(descriptor: MTL4CompilerDescriptor())
            func function(_ name: String) throws -> MTL4LibraryFunctionDescriptor {
                guard library.functionNames.contains(name) else {
                    throw MGError.pipelineSetupFailed("missing shader function \(name)")
                }
                let function = MTL4LibraryFunctionDescriptor()
                function.name = name
                function.library = library
                return function
            }
            func compute(_ name: String) throws -> any MTLComputePipelineState {
                let descriptor = MTL4ComputePipelineDescriptor()
                descriptor.computeFunctionDescriptor = try function(name)
                return try compiler.makeComputePipelineState(descriptor: descriptor)
            }

            let presentDescriptor = MTL4RenderPipelineDescriptor()
            presentDescriptor.vertexFunctionDescriptor = try function("present_vertex")
            presentDescriptor.fragmentFunctionDescriptor = try function("present_fragment")
            presentDescriptor.colorAttachments[0].pixelFormat = Self.drawablePixelFormat

            let pipelines = Pipelines(
                sharpen: try compute("contrastAdaptiveSharpening"),
                fxaa: try compute("fxaa"),
                smaaEdges: try compute("smaaEdgeDetection"),
                smaaWeights: try compute("smaaBlendingWeights"),
                smaaBlend: try compute("smaaBlend"),
                luma: try compute("bgraToLuma"),
                copyMotion: try compute("copyMotionField"),
                despeckle: try compute("despeckleMotion"),
                globalMotion: try compute("globalMotion"),
                stabilityBlend: try compute("blendTowardCaptures"),
                stabilityBlendFromYUV: try compute("blendTowardCapturesFromYUV"),
                stabilityBlendFromYUVResampled: try compute("blendTowardCapturesFromYUVResampled"),
                convertTo420: try compute("bgraTo420"),
                convertTo420Resampled: try compute("bgraTo420Resampled"),
                present: try compiler.makeRenderPipelineState(descriptor: presentDescriptor))
            return .success(GPUContext(device: device, compiler: compiler, pipelines: pipelines, reports: reports,
                                       capture: capture, interpolation: interpolation, render: render))
        } catch let error as MGError {
            return .failure(error)
        } catch {
            return .failure(.pipelineSetupFailed("\(error)"))
        }
    }

    /// What the overlay's layer and the pipeline's textures are built in.
    static let drawablePixelFormat: MTLPixelFormat = .bgra8Unorm

    // MARK: - Textures

    /// Returns `texture` if it already has this shape, otherwise a fresh private texture of
    /// that shape. Validity is checked against the texture's own dimensions rather than a
    /// shadow copy, so there is no second source of truth to drift out of sync.
    func ensureTexture(_ texture: inout (any MTLTexture)?, width: Int, height: Int,
                       pixelFormat: MTLPixelFormat = .bgra8Unorm,
                       usage: MTLTextureUsage = [.shaderRead, .shaderWrite]) -> (any MTLTexture)? {
        if let texture, texture.width == width, texture.height == height, texture.pixelFormat == pixelFormat {
            return texture
        }
        let descriptor = MTLTextureDescriptor.texture2DDescriptor(pixelFormat: pixelFormat,
                                                                  width: width, height: height, mipmapped: false)
        descriptor.usage = usage
        descriptor.storageMode = .private
        texture = device.makeTexture(descriptor: descriptor)
        return texture
    }

    /// A MetalFX spatial scaler for this exact shape, built once and reused. The capture path and
    /// the render path each keep their own cache because they run on different threads, but the
    /// construction is identical. Validity is checked against the scaler's own reported
    /// dimensions, so there is no shadow copy to drift out of sync with it.
    func ensureSpatialScaler(_ cache: inout (any MTL4FXSpatialScaler)?,
                             inputWidth: Int, inputHeight: Int,
                             outputWidth: Int, outputHeight: Int) -> (any MTL4FXSpatialScaler)? {
        if let scaler = cache,
           scaler.inputWidth == inputWidth, scaler.inputHeight == inputHeight,
           scaler.outputWidth == outputWidth, scaler.outputHeight == outputHeight {
            return scaler
        }
        let descriptor = MTLFXSpatialScalerDescriptor()
        descriptor.inputWidth = inputWidth
        descriptor.inputHeight = inputHeight
        descriptor.outputWidth = outputWidth
        descriptor.outputHeight = outputHeight
        descriptor.colorTextureFormat = .bgra8Unorm
        descriptor.outputTextureFormat = .bgra8Unorm
        // The captures are sRGB-encoded 8-bit. Perceptual measured 0.11 to 0.54 dB closer to the original than linear.
        descriptor.colorProcessingMode = .perceptual
        cache = descriptor.makeSpatialScaler(device: device, compiler: compiler)
        return cache
    }

    /// Encodes a MetalFX spatial upscale of `input` into `output` on `command`, between its passes.
    func encodeUpscale(_ scaler: any MTL4FXSpatialScaler, from input: any MTLTexture, to output: any MTLTexture,
                       on command: GPUCommand) {
        command.use(input)
        command.use(output)
        scaler.colorTexture = input
        scaler.outputTexture = output
        scaler.encode(commandBuffer: command.commandBuffer)
    }

    /// Threadgroup shape for a per-pixel kernel: as wide as the hardware's SIMD width, as tall
    /// as the pipeline allows. The shape was swept from 8x8 to 128x2 and moves the time by no
    /// more than measurement noise, so the hardware's own preference stands.
    func dispatch(_ pipeline: any MTLComputePipelineState, on pass: ComputePass, width: Int, height: Int) {
        let w = pipeline.threadExecutionWidth
        let h = pipeline.maxTotalThreadsPerThreadgroup / w
        pass.dispatchThreadgroups(MTLSize(width: (width + w - 1) / w, height: (height + h - 1) / h, depth: 1),
                                  threadsPerThreadgroup: MTLSize(width: w, height: h, depth: 1))
    }
}
