import Foundation
import os
@preconcurrency import Metal
@preconcurrency import MetalFX

/// The device, its queue, and every pipeline the engine runs. Built once; a pipeline that
/// cannot be built is a startup failure with the function's name on it, not an error that
/// surfaces later from whichever frame first needed it.
final class GPUContext: @unchecked Sendable {

    struct Pipelines {
        let sharpen: MTLComputePipelineState
        let fxaa: MTLComputePipelineState
        let smaaEdges: MTLComputePipelineState
        let smaaWeights: MTLComputePipelineState
        let smaaBlend: MTLComputePipelineState
        let luma: MTLComputePipelineState
        let copyMotion: MTLComputePipelineState
        let despeckle: MTLComputePipelineState
        let globalMotion: MTLComputePipelineState
        let stabilityBlend: MTLComputePipelineState
        let stabilityBlendFromYUV: MTLComputePipelineState
        let stabilityBlendFromYUVResampled: MTLComputePipelineState
        let convertTo420: MTLComputePipelineState
        let convertTo420Resampled: MTLComputePipelineState
        let present: MTLRenderPipelineState
    }

    let device: MTLDevice
    let queue: MTLCommandQueue
    let pipelines: Pipelines

    private let busy = OSAllocatedUnfairLock(initialState: 0.0)

    private init(device: MTLDevice, queue: MTLCommandQueue, pipelines: Pipelines) {
        self.device = device
        self.queue = queue
        self.pipelines = pipelines
    }

    /// `libraryURL` names a compiled shader library to use instead of the app bundle's default one,
    /// which is how the pipelines are built outside an app — by the benchmark and the tests.
    static func make(libraryURL: URL? = nil) -> Result<GPUContext, MGError> {
        guard let device = MTLCreateSystemDefaultDevice() else { return .failure(.metalDeviceUnavailable) }
        guard let queue = device.makeCommandQueue() else { return .failure(.commandQueueUnavailable) }
        let library: MTLLibrary
        do {
            guard let loaded = try libraryURL.map({ try device.makeLibrary(URL: $0) }) ?? device.makeDefaultLibrary() else {
                return .failure(.pipelineSetupFailed())
            }
            library = loaded
        } catch {
            return .failure(.pipelineSetupFailed("\(error)"))
        }

        do {
            func compute(_ name: String) throws -> MTLComputePipelineState {
                guard let function = library.makeFunction(name: name) else {
                    throw MGError.pipelineSetupFailed("missing shader function \(name)")
                }
                return try device.makeComputePipelineState(function: function)
            }

            guard let vertex = library.makeFunction(name: "present_vertex"),
                  let fragment = library.makeFunction(name: "present_fragment") else {
                throw MGError.pipelineSetupFailed("missing present shader")
            }
            let presentDescriptor = MTLRenderPipelineDescriptor()
            presentDescriptor.vertexFunction = vertex
            presentDescriptor.fragmentFunction = fragment
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
                present: try device.makeRenderPipelineState(descriptor: presentDescriptor))
            return .success(GPUContext(device: device, queue: queue, pipelines: pipelines))
        } catch let error as MGError {
            return .failure(error)
        } catch {
            return .failure(.pipelineSetupFailed("\(error)"))
        }
    }

    /// What the overlay's layer and the pipeline's textures are built in.
    static let drawablePixelFormat: MTLPixelFormat = .bgra8Unorm

    // MARK: - Command buffers

    /// A command buffer on the pipeline's queue that adds its own GPU time to the running total when it
    /// finishes. Every stage — capture, motion, interpolation, presentation — goes through this, so the
    /// total is the whole pipeline's share of the GPU and not just the last frame's.
    func makeCommandBuffer(_ label: String) -> MTLCommandBuffer? {
        guard let buffer = queue.makeCommandBuffer() else { return nil }
        buffer.label = label
        buffer.addCompletedHandler { [busy] finished in
            let elapsed = finished.gpuEndTime - finished.gpuStartTime
            if elapsed > 0 { busy.withLock { $0 += elapsed } }
        }
        return buffer
    }

    /// Seconds of GPU time the pipeline has used since it started.
    var busyTime: Double { busy.withLock { $0 } }

    // MARK: - Textures

    /// Returns `texture` if it already has this shape, otherwise a fresh private texture of
    /// that shape. Validity is checked against the texture's own dimensions rather than a
    /// shadow copy, so there is no second source of truth to drift out of sync.
    func ensureTexture(_ texture: inout MTLTexture?, width: Int, height: Int,
                       pixelFormat: MTLPixelFormat = .bgra8Unorm,
                       usage: MTLTextureUsage = [.shaderRead, .shaderWrite]) -> MTLTexture? {
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
    func ensureSpatialScaler(_ cache: inout MTLFXSpatialScaler?,
                             inputWidth: Int, inputHeight: Int,
                             outputWidth: Int, outputHeight: Int) -> MTLFXSpatialScaler? {
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
        cache = descriptor.makeSpatialScaler(device: device)
        return cache
    }

    /// Threadgroup shape for a per-pixel kernel: as wide as the hardware's SIMD width, as tall
    /// as the pipeline allows. The shape was swept from 8x8 to 128x2 and moves the time by no
    /// more than measurement noise, so the hardware's own preference stands.
    func dispatch(_ pipeline: MTLComputePipelineState, on encoder: MTLComputeCommandEncoder,
                  width: Int, height: Int) {
        let w = pipeline.threadExecutionWidth
        let h = pipeline.maxTotalThreadsPerThreadgroup / w
        encoder.dispatchThreadgroups(MTLSize(width: (width + w - 1) / w, height: (height + h - 1) / h, depth: 1),
                                     threadsPerThreadgroup: MTLSize(width: w, height: h, depth: 1))
    }
}
