import Foundation
import AppKit
@preconcurrency import Metal
@preconcurrency import MetalKit
@preconcurrency import MetalFX
@preconcurrency import IOSurface
import QuartzCore
import os
@preconcurrency import CoreVideo
@preconcurrency import VideoToolbox

struct CursorUniforms {
    var center: SIMD2<Float>
    var size: SIMD2<Float>
}

struct PipelineStats: @unchecked Sendable {
    var captureFPS: Float = 0
    var outputFPS: Float = 0
    /// Distinct images the generator produced per second — cache hits excluded.
    /// A generated image the panel shows twice is one image, so this counts the
    /// pair (or warp phase) that was actually synthesised, not the presents that
    /// carried it. `interpolatedFrameCount` is the present-side figure.
    var generatedFPS: Float = 0
    var frameTime: Float = 0
    var gpuTime: Float = 0
    var captureGPUTime: Float = 0
    var captureLatency: Float = 0
    var presentLatency: Float = 0
    var endToEndLatency: Float = 0
    var avgFrameTime: Float = 0
    var framePacingScore: Float = 100
    var frameCount: UInt64 = 0
    var outputFrameCount: UInt64 = 0
    var droppedFrames: UInt64 = 0
    /// Presents whose image came from the generator, however many of them shared
    /// one synthesised image. Together with `passthroughFrameCount` this adds up
    /// to `outputFrameCount`.
    var interpolatedFrameCount: UInt64 = 0
    var passthroughFrameCount: UInt64 = 0
    /// Distinct synthesised images, the cumulative form of `generatedFPS`.
    var generatedFrameCount: UInt64 = 0
    var gpuMemoryUsed: UInt64 = 0
    var gpuMemoryTotal: UInt64 = 0
    var processMemoryUsed: UInt64 = 0
    var cpuUsage: Float = 0
    var outputResolution: CGSize = .zero
    var screenRefreshRate: Int = 0
    var isProMotion: Bool = false
    var targetOutputFPS: Int = 0

}

final class GooseEngine: NSObject, MTKViewDelegate, @unchecked Sendable {

    private var _stats = PipelineStats()
    private let statsLock = OSAllocatedUnfairLock()
    private let renderStateLock = OSAllocatedUnfairLock()
    var stats: PipelineStats {
        statsLock.lock()
        defer { statsLock.unlock() }
        return _stats
    }

    private let errorLock = OSAllocatedUnfairLock()
    private var reportedErrors: Set<String> = []
    private var pendingErrors: [String] = []

    nonisolated(unsafe) static private(set) var lastInitError: String?

    private func reportError(_ message: String) {
        errorLock.lock()
        defer { errorLock.unlock() }
        guard reportedErrors.insert(message).inserted else { return }
        pendingErrors.append(message)
    }

    func consumePendingError() -> String? {
        errorLock.lock()
        defer { errorLock.unlock() }
        return pendingErrors.isEmpty ? nil : pendingErrors.removeFirst()
    }

    private func resetErrorReporting() {
        errorLock.lock()
        reportedErrors.removeAll()
        pendingErrors.removeAll()
        errorLock.unlock()
    }

    private let processingQueue = DispatchQueue(label: "com.metalgoose.processing", qos: .userInteractive)

    private static let maxInFlight = 3

    /// The engine's only tuned number. Capture rate, output interval, the frame
    /// clock and the rate-preference hysteresis are all first-order filters, and
    /// each derives its own coefficient from this window and the rate it is
    /// actually running at — so none of them carries a constant that assumes a
    /// particular display or capture speed.
    private static let measurementWindow: Double = 0.5

    /// Never replaced. A completion handler resolves this property when the GPU
    /// finishes, so swapping the object mid-flight would signal a semaphore the
    /// frame never waited on. Shallower pipelines park permits instead.
    private let inFlightSemaphore = DispatchSemaphore(value: GooseEngine.maxInFlight)

    /// Only touched from `processingQueue`, so it needs no lock.
    private var parkedPermits = 0

    var deviceName: String { device.name }

    private let device: MTLDevice
    private let commandQueue: MTLCommandQueue
    
    private var spatialScaler: MTLFXSpatialScaler?

    /// Capture-path scaler, separate from the render-path one because the two
    /// run on different threads. Brings a render-scaled capture back to the
    /// window's native size before the frame enters the history ring.
    private var captureScaler: MTLFXSpatialScaler?
    private var captureUpscaledTexture: MTLTexture?
    
    private var casPipeline: MTLComputePipelineState?
    private var fxaaPipeline: MTLComputePipelineState?
    private var smaaEdgePipeline: MTLComputePipelineState?
    private var smaaWeightPipeline: MTLComputePipelineState?
    private var smaaBlendPipeline: MTLComputePipelineState?
    private var copyPipeline: MTLComputePipelineState?
    private var lumaPipeline: MTLComputePipelineState?
    private var extrapolatePipeline: MTLComputePipelineState?

    /// Render-path only, exactly like the spatial scaler. The processing queue
    /// asks for a teardown through `frameGenNeedsTeardown` instead of touching
    /// these directly.
    private var frameInterpolator: MTLFXFrameInterpolator?
    private var frameInterpolatorNeedsHistoryReset: Bool = true
    private var frameGenNeedsTeardown: Bool = false

    /// MetalFX rejects a nil depth texture — its validation layer asserts with
    /// "Input content width exceed input texture dimension". Captured content is
    /// a flat 2D plane, so a constant far-plane depth is the honest answer rather
    /// than a workaround.
    private var flatDepthTexture: MTLTexture?

    /// Capture-path only: motion estimation runs where the frames are produced,
    /// so each ring entry carries the motion that belongs to it.
    private lazy var motionEstimator = MotionEstimator(device: device)
    private var extrapolatedTexture: MTLTexture?

    private var renderPipeline: MTLRenderPipelineState?
    private var cursorPipeline: MTLRenderPipelineState?
    private var cursorTexture: MTLTexture?
    private var cursorTextureSize: CGSize = .zero
    private weak var mtkView: MTKView?
    
    /// Render-path only: the spatial upscale now runs once per presented frame,
    /// so these are touched exclusively from `renderFrame` on the main thread.
    private var upscaledTexture: MTLTexture?
    private var upscaleCacheKey: CFTimeInterval = -1

    private var casTexture: MTLTexture?
    private var smaaEdgeTexture: MTLTexture?
    private var smaaWeightTexture: MTLTexture?

    struct FrameHistory {
        let texture: MTLTexture
        let timestamp: CFTimeInterval
        let isSceneCut: Bool
        /// Backward motion against the previous frame, in pixels, one vector per
        /// 16x16 block. Only produced in extrapolation mode.
        let motion: MTLTexture?
    }

    /// Wraps VTMotionEstimationSession, which runs on the media engine rather
    /// than the GPU. It takes single-component luma, so each frame is converted
    /// first, and it returns backward vectors in pixels at one vector per block.
    private final class MotionEstimator {
        private let device: MTLDevice
        private var session: __VTMotionEstimationSession?
        private var size: (width: Int, height: Int) = (0, 0)
        private var lumaBuffers: [CVPixelBuffer] = []
        private var lumaTextures: [MTLTexture] = []
        private var slot = 0
        private var hasReference = false

        init(device: MTLDevice) { self.device = device }

        func reset() {
            if let session { __VTMotionEstimationSessionInvalidate(session) }
            session = nil
            size = (0, 0)
            lumaBuffers.removeAll()
            lumaTextures.removeAll()
            slot = 0
            hasReference = false
        }

        /// Default block size and a single search pass: a 4x4 grid measures 12x
        /// slower, which no real-time budget can absorb.
        private func ensureSession(width: Int, height: Int) -> Bool {
            if session != nil, size == (width, height) { return true }
            reset()

            let options: [String: Any] = [
                kVTMotionEstimationSessionCreationOption_UseMultiPassSearch as String: false as CFBoolean
            ]
            var created: __VTMotionEstimationSession?
            guard __VTMotionEstimationSessionCreate(kCFAllocatorDefault, options as CFDictionary,
                                                    UInt32(width), UInt32(height), &created) == noErr,
                  let created else { return false }

            var attributes: CFDictionary?
            __VTMotionEstimationSessionCopySourcePixelBufferAttributes(created, &attributes)
            var descriptor = (attributes as? [String: Any]) ?? [:]
            descriptor[kCVPixelBufferIOSurfacePropertiesKey as String] = [:] as CFDictionary

            for _ in 0..<2 {
                var buffer: CVPixelBuffer?
                guard CVPixelBufferCreate(kCFAllocatorDefault, width, height,
                                          kCVPixelFormatType_OneComponent8,
                                          descriptor as CFDictionary, &buffer) == kCVReturnSuccess,
                      let buffer,
                      let surface = CVPixelBufferGetIOSurface(buffer)?.takeUnretainedValue() else { return false }

                let texDescriptor = MTLTextureDescriptor.texture2DDescriptor(
                    pixelFormat: .r8Unorm, width: width, height: height, mipmapped: false)
                texDescriptor.usage = [.shaderRead, .shaderWrite]
                guard let texture = device.makeTexture(descriptor: texDescriptor, iosurface: surface, plane: 0) else {
                    return false
                }
                lumaBuffers.append(buffer)
                lumaTextures.append(texture)
            }

            session = created
            size = (width, height)
            return true
        }

        /// Destination for this frame's luma conversion.
        func prepare(width: Int, height: Int) -> MTLTexture? {
            guard ensureSession(width: width, height: height) else { return nil }
            return lumaTextures[slot]
        }

        /// Must be called only once the luma write has actually landed.
        func estimate() -> CVPixelBuffer? {
            guard let session, lumaBuffers.count == 2 else { return nil }
            let current = lumaBuffers[slot]
            let reference = lumaBuffers[1 - slot]
            slot = 1 - slot

            guard hasReference else { hasReference = true; return nil }

            let semaphore = DispatchSemaphore(value: 0)
            var result: CVPixelBuffer?
            let status = __VTMotionEstimationSessionEstimateMotionVectors(
                session, reference, current, [], nil
            ) { status, _, _, vectors in
                if status == noErr { result = vectors }
                semaphore.signal()
            }
            guard status == noErr else { return nil }
            semaphore.wait()
            return result
        }

        func texture(for vectors: CVPixelBuffer) -> MTLTexture? {
            guard let surface = CVPixelBufferGetIOSurface(vectors)?.takeUnretainedValue() else { return nil }
            let descriptor = MTLTextureDescriptor.texture2DDescriptor(
                pixelFormat: .rg16Float,
                width: CVPixelBufferGetWidth(vectors),
                height: CVPixelBufferGetHeight(vectors),
                mipmapped: false)
            descriptor.usage = [.shaderRead]
            return device.makeTexture(descriptor: descriptor, iosurface: surface, plane: 0)
        }
    }
    
    private final class FrameRingBuffer: @unchecked Sendable {
        private var buffer: [FrameHistory] = []
        /// The render clock samples at most one capture interval behind the
        /// newest frame, so two entries always bracket it; the rest is headroom
        /// for captures still in flight behind the one being read.
        static let capacity = GooseEngine.maxInFlight + 1
        private let lock = NSLock()
        
        func push(_ frame: FrameHistory) {
            lock.lock()
            defer { lock.unlock() }
            buffer.append(frame)
            if buffer.count > Self.capacity {
                buffer.removeFirst()
            }
        }
        
        func getFramesForTime(_ targetTime: CFTimeInterval) -> (prev: FrameHistory, next: FrameHistory)? {
            lock.lock()
            defer { lock.unlock() }
            
            guard buffer.count >= 2 else { return nil }
            
            for i in 0..<(buffer.count - 1) {
                let prev = buffer[i]
                let next = buffer[i+1]
                if targetTime >= prev.timestamp && targetTime <= next.timestamp {
                    return (prev, next)
                }
            }
            
            if let last = buffer.last, targetTime > last.timestamp {
                return (buffer[buffer.count-2], last)
            }
            
            return (buffer[0], buffer[1])
        }
        
        var newestFrame: FrameHistory? {
            lock.lock()
            defer { lock.unlock() }
            return buffer.last
        }

        /// The two most recent captures. Interpolation works on this pair and no
        /// other: the midpoint MetalFX can add belongs between them, and asking
        /// for the pair directly cannot miss the way searching for a timestamp
        /// could when the estimated capture interval drifted from the real one.
        var newestPair: (prev: FrameHistory, next: FrameHistory)? {
            lock.lock()
            defer { lock.unlock() }
            guard buffer.count >= 2 else { return nil }
            return (buffer[buffer.count - 2], buffer[buffer.count - 1])
        }

        func clear() {
            lock.lock()
            defer { lock.unlock() }
            buffer.removeAll()
        }
    }
    
    private let frameBuffer = FrameRingBuffer()

    /// A slot must not be rewritten while the ring still hands it out, so the
    /// pool holds everything the ring can reference plus everything the pipeline
    /// can have in flight behind it. At six slots a full ring plus three
    /// in-flight captures wrapped onto a live frame and tore it.
    private static let texturePoolDepth = FrameRingBuffer.capacity + maxInFlight

    private var historyTextures: [MTLTexture?] = Array(repeating: nil, count: GooseEngine.texturePoolDepth)
    private var historyTextureIndex: Int = 0

    private var motionTextures: [MTLTexture?] = Array(repeating: nil, count: GooseEngine.texturePoolDepth)
    private var motionTextureIndex: Int = 0
    private let motionLock = OSAllocatedUnfairLock()
    private var _latestMotion: MTLTexture?

    private var interpolatedTexture: MTLTexture?
    private var cachedInterpPrevTimestamp: CFTimeInterval = -1
    private var cachedInterpNextTimestamp: CFTimeInterval = -1
    private var cachedExtrapSourceTimestamp: CFTimeInterval = -1
    private var cachedExtrapStep: Int = -1
    private var lastProcessedSize: CGSize = .zero

    private struct EngineConfig {
        var scalingType: CaptureSettings.ScalingType = .off
        var aaMode: CaptureSettings.AAMode = .off
        var frameGenEnabled: Bool = false
        var frameGenMode: CaptureSettings.FrameGenMode = .off
        /// Presented images per captured frame the pipeline should aim for.
        /// Interpolation is pinned to 2 by what MetalFX produces; extrapolation
        /// uses it to decide how many warp phases to sample in each gap.
        var frameGenMultiplier: Int = 2
        var vsyncEnabled: Bool = true
        /// Scale Factor is expressed in the overlay window's size and the upscale
        /// target is read from the live drawable, so the engine never needs the
        /// factor itself — only the profile the quality picker resolves to.
        var qualityProfile: QualityProfile = CaptureSettings.QualityMode.balanced.profile
        var captureCursorEnabled: Bool = false
        var bufferDepth: Int = GooseEngine.maxInFlight
    }

    private let configLock = OSAllocatedUnfairLock()
    private var _config = EngineConfig()

    /// Every consumer takes one snapshot at the top of its work unit, so a
    /// concurrent `updateSettings` cannot tear the pipeline state mid-frame.
    private var config: EngineConfig {
        configLock.lock()
        defer { configLock.unlock() }
        return _config
    }


    /// Written on the processing queue and read by the render thread every frame,
    /// so it cannot be a bare property — the frame clock is built on it.
    private let captureRateLock = OSAllocatedUnfairLock()
    private var _estimatedCaptureInterval: Double = 0
    private var estimatedCaptureInterval: Double {
        captureRateLock.lock()
        defer { captureRateLock.unlock() }
        return _estimatedCaptureInterval
    }

    private var lastCaptureTimestamp: CFTimeInterval = 0
    private var lastCPUSampleTime: CFTimeInterval = 0
    private var lastCPUTimeNanos: UInt64 = 0
    private var lastPreferredFPS: Int = 0
    private var lastPreferredUpdateTime: CFTimeInterval = 0

    private var windowCaptureManager: WindowCaptureManager?

    private var lastFrameTime: CFTimeInterval = 0
    private var frameCount: Int = 0
    private var fpsStartTime: CFTimeInterval = 0
    
    private var currentRefreshRate: Int = 0
    private var minRefreshRate: Int = 0

    static func make() -> GooseEngine? {
        guard let dev = MTLCreateSystemDefaultDevice() else {
            lastInitError = "Error Code: MG-ENG-002 Metal device not available."
            return nil
        }
        guard let queue = dev.makeCommandQueue() else {
            lastInitError = "Error Code: MG-ENG-003 Metal command queue not available."
            return nil
        }
        lastInitError = nil
        return GooseEngine(device: dev, commandQueue: queue)
    }

    private init(device: MTLDevice, commandQueue: MTLCommandQueue) {
        self.device = device
        self.commandQueue = commandQueue
        super.init()
        setupPipelines()
    }
    
    private func setupPipelines() {
        guard let library = device.makeDefaultLibrary() else {
            reportError("Error Code: MG-ENG-001 Metal pipeline setup failed.")
            return
        }
        
        func makeCompute(_ name: String) -> MTLComputePipelineState? {
            guard let function = library.makeFunction(name: name) else { return nil }
            return try? device.makeComputePipelineState(function: function)
        }

        casPipeline = makeCompute("contrastAdaptiveSharpening")
        copyPipeline = makeCompute("copyTexture")
        lumaPipeline = makeCompute("bgraToLuma")
        extrapolatePipeline = makeCompute("extrapolateFrame")
        
        fxaaPipeline = makeCompute("fxaa")
        smaaEdgePipeline = makeCompute("smaaEdgeDetection")
        smaaWeightPipeline = makeCompute("smaaBlendingWeights")
        smaaBlendPipeline = makeCompute("smaaBlend")

        do {
            if let vtx = library.makeFunction(name: "texture_vertex"),
               let frag = library.makeFunction(name: "texture_fragment") {
                let desc = MTLRenderPipelineDescriptor()
                desc.vertexFunction = vtx
                desc.fragmentFunction = frag
                desc.colorAttachments[0].pixelFormat = .bgra8Unorm
                renderPipeline = try device.makeRenderPipelineState(descriptor: desc)
            }
        } catch {
            reportError("Error Code: MG-ENG-001 Pipeline setup failed: \(error)")
        }

        do {
            if let vtx = library.makeFunction(name: "cursor_vertex"),
               let frag = library.makeFunction(name: "cursor_fragment") {
                let desc = MTLRenderPipelineDescriptor()
                desc.vertexFunction = vtx
                desc.fragmentFunction = frag
                desc.colorAttachments[0].pixelFormat = .bgra8Unorm
                desc.colorAttachments[0].isBlendingEnabled = true
                desc.colorAttachments[0].rgbBlendOperation = .add
                desc.colorAttachments[0].alphaBlendOperation = .add
                desc.colorAttachments[0].sourceRGBBlendFactor = .sourceAlpha
                desc.colorAttachments[0].sourceAlphaBlendFactor = .sourceAlpha
                desc.colorAttachments[0].destinationRGBBlendFactor = .oneMinusSourceAlpha
                desc.colorAttachments[0].destinationAlphaBlendFactor = .oneMinusSourceAlpha
                cursorPipeline = try device.makeRenderPipelineState(descriptor: desc)
            }
        } catch {
            reportError("Error Code: MG-ENG-011 Cursor pipeline setup failed: \(error)")
        }

        loadCursorTexture()
    }

    private func loadCursorTexture() {
        let cursorImage = NSCursor.arrow.image
        var rect = CGRect(origin: .zero, size: cursorImage.size)
        guard let cgImage = cursorImage.cgImage(forProposedRect: &rect, context: nil, hints: nil) else {
            return
        }
        let loader = MTKTextureLoader(device: device)
        let options: [MTKTextureLoader.Option: Any] = [
            .SRGB: false,
            .textureUsage: MTLTextureUsage.shaderRead.rawValue,
            .textureStorageMode: MTLStorageMode.private.rawValue
        ]
        cursorTexture = try? loader.newTexture(cgImage: cgImage, options: options)
        cursorTextureSize = cursorImage.size
    }
    
    /// Validity is checked against the scaler's own reported dimensions rather
    /// than a shadow copy, so there is no second source of truth to drift out of
    /// sync. The capture path and the render path each own a cache because they
    /// run on different threads, but the construction is identical.
    private func ensureSpatialScaler(_ cache: inout MTLFXSpatialScaler?,
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
        descriptor.colorProcessingMode = .perceptual

        guard let scaler = descriptor.makeSpatialScaler(device: device) else {
            reportError("Error Code: MG-ENG-004 MetalFX Spatial Scaler creation failed")
            return nil
        }

        cache = scaler
        return scaler
    }

    /// Cleared to the far plane once per size change: the capture is a flat plane,
    /// so a constant depth carries the correct meaning and costs one clear pass.
    @MainActor
    private func ensureFlatDepthTexture(width: Int, height: Int) -> MTLTexture? {
        if let tex = flatDepthTexture, tex.width == width, tex.height == height { return tex }

        let desc = MTLTextureDescriptor.texture2DDescriptor(pixelFormat: .r32Float,
                                                            width: width, height: height, mipmapped: false)
        desc.usage = [.shaderRead, .renderTarget]
        desc.storageMode = .private
        guard let tex = device.makeTexture(descriptor: desc),
              let commandBuffer = commandQueue.makeCommandBuffer() else { return nil }

        let pass = MTLRenderPassDescriptor()
        pass.colorAttachments[0].texture = tex
        pass.colorAttachments[0].loadAction = .clear
        pass.colorAttachments[0].storeAction = .store
        pass.colorAttachments[0].clearColor = MTLClearColor(red: 1, green: 0, blue: 0, alpha: 0)
        guard let encoder = commandBuffer.makeRenderCommandEncoder(descriptor: pass) else { return nil }
        encoder.endEncoding()
        commandBuffer.commit()

        flatDepthTexture = tex
        return tex
    }

    @MainActor
    private func ensureFrameInterpolator(width: Int, height: Int) -> MTLFXFrameInterpolator? {
        if let interpolator = frameInterpolator,
           interpolator.inputWidth == width, interpolator.inputHeight == height,
           interpolator.outputWidth == width, interpolator.outputHeight == height {
            return interpolator
        }

        let descriptor = MTLFXFrameInterpolatorDescriptor()
        descriptor.colorTextureFormat = .bgra8Unorm
        descriptor.outputTextureFormat = .bgra8Unorm
        descriptor.depthTextureFormat = .r32Float
        descriptor.inputWidth = width
        descriptor.inputHeight = height
        descriptor.outputWidth = width
        descriptor.outputHeight = height

        guard let interpolator = descriptor.makeFrameInterpolator(device: device) else {
            reportError("Error Code: MG-ENG-010 MetalFX Frame Interpolator creation failed")
            return nil
        }

        frameInterpolator = interpolator
        frameInterpolatorNeedsHistoryReset = true
        return interpolator
    }

    private func ensureTexture(_ texture: inout MTLTexture?, width: Int, height: Int,
                               pixelFormat: MTLPixelFormat = .bgra8Unorm,
                               usage: MTLTextureUsage = [.shaderRead, .shaderWrite]) -> MTLTexture? {
        if let tex = texture,
           tex.width == width,
           tex.height == height,
           tex.pixelFormat == pixelFormat {
            return tex
        }

        let desc = MTLTextureDescriptor.texture2DDescriptor(
            pixelFormat: pixelFormat,
            width: width,
            height: height,
            mipmapped: false
        )
        desc.usage = usage
        desc.storageMode = .private
        texture = device.makeTexture(descriptor: desc)
        return texture
    }

    private func encodeCopy(from input: MTLTexture,
                            to output: MTLTexture,
                            commandBuffer: MTLCommandBuffer) -> Bool {
        guard let copyPipeline = copyPipeline,
              let encoder = commandBuffer.makeComputeCommandEncoder() else {
            reportError("Error Code: MG-ENG-009 Copy pipeline unavailable")
            return false
        }
        encoder.setComputePipelineState(copyPipeline)
        encoder.setTexture(input, index: 0)
        encoder.setTexture(output, index: 1)
        dispatchThreads(pipeline: copyPipeline, encoder: encoder, width: output.width, height: output.height)
        encoder.endEncoding()
        return true
    }
    
    private func dispatchThreads(pipeline: MTLComputePipelineState,
                                 encoder: MTLComputeCommandEncoder,
                                 width: Int,
                                 height: Int) {
        let threadW = pipeline.threadExecutionWidth
        let threadH = pipeline.maxTotalThreadsPerThreadgroup / threadW
        let threadsPerGroup = MTLSize(width: threadW, height: threadH, depth: 1)
        let grid = MTLSize(width: (width + threadW - 1) / threadW,
                           height: (height + threadH - 1) / threadH,
                           depth: 1)
        encoder.dispatchThreadgroups(grid, threadsPerThreadgroup: threadsPerGroup)
    }
    
    private func resetProcessingState(clearFrames: Bool = true) {
        casTexture = nil
        captureUpscaledTexture = nil
        captureScaler = nil
        smaaEdgeTexture = nil
        smaaWeightTexture = nil
        historyTextures = Array(repeating: nil, count: historyTextures.count)
        historyTextureIndex = 0
        motionTextures = Array(repeating: nil, count: motionTextures.count)
        motionTextureIndex = 0
        motionLock.lock()
        _latestMotion = nil
        motionLock.unlock()
        motionEstimator.reset()
        lastProcessedSize = .zero

        // Frame-gen MetalFX state belongs to the render thread. Signal it here
        // and let renderFrame tear it down, so the interpolator is never created
        // on one thread and released on another mid-encode.
        renderStateLock.lock()
        frameGenNeedsTeardown = true
        renderStateLock.unlock()

        if clearFrames {
            frameBuffer.clear()
            statsLock.lock()
            _stats.droppedFrames = 0
            _stats.interpolatedFrameCount = 0
            _stats.passthroughFrameCount = 0
            _stats.generatedFrameCount = 0
            statsLock.unlock()
        }
    }

    private func resetProcessingStateAsync(clearFrames: Bool = true) {
        processingQueue.async { [weak self] in
            self?.resetProcessingState(clearFrames: clearFrames)
        }
    }

    private func resetFrameCounters() {
        statsLock.lock()
        _stats.frameCount = 0
        _stats.outputFrameCount = 0
        _stats.interpolatedFrameCount = 0
        _stats.passthroughFrameCount = 0
        _stats.generatedFrameCount = 0
        _stats.droppedFrames = 0
        _stats.captureFPS = 0
        _stats.outputFPS = 0
        _stats.generatedFPS = 0
        statsLock.unlock()

        captureRateLock.lock()
        _estimatedCaptureInterval = 0
        captureRateLock.unlock()

        frameCount = 0
        fpsStartTime = CACurrentMediaTime()

        // The frame clock and its history belong to the render thread. This runs
        // off it, so the reset is handed over rather than written across.
        DispatchQueue.main.async { [weak self] in
            guard let self else { return }
            self.renderFrameCount = 0
            self.interpolatedFrameCount = 0
            self.generatedFrameCount = 0
            self.renderFPSStartTime = CACurrentMediaTime()
            self.outputFrameTimeHistory.removeAll()
            self.lastRenderTime = 0
            self.lastPresentedCaptureTimestamp = -1
            self.pllPhase = 0
        }
    }

    private func measuredSourceFPS() -> Double {
        if estimatedCaptureInterval > 0 {
            return 1.0 / estimatedCaptureInterval
        }
        if stats.captureFPS > 0 {
            return Double(stats.captureFPS)
        }
        return 0
    }

    private func snapToRefreshDivisor(_ target: Double) -> Int {
        guard currentRefreshRate > 0, target > 0 else {
            return max(1, Int(round(target)))
        }
        let r = Double(currentRefreshRate)
        let n = max(1, Int(round(r / target)))
        return max(1, Int(round(r / Double(n))))
    }

    /// Frame generation exists to fill the gaps between captures, so the honest
    /// target is whatever the panel itself presents: each extra present is either
    /// a newly generated image or a repeat that keeps the cadence on the vsync
    /// grid. Deriving the target from a fixed multiplier instead — 2x at 30 fps
    /// capture asks for 60 on a 120 Hz panel — halves the present rate, so every
    /// missed deadline costs two refresh intervals rather than one, and the
    /// schedule has half as many phase slots to absorb clock jitter.
    ///
    /// Without frame generation the target is the source rate, snapped so the
    /// frame clock advances at a rate the display can deliver: asking for 50 on a
    /// 120 Hz panel yields 40, the clock drifts behind real time, and motion
    /// plays back slowly.
    private func desiredOutputFPS(_ config: EngineConfig) -> Int {
        let sourceFPS = measuredSourceFPS()
        let requested = config.frameGenEnabled
            ? Double(currentRefreshRate)
            : (sourceFPS > 0 ? sourceFPS : Double(currentRefreshRate))

        var target = config.vsyncEnabled ? snapToRefreshDivisor(requested) : Int(requested.rounded())

        if currentRefreshRate > 0 {
            target = min(target, currentRefreshRate)
        }
        // A variable-refresh panel cannot be driven below its own floor. A fixed
        // panel has no floor to respect — it repeats frames instead — so its
        // nominal minimum must not be allowed to force the target upwards.
        if minRefreshRate > 0, minRefreshRate < currentRefreshRate, target < minRefreshRate {
            target = minRefreshRate
        }
        return max(1, target)
    }

    /// The target is derived from a filter that settles over one measurement
    /// window, so it cannot honestly change faster than that — and re-tuning
    /// more often only resets CADisplayLink's pacing. The window is the whole
    /// hysteresis; no magnitude threshold is needed on top of it, and one would
    /// only mask real divisor changes.
    private func applyFrameRatePreference(_ preferred: Int) {
        guard preferred > 0, preferred != lastPreferredFPS else { return }
        let now = CACurrentMediaTime()
        if lastPreferredUpdateTime > 0, now - lastPreferredUpdateTime < Self.measurementWindow { return }

        lastPreferredFPS = preferred
        lastPreferredUpdateTime = now

        let view = mtkView
        DispatchQueue.main.async {
            view?.preferredFramesPerSecond = preferred
        }
    }

    private func applyDisplaySync(to view: MTKView, vsync: Bool) {
        MainActor.assumeIsolated {
            guard let layer = view.layer as? CAMetalLayer else { return }
            layer.displaySyncEnabled = vsync
            layer.presentsWithTransaction = false
        }
    }

    /// How far behind real time the frame clock samples. Interpolation blends
    /// between a pair that has already arrived, so the clock sits a full capture
    /// interval back: ring timestamps are arrival times, so with a delay of one
    /// interval the sample sweeps exactly from `prev` to `next` as the gap fills
    /// in, and every phase in the bracket is reachable.
    ///
    /// Half an interval — what this used to hold back — puts the sample past the
    /// newest frame for the back half of every capture period. The ratio clamps
    /// to 1, the schedule takes the passthrough branch, `prev` is never shown at
    /// all, and the generated midpoint reaches only a quarter of the presents
    /// instead of half. Extrapolation predicts forward from the newest frame and
    /// holds nothing back.
    private func interpolationDelay(outputInterval: Double, mode: CaptureSettings.FrameGenMode) -> Double {
        let captureInterval = estimatedCaptureInterval
        switch mode {
        case .interpolation: return max(outputInterval, captureInterval)
        case .extrapolation: return 0
        case .off:           return max(outputInterval, captureInterval)
        }
    }

    /// Emulates a shallower pipeline by parking permits on `processingQueue`
    /// rather than replacing the semaphore out from under in-flight frames.
    private func applyBufferDepth(_ depth: Int) {
        let wanted = Self.maxInFlight - max(2, min(Self.maxInFlight, depth))
        processingQueue.async { [weak self] in
            guard let self else { return }
            while self.parkedPermits < wanted {
                self.inFlightSemaphore.wait()
                self.parkedPermits += 1
            }
            while self.parkedPermits > wanted {
                self.inFlightSemaphore.signal()
                self.parkedPermits -= 1
            }
        }
    }

    /// `minRefreshRate` is the panel's own floor. A panel whose floor equals its
    /// ceiling is fixed-refresh; anything else is variable, so ProMotion needs no
    /// separate flag and cannot disagree with the two rates it is derived from.
    func attachToView(_ view: MTKView, displayRefreshRate: Int, minRefreshRate: Int) {
        let config = self.config
        MainActor.assumeIsolated {
            view.device = device
            view.delegate = self
            view.preferredFramesPerSecond = displayRefreshRate
            view.isPaused = false
            view.enableSetNeedsDisplay = false
            view.colorPixelFormat = .bgra8Unorm
            view.framebufferOnly = false
            if let layer = view.layer as? CAMetalLayer {
                layer.maximumDrawableCount = min(max(2, config.bufferDepth), Self.maxInFlight)
            }
        }
        applyDisplaySync(to: view, vsync: config.vsyncEnabled)
        self.mtkView = view
        self.currentRefreshRate = displayRefreshRate
        self.minRefreshRate = minRefreshRate > 0 ? min(minRefreshRate, displayRefreshRate) : displayRefreshRate
        statsLock.lock()
        _stats.screenRefreshRate = displayRefreshRate
        _stats.isProMotion = self.minRefreshRate < displayRefreshRate
        statsLock.unlock()
        applyFrameRatePreference(desiredOutputFPS(config))
    }

    func detachFromView() {
        MainActor.assumeIsolated {
            mtkView?.delegate = nil
            mtkView?.isPaused = true
            upscaledTexture = nil
            upscaleCacheKey = -1
            spatialScaler = nil
            frameInterpolator = nil
            frameInterpolatorNeedsHistoryReset = true
            interpolatedTexture = nil
            extrapolatedTexture = nil
            flatDepthTexture = nil
            cachedInterpPrevTimestamp = -1
            cachedInterpNextTimestamp = -1
            cachedExtrapSourceTimestamp = -1
            cachedExtrapStep = -1
        }
        mtkView = nil
    }

    nonisolated func draw(in view: MTKView) {
        MainActor.assumeIsolated {
            renderFrame(in: view)
        }
    }
    
    /// Render-thread only, published once a second alongside the FPS counters.
    private var renderFrameCount: Int = 0
    private var renderFPSStartTime: CFTimeInterval = 0
    private var interpolatedFrameCount: Int = 0
    /// Cache misses in the generator over the current measurement window — the
    /// count of images actually synthesised, as opposed to presents that showed
    /// one.
    private var generatedFrameCount: Int = 0
    private var outputFrameTimeHistory: [Double] = []
    private var lastRenderTime: CFTimeInterval = 0
    private var lastPresentedCaptureTimestamp: CFTimeInterval = -1

    /// One measurement window of output frames, at whatever rate the view is
    /// actually being driven — a fixed count would average over a quarter of a
    /// second on a 480 Hz panel and two seconds on a 60 Hz one.
    private var frameTimeHistoryCapacity: Int {
        // Eight is the smallest sample count the mean is meaningful over; it is a
        // statistics floor, not an assumption about the display.
        guard currentRefreshRate > 0 else { return Self.minFrameTimeSamples }
        return max(Self.minFrameTimeSamples,
                   Int((Double(currentRefreshRate) * Self.measurementWindow).rounded()))
    }

    private static let minFrameTimeSamples = 8

    private var pllPhase: CFTimeInterval = 0

    /// First-order loop: phase error decays by this fraction each output frame,
    /// so the gain is whatever settles inside one measurement window at the rate
    /// the view is running. A fixed gain locks four times too fast on a 480 Hz
    /// panel and four times too slowly on a 30 fps one.
    private func pllGain(outputInterval: Double) -> Double {
        guard outputInterval > 0 else { return 1 }
        return min(1, outputInterval / Self.measurementWindow)
    }

    @MainActor
    private func renderFrame(in view: MTKView) {
        // The drawable is deliberately not acquired here. `currentDrawable`
        // blocks the calling thread — this one, the main thread — for as long as
        // the layer's pool is empty, and a layer whose window has gone away never
        // refills it: recycling happens on display refresh, and a layer with
        // nowhere to present never gets one. Taking a drawable at the top of the
        // function put every early return on the wrong side of that, so closing
        // the window while the view was still being driven left the main thread
        // spinning inside nextDrawable until the app had to be force quit.
        //
        // So: establish there is somewhere to present first, do all the work that
        // does not need a drawable, and take one at the last possible moment.
        let drawableSize = view.drawableSize
        guard view.window != nil,
              drawableSize.width > 0, drawableSize.height > 0,
              let renderPipeline = renderPipeline,
              let commandBuffer = commandQueue.makeCommandBuffer() else { return }

        let config = self.config
        let currentTime = CACurrentMediaTime()

        // The processing queue only asks; the teardown itself happens here so
        // every MetalFX frame-gen object lives and dies on this thread.
        renderStateLock.lock()
        let needsTeardown = frameGenNeedsTeardown
        frameGenNeedsTeardown = false
        renderStateLock.unlock()
        if needsTeardown {
            frameInterpolator = nil
            frameInterpolatorNeedsHistoryReset = true
            interpolatedTexture = nil
            extrapolatedTexture = nil
            flatDepthTexture = nil
            cachedInterpPrevTimestamp = -1
            cachedInterpNextTimestamp = -1
            cachedExtrapSourceTimestamp = -1
            cachedExtrapStep = -1
        }

        if lastRenderTime > 0 {
            let interval = currentTime - lastRenderTime
            outputFrameTimeHistory.append(interval)
            if outputFrameTimeHistory.count > frameTimeHistoryCapacity {
                outputFrameTimeHistory.removeFirst()
            }
        }
        lastRenderTime = currentTime

        if renderFPSStartTime == 0 { renderFPSStartTime = currentTime }
        let elapsed = currentTime - renderFPSStartTime

        let targetFPS = desiredOutputFPS(config)
        applyFrameRatePreference(targetFPS)
        statsLock.lock()
        _stats.targetOutputFPS = targetFPS
        statsLock.unlock()

        // The frame clock has to advance at the rate the view is actually being
        // driven at. preferredFramesPerSecond is only a hint — a ProMotion panel
        // happily runs the callback at 120 while we asked for 60 — and a clock
        // built on the requested rate runs ahead of real time, pushes targetTime
        // past the newest captured frame, and turns every frame into passthrough.
        let measuredInterval = outputFrameTimeHistory.count >= Self.minFrameTimeSamples
            ? outputFrameTimeHistory.reduce(0, +) / Double(outputFrameTimeHistory.count)
            : 0
        let nominalInterval = measuredInterval > 0
            ? measuredInterval
            : (targetFPS > 0 ? 1.0 / Double(targetFPS) : 0)
        let sampleClock: CFTimeInterval
        // Several frames' worth of deviation is a stall, not phase error a
        // first-order loop should chase; resynchronise instead of crawling back.
        if pllPhase == 0 || nominalInterval <= 0
            || abs(currentTime - (pllPhase + nominalInterval)) > nominalInterval * 3 {
            pllPhase = currentTime
            sampleClock = currentTime
        } else {
            let advanced = pllPhase + nominalInterval
            let phaseError = currentTime - advanced
            pllPhase = advanced + phaseError * pllGain(outputInterval: nominalInterval)
            sampleClock = pllPhase
        }

        let targetTime = sampleClock - interpolationDelay(outputInterval: nominalInterval, mode: config.frameGenMode)

        var outputTex: MTLTexture?
        var sourceTimestamp: CFTimeInterval?
        var contentKey: CFTimeInterval = -1
        let frameGenActive = config.frameGenEnabled
        var isInterpolated = false
        var generatedNewImage = false

        if config.frameGenMode == .extrapolation, let newest = frameBuffer.newestFrame {
            sourceTimestamp = newest.timestamp
            let interval = estimatedCaptureInterval
            // The multiplier is how many images the gap should carry, so the gap
            // is cut into that many slots: slot 0 is the captured frame itself
            // and each later slot is one warp phase. Letting every present pick
            // its own continuous phase ignored the multiplier entirely and paid
            // for a fresh warp even where the panel was repeating an image.
            let steps = max(1, config.frameGenMultiplier)
            let rawPhase = interval > 0 ? min(max((currentTime - newest.timestamp) / interval, 0), 1) : 0
            let step = min(steps - 1, Int(rawPhase * Double(steps)))
            let phase = Float(step) / Float(steps)

            // A capture that has not been shown yet is always presented as it
            // was captured. Only the gaps between captures are generated —
            // warping real frames as well destroys the image for no benefit.
            if newest.timestamp != lastPresentedCaptureTimestamp {
                lastPresentedCaptureTimestamp = newest.timestamp
                outputTex = newest.texture
                contentKey = newest.timestamp
            } else if step > 0, !newest.isSceneCut, let motion = newest.motion,
                      let warped = encodeExtrapolation(source: newest.texture, motion: motion,
                                                       phase: phase,
                                                       sourceTimestamp: newest.timestamp, step: step,
                                                       commandBuffer: commandBuffer) {
                outputTex = warped.texture
                isInterpolated = true
                generatedNewImage = warped.isNew
                // Report the age of the newest real information on screen, not
                // the moment the warp predicts: extrapolated pixels are a guess,
                // and crediting them made the figure read as zero.
                sourceTimestamp = newest.timestamp
                contentKey = newest.timestamp + Double(phase)
            } else {
                outputTex = newest.texture
                contentKey = newest.timestamp
            }
        } else if config.frameGenMode == .interpolation,
           let (prev, next) = frameBuffer.getFramesForTime(targetTime),
           prev.texture.width == next.texture.width,
           prev.texture.height == next.texture.height {
            let duration = next.timestamp - prev.timestamp
            let ratio = duration > 0 ? ((targetTime - prev.timestamp) / duration) : 0
            let t = min(max(ratio, 0), 1)

            // MetalFX synthesises exactly one phase between a pair — the midpoint
            // — so the images available for this bracket are prev at 0, the
            // generated frame at 0.5 and next at 1. Each sample is served by
            // whichever of those sits closest to it, which puts the boundaries at
            // 0.25 and 0.75 for free. A tuned snap window in output-interval units
            // did the same job only while the output rate happened to match, and
            // dropped generated frames whenever it did not.
            let phase = duration > 0 ? (t * 2).rounded() / 2 : 0

            if phase == 0 {
                outputTex = prev.texture
                sourceTimestamp = prev.timestamp
                contentKey = prev.timestamp
            } else if phase == 1 {
                outputTex = next.texture
                sourceTimestamp = next.timestamp
                contentKey = next.timestamp
            } else if next.isSceneCut {
                outputTex = next.texture
                sourceTimestamp = next.timestamp
                contentKey = next.timestamp
                frameInterpolatorNeedsHistoryReset = true
            } else if let interpolated = interpolateFrame(prev: prev, next: next,
                                                          mode: config.frameGenMode,
                                                          commandBuffer: commandBuffer) {
                outputTex = interpolated.texture
                isInterpolated = true
                generatedNewImage = interpolated.isNew
                // The generated frame stands for targetTime, not for prev — using
                // prev here overstated latency by up to a whole capture interval.
                // Interpolation had to wait for `next` before it could produce
                // anything, so that is the honest age to report — and it makes
                // the two modes directly comparable.
                sourceTimestamp = next.timestamp
                // MetalFX yields one image per frame pair, so the pair identifies
                // the content however many times it is presented.
                contentKey = prev.timestamp + next.timestamp
            } else {
                outputTex = t < 0.5 ? prev.texture : next.texture
                sourceTimestamp = t < 0.5 ? prev.timestamp : next.timestamp
                contentKey = sourceTimestamp ?? -1
            }
        } else {
            outputTex = frameBuffer.newestFrame?.texture
            sourceTimestamp = frameBuffer.newestFrame?.timestamp
            contentKey = sourceTimestamp ?? -1
        }

        guard let finalTex = outputTex else {
            commandBuffer.commit()
            return
        }

        let presentTex = encodeUpscale(finalTex,
                                       drawableWidth: Int(drawableSize.width),
                                       drawableHeight: Int(drawableSize.height),
                                       contentKey: contentKey,
                                       config: config, commandBuffer: commandBuffer)
        statsLock.lock()
        _stats.outputResolution = CGSize(width: presentTex.width, height: presentTex.height)
        statsLock.unlock()

        if let sourceTimestamp {
            let presentLatencyMs = Float((currentTime - sourceTimestamp) * 1000.0)
            statsLock.lock()
            _stats.presentLatency = presentLatencyMs
            _stats.endToEndLatency = _stats.captureLatency + presentLatencyMs
            statsLock.unlock()
        }
        
        renderFrameCount += 1
        if isInterpolated { interpolatedFrameCount += 1 }
        // Only a cache miss put a new image on screen. Counting every present
        // that carried generated content instead reported the same synthesised
        // image once per present, which at 120 Hz over a 15 fps capture inflated
        // the figure by roughly the refresh ratio.
        if generatedNewImage { generatedFrameCount += 1 }
        if elapsed >= 1.0 {
            statsLock.lock()
            _stats.outputFPS = Float(renderFrameCount) / Float(elapsed)
            _stats.generatedFPS = Float(generatedFrameCount) / Float(elapsed)
            statsLock.unlock()
            renderFrameCount = 0
            interpolatedFrameCount = 0
            generatedFrameCount = 0
            renderFPSStartTime = currentTime

            updateFramePacingStats()
        }
        
        // Last possible moment: everything above is encoded and the only thing
        // left is the pass that writes into the drawable and presents it.
        guard let drawable = view.currentDrawable else {
            commandBuffer.commit()
            return
        }

        let renderPassDesc = MTLRenderPassDescriptor()
        renderPassDesc.colorAttachments[0].texture = drawable.texture
        renderPassDesc.colorAttachments[0].loadAction = .clear
        renderPassDesc.colorAttachments[0].storeAction = .store
        renderPassDesc.colorAttachments[0].clearColor = MTLClearColor(red: 0, green: 0, blue: 0, alpha: 1)
        
        guard let renEncoder = commandBuffer.makeRenderCommandEncoder(descriptor: renderPassDesc) else {
            commandBuffer.commit()
            return
        }
        renEncoder.setRenderPipelineState(renderPipeline)
        renEncoder.setFragmentTexture(presentTex, index: 0)
        renEncoder.drawPrimitives(type: .triangleStrip, vertexStart: 0, vertexCount: 4)

        if config.captureCursorEnabled {
            drawSyntheticCursor(in: view, encoder: renEncoder, drawableSize: drawableSize)
        }

        renEncoder.endEncoding()
        
        let frameWasInterpolated = isInterpolated
        let frameWasGenerated = generatedNewImage
        commandBuffer.present(drawable)
        commandBuffer.addCompletedHandler { [weak self] buffer in
            guard let self = self else { return }
            // Interpolation and the upscale live here now, so the capture buffer
            // alone no longer represents the pipeline's GPU cost.
            let renderGPU = Float((buffer.gpuEndTime - buffer.gpuStartTime) * 1000.0)
            self.statsLock.lock()
            self._stats.gpuTime = self._stats.captureGPUTime + renderGPU
            self._stats.outputFrameCount += 1
            if frameWasGenerated { self._stats.generatedFrameCount += 1 }
            if frameGenActive {
                if frameWasInterpolated {
                    self._stats.interpolatedFrameCount += 1
                } else {
                    self._stats.passthroughFrameCount += 1
                }
            } else {
                self._stats.passthroughFrameCount += 1
            }
            self.statsLock.unlock()
        }
        commandBuffer.commit()
    }
    
    /// Spatial upscale lives on the render path so that frame interpolation can
    /// run at capture resolution. Interpolation cost scales with pixel count, so
    /// paying for one upscale per presented frame is far cheaper than making
    /// every generated frame an output-resolution interpolation.
    @MainActor
    private func encodeUpscale(_ input: MTLTexture, drawableWidth: Int, drawableHeight: Int,
                               contentKey: CFTimeInterval, config: EngineConfig,
                               commandBuffer: MTLCommandBuffer) -> MTLTexture {
        guard config.scalingType == .mgup1, drawableWidth > 0, drawableHeight > 0 else { return input }

        // Frame generation presents the same content more than once per capture,
        // and upscaling it again produces an identical image for the full cost.
        if let cached = upscaledTexture, upscaleCacheKey == contentKey,
           cached.width == drawableWidth, cached.height == drawableHeight {
            return cached
        }

        // Scale Factor is already expressed in the overlay's size, so the target
        // is simply the drawable: MetalFX covers the whole gap and the final blit
        // stays 1:1 instead of bilinearly stretching on top of the upscale. The
        // overlay frame is capped at the screen, so the drawable is bounded by
        // the panel and can never approach Metal's texture limit.
        guard drawableWidth > input.width || drawableHeight > input.height else { return input }

        guard let out = ensureTexture(&upscaledTexture, width: drawableWidth, height: drawableHeight,
                                      usage: [.shaderRead, .shaderWrite, .renderTarget]),
              let scaler = ensureSpatialScaler(&spatialScaler,
                                               inputWidth: input.width, inputHeight: input.height,
                                               outputWidth: drawableWidth, outputHeight: drawableHeight) else {
            return input
        }
        scaler.colorTexture = input
        scaler.outputTexture = out
        scaler.encode(commandBuffer: commandBuffer)
        upscaleCacheKey = contentKey
        return out
    }

    @MainActor
    private func drawSyntheticCursor(in view: MTKView, encoder: MTLRenderCommandEncoder, drawableSize: CGSize) {
        guard let pipeline = cursorPipeline,
              let texture = cursorTexture,
              drawableSize.width > 0, drawableSize.height > 0,
              cursorTextureSize.width > 0, cursorTextureSize.height > 0,
              let fraction = MouseConstraintManager.shared.currentCursorFraction() else {
            return
        }

        let scale = view.window?.screen?.backingScaleFactor ?? 2.0

        let widthNDC = Float((cursorTextureSize.width * scale) / drawableSize.width * 2.0)
        let heightNDC = Float((cursorTextureSize.height * scale) / drawableSize.height * 2.0)
        let centerX = Float(-1.0 + 2.0 * fraction.x)
        let centerY = Float(1.0 - 2.0 * fraction.y)

        var uniforms = CursorUniforms(center: SIMD2<Float>(centerX, centerY),
                                       size: SIMD2<Float>(widthNDC, heightNDC))

        encoder.setRenderPipelineState(pipeline)
        encoder.setVertexBytes(&uniforms, length: MemoryLayout<CursorUniforms>.stride, index: 0)
        encoder.setFragmentTexture(texture, index: 0)
        encoder.drawPrimitives(type: .triangleStrip, vertexStart: 0, vertexCount: 4)
    }

    private func updateFramePacingStats() {
        guard outputFrameTimeHistory.count >= 2 else { return }

        let avg = outputFrameTimeHistory.reduce(0, +) / Double(outputFrameTimeHistory.count)

        var jitterSum = 0.0
        for i in 1..<outputFrameTimeHistory.count {
            jitterSum += abs(outputFrameTimeHistory[i] - outputFrameTimeHistory[i - 1])
        }
        let jitter = jitterSum / Double(outputFrameTimeHistory.count - 1)

        let pacingScore = avg > 0 ? max(0, 100 * (1 - min(1, jitter / avg))) : 100

        statsLock.lock()
        _stats.avgFrameTime = Float(avg * 1000.0)
        _stats.framePacingScore = Float(pacingScore)
        statsLock.unlock()
    }

    private func updateCaptureStats(currentTime: CFTimeInterval, captureTimestamp: CFTimeInterval?) {
        frameCount += 1

        statsLock.lock()
        _stats.frameCount += 1
        _stats.gpuMemoryUsed = UInt64(device.currentAllocatedSize)
        _stats.gpuMemoryTotal = UInt64(device.recommendedMaxWorkingSetSize)
        _stats.processMemoryUsed = Self.currentProcessMemoryFootprint()
        statsLock.unlock()

        sampleCPUUsage(now: currentTime)

        if lastCaptureTimestamp > 0 {
            let interval = currentTime - lastCaptureTimestamp
            if interval > 0 {
                // Time-constant filter rather than a fixed blend: the weight of
                // one sample is its own share of the measurement window, so the
                // estimate settles in the same wall-clock time at 5 fps as at 240.
                let alpha = min(1, interval / Self.measurementWindow)
                captureRateLock.lock()
                _estimatedCaptureInterval += (interval - _estimatedCaptureInterval) * alpha
                captureRateLock.unlock()
            }
        }
        lastCaptureTimestamp = currentTime

        let elapsed = currentTime - fpsStartTime
        if elapsed >= 1.0 {
            statsLock.lock()
            _stats.captureFPS = Float(frameCount) / Float(elapsed)
            statsLock.unlock()
            frameCount = 0
            fpsStartTime = currentTime
            DispatchQueue.main.async { [weak self] in
                guard let self else { return }
                self.applyFrameRatePreference(self.desiredOutputFPS(self.config))
            }
        }

        let delta = currentTime - lastFrameTime
        statsLock.lock()
        if lastFrameTime > 0 {
            _stats.frameTime = Float(delta * 1000.0)
        }
        if let captureTimestamp, captureTimestamp > 0 {
            _stats.captureLatency = Float((currentTime - captureTimestamp) * 1000.0)
        } else {
            _stats.captureLatency = Float(delta * 1000.0)
        }
        statsLock.unlock()
        lastFrameTime = currentTime
    }

    private static func currentProcessMemoryFootprint() -> UInt64 {
        var info = task_vm_info_data_t()
        var count = mach_msg_type_number_t(MemoryLayout<task_vm_info_data_t>.size / MemoryLayout<integer_t>.size)
        let result = withUnsafeMutablePointer(to: &info) {
            $0.withMemoryRebound(to: integer_t.self, capacity: Int(count)) {
                task_info(mach_task_self_, task_flavor_t(TASK_VM_INFO), $0, &count)
            }
        }
        guard result == KERN_SUCCESS else { return 0 }
        return info.phys_footprint
    }

    private func sampleCPUUsage(now: CFTimeInterval) {
        if lastCPUSampleTime > 0, (now - lastCPUSampleTime) < Self.measurementWindow { return }

        var usage = rusage_info_current()
        let result = withUnsafeMutablePointer(to: &usage) { ptr -> Int32 in
            ptr.withMemoryRebound(to: rusage_info_t?.self, capacity: 1) {
                proc_pid_rusage(getpid(), RUSAGE_INFO_CURRENT, $0)
            }
        }
        guard result == 0 else { return }

        let cpuNanos = usage.ri_user_time + usage.ri_system_time
        if lastCPUSampleTime > 0 {
            let wall = now - lastCPUSampleTime
            if wall > 0, cpuNanos >= lastCPUTimeNanos {
                let cpuDelta = Double(cpuNanos - lastCPUTimeNanos)
                let pct = (cpuDelta / 1_000_000_000.0) / wall * 100.0
                statsLock.lock()
                _stats.cpuUsage = Float(pct)
                statsLock.unlock()
            }
        }
        lastCPUSampleTime = now
        lastCPUTimeNanos = cpuNanos
    }

    private func processSurface(_ surface: IOSurfaceRef, pixelBuffer: CVPixelBuffer, timestamp: CFTimeInterval, isSceneCut: Bool) {
        let w = IOSurfaceGetWidth(surface)
        let h = IOSurfaceGetHeight(surface)

        let desc = MTLTextureDescriptor.texture2DDescriptor(
            pixelFormat: .bgra8Unorm,
            width: w,
            height: h,
            mipmapped: false
        )
        desc.usage = [.shaderRead]
        guard let inputTex = device.makeTexture(descriptor: desc, iosurface: surface, plane: 0) else {
            reportError("Error Code: MG-ENG-008 IOSurface texture creation failed")
            inFlightSemaphore.signal()
            return
        }

        processCapturedTexture(inputTex, pixelBuffer: pixelBuffer, timestamp: timestamp, isSceneCut: isSceneCut)
    }

    private func processCapturedTexture(_ inputTex: MTLTexture, pixelBuffer: CVPixelBuffer, timestamp: CFTimeInterval, isSceneCut: Bool) {
        guard let commandBuffer = commandQueue.makeCommandBuffer() else {
            inFlightSemaphore.signal()
            return
        }

        nonisolated(unsafe) let retainedPixelBuffer = pixelBuffer
        commandBuffer.addCompletedHandler { [weak self] _ in
            _ = retainedPixelBuffer
            self?.inFlightSemaphore.signal()
        }

        let config = self.config

        // ScreenCaptureKit now delivers the render resolution directly and the
        // spatial upscale runs once per presented frame, so the capture path
        // only sharpens and anti-aliases before the frame enters the ring.
        // Render scale reduces the capture, but frame generation must not inherit
        // that reduction: interpolation and the motion field would then work on a
        // fraction of the pixels and smear. Bring the frame back to the window's
        // native size first, exactly where the ring used to sit.
        let native = windowCaptureManager?.nativePixelSize ?? .zero
        let restoreActive = config.scalingType == .mgup1
            && native.width >= CGFloat(inputTex.width) + 1
            && native.height >= CGFloat(inputTex.height) + 1

        let width = restoreActive ? Int(native.width) : inputTex.width
        let height = restoreActive ? Int(native.height) : inputTex.height

        let size = CGSize(width: width, height: height)
        if size != lastProcessedSize {
            resetProcessingState(clearFrames: true)
            lastProcessedSize = size
        }

        let sharpness = config.scalingType == .mgup1 ? config.qualityProfile.sharpnessScale : 0
        let casActive = sharpness > 0.01
        let aaActive = config.aaMode != .off

        // Every scratch texture is reused next frame, so the LAST active stage
        // writes straight into the ring slot; a passthrough config needs a copy.
        let slot = historyTextureIndex % historyTextures.count
        historyTextureIndex += 1
        guard let historyTex = ensureTexture(&historyTextures[slot], width: width, height: height,
                                             usage: [.shaderRead, .shaderWrite, .renderTarget]) else {
            statsLock.lock()
            _stats.droppedFrames += 1
            statsLock.unlock()
            commandBuffer.commit()
            return
        }

        var workingTex = inputTex

        if restoreActive {
            let destination: MTLTexture
            if casActive || aaActive {
                guard let scratch = ensureTexture(&captureUpscaledTexture, width: width, height: height,
                                                  usage: [.shaderRead, .shaderWrite, .renderTarget]) else {
                    commandBuffer.commit()
                    return
                }
                destination = scratch
            } else {
                destination = historyTex
            }
            guard let scaler = ensureSpatialScaler(&captureScaler,
                                                   inputWidth: inputTex.width, inputHeight: inputTex.height,
                                                   outputWidth: width, outputHeight: height) else {
                commandBuffer.commit()
                return
            }
            scaler.colorTexture = inputTex
            scaler.outputTexture = destination
            scaler.encode(commandBuffer: commandBuffer)
            workingTex = destination
        }

        if casActive {
            let casDest: MTLTexture
            if aaActive {
                guard let scratch = ensureTexture(&casTexture, width: width, height: height) else {
                    reportError("Error Code: MG-ENG-007 CAS pipeline unavailable")
                    commandBuffer.commit()
                    return
                }
                casDest = scratch
            } else {
                casDest = historyTex
            }

            guard let casPipeline = casPipeline,
                  let encoder = commandBuffer.makeComputeCommandEncoder() else {
                reportError("Error Code: MG-ENG-007 CAS pipeline unavailable")
                commandBuffer.commit()
                return
            }
            var params = SharpenParams(sharpness: sharpness)
            encoder.setComputePipelineState(casPipeline)
            encoder.setTexture(workingTex, index: 0)
            encoder.setTexture(casDest, index: 1)
            encoder.setBytes(&params, length: MemoryLayout<SharpenParams>.size, index: 0)
            dispatchThreads(pipeline: casPipeline, encoder: encoder, width: width, height: height)
            encoder.endEncoding()
            workingTex = casDest
        }

        if aaActive {
            guard encodeAntiAliasing(from: workingTex, to: historyTex,
                                     mode: config.aaMode, profile: config.qualityProfile,
                                     commandBuffer: commandBuffer) else {
                statsLock.lock()
                _stats.droppedFrames += 1
                statsLock.unlock()
                commandBuffer.commit()
                return
            }
        } else if !casActive && !restoreActive {
            guard encodeCopy(from: workingTex, to: historyTex, commandBuffer: commandBuffer) else {
                statsLock.lock()
                _stats.droppedFrames += 1
                statsLock.unlock()
                commandBuffer.commit()
                return
            }
        }

        commandBuffer.addCompletedHandler { [weak self] buffer in
            guard let self = self else { return }
            let gpuTime = buffer.gpuEndTime - buffer.gpuStartTime
            self.statsLock.lock()
            self._stats.captureGPUTime = Float(gpuTime * 1000.0)
            self.statsLock.unlock()
        }
        commandBuffer.commit()

        let motionTexture = config.frameGenMode == .extrapolation
            ? encodeMotion(from: historyTex, width: width, height: height)
            : nil

        frameBuffer.push(FrameHistory(texture: historyTex, timestamp: timestamp,
                                      isSceneCut: isSceneCut, motion: motionTexture))
    }

    /// Converts the frame to luma, hands it to the media engine, and copies the
    /// resulting vectors into a slot we own — VideoToolbox recycles its own
    /// buffers, and the ring holds each field until the frame leaves it.
    private func encodeMotion(from source: MTLTexture, width: Int, height: Int) -> MTLTexture? {
        guard let lumaPipeline,
              let lumaTarget = motionEstimator.prepare(width: width, height: height),
              let lumaBuffer = commandQueue.makeCommandBuffer(),
              let encoder = lumaBuffer.makeComputeCommandEncoder() else { return latestMotion() }

        encoder.setComputePipelineState(lumaPipeline)
        encoder.setTexture(source, index: 0)
        encoder.setTexture(lumaTarget, index: 1)
        dispatchThreads(pipeline: lumaPipeline, encoder: encoder, width: width, height: height)
        encoder.endEncoding()

        // VideoToolbox reads the IOSurface outside Metal's ordering, so the
        // estimate has to follow the write — but waiting for it here would
        // serialise the capture pipeline, so it runs on the completion handler
        // and the result is picked up by the next frame instead.
        lumaBuffer.addCompletedHandler { [weak self] _ in
            guard let self else { return }
            self.processingQueue.async {
                guard let vectors = self.motionEstimator.estimate(),
                      let field = self.motionEstimator.texture(for: vectors) else { return }
                self.storeMotion(field)
            }
        }
        lumaBuffer.commit()

        return latestMotion()
    }

    /// Copies the estimator's output into a slot we own, since VideoToolbox
    /// recycles its buffers while the ring still references the field.
    private func storeMotion(_ field: MTLTexture) {
        let slot = motionTextureIndex % motionTextures.count
        motionTextureIndex += 1
        guard let destination = ensureTexture(&motionTextures[slot],
                                              width: field.width, height: field.height,
                                              pixelFormat: .rg16Float,
                                              usage: [.shaderRead, .shaderWrite]),
              let blitBuffer = commandQueue.makeCommandBuffer(),
              let blit = blitBuffer.makeBlitCommandEncoder() else { return }

        blit.copy(from: field, sourceSlice: 0, sourceLevel: 0,
                  sourceOrigin: MTLOrigin(x: 0, y: 0, z: 0),
                  sourceSize: MTLSize(width: field.width, height: field.height, depth: 1),
                  to: destination, destinationSlice: 0, destinationLevel: 0,
                  destinationOrigin: MTLOrigin(x: 0, y: 0, z: 0))
        blit.endEncoding()
        blitBuffer.commit()

        motionLock.lock()
        _latestMotion = destination
        motionLock.unlock()
    }

    private func latestMotion() -> MTLTexture? {
        motionLock.lock()
        defer { motionLock.unlock() }
        return _latestMotion
    }

    private struct SharpenParams {
        var sharpness: Float
    }

    private struct AntiAliasParams {
        var threshold: Float
        var maxSearchSteps: Int32
    }


    private func encodeAntiAliasing(from input: MTLTexture,
                                    to output: MTLTexture,
                                    mode: CaptureSettings.AAMode,
                                    profile: QualityProfile,
                                    commandBuffer: MTLCommandBuffer) -> Bool {
        let width = output.width
        let height = output.height

        switch mode {
        case .off:
            return encodeCopy(from: input, to: output, commandBuffer: commandBuffer)

        case .fxaa:
            guard let fxaaPipeline = fxaaPipeline,
                  let encoder = commandBuffer.makeComputeCommandEncoder() else {
                reportError("Error Code: MG-ENG-005 Anti-aliasing pipeline unavailable (FXAA)")
                return false
            }
            var threshold = profile.aaThreshold
            encoder.setComputePipelineState(fxaaPipeline)
            encoder.setTexture(input, index: 0)
            encoder.setTexture(output, index: 1)
            encoder.setBytes(&threshold, length: MemoryLayout<Float>.size, index: 0)
            dispatchThreads(pipeline: fxaaPipeline, encoder: encoder, width: width, height: height)
            encoder.endEncoding()
            return true

        case .smaa:
            guard let edgePipe = smaaEdgePipeline,
                  let weightPipe = smaaWeightPipeline,
                  let blendPipe = smaaBlendPipeline,
                  let edges = ensureTexture(&smaaEdgeTexture, width: width, height: height),
                  let weights = ensureTexture(&smaaWeightTexture, width: width, height: height),
                  let encoder = commandBuffer.makeComputeCommandEncoder() else {
                reportError("Error Code: MG-ENG-005 Anti-aliasing pipeline unavailable (SMAA)")
                return false
            }
            var params = AntiAliasParams(
                threshold: profile.aaThreshold,
                maxSearchSteps: Int32(profile.smaaSearchSteps)
            )
            encoder.setComputePipelineState(edgePipe)
            encoder.setTexture(input, index: 0)
            encoder.setTexture(edges, index: 1)
            encoder.setBytes(&params, length: MemoryLayout<AntiAliasParams>.size, index: 0)
            dispatchThreads(pipeline: edgePipe, encoder: encoder, width: width, height: height)

            encoder.setComputePipelineState(weightPipe)
            encoder.setTexture(edges, index: 0)
            encoder.setTexture(weights, index: 1)
            encoder.setBytes(&params, length: MemoryLayout<AntiAliasParams>.size, index: 0)
            dispatchThreads(pipeline: weightPipe, encoder: encoder, width: width, height: height)

            encoder.setComputePipelineState(blendPipe)
            encoder.setTexture(input, index: 0)
            encoder.setTexture(weights, index: 1)
            encoder.setTexture(output, index: 2)
            dispatchThreads(pipeline: blendPipe, encoder: encoder, width: width, height: height)
            encoder.endEncoding()
            return true
        }
    }

    @MainActor
    private func interpolateFrame(prev: FrameHistory,
                                  next: FrameHistory,
                                  mode: CaptureSettings.FrameGenMode,
                                  commandBuffer: MTLCommandBuffer) -> (texture: MTLTexture, isNew: Bool)? {
        let prevTex = prev.texture
        let nextTex = next.texture
        guard prevTex.width == nextTex.width, prevTex.height == nextTex.height else {
            return nil
        }

        guard mode == .interpolation else { return nil }

        do {
            guard let output = ensureTexture(&interpolatedTexture, width: prevTex.width, height: prevTex.height,
                                             usage: [.shaderRead, .shaderWrite, .renderTarget]) else { return nil }

            if cachedInterpPrevTimestamp == prev.timestamp,
               cachedInterpNextTimestamp == next.timestamp,
               output.width == prevTex.width, output.height == prevTex.height {
                return (output, false)
            }

            guard let interpolator = ensureFrameInterpolator(width: prevTex.width, height: prevTex.height) else {
                return nil
            }

            let shouldResetHistory = frameInterpolatorNeedsHistoryReset
            frameInterpolatorNeedsHistoryReset = false

            guard prevTex.width == output.width, prevTex.height == output.height,
                  nextTex.width == output.width, nextTex.height == output.height,
                  let depthTex = ensureFlatDepthTexture(width: prevTex.width, height: prevTex.height) else {
                return nil
            }

            interpolator.colorTexture = nextTex
            interpolator.prevColorTexture = prevTex
            interpolator.outputTexture = output
            interpolator.depthTexture = depthTex
            interpolator.motionTexture = nil
            interpolator.isDepthReversed = false
            interpolator.nearPlane = 0.1
            interpolator.farPlane = 1000.0
            interpolator.fieldOfView = 1.0
            interpolator.aspectRatio = Float(prevTex.width) / Float(max(1, prevTex.height))
            interpolator.deltaTime = Float(max(0.0001, next.timestamp - prev.timestamp))
            interpolator.shouldResetHistory = shouldResetHistory
            interpolator.encode(commandBuffer: commandBuffer)

            cachedInterpPrevTimestamp = prev.timestamp
            cachedInterpNextTimestamp = next.timestamp
            return (output, true)
        }
    }

    /// Warps the newest frame forward along its motion field. `phase` is how far
    /// into the next capture interval we are, so 0 reproduces the source frame.
    @MainActor
    /// `isNew` distinguishes a warp that was actually encoded from one served out
    /// of the cache, which is what makes the generated-image rate countable: the
    /// panel presents each phase more than once at high refresh rates, and
    /// counting those presents overstates how much new information is on screen.
    private func encodeExtrapolation(source: MTLTexture, motion: MTLTexture, phase: Float,
                                     sourceTimestamp: CFTimeInterval, step: Int,
                                     commandBuffer: MTLCommandBuffer) -> (texture: MTLTexture, isNew: Bool)? {
        guard let extrapolatePipeline,
              let output = ensureTexture(&extrapolatedTexture,
                                         width: source.width, height: source.height,
                                         usage: [.shaderRead, .shaderWrite, .renderTarget]) else { return nil }

        // The warp for a given capture and phase step is the same image every
        // time it is asked for, so it is encoded once and held. Without this the
        // pipeline re-warped for every present, paying full cost for an image it
        // had already produced.
        if cachedExtrapSourceTimestamp == sourceTimestamp, cachedExtrapStep == step,
           output.width == source.width, output.height == source.height {
            return (output, false)
        }

        guard let encoder = commandBuffer.makeComputeCommandEncoder() else { return nil }

        // Everything the warp is allowed to do is expressed in blocks, and the
        // block size comes from the field the estimator actually returned rather
        // than a fraction of the frame — VideoToolbox picks the grid, and it does
        // not have to be the 16x16 default at every resolution.
        var value = phase
        var blockSize = Float(source.width) / Float(max(1, motion.width))
        encoder.setComputePipelineState(extrapolatePipeline)
        encoder.setTexture(source, index: 0)
        encoder.setTexture(motion, index: 1)
        encoder.setTexture(output, index: 2)
        encoder.setBytes(&value, length: MemoryLayout<Float>.size, index: 0)
        encoder.setBytes(&blockSize, length: MemoryLayout<Float>.size, index: 1)
        dispatchThreads(pipeline: extrapolatePipeline, encoder: encoder,
                        width: source.width, height: source.height)
        encoder.endEncoding()
        cachedExtrapSourceTimestamp = sourceTimestamp
        cachedExtrapStep = step
        return (output, true)
    }


    func updateSettings(_ settings: CaptureSettings) {
        var next = EngineConfig()
        next.scalingType = settings.scalingType
        next.aaMode = settings.aaMode
        next.qualityProfile = settings.qualityMode.profile
        next.frameGenMode = settings.frameGenMode
        next.frameGenEnabled = settings.frameGenMode != .off
        next.frameGenMultiplier = settings.effectiveFrameGenMultiplier
        next.vsyncEnabled = settings.vsync
        next.captureCursorEnabled = settings.captureCursor
        next.bufferDepth = max(2, min(Self.maxInFlight, settings.bufferCount))

        configLock.lock()
        let previous = _config
        _config = next
        configLock.unlock()

        // Only the stages that change which textures the capture path writes
        // need the scratch pool rebuilt. Scale Factor lives in the overlay's
        // geometry and the quality profile is read per frame, so neither does.
        let pipelineChanged =
            next.scalingType != previous.scalingType ||
            next.aaMode != previous.aaMode

        if pipelineChanged {
            resetProcessingStateAsync(clearFrames: true)
        }

        if next.bufferDepth != previous.bufferDepth {
            applyBufferDepth(next.bufferDepth)
            let view = mtkView
            let drawableCount = min(max(2, next.bufferDepth), Self.maxInFlight)
            DispatchQueue.main.async {
                (view?.layer as? CAMetalLayer)?.maximumDrawableCount = drawableCount
            }
        }

        applyFrameRatePreference(desiredOutputFPS(next))
        if let view = mtkView {
            applyDisplaySync(to: view, vsync: next.vsyncEnabled)
        }

        // The cumulative counters describe one mode's behaviour. Carrying them
        // across a mode switch mixes two schedules into one set of totals and
        // the generated/passthrough split stops meaning anything.
        // A multiplier change reshapes the same split, so it invalidates the
        // totals exactly the way a mode change does.
        if next.frameGenMode != previous.frameGenMode
            || next.frameGenMultiplier != previous.frameGenMultiplier {
            statsLock.lock()
            _stats.droppedFrames = 0
            _stats.frameCount = 0
            _stats.outputFrameCount = 0
            _stats.interpolatedFrameCount = 0
            _stats.passthroughFrameCount = 0
            _stats.generatedFrameCount = 0
            statsLock.unlock()
            lastPresentedCaptureTimestamp = -1
        }
    }

    func startCaptureFromWindow(_ manager: WindowCaptureManager) async {
        resetProcessingStateAsync(clearFrames: true)
        resetFrameCounters()
        resetErrorReporting()
        applyBufferDepth(config.bufferDepth)
        self.windowCaptureManager = manager
        
        manager.onFrameReceived = { [weak self] surface, pixelBuffer, timestamp, isSceneCut in
            nonisolated(unsafe) let surface = surface
            nonisolated(unsafe) let pixelBuffer = pixelBuffer
            guard let self else { return }
            self.processingQueue.async {
                self.processIOSurfaceFrame(surface: surface, pixelBuffer: pixelBuffer, timestamp: timestamp, isSceneCut: isSceneCut)
            }
        }
        
        self.frameCount = 0
        self.fpsStartTime = CACurrentMediaTime()
        self.lastCaptureTimestamp = 0
        self.lastCPUSampleTime = 0
        self.lastCPUTimeNanos = 0
    }
    
    private func processIOSurfaceFrame(surface: IOSurfaceRef, pixelBuffer: CVPixelBuffer, timestamp: Double, isSceneCut: Bool) {
        inFlightSemaphore.wait()
        let currentTime = CACurrentMediaTime()
        updateCaptureStats(currentTime: currentTime, captureTimestamp: timestamp)
        processSurface(surface, pixelBuffer: pixelBuffer, timestamp: currentTime, isSceneCut: isSceneCut)
    }
    
    func stopCapture() {
        windowCaptureManager?.onFrameReceived = nil
        windowCaptureManager = nil
        lastCaptureTimestamp = 0
        // The ring holds a texture from the pool for every entry; leaving them
        // there keeps the whole pool alive until the next capture overwrites it.
        resetProcessingStateAsync(clearFrames: true)
    }
    
    /// Nothing to do: the upscale target is derived from the live drawable every
    /// frame, and `ensureTexture` / `ensureMetalFXSpatialScaler` rebuild on any
    /// size change on their own.
    nonisolated func mtkView(_ view: MTKView, drawableSizeWillChange size: CGSize) {}
}
