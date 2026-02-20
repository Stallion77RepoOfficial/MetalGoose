import Foundation
import AppKit
@preconcurrency import Metal
@preconcurrency import MetalKit
@preconcurrency import MetalFX
import IOSurface
import QuartzCore
import os
import Vision
import CoreVideo

struct PipelineStats: @unchecked Sendable {
    var captureFPS: Float = 0
    var outputFPS: Float = 0
    var interpolatedFPS: Float = 0
    var frameTime: Float = 0
    var gpuTime: Float = 0
    var captureLatency: Float = 0
    var frameCount: UInt64 = 0
    var outputFrameCount: UInt64 = 0
    var droppedFrames: UInt64 = 0
    var interpolatedFrameCount: UInt64 = 0
    var passthroughFrameCount: UInt64 = 0
    var gpuMemoryUsed: UInt64 = 0
    var gpuMemoryTotal: UInt64 = 0
    var isUsingVirtualDisplay: Bool = false
    var virtualResolution: CGSize = .zero
    var outputResolution: CGSize = .zero
}

@available(macOS 26.0, *)
final class GooseEngine: NSObject, ObservableObject, MTKViewDelegate, @unchecked Sendable {
    
    // MARK: - Thread-safe published state
    @Published private(set) var isCapturing: Bool = false
    @Published private(set) var lastError: String?
    
    // Stats are read/written from processing queue, published to main for UI
    private var _stats = PipelineStats()
    private var statsLock = os_unfair_lock()
    var stats: PipelineStats {
        os_unfair_lock_lock(&statsLock)
        defer { os_unfair_lock_unlock(&statsLock) }
        return _stats
    }
    
    // Dedicated GPU processing queue — all frame work happens here
    private let processingQueue = DispatchQueue(label: "com.metalgoose.processing", qos: .userInteractive)
    
    // Triple buffering: allow up to 3 frames in-flight simultaneously
    private let inFlightSemaphore = DispatchSemaphore(value: 3)
    
    // IOSurface texture pool — avoid per-frame MTLTexture allocation
    private var ioSurfaceTexturePool: [MTLTexture?] = [nil, nil, nil]
    private var ioSurfacePoolIndex: Int = 0
    private var ioSurfacePoolSize: (width: Int, height: Int) = (0, 0)
    
    var deviceName: String { device.name }
    
    var onFrameReady: ((MTLTexture) -> Void)?
    private let device: MTLDevice
    private let commandQueue: MTLCommandQueue
    
    // MetalFX Spatial Scaler
    private var spatialScaler: MTLFXSpatialScaler?
    private var spatialScalerInputSize: (width: Int, height: Int) = (0, 0)
    private var spatialScalerOutputSize: (width: Int, height: Int) = (0, 0)
    
    // Retained pipelines
    private var scalePipeline: MTLComputePipelineState?
    private var casPipeline: MTLComputePipelineState?
    private var fxaaPipeline: MTLComputePipelineState?
    private var smaaEdgePipeline: MTLComputePipelineState?
    private var smaaWeightPipeline: MTLComputePipelineState?
    private var smaaBlendPipeline: MTLComputePipelineState?
    private var msaaPipeline: MTLComputePipelineState?
    private var temporalPipeline: MTLComputePipelineState?
    private var copyPipeline: MTLComputePipelineState?

    // Optical flow / frame generation pipelines
    private var flowWarpPipeline: MTLComputePipelineState?
    private var flowComposePipeline: MTLComputePipelineState?
    private var flowOcclusionPipeline: MTLComputePipelineState?
    
    private var renderPipeline: MTLRenderPipelineState?
    private weak var mtkView: MTKView?
    
    private var renderTexture: MTLTexture?
    private var scaledTexture: MTLTexture?
    private var casTexture: MTLTexture?
    private var usmTexture: MTLTexture?
    private var fxaaTexture: MTLTexture?
    private var smaaEdgeTexture: MTLTexture?
    private var smaaWeightTexture: MTLTexture?
    private var smaaOutputTexture: MTLTexture?
    private var msaaTexture: MTLTexture?
    private var taaHistoryTexture: MTLTexture?
    private var taaOutputTexture: MTLTexture?
    
    // Vision framework optical flow (ANE/GPU-powered)
    // Managed by VisionFlowProvider class — no instance variables needed here
    
    private var occlusionTexture: MTLTexture?
    private var warpedPrevTexture: MTLTexture?
    private var warpedNextTexture: MTLTexture?
    
    struct FrameHistory {
        let texture: MTLTexture
        let timestamp: CFTimeInterval
        let flowFromPrev: MTLTexture?
        let flowToPrev: MTLTexture?
    }
    
    private final class FrameRingBuffer: @unchecked Sendable {
        private var buffer: [FrameHistory] = []
        private let capacity = 4
        private let lock = NSLock()
        
        func push(_ frame: FrameHistory) {
            lock.lock()
            defer { lock.unlock() }
            buffer.append(frame)
            if buffer.count > capacity {
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
        
        var count: Int {
            lock.lock()
            defer { lock.unlock() }
            return buffer.count
        }
        
        var newestFrame: FrameHistory? {
            lock.lock()
            defer { lock.unlock() }
            return buffer.last
        }

        func clear() {
            lock.lock()
            defer { lock.unlock() }
            buffer.removeAll()
        }
    }
    
    private let frameBuffer = FrameRingBuffer()
    private var blendTexture: MTLTexture?
    private var hasTAAHistory: Bool = false
    private var lastProcessedSize: CGSize = .zero

    private var scalingType: CaptureSettings.ScalingType = .off
    private var qualityMode: CaptureSettings.QualityMode = .balanced
    private var aaMode: CaptureSettings.AAMode = .off
    private var renderScaleFactor: Float = 1.0
    private var scaleFactor: Float = 1.0
    private var sharpness: Float = 0.5
    private var temporalBlend: Float = 0.1
    private var motionScale: Float = 1.0
    private var captureCursor: Bool = true
    private var frameGenEnabled: Bool = false
    private var frameGenMode: CaptureSettings.FrameGenMode = .off
    private var frameGenType: CaptureSettings.FrameGenType = .adaptive
    private var targetFPS: Int = 120
    private var frameGenMultiplier: Int = 2
    private var adaptiveSync: Bool = true
    private var vsyncEnabled: Bool = true
    private var qualityProfile: QualityProfile = CaptureSettings.QualityMode.balanced.profile
    
    private let flowOcclusionThreshold: Float = 1.5

    private var estimatedCaptureInterval: Double = 0
    private var lastCaptureTimestamp: CFTimeInterval = 0
    private var lastPreferredFPS: Int = 0
    private var lastPreferredUpdateTime: CFTimeInterval = 0

    private var virtualDisplayManager: VirtualDisplayManager?
    private var captureRefreshRate: Int = 0
    
    private var lastFrameTime: CFTimeInterval = 0
    private var frameCount: Int = 0
    private var fpsStartTime: CFTimeInterval = 0
    
    private var outputSize: CGSize = .zero
    private var currentRefreshRate: Int = 0
    
    override init() {
        guard let dev = MTLCreateSystemDefaultDevice() else {
            fatalError("Error Code: MG-ENG-002 Metal device not available")
        }
        guard let queue = dev.makeCommandQueue() else {
            fatalError("Error Code: MG-ENG-003 Metal command queue not available")
        }
        
        self.device = dev
        self.commandQueue = queue
        
        super.init()
        setupPipelines()
    }
    
    private func setupPipelines() {
        guard let library = device.makeDefaultLibrary() else { return }
        
        func makeCompute(_ name: String) -> MTLComputePipelineState? {
            guard let function = library.makeFunction(name: name) else { return nil }
            return try? device.makeComputePipelineState(function: function)
        }

        // Core pipelines
        scalePipeline = makeCompute("blitScaleBilinear")
        casPipeline = makeCompute("contrastAdaptiveSharpening")
        copyPipeline = makeCompute("copyTexture")
        
        // Anti-aliasing pipelines
        fxaaPipeline = makeCompute("fxaa")
        smaaEdgePipeline = makeCompute("smaaEdgeDetection")
        smaaWeightPipeline = makeCompute("smaaBlendingWeights")
        smaaBlendPipeline = makeCompute("smaaBlend")
        msaaPipeline = makeCompute("msaa")
        temporalPipeline = makeCompute("temporalReproject")

        // Optical flow / frame generation pipelines
        flowWarpPipeline = makeCompute("flowWarp")
        flowComposePipeline = makeCompute("flowCompose")
        flowOcclusionPipeline = makeCompute("flowOcclusion")
        
        // Async Vision optical flow provider
        visionFlowProvider = VisionFlowProvider(device: device, commandQueue: commandQueue)

        // Render pipeline
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
            lastError = "Error Code: MG-ENG-001 Pipeline setup failed: \(error)"
        }
        
    }
    
    private func ensureMetalFXSpatialScaler(inputWidth: Int, inputHeight: Int,
                                            outputWidth: Int, outputHeight: Int) -> MTLFXSpatialScaler? {
        // Reuse existing scaler if dimensions match
        if let scaler = spatialScaler,
           spatialScalerInputSize == (inputWidth, inputHeight),
           spatialScalerOutputSize == (outputWidth, outputHeight) {
            return scaler
        }
        
        // Create new scaler
        let descriptor = MTLFXSpatialScalerDescriptor()
        descriptor.inputWidth = inputWidth
        descriptor.inputHeight = inputHeight
        descriptor.outputWidth = outputWidth
        descriptor.outputHeight = outputHeight
        descriptor.colorTextureFormat = .bgra8Unorm
        descriptor.outputTextureFormat = .bgra8Unorm
        descriptor.colorProcessingMode = .perceptual
        
        guard let scaler = descriptor.makeSpatialScaler(device: device) else {
            lastError = "Error Code: MG-ENG-004 MetalFX Spatial Scaler creation failed"
            return nil
        }
        
        spatialScaler = scaler
        spatialScalerInputSize = (inputWidth, inputHeight)
        spatialScalerOutputSize = (outputWidth, outputHeight)
        
        return scaler
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

    private func makeFlowTexture(width: Int, height: Int) -> MTLTexture? {
        let desc = MTLTextureDescriptor.texture2DDescriptor(
            pixelFormat: .rg16Float,
            width: width,
            height: height,
            mipmapped: false
        )
        desc.usage = [.shaderRead, .shaderWrite]
        desc.storageMode = .private
        return device.makeTexture(descriptor: desc)
    }

    private func encodeCopy(from input: MTLTexture,
                            to output: MTLTexture,
                            commandBuffer: MTLCommandBuffer) -> Bool {
        guard let copyPipeline = copyPipeline,
              let encoder = commandBuffer.makeComputeCommandEncoder() else {
            lastError = "Error Code: MG-ENG-011 Optical flow pipeline unavailable"
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
    
    // MARK: - Async Vision Optical Flow Provider
    
    /// Async flow provider: runs VNGenerateOpticalFlowRequest on a dedicated queue.
    /// The frame pipeline submits frame pairs and queries the latest available result.
    private final class VisionFlowProvider {
        private let device: MTLDevice
        private let commandQueue: MTLCommandQueue
        private let flowQueue = DispatchQueue(label: "com.metalgoose.visionflow", qos: .userInitiated)
        private var textureCache: CVMetalTextureCache?
        
        // Latest computed flow pair (protected by lock)
        private var lock = os_unfair_lock()
        private var _forwardFlow: MTLTexture?
        private var _backwardFlow: MTLTexture?
        private var _flowFrameID: UInt64 = 0
        private var isComputing = false
        
        // Track submitted frame ID to avoid redundant computation
        private var submittedFrameID: UInt64 = 0
        
        struct FlowResult {
            let forward: MTLTexture
            let backward: MTLTexture
            let frameID: UInt64
        }
        
        init(device: MTLDevice, commandQueue: MTLCommandQueue) {
            self.device = device
            self.commandQueue = commandQueue
            var cache: CVMetalTextureCache?
            CVMetalTextureCacheCreate(kCFAllocatorDefault, nil, device, nil, &cache)
            self.textureCache = cache
        }
        
        /// Returns the latest computed flow pair, or nil if not yet available
        func latestFlow() -> FlowResult? {
            os_unfair_lock_lock(&lock)
            defer { os_unfair_lock_unlock(&lock) }
            guard let fwd = _forwardFlow, let bwd = _backwardFlow else { return nil }
            return FlowResult(forward: fwd, backward: bwd, frameID: _flowFrameID)
        }
        
        /// Submit a frame pair for async flow computation (non-blocking)
        func submitFlowRequest(prev: MTLTexture, next: MTLTexture, frameID: UInt64) {
            os_unfair_lock_lock(&lock)
            let alreadyComputing = isComputing
            os_unfair_lock_unlock(&lock)
            
            // Skip if already computing — we'll pick up the next frame
            if alreadyComputing { return }
            
            os_unfair_lock_lock(&lock)
            isComputing = true
            os_unfair_lock_unlock(&lock)
            
            // Read texture data to CPU-accessible buffers on the current thread
            // (this is fast — just creates IOSurface-backed pixel buffers and blits)
            guard let prevPB = createPixelBuffer(width: prev.width, height: prev.height),
                  let nextPB = createPixelBuffer(width: next.width, height: next.height),
                  let blitBuf = commandQueue.makeCommandBuffer() else {
                os_unfair_lock_lock(&lock)
                isComputing = false
                os_unfair_lock_unlock(&lock)
                return
            }
            
            blitTexture(prev, to: prevPB, commandBuffer: blitBuf)
            blitTexture(next, to: nextPB, commandBuffer: blitBuf)
            blitBuf.commit()
            
            // Run Vision flow async on dedicated queue
            let width = prev.width
            let height = prev.height
            flowQueue.async { [weak self] in
                guard let self = self else { return }
                defer {
                    os_unfair_lock_lock(&self.lock)
                    self.isComputing = false
                    os_unfair_lock_unlock(&self.lock)
                }
                
                // Wait for blit to finish
                blitBuf.waitUntilCompleted()
                
                // Forward flow: prev → next
                let forwardFlow = self.runVisionFlow(source: prevPB, target: nextPB, width: width, height: height)
                // Backward flow: next → prev
                let backwardFlow = self.runVisionFlow(source: nextPB, target: prevPB, width: width, height: height)
                
                guard let fwd = forwardFlow, let bwd = backwardFlow else { return }
                
                // Store results
                os_unfair_lock_lock(&self.lock)
                self._forwardFlow = fwd
                self._backwardFlow = bwd
                self._flowFrameID = frameID
                os_unfair_lock_unlock(&self.lock)
            }
        }
        
        func reset() {
            os_unfair_lock_lock(&lock)
            _forwardFlow = nil
            _backwardFlow = nil
            _flowFrameID = 0
            os_unfair_lock_unlock(&lock)
        }
        
        private func runVisionFlow(source: CVPixelBuffer, target: CVPixelBuffer,
                                   width: Int, height: Int) -> MTLTexture? {
            let request = VNGenerateOpticalFlowRequest(targetedCVPixelBuffer: target)
            request.computationAccuracy = .medium
            request.outputPixelFormat = kCVPixelFormatType_TwoComponent16Half
            
            let handler = VNImageRequestHandler(cvPixelBuffer: source, options: [:])
            do {
                try handler.perform([request])
            } catch {
                return nil
            }
            
            guard let observation = request.results?.first as? VNPixelBufferObservation,
                  let cache = textureCache else { return nil }
            
            let flowBuffer = observation.pixelBuffer
            let flowW = CVPixelBufferGetWidth(flowBuffer)
            let flowH = CVPixelBufferGetHeight(flowBuffer)
            
            var cvTex: CVMetalTexture?
            let status = CVMetalTextureCacheCreateTextureFromImage(
                kCFAllocatorDefault, cache, flowBuffer, nil,
                .rg16Float, flowW, flowH, 0, &cvTex
            )
            guard status == kCVReturnSuccess, let tex = cvTex,
                  let flowTexture = CVMetalTextureGetTexture(tex) else { return nil }
            
            // Copy to a persistent GPU texture (CVMetalTexture is transient)
            let desc = MTLTextureDescriptor.texture2DDescriptor(
                pixelFormat: .rg16Float, width: flowW, height: flowH, mipmapped: false
            )
            desc.usage = [.shaderRead, .shaderWrite]
            desc.storageMode = .private
            guard let persistentTex = device.makeTexture(descriptor: desc),
                  let copyBuf = commandQueue.makeCommandBuffer(),
                  let blit = copyBuf.makeBlitCommandEncoder() else { return nil }
            blit.copy(from: flowTexture, to: persistentTex)
            blit.endEncoding()
            copyBuf.commit()
            copyBuf.waitUntilCompleted()
            
            return persistentTex
        }
        
        private func createPixelBuffer(width: Int, height: Int) -> CVPixelBuffer? {
            var pb: CVPixelBuffer?
            let attrs: [String: Any] = [
                kCVPixelBufferMetalCompatibilityKey as String: true,
                kCVPixelBufferIOSurfacePropertiesKey as String: [:] as [String: Any]
            ]
            CVPixelBufferCreate(kCFAllocatorDefault, width, height,
                               kCVPixelFormatType_32BGRA, attrs as CFDictionary, &pb)
            return pb
        }
        
        private func blitTexture(_ texture: MTLTexture, to pixelBuffer: CVPixelBuffer,
                                 commandBuffer: MTLCommandBuffer) {
            guard let surface = CVPixelBufferGetIOSurface(pixelBuffer)?.takeUnretainedValue() else { return }
            let desc = MTLTextureDescriptor.texture2DDescriptor(
                pixelFormat: .bgra8Unorm, width: texture.width, height: texture.height, mipmapped: false
            )
            desc.usage = [.shaderWrite]
            guard let destTex = device.makeTexture(descriptor: desc, iosurface: surface, plane: 0),
                  let blit = commandBuffer.makeBlitCommandEncoder() else { return }
            blit.copy(from: texture, to: destTex)
            blit.endEncoding()
        }
    }
    
    private var visionFlowProvider: VisionFlowProvider?
    private var flowFrameCounter: UInt64 = 0

    private func resetProcessingState(clearFrames: Bool = true) {
        renderTexture = nil
        scaledTexture = nil
        casTexture = nil
        usmTexture = nil
        fxaaTexture = nil
        smaaEdgeTexture = nil
        smaaWeightTexture = nil
        smaaOutputTexture = nil
        msaaTexture = nil
        taaHistoryTexture = nil
        taaOutputTexture = nil
        visionFlowProvider?.reset()
        occlusionTexture = nil
        warpedPrevTexture = nil
        warpedNextTexture = nil
        blendTexture = nil
        hasTAAHistory = false
        lastProcessedSize = .zero
        if clearFrames {
            frameBuffer.clear()
            os_unfair_lock_lock(&statsLock)
            _stats.droppedFrames = 0
            _stats.interpolatedFrameCount = 0
            _stats.passthroughFrameCount = 0
            os_unfair_lock_unlock(&statsLock)
        }
    }

    private func resetFrameCounters() {
        os_unfair_lock_lock(&statsLock)
        _stats.frameCount = 0
        _stats.outputFrameCount = 0
        _stats.interpolatedFrameCount = 0
        _stats.passthroughFrameCount = 0
        _stats.droppedFrames = 0
        _stats.captureFPS = 0
        _stats.outputFPS = 0
        _stats.interpolatedFPS = 0
        os_unfair_lock_unlock(&statsLock)
        frameCount = 0
        renderFrameCount = 0
        interpolatedFrameCount = 0
        fpsStartTime = CACurrentMediaTime()
        renderFPSStartTime = CACurrentMediaTime()
    }

    private func effectiveSharpness() -> Float {
        return sharpness * qualityProfile.sharpnessScale
    }

    private func effectiveTemporalBlend() -> Float {
        return temporalBlend * qualityProfile.temporalBlendScale
    }

    private func desiredOutputFPS() -> Int {
        var target: Int
        let captureFPS: Double = {
            if estimatedCaptureInterval > 0 {
                return 1.0 / estimatedCaptureInterval
            }
            if stats.captureFPS > 0 {
                return Double(stats.captureFPS)
            }
            return 0
        }()
        if frameGenEnabled {
            switch frameGenType {
            case .adaptive:
                let maxGenFPS = captureFPS > 0 ? Int(round(captureFPS * Double(frameGenMultiplier))) : targetFPS
                target = min(targetFPS, maxGenFPS)
            case .fixed:
                let maxGenFPS = captureFPS > 0 ? Int(round(captureFPS * Double(frameGenMultiplier))) : targetFPS
                target = maxGenFPS
            }
        } else if adaptiveSync {
            let capture = Int(round(captureFPS))
            target = capture
        } else {
            target = currentRefreshRate
        }

        if currentRefreshRate > 0 {
            target = min(target, currentRefreshRate)
        }
        return target
    }

    private func applyFrameRatePreference(_ preferred: Int) {
        let target = preferred
        guard target > 0 else { return }
        let now = CACurrentMediaTime()
        if abs(target - lastPreferredFPS) < 3 { return }
        if now - lastPreferredUpdateTime < 0.5 { return }
        if target != lastPreferredFPS {
            mtkView?.preferredFramesPerSecond = target
            lastPreferredFPS = target
            lastPreferredUpdateTime = now
        }
    }
    
    private func applyDisplaySync(to view: MTKView) {
        guard let layer = view.layer as? CAMetalLayer else { return }
        if #available(macOS 10.13, *) {
            layer.displaySyncEnabled = vsyncEnabled
        }
        layer.presentsWithTransaction = false
    }

    private func interpolationDelay(for targetFPS: Int) -> Double {
        guard targetFPS > 0 else { return 0 }
        let outputInterval = 1.0 / Double(targetFPS)
        let captureInterval = estimatedCaptureInterval
        if frameGenEnabled {
            let genInterval = captureInterval / Double(frameGenMultiplier)
            return outputInterval >= genInterval ? outputInterval : genInterval
        }
        return captureInterval >= outputInterval ? captureInterval : outputInterval
    }
    
    func attachToView(_ view: MTKView, refreshRate: Int) {
        view.device = device
        view.delegate = self
        view.preferredFramesPerSecond = refreshRate
        view.isPaused = false
        view.enableSetNeedsDisplay = false
        view.colorPixelFormat = .bgra8Unorm
        view.framebufferOnly = false
        applyDisplaySync(to: view)
        self.mtkView = view
        self.currentRefreshRate = refreshRate
        applyFrameRatePreference(desiredOutputFPS())
    }
    
    func detachFromView() {
        mtkView?.delegate = nil
        mtkView?.isPaused = true
        mtkView = nil
    }
    
    nonisolated func draw(in view: MTKView) {
        renderFrame(in: view)
    }
    
    private var renderFrameCount: Int = 0
    private var renderFPSStartTime: CFTimeInterval = 0
    private var interpolatedFrameCount: Int = 0
    
    private func renderFrame(in view: MTKView) {
        guard let drawable = view.currentDrawable,
              let renderPipeline = renderPipeline,
              let commandBuffer = commandQueue.makeCommandBuffer() else { return }
        
        let currentTime = CACurrentMediaTime()
        
        if renderFPSStartTime == 0 { renderFPSStartTime = currentTime }
        let elapsed = currentTime - renderFPSStartTime

        let targetFPS = desiredOutputFPS()
        applyFrameRatePreference(targetFPS)
        let targetTime = currentTime - interpolationDelay(for: targetFPS)

        var outputTex: MTLTexture?
        let frameGenActive = frameGenEnabled
        var isInterpolated = false

        if frameGenEnabled {
            guard let (prev, next) = frameBuffer.getFramesForTime(targetTime) else {
                os_unfair_lock_lock(&statsLock)
                _stats.droppedFrames += 1
                os_unfair_lock_unlock(&statsLock)
                lastError = "Error Code: MG-ENG-006 Frame interpolation failed: missing frame pair"
                commandBuffer.commit()
                return
            }
            let sameSize = prev.texture.width == next.texture.width &&
                           prev.texture.height == next.texture.height
            guard sameSize else {
                os_unfair_lock_lock(&statsLock)
                _stats.droppedFrames += 1
                os_unfair_lock_unlock(&statsLock)
                lastError = "Error Code: MG-ENG-006 Frame interpolation failed: size mismatch"
                commandBuffer.commit()
                return
            }
            let duration = next.timestamp - prev.timestamp
            let timeSincePrev = targetTime - prev.timestamp
            let ratio = duration > 0 ? (timeSincePrev / duration) : 0
            let t = Float(min(max(ratio, 0), 1))
            guard let interpolated = interpolateFrame(prev: prev, next: next, t: t, commandBuffer: commandBuffer) else {
                os_unfair_lock_lock(&statsLock)
                _stats.droppedFrames += 1
                os_unfair_lock_unlock(&statsLock)
                lastError = "Error Code: MG-ENG-006 Frame interpolation failed: pipeline error"
                commandBuffer.commit()
                return
            }
            outputTex = interpolated
            isInterpolated = true
        } else {
            outputTex = frameBuffer.newestFrame?.texture
        }
        
        guard let finalTex = outputTex else {
            commandBuffer.commit()
            return
        }
        
        renderFrameCount += 1
        if isInterpolated { interpolatedFrameCount += 1 }
        if elapsed >= 1.0 {
            os_unfair_lock_lock(&statsLock)
            _stats.outputFPS = Float(renderFrameCount) / Float(elapsed)
            _stats.interpolatedFPS = Float(interpolatedFrameCount) / Float(elapsed)
            os_unfair_lock_unlock(&statsLock)
            renderFrameCount = 0
            interpolatedFrameCount = 0
            renderFPSStartTime = currentTime
        }
        
        let renderPassDesc = MTLRenderPassDescriptor()
        renderPassDesc.colorAttachments[0].texture = drawable.texture
        renderPassDesc.colorAttachments[0].loadAction = .clear
        renderPassDesc.colorAttachments[0].storeAction = .store
        renderPassDesc.colorAttachments[0].clearColor = MTLClearColor(red: 0, green: 0, blue: 0, alpha: 1)
        
        guard let renEncoder = commandBuffer.makeRenderCommandEncoder(descriptor: renderPassDesc) else { return }
        renEncoder.setRenderPipelineState(renderPipeline)
        renEncoder.setFragmentTexture(finalTex, index: 0)
        renEncoder.drawPrimitives(type: .triangleStrip, vertexStart: 0, vertexCount: 4)
        renEncoder.endEncoding()
        
        commandBuffer.present(drawable)
        commandBuffer.addCompletedHandler { [weak self] _ in
            guard let self = self else { return }
            os_unfair_lock_lock(&self.statsLock)
            self._stats.outputFrameCount += 1
            if frameGenActive {
                if isInterpolated {
                    self._stats.interpolatedFrameCount += 1
                } else {
                    self._stats.passthroughFrameCount += 1
                }
            } else {
                self._stats.passthroughFrameCount += 1
            }
            os_unfair_lock_unlock(&self.statsLock)
        }
        commandBuffer.commit()
    }
    
    private func updateCaptureStats(currentTime: CFTimeInterval, captureTimestamp: CFTimeInterval?) {
        frameCount += 1

        os_unfair_lock_lock(&statsLock)
        _stats.frameCount += 1
        _stats.gpuMemoryUsed = UInt64(device.currentAllocatedSize)
        _stats.gpuMemoryTotal = UInt64(device.recommendedMaxWorkingSetSize)
        os_unfair_lock_unlock(&statsLock)

        if lastCaptureTimestamp > 0 {
            let interval = currentTime - lastCaptureTimestamp
            if interval > 0 {
                estimatedCaptureInterval = (estimatedCaptureInterval * 0.9) + (interval * 0.1)
            }
        }
        lastCaptureTimestamp = currentTime

        let elapsed = currentTime - fpsStartTime
        if elapsed >= 1.0 {
            os_unfair_lock_lock(&statsLock)
            _stats.captureFPS = Float(frameCount) / Float(elapsed)
            os_unfair_lock_unlock(&statsLock)
            frameCount = 0
            fpsStartTime = currentTime
            DispatchQueue.main.async { [weak self] in
                self?.applyFrameRatePreference(self?.desiredOutputFPS() ?? 0)
            }
        }

        let delta = currentTime - lastFrameTime
        os_unfair_lock_lock(&statsLock)
        if lastFrameTime > 0 {
            _stats.frameTime = Float(delta * 1000.0)
        }
        if let captureTimestamp, captureTimestamp > 0 {
            _stats.captureLatency = Float((currentTime - captureTimestamp) * 1000.0)
        } else {
            _stats.captureLatency = Float(delta * 1000.0)
        }
        os_unfair_lock_unlock(&statsLock)
        lastFrameTime = currentTime
    }

    private func processSurface(_ surface: IOSurfaceRef, timestamp: CFTimeInterval) {
        let w = IOSurfaceGetWidth(surface)
        let h = IOSurfaceGetHeight(surface)

        // Re-create pool if resolution changed
        if w != ioSurfacePoolSize.width || h != ioSurfacePoolSize.height {
            ioSurfacePoolSize = (w, h)
            ioSurfaceTexturePool = [nil, nil, nil]
        }

        // Reuse texture from pool or create new one backed by IOSurface
        let poolIdx = ioSurfacePoolIndex % ioSurfaceTexturePool.count
        ioSurfacePoolIndex += 1

        let desc = MTLTextureDescriptor.texture2DDescriptor(
            pixelFormat: .bgra8Unorm,
            width: w,
            height: h,
            mipmapped: false
        )
        desc.usage = [.shaderRead]
        guard let inputTex = device.makeTexture(descriptor: desc, iosurface: surface, plane: 0) else {
            lastError = "Error Code: MG-ENG-010 IOSurface texture creation failed"
            return
        }
        ioSurfaceTexturePool[poolIdx] = inputTex

        processCapturedTexture(inputTex, timestamp: timestamp)
    }

    private func processCapturedTexture(_ inputTex: MTLTexture, timestamp: CFTimeInterval) {
        guard let commandBuffer = commandQueue.makeCommandBuffer() else {
            inFlightSemaphore.signal()
            return
        }
        
        // Signal semaphore when GPU finishes this frame
        commandBuffer.addCompletedHandler { [weak self] _ in
            self?.inFlightSemaphore.signal()
        }

        let inputWidth = inputTex.width
        let inputHeight = inputTex.height
        let shouldScale = scalingType != .off
        let renderScale = shouldScale ? renderScaleFactor : 1.0
        let renderWidth = Int(round(Float(inputWidth) * renderScale))
        let renderHeight = Int(round(Float(inputHeight) * renderScale))

        let targetScale = shouldScale ? scaleFactor : 1.0
        let targetWidth = Int(round(Float(renderWidth) * targetScale))
        let targetHeight = Int(round(Float(renderHeight) * targetScale))

        let targetSize = CGSize(width: targetWidth, height: targetHeight)
        if targetSize != lastProcessedSize {
            resetProcessingState(clearFrames: true)
            lastProcessedSize = targetSize
        }

        var workingTex = inputTex

        if renderWidth != inputWidth || renderHeight != inputHeight {
            guard let scalePipeline = scalePipeline,
                  let renderTex = ensureTexture(&renderTexture, width: renderWidth, height: renderHeight),
                  let encoder = commandBuffer.makeComputeCommandEncoder() else {
                lastError = "Error Code: MG-ENG-008 Scale pipeline unavailable"
                commandBuffer.commit()
                return
            }
            encoder.setComputePipelineState(scalePipeline)
            encoder.setTexture(inputTex, index: 0)
            encoder.setTexture(renderTex, index: 1)
            dispatchThreads(pipeline: scalePipeline, encoder: encoder, width: renderWidth, height: renderHeight)
            encoder.endEncoding()
            workingTex = renderTex
        }

        var scaledTex = workingTex
        if shouldScale || targetWidth != workingTex.width || targetHeight != workingTex.height {
            // MetalFX Spatial Scaler requires .renderTarget usage flag
            guard let outputTex = ensureTexture(&scaledTexture, width: targetWidth, height: targetHeight,
                                                 usage: [.shaderRead, .shaderWrite, .renderTarget]) else {
                lastError = "Error Code: MG-ENG-008 Scale pipeline unavailable"
                commandBuffer.commit()
                return
            }
            let baseSharpness = effectiveSharpness()

            switch scalingType {
            case .mgup1:
                // MGUP-1: MetalFX Spatial Scaler + quality-based CAS sharpening
                guard let scaler = ensureMetalFXSpatialScaler(
                    inputWidth: workingTex.width, inputHeight: workingTex.height,
                    outputWidth: targetWidth, outputHeight: targetHeight
                ) else {
                    lastError = "Error Code: MG-ENG-004 MetalFX Spatial Scaler creation failed"
                    commandBuffer.commit()
                    return
                }
                scaler.colorTexture = workingTex
                scaler.outputTexture = outputTex
                scaler.encode(commandBuffer: commandBuffer)
                scaledTex = outputTex

                // Apply CAS sharpening based on quality mode
                if baseSharpness > 0.01 {
                    guard let casPipeline = casPipeline,
                          let casOut = ensureTexture(&casTexture, width: targetWidth, height: targetHeight),
                          let encoder = commandBuffer.makeComputeCommandEncoder() else {
                        lastError = "Error Code: MG-ENG-009 CAS pipeline unavailable"
                        commandBuffer.commit()
                        return
                    }
                    var params = SharpenParams(sharpness: baseSharpness, radius: 1.0)
                    encoder.setComputePipelineState(casPipeline)
                    encoder.setTexture(scaledTex, index: 0)
                    encoder.setTexture(casOut, index: 1)
                    encoder.setBytes(&params, length: MemoryLayout<SharpenParams>.size, index: 0)
                    dispatchThreads(pipeline: casPipeline, encoder: encoder, width: targetWidth, height: targetHeight)
                    encoder.endEncoding()
                    scaledTex = casOut
                }

            case .off:
                scaledTex = workingTex
            }
        }

        guard let finalTex = applyAntiAliasing(to: scaledTex, commandBuffer: commandBuffer) else {
            os_unfair_lock_lock(&statsLock)
            _stats.droppedFrames += 1
            os_unfair_lock_unlock(&statsLock)
            commandBuffer.commit()
            return
        }

        os_unfair_lock_lock(&statsLock)
        _stats.outputResolution = CGSize(width: finalTex.width, height: finalTex.height)
        os_unfair_lock_unlock(&statsLock)

        var flowFromPrev: MTLTexture?
        var flowToPrev: MTLTexture?
        if frameGenEnabled, let prevFrame = frameBuffer.newestFrame {
            let prevTex = prevFrame.texture
            if prevTex.width == finalTex.width && prevTex.height == finalTex.height {
                // Submit frame pair for async Vision flow (non-blocking)
                flowFrameCounter += 1
                visionFlowProvider?.submitFlowRequest(
                    prev: prevTex, next: finalTex, frameID: flowFrameCounter
                )
                
                // Use latest available flow result (may be from previous frame pair)
                if let cachedFlow = visionFlowProvider?.latestFlow() {
                    flowFromPrev = cachedFlow.forward
                    flowToPrev = cachedFlow.backward
                }
                // If no flow yet, frame will be pushed without flow — interpolation will use
                // nearest available frame without motion compensation
            } else {
                lastError = "Error Code: MG-ENG-006 Frame interpolation failed: size mismatch"
            }
        }

        commandBuffer.addCompletedHandler { [weak self] buffer in
            guard let self = self else { return }
            let gpuTime = buffer.gpuEndTime - buffer.gpuStartTime
            os_unfair_lock_lock(&self.statsLock)
            self._stats.gpuTime = Float(gpuTime * 1000.0)
            os_unfair_lock_unlock(&self.statsLock)
        }
        commandBuffer.commit()

        frameBuffer.push(FrameHistory(texture: finalTex, timestamp: timestamp, flowFromPrev: flowFromPrev, flowToPrev: flowToPrev))
    }

    private struct UpscaleParams {
        var sharpness: Float
        var inputSize: SIMD2<UInt32>
        var outputSize: SIMD2<UInt32>
    }

    private struct SharpenParams {
        var sharpness: Float
        var radius: Float
    }

    private struct AntiAliasParams {
        var threshold: Float
        var depthThreshold: Float
        var maxSearchSteps: Int32
        var subpixelBlend: Float
    }


    private struct FlowWarpParams {
        var scale: Float
    }

    private struct FlowComposeParams {
        var t: Float
        var errorThreshold: Float
        var flowThreshold: Float
    }

    private struct FlowOcclusionParams {
        var threshold: Float
    }
    
    private struct TemporalParams {
        var blendFactor: Float
    }

    private func applyAntiAliasing(to input: MTLTexture,
                                   commandBuffer: MTLCommandBuffer) -> MTLTexture? {
        switch aaMode {
        case .off:
            return input
        case .fxaa:
            guard let fxaaPipeline = fxaaPipeline,
                  let out = ensureTexture(&fxaaTexture, width: input.width, height: input.height),
                  let encoder = commandBuffer.makeComputeCommandEncoder() else {
                lastError = "Error Code: MG-ENG-007 Anti-aliasing pipeline unavailable (FXAA)"
                return nil
            }
            var threshold = qualityProfile.aaThreshold
            encoder.setComputePipelineState(fxaaPipeline)
            encoder.setTexture(input, index: 0)
            encoder.setTexture(out, index: 1)
            encoder.setBytes(&threshold, length: MemoryLayout<Float>.size, index: 0)
            dispatchThreads(pipeline: fxaaPipeline, encoder: encoder, width: input.width, height: input.height)
            encoder.endEncoding()
            return out
        case .smaa:
            guard let edgePipe = smaaEdgePipeline,
                  let weightPipe = smaaWeightPipeline,
                  let blendPipe = smaaBlendPipeline,
                  let edges = ensureTexture(&smaaEdgeTexture, width: input.width, height: input.height),
                  let weights = ensureTexture(&smaaWeightTexture, width: input.width, height: input.height),
                  let out = ensureTexture(&smaaOutputTexture, width: input.width, height: input.height) else {
                lastError = "Error Code: MG-ENG-007 Anti-aliasing pipeline unavailable (SMAA)"
                return nil
            }
            var params = AntiAliasParams(
                threshold: qualityProfile.aaThreshold,
                depthThreshold: 0.1,
                maxSearchSteps: Int32(qualityProfile.smaaSearchSteps),
                subpixelBlend: 0.75
            )
            guard let encoder = commandBuffer.makeComputeCommandEncoder() else {
                lastError = "Error Code: MG-ENG-007 Anti-aliasing pipeline unavailable (SMAA)"
                return nil
            }
            // Edge detection
            encoder.setComputePipelineState(edgePipe)
            encoder.setTexture(input, index: 0)
            encoder.setTexture(edges, index: 1)
            encoder.setBytes(&params, length: MemoryLayout<AntiAliasParams>.size, index: 0)
            dispatchThreads(pipeline: edgePipe, encoder: encoder, width: input.width, height: input.height)

            // Weight calculation
            encoder.setComputePipelineState(weightPipe)
            encoder.setTexture(edges, index: 0)
            encoder.setTexture(weights, index: 1)
            encoder.setBytes(&params, length: MemoryLayout<AntiAliasParams>.size, index: 0)
            dispatchThreads(pipeline: weightPipe, encoder: encoder, width: input.width, height: input.height)

            // Neighborhood blending
            encoder.setComputePipelineState(blendPipe)
            encoder.setTexture(input, index: 0)
            encoder.setTexture(weights, index: 1)
            encoder.setTexture(out, index: 2)
            dispatchThreads(pipeline: blendPipe, encoder: encoder, width: input.width, height: input.height)
            encoder.endEncoding()
            return out
        case .msaa:
            guard let msaaPipeline = msaaPipeline,
                  let out = ensureTexture(&msaaTexture, width: input.width, height: input.height),
                  let encoder = commandBuffer.makeComputeCommandEncoder() else {
                lastError = "Error Code: MG-ENG-007 Anti-aliasing pipeline unavailable (MSAA)"
                return nil
            }
            var params = AntiAliasParams(
                threshold: qualityProfile.aaThreshold,
                depthThreshold: 0.1,
                maxSearchSteps: 8,
                subpixelBlend: 0.5
            )
            encoder.setComputePipelineState(msaaPipeline)
            encoder.setTexture(input, index: 0)
            encoder.setTexture(out, index: 1)
            encoder.setBytes(&params, length: MemoryLayout<AntiAliasParams>.size, index: 0)
            dispatchThreads(pipeline: msaaPipeline, encoder: encoder, width: input.width, height: input.height)
            encoder.endEncoding()
            return out
        case .taa:
            guard let temporalPipeline = temporalPipeline,
                  let copyPipeline = copyPipeline,
                  let history = ensureTexture(&taaHistoryTexture, width: input.width, height: input.height),
                  let out = ensureTexture(&taaOutputTexture, width: input.width, height: input.height) else {
                lastError = "Error Code: MG-ENG-007 Anti-aliasing pipeline unavailable (TAA)"
                return nil
            }

            if !hasTAAHistory {
                guard let encoder = commandBuffer.makeComputeCommandEncoder() else {
                    lastError = "Error Code: MG-ENG-007 Anti-aliasing pipeline unavailable (TAA)"
                    return nil
                }
                encoder.setComputePipelineState(copyPipeline)
                encoder.setTexture(input, index: 0)
                encoder.setTexture(history, index: 1)
                dispatchThreads(pipeline: copyPipeline, encoder: encoder, width: input.width, height: input.height)
                encoder.endEncoding()
                hasTAAHistory = true
                return input
            }

            // TAA without motion vectors — simple temporal blend
            // (Vision flow is async & used for frame gen; TAA uses motionless reprojection)
            var params = TemporalParams(blendFactor: effectiveTemporalBlend())
            guard let encoder = commandBuffer.makeComputeCommandEncoder() else {
                lastError = "Error Code: MG-ENG-007 Anti-aliasing pipeline unavailable (TAA)"
                return nil
            }
            // Temporal reproject
            encoder.setComputePipelineState(temporalPipeline)
            encoder.setTexture(input, index: 0)
            encoder.setTexture(history, index: 1)
            encoder.setTexture(input, index: 2)  // Use input as dummy flow (zero motion)
            encoder.setTexture(out, index: 3)
            encoder.setBytes(&params, length: MemoryLayout<TemporalParams>.size, index: 0)
            dispatchThreads(pipeline: temporalPipeline, encoder: encoder, width: input.width, height: input.height)

            // Copy to history
            encoder.setComputePipelineState(copyPipeline)
            encoder.setTexture(out, index: 0)
            encoder.setTexture(history, index: 1)
            dispatchThreads(pipeline: copyPipeline, encoder: encoder, width: input.width, height: input.height)
            encoder.endEncoding()
            return out
        }
    }

    private func interpolateFrame(prev: FrameHistory,
                                  next: FrameHistory,
                                  t: Float,
                                  commandBuffer: MTLCommandBuffer) -> MTLTexture? {
        let prevTex = prev.texture
        let nextTex = next.texture
        guard let output = ensureTexture(&blendTexture, width: prevTex.width, height: prevTex.height) else { return nil }

        switch frameGenMode {
        case .mgfg1:
            guard let flowWarpPipeline = flowWarpPipeline,
                  let flowComposePipeline = flowComposePipeline,
                  let flowOcclusionPipeline = flowOcclusionPipeline else {
                lastError = "Error Code: MG-ENG-013 Frame generation pipeline unavailable"
                return nil
            }
            guard let flowForward = next.flowFromPrev,
                  let flowBackward = next.flowToPrev else {
                lastError = "Error Code: MG-ENG-013 Frame generation pipeline unavailable"
                return nil
            }
            guard flowForward.width == prevTex.width,
                  flowForward.height == prevTex.height,
                  flowBackward.width == prevTex.width,
                  flowBackward.height == prevTex.height else {
                lastError = "Error Code: MG-ENG-006 Frame interpolation failed: size mismatch"
                return nil
            }
            guard let occlusion = ensureTexture(&occlusionTexture, width: prevTex.width, height: prevTex.height, pixelFormat: .r16Float),
                  let warpPrev = ensureTexture(&warpedPrevTexture, width: prevTex.width, height: prevTex.height),
                  let warpNext = ensureTexture(&warpedNextTexture, width: prevTex.width, height: prevTex.height) else {
                lastError = "Error Code: MG-ENG-013 Frame generation pipeline unavailable"
                return nil
            }

            var occParams = FlowOcclusionParams(threshold: flowOcclusionThreshold)
            guard let encoder = commandBuffer.makeComputeCommandEncoder() else {
                lastError = "Error Code: MG-ENG-013 Frame generation pipeline unavailable"
                return nil
            }
            
            // Occlusion detection
            encoder.setComputePipelineState(flowOcclusionPipeline)
            encoder.setTexture(flowForward, index: 0)
            encoder.setTexture(flowBackward, index: 1)
            encoder.setTexture(occlusion, index: 2)
            encoder.setBytes(&occParams, length: MemoryLayout<FlowOcclusionParams>.size, index: 0)
            dispatchThreads(pipeline: flowOcclusionPipeline, encoder: encoder, width: occlusion.width, height: occlusion.height)

            // Warp prev frame forward
            var warpPrevParams = FlowWarpParams(scale: t)
            encoder.setComputePipelineState(flowWarpPipeline)
            encoder.setTexture(prevTex, index: 0)
            encoder.setTexture(flowForward, index: 1)
            encoder.setTexture(warpPrev, index: 2)
            encoder.setBytes(&warpPrevParams, length: MemoryLayout<FlowWarpParams>.size, index: 0)
            dispatchThreads(pipeline: flowWarpPipeline, encoder: encoder, width: warpPrev.width, height: warpPrev.height)

            // Warp next frame backward
            var warpNextParams = FlowWarpParams(scale: (1.0 - t))
            encoder.setComputePipelineState(flowWarpPipeline)
            encoder.setTexture(nextTex, index: 0)
            encoder.setTexture(flowBackward, index: 1)
            encoder.setTexture(warpNext, index: 2)
            encoder.setBytes(&warpNextParams, length: MemoryLayout<FlowWarpParams>.size, index: 0)
            dispatchThreads(pipeline: flowWarpPipeline, encoder: encoder, width: warpNext.width, height: warpNext.height)

            // Compose final interpolated frame
            var composeParams = FlowComposeParams(
                t: t, 
                errorThreshold: qualityProfile.frameGenGradientThreshold,
                flowThreshold: flowOcclusionThreshold
            )
            encoder.setComputePipelineState(flowComposePipeline)
            encoder.setTexture(warpPrev, index: 0)
            encoder.setTexture(warpNext, index: 1)
            encoder.setTexture(occlusion, index: 2)
            encoder.setTexture(output, index: 3)
            encoder.setBytes(&composeParams, length: MemoryLayout<FlowComposeParams>.size, index: 0)
            dispatchThreads(pipeline: flowComposePipeline, encoder: encoder, width: output.width, height: output.height)
            encoder.endEncoding()
            return output
        case .off:
            return nil
        }
    }
    
    func configure(
        virtualResolution: CGSize,
        outputSize: CGSize
    ) {
        if self.outputSize != outputSize {
            resetProcessingState(clearFrames: true)
        }
        self.outputSize = outputSize
        os_unfair_lock_lock(&statsLock)
        _stats.virtualResolution = virtualResolution
        _stats.outputResolution = outputSize
        _stats.isUsingVirtualDisplay = true
        os_unfair_lock_unlock(&statsLock)
    }
    
    func updateSettings(_ settings: CaptureSettings) {
        let cursorChanged = settings.captureCursor != captureCursor
        let newScalingType = settings.scalingType
        let newQualityMode = settings.qualityMode
        let newAAMode = settings.aaMode
        let shouldScale = newScalingType != .off
        let newRenderScale = shouldScale ? settings.renderScale.multiplier : 1.0
        let newScaleFactor = shouldScale ? settings.scaleFactor.floatValue : 1.0
        let newProfile = newQualityMode.profile
        
        let pipelineChanged =
            newScalingType != scalingType ||
            newQualityMode != qualityMode ||
            newAAMode != aaMode ||
            abs(newRenderScale - renderScaleFactor) > 0.001 ||
            abs(newScaleFactor - scaleFactor) > 0.001
        
        if pipelineChanged {
            resetProcessingState(clearFrames: true)
        }
        
        scalingType = newScalingType
        qualityMode = newQualityMode
        aaMode = newAAMode
        renderScaleFactor = newRenderScale
        scaleFactor = newScaleFactor
        qualityProfile = newProfile
        sharpness = settings.sharpening
        temporalBlend = settings.temporalBlend
        motionScale = settings.motionScale
        captureCursor = settings.captureCursor
        frameGenMode = settings.frameGenMode
        frameGenEnabled = settings.frameGenMode != .off
        frameGenType = settings.frameGenType
        targetFPS = settings.targetFPS.intValue
        frameGenMultiplier = settings.frameGenMultiplier.intValue
        adaptiveSync = settings.adaptiveSync
        vsyncEnabled = settings.vsync
        
        applyFrameRatePreference(desiredOutputFPS())
        if let view = mtkView {
            applyDisplaySync(to: view)
        }

        if cursorChanged, isCapturing {
            restartCaptureForCursorChange()
        }

        if !frameGenEnabled {
            os_unfair_lock_lock(&statsLock)
            _stats.droppedFrames = 0
            _stats.interpolatedFrameCount = 0
            _stats.passthroughFrameCount = 0
            os_unfair_lock_unlock(&statsLock)
        }
    }

    private func restartCaptureForCursorChange() {
        if let manager = virtualDisplayManager {
            Task {
                _ = await self.startCaptureFromVirtualDisplay(manager, refreshRate: self.captureRefreshRate)
            }
        }
    }
    
    func startCaptureFromVirtualDisplay(_ virtualDisplayManager: VirtualDisplayManager, refreshRate: Int) async -> Bool {
        guard virtualDisplayManager.isActive else {
            lastError = "Error Code: MG-VD-006 No active virtual display"
            return false
        }
        
        resetProcessingState(clearFrames: true)
        resetFrameCounters()
        self.virtualDisplayManager = virtualDisplayManager
        
        virtualDisplayManager.onFrameReceived = { [weak self] surface, timestamp in
            self?.processingQueue.async {
                self?.processIOSurfaceFrame(surface: surface, timestamp: timestamp)
            }
        }
        
        guard await virtualDisplayManager.startFrameCapture(refreshRate: refreshRate, showsCursor: captureCursor) else {
            lastError = virtualDisplayManager.lastError!
            isCapturing = false
            return false
        }
        
        self.isCapturing = true
        self.captureRefreshRate = refreshRate
        self.frameCount = 0
        self.fpsStartTime = CACurrentMediaTime()
        self.lastCaptureTimestamp = 0
        
        return true
    }
    
    private func processIOSurfaceFrame(surface: IOSurfaceRef, timestamp: Double) {
        // Triple buffering: wait for a slot
        inFlightSemaphore.wait()
        let currentTime = CACurrentMediaTime()
        updateCaptureStats(currentTime: currentTime, captureTimestamp: timestamp)
        processSurface(surface, timestamp: currentTime)
    }
    
    func stopCapture() async {
        isCapturing = false
        if let manager = virtualDisplayManager {
            await manager.stopFrameCapture()
            manager.onFrameReceived = nil
        }
        lastCaptureTimestamp = 0
    }
    
    nonisolated func mtkView(_ view: MTKView, drawableSizeWillChange size: CGSize) {}
}
