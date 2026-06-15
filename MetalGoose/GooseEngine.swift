import Foundation
import AppKit
@preconcurrency import Metal
@preconcurrency import MetalKit
@preconcurrency import MetalFX
@preconcurrency import IOSurface
import QuartzCore
import os
@preconcurrency import CoreVideo

struct CursorUniforms {
    var center: SIMD2<Float>
    var size: SIMD2<Float>
}

struct PipelineStats: @unchecked Sendable {
    var captureFPS: Float = 0
    var outputFPS: Float = 0
    var interpolatedFPS: Float = 0
    var frameTime: Float = 0
    var gpuTime: Float = 0
    var captureLatency: Float = 0
    var presentLatency: Float = 0
    var endToEndLatency: Float = 0
    var avgFrameTime: Float = 0
    var frameTimeJitter: Float = 0
    var framePacingScore: Float = 100
    var frameCount: UInt64 = 0
    var outputFrameCount: UInt64 = 0
    var droppedFrames: UInt64 = 0
    var interpolatedFrameCount: UInt64 = 0
    var passthroughFrameCount: UInt64 = 0
    var gpuMemoryUsed: UInt64 = 0
    var gpuMemoryTotal: UInt64 = 0
    var outputResolution: CGSize = .zero
    var screenRefreshRate: Int = 0
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

    // Error surfacing. Engine errors are never printed to the terminal; they are
    // queued here (deduplicated) and drained by the UI, which shows them as alerts.
    private let errorLock = OSAllocatedUnfairLock()
    private var reportedErrors: Set<String> = []
    private var pendingErrors: [String] = []

    // Set by the factory when the engine cannot be created at all (no Metal device /
    // command queue), so the UI can surface the reason instead of crashing.
    nonisolated(unsafe) static private(set) var lastInitError: String?

    private func reportError(_ message: String) {
        errorLock.lock()
        defer { errorLock.unlock() }
        // Each distinct error surfaces once per capture session (frame-loop failures
        // would otherwise fire every frame).
        guard reportedErrors.insert(message).inserted else { return }
        pendingErrors.append(message)
    }

    // Drained by the UI's stats timer; returns one queued error per call (FIFO).
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
    
    private var inFlightSemaphore = DispatchSemaphore(value: 3)
    private var bufferDepth: Int = 3

    var deviceName: String { device.name }

    private let device: MTLDevice
    private let commandQueue: MTLCommandQueue
    
    private var spatialScaler: MTLFXSpatialScaler?
    private var spatialScalerInputSize: (width: Int, height: Int) = (0, 0)
    private var spatialScalerOutputSize: (width: Int, height: Int) = (0, 0)
    
    private var scalePipeline: MTLComputePipelineState?
    private var casPipeline: MTLComputePipelineState?
    private var fxaaPipeline: MTLComputePipelineState?
    private var smaaEdgePipeline: MTLComputePipelineState?
    private var smaaWeightPipeline: MTLComputePipelineState?
    private var smaaBlendPipeline: MTLComputePipelineState?
    private var copyPipeline: MTLComputePipelineState?

    private var frameInterpolator: MTLFXFrameInterpolator?
    private var frameInterpolatorSize: (width: Int, height: Int) = (0, 0)
    private var frameInterpolatorNeedsHistoryReset: Bool = true

    private var renderPipeline: MTLRenderPipelineState?
    private var cursorPipeline: MTLRenderPipelineState?
    private var cursorTexture: MTLTexture?
    private var cursorTextureSize: CGSize = .zero
    private weak var mtkView: MTKView?
    
    private var renderTexture: MTLTexture?
    private var scaledTexture: MTLTexture?
    private var casTexture: MTLTexture?
    private var fxaaTexture: MTLTexture?
    private var smaaEdgeTexture: MTLTexture?
    private var smaaWeightTexture: MTLTexture?
    private var smaaOutputTexture: MTLTexture?

    struct FrameHistory {
        let texture: MTLTexture
        let timestamp: CFTimeInterval
        let isSceneCut: Bool
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

    private var historyTextures: [MTLTexture?] = Array(repeating: nil, count: 6)
    private var historyTextureIndex: Int = 0

    private var blendTexture: MTLTexture?
    private var lastProcessedSize: CGSize = .zero

    private var scalingType: CaptureSettings.ScalingType = .off
    private var qualityMode: CaptureSettings.QualityMode = .balanced
    private var aaMode: CaptureSettings.AAMode = .off
    private var renderScaleFactor: Float = 1.0
    private var scaleFactor: Float = 1.0
    private var frameGenEnabled: Bool = false
    private var frameGenMode: CaptureSettings.FrameGenMode = .off
    private var vsyncEnabled: Bool = true
    private var qualityProfile: QualityProfile = CaptureSettings.QualityMode.balanced.profile
    private var captureCursorEnabled: Bool = false
    

    private var estimatedCaptureInterval: Double = 0
    private var lastCaptureTimestamp: CFTimeInterval = 0
    private var lastPreferredFPS: Int = 0
    private var lastPreferredUpdateTime: CFTimeInterval = 0

    private var windowCaptureManager: WindowCaptureManager?

    private var lastFrameTime: CFTimeInterval = 0
    private var frameCount: Int = 0
    private var fpsStartTime: CFTimeInterval = 0
    
    private var outputSize: CGSize = .zero
    private var currentRefreshRate: Int = 0
    
    // Use this instead of init(): it fails gracefully (returning nil + setting
    // lastInitError) rather than crashing when Metal is unavailable, so the UI can
    // present MG-ENG-002 / MG-ENG-003 as an alert.
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

        scalePipeline = makeCompute("blitScaleBilinear")
        casPipeline = makeCompute("contrastAdaptiveSharpening")
        copyPipeline = makeCompute("copyTexture")
        
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
    
    private func ensureMetalFXSpatialScaler(inputWidth: Int, inputHeight: Int,
                                            outputWidth: Int, outputHeight: Int) -> MTLFXSpatialScaler? {

        if let scaler = spatialScaler,
           spatialScalerInputSize == (inputWidth, inputHeight),
           spatialScalerOutputSize == (outputWidth, outputHeight) {
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
        
        spatialScaler = scaler
        spatialScalerInputSize = (inputWidth, inputHeight)
        spatialScalerOutputSize = (outputWidth, outputHeight)
        
        return scaler
    }

    private func ensureFrameInterpolator(width: Int, height: Int) -> MTLFXFrameInterpolator? {
        renderStateLock.lock()
        defer { renderStateLock.unlock() }

        if let interpolator = frameInterpolator, frameInterpolatorSize == (width, height) {
            return interpolator
        }

        let descriptor = MTLFXFrameInterpolatorDescriptor()
        descriptor.colorTextureFormat = .bgra8Unorm
        descriptor.outputTextureFormat = .bgra8Unorm
        descriptor.inputWidth = width
        descriptor.inputHeight = height
        descriptor.outputWidth = width
        descriptor.outputHeight = height

        guard let interpolator = descriptor.makeFrameInterpolator(device: device) else {
            reportError("Error Code: MG-ENG-010 MetalFX Frame Interpolator creation failed")
            return nil
        }

        frameInterpolator = interpolator
        frameInterpolatorSize = (width, height)
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
        renderTexture = nil
        scaledTexture = nil
        casTexture = nil
        fxaaTexture = nil
        smaaEdgeTexture = nil
        smaaWeightTexture = nil
        smaaOutputTexture = nil
        historyTextures = Array(repeating: nil, count: historyTextures.count)
        historyTextureIndex = 0
        lastProcessedSize = .zero

        renderStateLock.lock()
        frameInterpolator = nil
        frameInterpolatorSize = (0, 0)
        frameInterpolatorNeedsHistoryReset = true
        blendTexture = nil
        renderStateLock.unlock()

        if clearFrames {
            frameBuffer.clear()
            statsLock.lock()
            _stats.droppedFrames = 0
            _stats.interpolatedFrameCount = 0
            _stats.passthroughFrameCount = 0
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
        _stats.droppedFrames = 0
        _stats.captureFPS = 0
        _stats.outputFPS = 0
        _stats.interpolatedFPS = 0
        statsLock.unlock()
        frameCount = 0
        renderFrameCount = 0
        interpolatedFrameCount = 0
        fpsStartTime = CACurrentMediaTime()
        renderFPSStartTime = CACurrentMediaTime()
        outputFrameTimeHistory.removeAll()
        lastRenderTime = 0
    }

    private func effectiveSharpness() -> Float {
        return qualityProfile.sharpnessScale
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

    private func desiredOutputFPS() -> Int {
        let sourceFPS = measuredSourceFPS()
        var target: Int

        if frameGenEnabled {
            let doubled = sourceFPS > 0 ? sourceFPS * 2.0 : Double(currentRefreshRate)
            let fill = currentRefreshRate > 0 ? Double(currentRefreshRate) : doubled
            target = snapToRefreshDivisor(max(fill, doubled))
        } else {
            target = sourceFPS > 0 ? Int(round(sourceFPS)) : currentRefreshRate
        }

        if currentRefreshRate > 0 {
            target = min(target, currentRefreshRate)
        }
        return max(1, target)
    }

    private func applyFrameRatePreference(_ preferred: Int) {
        let target = preferred
        guard target > 0 else { return }
        let now = CACurrentMediaTime()
        if abs(target - lastPreferredFPS) < 3 { return }
        if now - lastPreferredUpdateTime < 0.5 { return }
        if target != lastPreferredFPS {
            lastPreferredFPS = target
            lastPreferredUpdateTime = now

            let view = mtkView
            DispatchQueue.main.async {
                view?.preferredFramesPerSecond = target
            }
        }
    }
    
    private func applyDisplaySync(to view: MTKView) {
        MainActor.assumeIsolated {
            guard let layer = view.layer as? CAMetalLayer else { return }
            layer.displaySyncEnabled = vsyncEnabled
            layer.presentsWithTransaction = false
        }
    }

    private func interpolationDelay(for targetFPS: Int) -> Double {
        guard targetFPS > 0 else { return 0 }
        let outputInterval = 1.0 / Double(targetFPS)
        let captureInterval = estimatedCaptureInterval
        if frameGenEnabled {
            let genInterval = captureInterval / 2.0
            return outputInterval >= genInterval ? outputInterval : genInterval
        }
        return captureInterval >= outputInterval ? captureInterval : outputInterval
    }
    
    func attachToView(_ view: MTKView, displayRefreshRate: Int) {
        MainActor.assumeIsolated {
            view.device = device
            view.delegate = self
            view.preferredFramesPerSecond = displayRefreshRate
            view.isPaused = false
            view.enableSetNeedsDisplay = false
            view.colorPixelFormat = .bgra8Unorm
            view.framebufferOnly = false
            // Presentation latency knob. CAMetalLayer accepts only 2 or 3 drawables;
            // the processing-pipeline depth (inFlightSemaphore) carries the rest.
            if let layer = view.layer as? CAMetalLayer {
                layer.maximumDrawableCount = min(max(2, bufferDepth), 3)
            }
        }
        applyDisplaySync(to: view)
        self.mtkView = view
        self.currentRefreshRate = displayRefreshRate
        statsLock.lock()
        _stats.screenRefreshRate = displayRefreshRate
        statsLock.unlock()
        applyFrameRatePreference(desiredOutputFPS())
    }

    func detachFromView() {
        MainActor.assumeIsolated {
            mtkView?.delegate = nil
            mtkView?.isPaused = true
        }
        mtkView = nil
    }

    nonisolated func draw(in view: MTKView) {
        MainActor.assumeIsolated {
            renderFrame(in: view)
        }
    }
    
    private var renderFrameCount: Int = 0
    private var renderFPSStartTime: CFTimeInterval = 0
    private var interpolatedFrameCount: Int = 0
    private var outputFrameTimeHistory: [Double] = []
    private var lastRenderTime: CFTimeInterval = 0
    private let frameTimeHistoryCapacity = 120

    @MainActor
    private func renderFrame(in view: MTKView) {
        guard let drawable = view.currentDrawable,
              let renderPipeline = renderPipeline,
              let commandBuffer = commandQueue.makeCommandBuffer() else { return }

        let currentTime = CACurrentMediaTime()

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

        let targetFPS = desiredOutputFPS()
        applyFrameRatePreference(targetFPS)
        statsLock.lock()
        _stats.targetOutputFPS = targetFPS
        statsLock.unlock()
        let targetTime = currentTime - interpolationDelay(for: targetFPS)

        var outputTex: MTLTexture?
        var sourceTimestamp: CFTimeInterval?
        let frameGenActive = frameGenEnabled
        var isInterpolated = false

        if frameGenEnabled, let (prev, next) = frameBuffer.getFramesForTime(targetTime),
           prev.texture.width == next.texture.width,
           prev.texture.height == next.texture.height {
            let duration = next.timestamp - prev.timestamp
            let timeSincePrev = targetTime - prev.timestamp
            let ratio = duration > 0 ? (timeSincePrev / duration) : 0
            let t = Float(min(max(ratio, 0), 1))

            let outputInterval = targetFPS > 0 ? 1.0 / Double(targetFPS) : 0
            let snap = outputInterval * 0.25

            if duration <= 0 || timeSincePrev <= snap {
                outputTex = prev.texture
                sourceTimestamp = prev.timestamp
            } else if (duration - timeSincePrev) <= snap {
                outputTex = next.texture
                sourceTimestamp = next.timestamp
            } else if next.isSceneCut {
                outputTex = t < 0.5 ? prev.texture : next.texture
                sourceTimestamp = t < 0.5 ? prev.timestamp : next.timestamp
                renderStateLock.lock()
                frameInterpolatorNeedsHistoryReset = true
                renderStateLock.unlock()
            } else if let interpolated = interpolateFrame(prev: prev, next: next, commandBuffer: commandBuffer) {
                outputTex = interpolated
                isInterpolated = true
                sourceTimestamp = prev.timestamp
            } else {
                outputTex = t < 0.5 ? prev.texture : next.texture
                sourceTimestamp = t < 0.5 ? prev.timestamp : next.timestamp
            }
        } else {
            outputTex = frameBuffer.newestFrame?.texture
            sourceTimestamp = frameBuffer.newestFrame?.timestamp
        }

        guard let finalTex = outputTex else {
            commandBuffer.commit()
            return
        }

        if let sourceTimestamp {
            let presentLatencyMs = Float((currentTime - sourceTimestamp) * 1000.0)
            statsLock.lock()
            _stats.presentLatency = presentLatencyMs
            _stats.endToEndLatency = _stats.captureLatency + presentLatencyMs
            statsLock.unlock()
        }
        
        renderFrameCount += 1
        if isInterpolated { interpolatedFrameCount += 1 }
        if elapsed >= 1.0 {
            statsLock.lock()
            _stats.outputFPS = Float(renderFrameCount) / Float(elapsed)
            _stats.interpolatedFPS = Float(interpolatedFrameCount) / Float(elapsed)
            statsLock.unlock()
            renderFrameCount = 0
            interpolatedFrameCount = 0
            renderFPSStartTime = currentTime

            updateFramePacingStats()
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

        drawSyntheticCursor(in: view, encoder: renEncoder, drawableSize: drawable.texture.width != 0 ?
            CGSize(width: drawable.texture.width, height: drawable.texture.height) : .zero)

        renEncoder.endEncoding()
        
        let frameWasInterpolated = isInterpolated
        commandBuffer.present(drawable)
        commandBuffer.addCompletedHandler { [weak self] _ in
            guard let self = self else { return }
            self.statsLock.lock()
            self._stats.outputFrameCount += 1
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
    
    @MainActor
    private func drawSyntheticCursor(in view: MTKView, encoder: MTLRenderCommandEncoder, drawableSize: CGSize) {
        guard captureCursorEnabled,
              let pipeline = cursorPipeline,
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
        _stats.frameTimeJitter = Float(jitter * 1000.0)
        _stats.framePacingScore = Float(pacingScore)
        statsLock.unlock()
    }

    private func updateCaptureStats(currentTime: CFTimeInterval, captureTimestamp: CFTimeInterval?) {
        frameCount += 1

        statsLock.lock()
        _stats.frameCount += 1
        _stats.gpuMemoryUsed = UInt64(device.currentAllocatedSize)
        _stats.gpuMemoryTotal = UInt64(device.recommendedMaxWorkingSetSize)
        statsLock.unlock()

        if lastCaptureTimestamp > 0 {
            let interval = currentTime - lastCaptureTimestamp
            if interval > 0 {
                estimatedCaptureInterval = (estimatedCaptureInterval * 0.9) + (interval * 0.1)
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
                self?.applyFrameRatePreference(self?.desiredOutputFPS() ?? 0)
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
                reportError("Error Code: MG-ENG-006 Scale pipeline unavailable")
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
            guard let outputTex = ensureTexture(&scaledTexture, width: targetWidth, height: targetHeight,
                                                 usage: [.shaderRead, .shaderWrite, .renderTarget]) else {
                reportError("Error Code: MG-ENG-006 Scale pipeline unavailable")
                commandBuffer.commit()
                return
            }
            let baseSharpness = effectiveSharpness()

            switch scalingType {
            case .mgup1:
                guard let scaler = ensureMetalFXSpatialScaler(
                    inputWidth: workingTex.width, inputHeight: workingTex.height,
                    outputWidth: targetWidth, outputHeight: targetHeight
                ) else {
                    reportError("Error Code: MG-ENG-004 MetalFX Spatial Scaler creation failed")
                    commandBuffer.commit()
                    return
                }
                scaler.colorTexture = workingTex
                scaler.outputTexture = outputTex
                scaler.encode(commandBuffer: commandBuffer)
                scaledTex = outputTex

                if baseSharpness > 0.01 {
                    guard let casPipeline = casPipeline,
                          let casOut = ensureTexture(&casTexture, width: targetWidth, height: targetHeight),
                          let encoder = commandBuffer.makeComputeCommandEncoder() else {
                        reportError("Error Code: MG-ENG-007 CAS pipeline unavailable")
                        commandBuffer.commit()
                        return
                    }
                    var params = SharpenParams(sharpness: baseSharpness)
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
            statsLock.lock()
            _stats.droppedFrames += 1
            statsLock.unlock()
            commandBuffer.commit()
            return
        }

        statsLock.lock()
        _stats.outputResolution = CGSize(width: finalTex.width, height: finalTex.height)
        statsLock.unlock()

        let slot = historyTextureIndex % historyTextures.count
        historyTextureIndex += 1
        guard let historyTex = ensureTexture(&historyTextures[slot], width: finalTex.width, height: finalTex.height,
                                             usage: [.shaderRead, .shaderWrite, .renderTarget]),
              encodeCopy(from: finalTex, to: historyTex, commandBuffer: commandBuffer) else {
            statsLock.lock()
            _stats.droppedFrames += 1
            statsLock.unlock()
            commandBuffer.commit()
            return
        }

        commandBuffer.addCompletedHandler { [weak self] buffer in
            guard let self = self else { return }
            let gpuTime = buffer.gpuEndTime - buffer.gpuStartTime
            self.statsLock.lock()
            self._stats.gpuTime = Float(gpuTime * 1000.0)
            self.statsLock.unlock()
        }
        commandBuffer.commit()

        frameBuffer.push(FrameHistory(texture: historyTex, timestamp: timestamp, isSceneCut: isSceneCut))
    }

    private struct SharpenParams {
        var sharpness: Float
    }

    private struct AntiAliasParams {
        var threshold: Float
        var maxSearchSteps: Int32
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
                reportError("Error Code: MG-ENG-005 Anti-aliasing pipeline unavailable (FXAA)")
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
                reportError("Error Code: MG-ENG-005 Anti-aliasing pipeline unavailable (SMAA)")
                return nil
            }
            var params = AntiAliasParams(
                threshold: qualityProfile.aaThreshold,
                maxSearchSteps: Int32(qualityProfile.smaaSearchSteps)
            )
            guard let encoder = commandBuffer.makeComputeCommandEncoder() else {
                reportError("Error Code: MG-ENG-005 Anti-aliasing pipeline unavailable (SMAA)")
                return nil
            }
            encoder.setComputePipelineState(edgePipe)
            encoder.setTexture(input, index: 0)
            encoder.setTexture(edges, index: 1)
            encoder.setBytes(&params, length: MemoryLayout<AntiAliasParams>.size, index: 0)
            dispatchThreads(pipeline: edgePipe, encoder: encoder, width: input.width, height: input.height)

            encoder.setComputePipelineState(weightPipe)
            encoder.setTexture(edges, index: 0)
            encoder.setTexture(weights, index: 1)
            encoder.setBytes(&params, length: MemoryLayout<AntiAliasParams>.size, index: 0)
            dispatchThreads(pipeline: weightPipe, encoder: encoder, width: input.width, height: input.height)

            encoder.setComputePipelineState(blendPipe)
            encoder.setTexture(input, index: 0)
            encoder.setTexture(weights, index: 1)
            encoder.setTexture(out, index: 2)
            dispatchThreads(pipeline: blendPipe, encoder: encoder, width: input.width, height: input.height)
            encoder.endEncoding()
            return out
        }
    }

    private func interpolateFrame(prev: FrameHistory,
                                  next: FrameHistory,
                                  commandBuffer: MTLCommandBuffer) -> MTLTexture? {
        let prevTex = prev.texture
        let nextTex = next.texture
        guard prevTex.width == nextTex.width, prevTex.height == nextTex.height else {
            return nil
        }

        switch frameGenMode {
        case .mgfg1:
            renderStateLock.lock()
            let output = ensureTexture(&blendTexture, width: prevTex.width, height: prevTex.height,
                                        usage: [.shaderRead, .shaderWrite, .renderTarget])
            renderStateLock.unlock()
            guard let output else { return nil }

            guard let interpolator = ensureFrameInterpolator(width: prevTex.width, height: prevTex.height) else {
                return nil
            }

            renderStateLock.lock()
            let shouldResetHistory = frameInterpolatorNeedsHistoryReset
            frameInterpolatorNeedsHistoryReset = false
            renderStateLock.unlock()

            guard prevTex.width == output.width, prevTex.height == output.height,
                  nextTex.width == output.width, nextTex.height == output.height else {
                return nil
            }

            interpolator.colorTexture = nextTex
            interpolator.prevColorTexture = prevTex
            interpolator.outputTexture = output
            interpolator.depthTexture = nil
            interpolator.motionTexture = nil
            interpolator.deltaTime = Float(max(0.0001, next.timestamp - prev.timestamp))
            interpolator.shouldResetHistory = shouldResetHistory
            interpolator.encode(commandBuffer: commandBuffer)
            return output
        case .off:
            return nil
        }
    }
    
    func configure(outputSize: CGSize) {
        if self.outputSize != outputSize {
            resetProcessingStateAsync(clearFrames: true)
        }
        self.outputSize = outputSize
    }
    
    func updateSettings(_ settings: CaptureSettings) {
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
            resetProcessingStateAsync(clearFrames: true)
        }

        scalingType = newScalingType
        qualityMode = newQualityMode
        aaMode = newAAMode
        renderScaleFactor = newRenderScale
        scaleFactor = newScaleFactor
        qualityProfile = newProfile
        frameGenMode = settings.frameGenMode
        frameGenEnabled = settings.frameGenMode != .off
        vsyncEnabled = settings.vsync
        captureCursorEnabled = settings.captureCursor
        bufferDepth = max(2, min(4, settings.bufferCount))

        applyFrameRatePreference(desiredOutputFPS())
        if let view = mtkView {
            applyDisplaySync(to: view)
        }

        if !frameGenEnabled {
            statsLock.lock()
            _stats.droppedFrames = 0
            _stats.interpolatedFrameCount = 0
            _stats.passthroughFrameCount = 0
            statsLock.unlock()
        }
    }

    func startCaptureFromWindow(_ manager: WindowCaptureManager, refreshRate _: Int) async {
        resetProcessingStateAsync(clearFrames: true)
        resetFrameCounters()
        resetErrorReporting()
        // Rebuild the in-flight semaphore at the chosen depth. Safe here because a
        // fresh capture has no frames in flight (start is gated behind a stopped state).
        inFlightSemaphore = DispatchSemaphore(value: max(2, min(4, bufferDepth)))
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
    }
    
    private func processIOSurfaceFrame(surface: IOSurfaceRef, pixelBuffer: CVPixelBuffer, timestamp: Double, isSceneCut: Bool) {
        inFlightSemaphore.wait()
        let currentTime = CACurrentMediaTime()
        updateCaptureStats(currentTime: currentTime, captureTimestamp: timestamp)
        processSurface(surface, pixelBuffer: pixelBuffer, timestamp: currentTime, isSceneCut: isSceneCut)
    }
    
    func stopCapture() async {
        if let manager = windowCaptureManager {
            manager.onFrameReceived = nil
        }
        lastCaptureTimestamp = 0
    }
    
    nonisolated func mtkView(_ view: MTKView, drawableSizeWillChange size: CGSize) {}
}
