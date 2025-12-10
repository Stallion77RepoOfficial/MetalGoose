import Foundation
import AppKit
import Metal
import MetalKit
import IOSurface
import QuartzCore

@available(macOS 26.0, *)
@MainActor
final class DirectRenderer: NSObject, MTKViewDelegate {
    private(set) var currentFPS: Float = 0
    private(set) var processingTime: Double = 0
    private(set) var currentStats: DirectEngineStats = DirectEngineStats()
    private(set) var captureFPS: Float = 0
    private(set) var interpolatedFPS: Float = 0
    
    var onWindowLost: (() -> Void)?
    var onWindowMoved: ((CGRect) -> Void)?
    var onStatsUpdated: ((DirectEngineStats) -> Void)?
    
    private let device: MTLDevice
    private let commandQueue: MTLCommandQueue
    private var engine: DirectEngineRef?
    private var overlayManager: OverlayWindowManager?
    private var outputWindow: NSWindow?
    private var mtkView: MTKView?
    private var frameGenEnabled: Bool = false
    private var frameGenMultiplier: Int = 2
    private var frameGenType: CaptureSettings.FrameGenType = .fixed
    private var frameGenCycle: Int = 0
    private var lastRealFrameTime: CFTimeInterval = 0
    private var previousTexture: MTLTexture?
    private var currentTexture: MTLTexture?
    private var displayTexture: MTLTexture?
    private var reduceLatencyEnabled: Bool = true
    private var vsyncEnabled: Bool = true
    private var adaptiveSyncEnabled: Bool = true
    private var lastFrameTime: CFTimeInterval = 0
    private var frameTimeAccumulator: Double = 0
    private var frameCount: Int = 0
    private var renderPipelineState: MTLRenderPipelineState?
    private var sampler: MTLSamplerState?
    
    init?(device: MTLDevice? = nil, commandQueue: MTLCommandQueue? = nil) {
        guard let dev = device ?? MTLCreateSystemDefaultDevice() else {
            return nil
        }
        guard let queue = commandQueue ?? dev.makeCommandQueue() else {
            return nil
        }
        
        self.device = dev
        self.commandQueue = queue
        
        super.init()
        
        guard let createdEngine = DirectEngine_Create(dev, queue) else {
            return nil
        }
        self.engine = createdEngine
        
        guard let overlay = OverlayWindowManager(device: dev) else {
            DirectEngine_Destroy(createdEngine)
            return nil
        }
        self.overlayManager = overlay
        
        if let library = dev.makeDefaultLibrary(),
           let vertexFunc = library.makeFunction(name: "texture_vertex"),
           let fragmentFunc = library.makeFunction(name: "texture_fragment") {
            let pipelineDesc = MTLRenderPipelineDescriptor()
            pipelineDesc.vertexFunction = vertexFunc
            pipelineDesc.fragmentFunction = fragmentFunc
            pipelineDesc.colorAttachments[0].pixelFormat = .bgra8Unorm
            pipelineDesc.colorAttachments[0].isBlendingEnabled = false
            self.renderPipelineState = try? dev.makeRenderPipelineState(descriptor: pipelineDesc)
        }
        
        let samplerDesc = MTLSamplerDescriptor()
        samplerDesc.minFilter = .linear
        samplerDesc.magFilter = .linear
        samplerDesc.sAddressMode = .clampToEdge
        samplerDesc.tAddressMode = .clampToEdge
        self.sampler = dev.makeSamplerState(descriptor: samplerDesc)
    }
    
    deinit {
        let eng = engine
        if let eng {
            DirectEngine_StopCapture(eng)
            DirectEngine_Destroy(eng)
        }
    }
    
    func configure(from settings: CaptureSettings,
                   targetFPS: Int = 120,
                   sourceSize: CGSize? = nil,
                   outputSize: CGSize? = nil) {
        
        guard let engine else { return }
        
        var cfg = DirectEngineConfig()
        
        cfg.upscaleMode = mapUpscale(settings.scalingType)
        cfg.renderScale = mapRenderScale(settings.renderScale)
        cfg.scaleFactor = settings.effectiveUpscaleFactor
        cfg.frameGenMode = mapFrameGen(settings.frameGenMode)
        cfg.frameGenType = mapFrameGenType(settings.frameGenType)
        cfg.frameGenQuality = mapFrameGenQuality(settings.qualityMode)
        cfg.frameGenMultiplier = Int32(settings.frameGenMultiplier.intValue)
        cfg.adaptiveTargetFPS = Int32(settings.targetFPS.intValue)
        cfg.aaMode = mapAA(settings.aaMode)
        cfg.aaThreshold = 0.166
        cfg.baseWidth = Int32(sourceSize?.width ?? 0)
        cfg.baseHeight = Int32(sourceSize?.height ?? 0)
        cfg.outputWidth = Int32(outputSize?.width ?? 0)
        cfg.outputHeight = Int32(outputSize?.height ?? 0)
        cfg.targetFPS = Int32(targetFPS)
        cfg.useMotionVectors = true
        cfg.vsyncEnabled = settings.vsync
        cfg.reduceLatency = settings.reduceLatency
        cfg.adaptiveSync = settings.adaptiveSync
        cfg.captureMouseCursor = settings.captureCursor
        cfg.sharpness = settings.sharpening
        cfg.temporalBlend = settings.temporalBlend
        cfg.motionScale = settings.motionScale
        
        DirectEngine_SetConfig(engine, cfg)
        
        frameGenEnabled = settings.isFrameGenEnabled
        frameGenType = settings.frameGenType
        frameGenMultiplier = settings.frameGenMultiplier.intValue
        reduceLatencyEnabled = settings.reduceLatency
        vsyncEnabled = settings.vsync
        adaptiveSyncEnabled = settings.adaptiveSync
    }
    
    func startCapture(windowID: CGWindowID, pid: pid_t = 0) -> Bool {
        guard let engine else { return false }
        
        guard DirectEngine_SetTargetWindow(engine, windowID) else {
            return false
        }
        
        overlayManager?.setTargetWindow(windowID, pid: pid)
        
        let success = DirectEngine_StartCapture(engine)
        
        if success {
            lastFrameTime = CACurrentMediaTime()
            frameGenCycle = 0
        }
        
        return success
    }
    
    func stopCapture() {
        guard let engine else { return }
        DirectEngine_StopCapture(engine)
    }
    
    func pauseCapture() {
        guard let engine else { return }
        DirectEngine_PauseCapture(engine)
    }
    
    func resumeCapture() {
        guard let engine else { return }
        DirectEngine_ResumeCapture(engine)
    }
    
    func attachToScreen(_ screen: NSScreen? = nil, size: CGSize? = nil, windowFrame: CGRect? = nil) {
        let targetScreen = screen ?? NSScreen.main ?? NSScreen.screens.first
        guard let targetScreen else { return }
        
        let displaySize = size ?? targetScreen.frame.size
        
        let config = OverlayWindowConfig(
            targetScreen: targetScreen,
            windowFrame: windowFrame,
            size: displaySize,
            refreshRate: frameGenEnabled ? 120.0 : 60.0,
            vsyncEnabled: vsyncEnabled,
            adaptiveSyncEnabled: adaptiveSyncEnabled,
            passThrough: true
        )
        
        guard let overlayManager, overlayManager.createOverlay(config: config) else {
            return
        }
        
        let view = MTKView(frame: CGRect(origin: .zero, size: displaySize), device: device)
        view.clearColor = MTLClearColorMake(0, 0, 0, 0)
        view.colorPixelFormat = .bgra8Unorm
        view.framebufferOnly = true
        view.autoResizeDrawable = true
        view.layer?.isOpaque = false
        view.enableSetNeedsDisplay = false
        view.isPaused = false
        
        let targetFPS = frameGenEnabled ? 120 : 60
        view.preferredFramesPerSecond = targetFPS
        view.delegate = self
        
        if let layer = view.layer as? CAMetalLayer {
            layer.displaySyncEnabled = vsyncEnabled
            layer.presentsWithTransaction = false
            layer.wantsExtendedDynamicRangeContent = false
            layer.maximumDrawableCount = 3
            layer.allowsNextDrawableTimeout = true
            layer.pixelFormat = .bgra8Unorm
            layer.isOpaque = false
            let backingScale = targetScreen.backingScaleFactor
            layer.drawableSize = CGSize(width: displaySize.width * backingScale, height: displaySize.height * backingScale)
        }
        
        overlayManager.setMTKView(view)
        mtkView = view
    }
    
    func attachToVirtualDisplay(_ displayID: CGDirectDisplayID, size: CGSize) {
        attachToScreen(nil, size: size)
    }
    
    func detachWindow() {
        mtkView?.isPaused = true
        mtkView?.delegate = nil
        mtkView = nil
        overlayManager?.destroyOverlay()
        outputWindow?.orderOut(nil)
        outputWindow = nil
        previousTexture = nil
        currentTexture = nil
    }
    
    func mtkView(_ view: MTKView, drawableSizeWillChange size: CGSize) {
    }
    
    func draw(in view: MTKView) {
        guard let engine else { return }
        
        overlayManager?.updateWindowPosition()
        
        autoreleasepool {
            drawFrame(in: view, engine: engine)
        }
    }
    
    private func drawFrame(in view: MTKView, engine: DirectEngineRef) {
        guard let commandBuffer = commandQueue.makeCommandBuffer() else { return }
        commandBuffer.label = "MetalGoose Frame"
        
        let startTime = CACurrentMediaTime()
        let hasNew = DirectEngine_HasNewFrame(engine)
        
        var outputTexture: MTLTexture?
        var isInterpolatedFrame = false
        
        if hasNew {
            if let processed = DirectEngine_ProcessFrame(engine, device, commandBuffer) {
                previousTexture = currentTexture
                currentTexture = processed
                displayTexture = processed
                
                let frameDelta = startTime - lastRealFrameTime
                lastRealFrameTime = startTime
                
                if frameDelta > 0.001 && frameDelta < 1.0 {
                    frameTimeAccumulator = frameTimeAccumulator * 0.8 + frameDelta * 0.2
                }
                
                if frameGenEnabled {
                    let gameFPS = 1.0 / (frameTimeAccumulator > 0 ? frameTimeAccumulator : 0.033)
                    let targetFPS = frameGenType == .adaptive ? Double(DirectEngine_GetFrameGenMultiplier(engine)) * 30.0 : Double(frameGenMultiplier) * gameFPS
                    let calculatedMultiplier = max(1, Int(ceil(targetFPS / gameFPS)))
                    DirectEngine_SetTargetFrameMultiplier(engine, Int32(min(calculatedMultiplier, 4)))
                    
                    let effectiveMultiplier = frameGenType == .adaptive ? Int(DirectEngine_GetFrameGenMultiplier(engine)) : frameGenMultiplier
                    
                    if effectiveMultiplier > 1 && previousTexture != nil {
                        frameGenCycle += 1
                        
                        if frameGenCycle >= effectiveMultiplier {
                            frameGenCycle = 0
                            outputTexture = processed
                        } else {
                            let t = Float(frameGenCycle) / Float(effectiveMultiplier)
                            var isInterpolated = false
                            if let interpTex = DirectEngine_GetInterpolatedFrameWithT(engine, device, commandBuffer, t, &isInterpolated) {
                                outputTexture = interpTex
                                isInterpolatedFrame = isInterpolated
                            } else {
                                outputTexture = processed
                            }
                        }
                    } else {
                        frameGenCycle = 0
                        outputTexture = processed
                    }
                } else {
                    frameGenCycle = 0
                    outputTexture = processed
                }
            } else {
                outputTexture = displayTexture
            }
        } else {
            outputTexture = displayTexture
        }
        
        guard let texture = outputTexture,
              let pipelineState = renderPipelineState,
              let drawable = view.currentDrawable else {
            commandBuffer.commit()
            return
        }
        
        let renderPassDesc = MTLRenderPassDescriptor()
        renderPassDesc.colorAttachments[0].texture = drawable.texture
        renderPassDesc.colorAttachments[0].loadAction = .clear
        renderPassDesc.colorAttachments[0].storeAction = .store
        renderPassDesc.colorAttachments[0].clearColor = MTLClearColorMake(0, 0, 0, 0)
        
        if let renderEncoder = commandBuffer.makeRenderCommandEncoder(descriptor: renderPassDesc) {
            renderEncoder.label = isInterpolatedFrame ? "Interpolated Render" : "Real Frame Render"
            renderEncoder.setRenderPipelineState(pipelineState)
            renderEncoder.setFragmentTexture(texture, index: 0)
            if let sampler = sampler {
                renderEncoder.setFragmentSamplerState(sampler, index: 0)
            }
            renderEncoder.drawPrimitives(type: .triangleStrip, vertexStart: 0, vertexCount: 4)
            renderEncoder.endEncoding()
        }
        
        commandBuffer.present(drawable)
        
        let capturedStartTime = startTime
        let capturedIsInterpolated = isInterpolatedFrame
        commandBuffer.addCompletedHandler { [weak self] buffer in
            Task { @MainActor [weak self] in
                guard let self, let engine = self.engine else { return }
                
                let endTime = CACurrentMediaTime()
                let gpuEndTime = buffer.gpuEndTime
                let gpuStartTime = buffer.gpuStartTime
                
                if gpuEndTime > gpuStartTime && gpuStartTime > 0 {
                    self.processingTime = (gpuEndTime - gpuStartTime) * 1000.0
                } else {
                    self.processingTime = (endTime - capturedStartTime) * 1000.0
                }
                
                self.currentStats = DirectEngine_GetStats(engine)
                self.currentFPS = DirectEngine_GetCurrentFPS(engine)
                self.captureFPS = DirectEngine_GetCaptureFPS(engine)
                let rawInterpFPS = DirectEngine_GetInterpolatedFPS(engine)
                if self.interpolatedFPS > 0 {
                    self.interpolatedFPS = self.interpolatedFPS * 0.85 + rawInterpFPS * 0.15
                } else {
                    self.interpolatedFPS = rawInterpFPS
                }
                
                if capturedIsInterpolated {
                    self.frameCount += 1
                }
                
                self.onStatsUpdated?(self.currentStats)
            }
        }
        commandBuffer.commit()
    }
    
    private func mapUpscale(_ type: CaptureSettings.ScalingType) -> DirectUpscaleMode {
        switch type {
        case .off: return DirectUpscaleMode.off
        case .mgup1: return DirectUpscaleMode(rawValue: 1)!
        case .mgup1Fast: return DirectUpscaleMode(rawValue: 2)!
        case .mgup1Quality: return DirectUpscaleMode(rawValue: 3)!
        }
    }
    
    private func mapFrameGen(_ mode: CaptureSettings.FrameGenMode) -> DirectFrameGen {
        switch mode {
        case .off: return DirectFrameGen.off
        case .mgfg1: return DirectFrameGen(rawValue: 1)!
        }
    }
    
    private func mapFrameGenType(_ type: CaptureSettings.FrameGenType) -> DirectFrameGenType {
        switch type {
        case .adaptive: return DirectFrameGenType.adaptive
        case .fixed: return DirectFrameGenType.fixed
        }
    }
    
    private func mapFrameGenQuality(_ quality: CaptureSettings.QualityMode) -> DirectFrameGenQuality {
        switch quality {
        case .performance: return DirectFrameGenQuality.performance
        case .balanced: return DirectFrameGenQuality.balanced
        case .ultra: return DirectFrameGenQuality.quality
        }
    }
    
    private func mapRenderScale(_ scale: CaptureSettings.RenderScale) -> DirectRenderScale {
        switch scale {
        case .native: return DirectRenderScale(rawValue: 0)!
        case .p75: return DirectRenderScale(rawValue: 1)!
        case .p67: return DirectRenderScale(rawValue: 2)!
        case .p50: return DirectRenderScale(rawValue: 3)!
        case .p33: return DirectRenderScale(rawValue: 4)!
        }
    }
    
    private func mapAA(_ mode: CaptureSettings.AAMode) -> DirectAAMode {
        switch mode {
        case .off: return DirectAAMode.off
        case .fxaa: return DirectAAMode(rawValue: 1)!
        case .smaa: return DirectAAMode(rawValue: 2)!
        case .msaa: return DirectAAMode(rawValue: 3)!
        case .taa: return DirectAAMode(rawValue: 4)!
        }
    }
    
    func getDebugInfo() -> String? {
        guard let engine else { return nil }
        return DirectEngine_GetDebugInfo(engine) as String?
    }
    
    func setDebugMode(_ enabled: Bool) {
        guard let engine else { return }
        DirectEngine_SetDebugMode(engine, enabled)
    }
    
    func getStats() -> DirectEngineStats {
        guard let engine else { return DirectEngineStats() }
        return DirectEngine_GetStats(engine)
    }
    
    func resetStats() {
        guard let engine else { return }
        DirectEngine_ResetStats(engine)
    }
}
