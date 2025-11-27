import MetalKit
import CoreVideo
import Vision
import simd
import Metal
import CoreGraphics
import ScreenCaptureKit
#if os(macOS)
import AppKit
#endif

// Finalized: ScreenCaptureKit (display/window), Integer/Bicubic/EASU Upscale, RCAS Sharpen, Vision Optical Flow + FlowWarp (macOS 26+)

final class Renderer: NSObject, MTKViewDelegate {
    private let device: MTLDevice
    private let commandQueue: MTLCommandQueue
    private var textureCache: CVMetalTextureCache?
    private var pixelBuffer: CVPixelBuffer?
    private var previousPixelBuffer: CVPixelBuffer?
    
    private var integerPipeline: MTLComputePipelineState?
    private var easuPipeline: MTLComputePipelineState?
    private var rcasPipeline: MTLComputePipelineState?
    private var library: MTLLibrary?
    
    // MARK: - Optical Flow / FrameGen
    private var warpPipeline: MTLComputePipelineState?
    private var flowConsistentPipeline: MTLComputePipelineState?
    private var bicubicPipeline: MTLComputePipelineState?
    private var convertPipeline: MTLComputePipelineState?

    // MARK: - Capture Engine
    #if os(macOS)
    private var scStream: SCStream?
    private var scOutputProxy: AnyObject?
    private let captureQueue = DispatchQueue(label: "capture.queue", qos: .userInteractive)
    private var overlayWindow: NSWindow?
    private weak var overlayView: MTKView?
    private var overlayTrackingTimer: Timer?
    private var trackedWindowID: CGWindowID?
    private var eventTap: CFMachPort?
    private var eventTapRunLoopSource: CFRunLoopSource?
    private var targetPID: pid_t?

    /// Convert a CGWindow rect (global pixels, origin at top-left of main display) to a local NSView rect (points, origin at bottom-left of the given screen)
    private func localRect(from cgRect: CGRect, on screen: NSScreen) -> CGRect {
        guard let displayID = screen.deviceDescription[NSDeviceDescriptionKey("NSScreenNumber")] as? CGDirectDisplayID else {
            return .zero
        }
        let scale = screen.backingScaleFactor
        let screenPix = CGDisplayBounds(displayID)
        // Position of window within this screen in pixels (top-left origin)
        let localXpix = cgRect.origin.x - screenPix.origin.x
        let localYTopPix = cgRect.origin.y - screenPix.origin.y
        // Convert to bottom-left origin within this screen
        let localYBottomPix = screenPix.size.height - (localYTopPix + cgRect.size.height)
        // Convert pixels -> points for NSView frame
        let xPts = localXpix / scale
        let yPts = localYBottomPix / scale
        let wPts = cgRect.size.width / scale
        let hPts = cgRect.size.height / scale
        return CGRect(x: xPts, y: yPts, width: wPts, height: hPts)
    }
    
    private func eventMask(for types: [CGEventType]) -> CGEventMask {
        var m: CGEventMask = 0
        for t in types {
            m |= (CGEventMask(1) << CGEventMask(t.rawValue))
        }
        return m
    }

    #endif

    // MARK: - Configuration
    var enableFrameGeneration: Bool = false
    enum ScalingMode { case integer, bicubic, easu }
    var scalingMode: ScalingMode = .bicubic
    var useHalfFloatPipeline: Bool = false

    private var sharpenAmount: Float = 0

    init?(view: MTKView) {
        guard let device = view.device ?? MTLCreateSystemDefaultDevice(),
              let commandQueue = device.makeCommandQueue() else {
            return nil
        }

        self.device = device
        self.commandQueue = commandQueue
        
        view.isPaused = false
        view.enableSetNeedsDisplay = false
        view.preferredFramesPerSecond = 60
        
        view.framebufferOnly = false
        
        do {
            self.library = try device.makeDefaultLibrary(bundle: .main)
            if let lib = self.library {
                if let fn = lib.makeFunction(name: "integer_nearest_upscale") {
                    self.integerPipeline = try device.makeComputePipelineState(function: fn)
                }
                if let fn = lib.makeFunction(name: "bicubic_upscale") {
                    self.easuPipeline = try device.makeComputePipelineState(function: fn)
                }
                if let fn = lib.makeFunction(name: "rcas_stub") {
                    self.rcasPipeline = try device.makeComputePipelineState(function: fn)
                }
                if let fn = lib.makeFunction(name: "flow_warp") {
                    self.warpPipeline = try? device.makeComputePipelineState(function: fn)
                }
                if let fn = lib.makeFunction(name: "flow_warp_consistent") {
                    self.flowConsistentPipeline = try? device.makeComputePipelineState(function: fn)
                }
                if let fn = lib.makeFunction(name: "bicubic_upscale") {
                    self.bicubicPipeline = try? device.makeComputePipelineState(function: fn)
                }
                if let fn = lib.makeFunction(name: "convert_float_to_unorm") {
                    self.convertPipeline = try? device.makeComputePipelineState(function: fn)
                }
            }
        } catch {
            // Failed to create compute pipelines, but continue
        }

        super.init()

        CVMetalTextureCacheCreate(kCFAllocatorDefault, nil, device, nil, &textureCache)
        // Capture will be started explicitly via startCapture(for:) when needed
    }

    // MARK: - Capture Control (ScreenCaptureKit)
    #if os(macOS)
    private final class StreamOutputProxy: NSObject, SCStreamOutput {
        weak var owner: Renderer?
        init(owner: Renderer) { self.owner = owner }
        func stream(_ stream: SCStream, didOutputSampleBuffer sampleBuffer: CMSampleBuffer, of outputType: SCStreamOutputType) {
            guard outputType == .screen, let pb = CMSampleBufferGetImageBuffer(sampleBuffer) else { return }
            owner?.update(with: pb)
        }
    }

    func startCaptureForDisplay(_ displayID: CGDirectDisplayID) {
        stopCapture()
        Task { @MainActor in
            do {
                let content = try await SCShareableContent.current
                guard let scDisplay = content.displays.first(where: { $0.displayID == displayID }) else { return }
                let filter = SCContentFilter(display: scDisplay, excludingWindows: [])
                let config = SCStreamConfiguration()
                config.pixelFormat = kCVPixelFormatType_32BGRA
                config.colorSpaceName = CGColorSpace.sRGB
                config.scalesToFit = true
                config.capturesAudio = false
                config.minimumFrameInterval = CMTime(value: 1, timescale: 60)
                let stream = SCStream(filter: filter, configuration: config, delegate: nil)
                let proxy = StreamOutputProxy(owner: self)
                try stream.addStreamOutput(proxy, type: .screen, sampleHandlerQueue: captureQueue)
                try await stream.startCapture()
                self.scStream = stream
                self.scOutputProxy = proxy
                self.createOverlayWindow(forDisplay: scDisplay)
            } catch {
                print("ScreenCaptureKit display capture error: \(error)")
            }
        }
    }

    func startCaptureForWindow(windowID: CGWindowID) {
        stopCapture()
        Task { @MainActor in
            do {
                let content = try await SCShareableContent.current
                guard let scWindow = content.windows.first(where: { $0.windowID == windowID }) else { return }
                let filter = SCContentFilter(desktopIndependentWindow: scWindow)
                let config = SCStreamConfiguration()
                config.pixelFormat = kCVPixelFormatType_32BGRA
                config.colorSpaceName = CGColorSpace.sRGB
                config.scalesToFit = true
                config.capturesAudio = false
                config.minimumFrameInterval = CMTime(value: 1, timescale: 60)
                let stream = SCStream(filter: filter, configuration: config, delegate: nil)
                let proxy = StreamOutputProxy(owner: self)
                try stream.addStreamOutput(proxy, type: .screen, sampleHandlerQueue: captureQueue)
                try await stream.startCapture()
                self.scStream = stream
                self.scOutputProxy = proxy
                self.createOverlayWindow(forWindow: scWindow)
                self.trackedWindowID = windowID
                self.startOverlayTracking()
                if let owning = scWindow.owningApplication {
                    let pid = owning.processID
                    if let running = NSRunningApplication(processIdentifier: pid) {
                        running.activate(options: [.activateIgnoringOtherApps])
                    }
                    if let running = NSRunningApplication(processIdentifier: pid_t(owning.processID)) {
                        self.startEventForwarding(to: running.processIdentifier)
                    }
                }
            } catch {
                print("ScreenCaptureKit window capture error: \(error)")
            }
        }
    }

    func stopCapture() {
        guard let stream = scStream else { return }
        Task { @MainActor in
            do { try await stream.stopCapture() } catch { print("stopCapture error: \(error)") }
            self.scStream = nil
            self.scOutputProxy = nil
            self.destroyOverlayWindow()
            self.stopOverlayTracking()
            self.trackedWindowID = nil
            self.stopEventForwarding()
        }
    }

    // MARK: - Overlay Window (Lossless-like scaling)
    private func screenContaining(_ rect: CGRect) -> NSScreen? {
        let center = CGPoint(x: rect.midX, y: rect.midY)
        for screen in NSScreen.screens {
            if screen.frame.contains(center) { return screen }
        }
        return NSScreen.main
    }

    private func createOverlayWindow(forDisplay display: SCDisplay) {
        destroyOverlayWindow()
        let screen = NSScreen.screens.first { screen in
            if let num = screen.deviceDescription[NSDeviceDescriptionKey("NSScreenNumber")] as? UInt32 {
                return num == display.displayID
            }
            return false
        } ?? NSScreen.main
        let frame = screen?.frame ?? NSScreen.main?.frame ?? .zero
        let window = NSWindow(contentRect: frame, styleMask: [.borderless], backing: .buffered, defer: false)
        window.isOpaque = false
        window.backgroundColor = .clear
        window.hasShadow = false
        window.ignoresMouseEvents = true // click-through
        window.level = .popUpMenu
        window.collectionBehavior = [.canJoinAllSpaces, .fullScreenAuxiliary, .ignoresCycle]

        let container = NSView(frame: frame)
        container.wantsLayer = true
        container.layer?.backgroundColor = NSColor.clear.cgColor
        window.contentView = container

        let mtk = MTKView(frame: frame, device: device)
        mtk.isPaused = false
        mtk.enableSetNeedsDisplay = false
        mtk.preferredFramesPerSecond = 60
        mtk.framebufferOnly = false
        mtk.delegate = self
        mtk.autoresizingMask = [.width, .height]
        container.addSubview(mtk)

        window.orderFrontRegardless()
        self.overlayWindow = window
        self.overlayView = mtk
    }

    private func createOverlayWindow(forWindow window: SCWindow) {
        destroyOverlayWindow()
        let screen = screenContaining(window.frame) ?? NSScreen.main
        let frame = screen?.frame ?? NSScreen.main?.frame ?? .zero
        let ow = NSWindow(contentRect: frame, styleMask: [.borderless], backing: .buffered, defer: false)
        ow.isOpaque = false
        ow.backgroundColor = .clear
        ow.hasShadow = false
        ow.ignoresMouseEvents = true // click-through
        ow.level = .popUpMenu
        ow.collectionBehavior = [.canJoinAllSpaces, .fullScreenAuxiliary, .ignoresCycle]

        let container = NSView(frame: frame)
        container.wantsLayer = true
        container.layer?.backgroundColor = NSColor.clear.cgColor
        ow.contentView = container

        let local = localRect(from: window.frame, on: screen!)

        let mtk = MTKView(frame: local, device: device)
        mtk.isPaused = false
        mtk.enableSetNeedsDisplay = false
        mtk.preferredFramesPerSecond = 60
        mtk.framebufferOnly = false
        mtk.delegate = self
        mtk.autoresizingMask = []
        container.addSubview(mtk)

        ow.orderFrontRegardless()
        self.overlayWindow = ow
        self.overlayView = mtk

        if let owning = window.owningApplication {
            let pid = owning.processID
            if let running = NSRunningApplication(processIdentifier: pid) {
                running.activate(options: [.activateIgnoringOtherApps])
            }
        }
    }

    private func destroyOverlayWindow() {
        if let win = overlayWindow {
            win.orderOut(nil)
        }
        overlayView?.delegate = nil
        overlayView = nil
        overlayWindow = nil
    }
    
    // MARK: - Event Forwarding (mouse/keyboard)
    private func startEventForwarding(to pid: pid_t) {
        stopEventForwarding()
        targetPID = pid
        let mouseMask = eventMask(for: [
            .leftMouseDown, .leftMouseUp,
            .rightMouseDown, .rightMouseUp,
            .otherMouseDown, .otherMouseUp,
            .mouseMoved,
            .leftMouseDragged, .rightMouseDragged, .otherMouseDragged,
            .scrollWheel
        ])
        let keyMask = eventMask(for: [.keyDown, .keyUp])
        let mask = mouseMask | keyMask
        let callback: CGEventTapCallBack = { proxy, type, event, refcon in
            guard let refcon = refcon else { return Unmanaged.passUnretained(event) }
            let me = Unmanaged<Renderer>.fromOpaque(refcon).takeUnretainedValue()
            // Avoid re-forwarding events we injected
            let marker: Int64 = 0x53434C46 // 'SCLF'
            if event.getIntegerValueField(.eventSourceUserData) == marker {
                return Unmanaged.passUnretained(event)
            }
            guard let pid = me.targetPID else { return Unmanaged.passUnretained(event) }
            if let e2 = event.copy() {
                e2.setIntegerValueField(.eventSourceUserData, value: marker)
                // Post directly to target app
                e2.postToPid(pid)
            }
            // Swallow original so topmost overlay never interferes
            return nil
        }
        guard let tap = CGEvent.tapCreate(tap: .cgSessionEventTap,
                                          place: .headInsertEventTap,
                                          options: .defaultTap,
                                          eventsOfInterest: mask,
                                          callback: callback,
                                          userInfo: Unmanaged.passUnretained(self).toOpaque()) else {
            print("[Renderer] Failed to create event tap (Accessibility permission required)")
            return
        }
        eventTap = tap
        let src = CFMachPortCreateRunLoopSource(kCFAllocatorDefault, tap, 0)
        eventTapRunLoopSource = src
        CFRunLoopAddSource(CFRunLoopGetMain(), src, .commonModes)
        CGEvent.tapEnable(tap: tap, enable: true)
    }

    private func stopEventForwarding() {
        if let tap = eventTap {
            CGEvent.tapEnable(tap: tap, enable: false)
        }
        if let src = eventTapRunLoopSource {
            CFRunLoopRemoveSource(CFRunLoopGetMain(), src, .commonModes)
        }
        eventTapRunLoopSource = nil
        eventTap = nil
        targetPID = nil
    }
    
    // MARK: - Overlay Tracking
    private func startOverlayTracking() {
        stopOverlayTracking()
        DispatchQueue.main.async {
            self.overlayTrackingTimer = Timer.scheduledTimer(withTimeInterval: 1.0/30.0, repeats: true) { [weak self] _ in
                self?.updateOverlayFrameForTrackedWindow()
            }
        }
    }

    private func stopOverlayTracking() {
        overlayTrackingTimer?.invalidate()
        overlayTrackingTimer = nil
    }

    private func updateOverlayFrameForTrackedWindow() {
        guard let targetID = trackedWindowID, let ow = overlayWindow else { return }
        let options: CGWindowListOption = [.optionOnScreenOnly, .excludeDesktopElements]
        guard let infoList = CGWindowListCopyWindowInfo(options, kCGNullWindowID) as? [[String: Any]] else { return }
        guard let info = infoList.first(where: { ($0[kCGWindowNumber as String] as? UInt32) == targetID }),
              let boundsDict = info[kCGWindowBounds as String] as? [String: CGFloat],
              let x = boundsDict["X"],
              let y = boundsDict["Y"],
              let w = boundsDict["Width"],
              let h = boundsDict["Height"] else { return }
        let cgFrame = CGRect(x: x, y: y, width: w, height: h)
        guard let screen = screenContaining(cgFrame) ?? NSScreen.main else { return }
        let screenFrame = screen.frame
        let local = localRect(from: cgFrame, on: screen)
        if ow.frame != screenFrame { ow.setFrame(screenFrame, display: false) }
        if let mtk = overlayView, mtk.frame != local { mtk.frame = local }
    }
    #endif
    
    private func integerScaleAndOffset(src: MTLTexture, dst: MTLTexture) -> (UInt32, UInt32, UInt32)? {
        let scaleX = dst.width / src.width
        let scaleY = dst.height / src.height
        let scale = min(scaleX, scaleY)
        guard scale >= 2 else { return nil }
        let usedW = src.width * scale
        let usedH = src.height * scale
        let offX = (dst.width - usedW) / 2
        let offY = (dst.height - usedH) / 2
        return (UInt32(scale), UInt32(offX), UInt32(offY))
    }

    func update(with pixelBuffer: CVPixelBuffer) {
        // Keep previous for frame generation
        previousPixelBuffer = self.pixelBuffer
        // Optionally generate intermediate
        let processed = generateIntermediateFrameIfNeeded(current: pixelBuffer)
        self.pixelBuffer = processed
    }

    func generateIntermediateFrameIfNeeded(current: CVPixelBuffer) -> CVPixelBuffer {
        guard enableFrameGeneration, let prev = previousPixelBuffer, let textureCache else { return current }

        let w = CVPixelBufferGetWidth(current)
        let h = CVPixelBufferGetHeight(current)

        // Build Vision optical flow requests (forward: prev->curr, backward: curr->prev)
        let requestFwd = VNGenerateOpticalFlowRequest(targetedCVPixelBuffer: current, options: [:])
        requestFwd.computationAccuracy = .high
        requestFwd.outputPixelFormat = kCVPixelFormatType_TwoComponent32Float
        let requestBwd = VNGenerateOpticalFlowRequest(targetedCVPixelBuffer: prev, options: [:])
        requestBwd.computationAccuracy = .high
        requestBwd.outputPixelFormat = kCVPixelFormatType_TwoComponent32Float

        do {
            // Forward flow uses prev as source
            try VNSequenceRequestHandler().perform([requestFwd], on: prev)
            // Backward flow uses current as source
            try VNSequenceRequestHandler().perform([requestBwd], on: current)
        } catch {
            return current
        }
        guard let obsFwd = requestFwd.results?.first as? VNPixelBufferObservation,
              let obsBwd = requestBwd.results?.first as? VNPixelBufferObservation else { return current }

        let flowFwdPB = obsFwd.pixelBuffer
        let flowBwdPB = obsBwd.pixelBuffer

        // Create textures for prev, curr, flows, and output
        var cvTexPrev: CVMetalTexture?
        var cvTexCurr: CVMetalTexture?
        var cvTexFlowFwd: CVMetalTexture?
        var cvTexFlowBwd: CVMetalTexture?
        CVMetalTextureCacheCreateTextureFromImage(kCFAllocatorDefault, textureCache, prev, nil, .bgra8Unorm, w, h, 0, &cvTexPrev)
        CVMetalTextureCacheCreateTextureFromImage(kCFAllocatorDefault, textureCache, current, nil, .bgra8Unorm, w, h, 0, &cvTexCurr)
        CVMetalTextureCacheCreateTextureFromImage(kCFAllocatorDefault, textureCache, flowFwdPB, nil, .rg32Float, w, h, 0, &cvTexFlowFwd)
        CVMetalTextureCacheCreateTextureFromImage(kCFAllocatorDefault, textureCache, flowBwdPB, nil, .rg32Float, w, h, 0, &cvTexFlowBwd)
        guard let tPrev = cvTexPrev.flatMap({ CVMetalTextureGetTexture($0) }),
              let tCurr = cvTexCurr.flatMap({ CVMetalTextureGetTexture($0) }),
              let tFlowFwd = cvTexFlowFwd.flatMap({ CVMetalTextureGetTexture($0) }),
              let tFlowBwd = cvTexFlowBwd.flatMap({ CVMetalTextureGetTexture($0) }),
              let commandBuffer = commandQueue.makeCommandBuffer() else { return current }

        // Output pixel buffer + texture
        var outPB: CVPixelBuffer?
        var outTexRef: CVMetalTexture?
        let attrs: [CFString: Any] = [kCVPixelBufferMetalCompatibilityKey: true]
        CVPixelBufferCreate(kCFAllocatorDefault, w, h, kCVPixelFormatType_32BGRA, attrs as CFDictionary, &outPB)
        guard let outPBUnwrapped = outPB else { return current }
        CVMetalTextureCacheCreateTextureFromImage(kCFAllocatorDefault, textureCache, outPBUnwrapped, nil, .bgra8Unorm, w, h, 0, &outTexRef)
        guard let outTex = outTexRef.flatMap({ CVMetalTextureGetTexture($0) }) else { return current }

        let t: Float = 0.5
        let consistencyThreshold: Float = 1.0 // pixels; tweakable

        if let pipeline = self.flowConsistentPipeline, let encoder = commandBuffer.makeComputeCommandEncoder() {
            encoder.setComputePipelineState(pipeline)
            encoder.setTexture(tPrev, index: 0)
            encoder.setTexture(tCurr, index: 1)
            encoder.setTexture(tFlowFwd, index: 2)
            encoder.setTexture(tFlowBwd, index: 3)
            encoder.setTexture(outTex, index: 4)
            var tVal = t
            var thr = consistencyThreshold
            encoder.setBytes(&tVal, length: MemoryLayout<Float>.size, index: 0)
            encoder.setBytes(&thr, length: MemoryLayout<Float>.size, index: 1)
            let tg = MTLSize(width: 16, height: 16, depth: 1)
            let ng = MTLSize(width: (w + tg.width - 1) / tg.width, height: (h + tg.height - 1) / tg.height, depth: 1)
            encoder.dispatchThreadgroups(ng, threadsPerThreadgroup: tg)
            encoder.endEncoding()
        } else if let pipeline = self.warpPipeline, let encoder = commandBuffer.makeComputeCommandEncoder() {
            // Fallback: single-flow warp at t=0.5 using forward flow only
            encoder.setComputePipelineState(pipeline)
            encoder.setTexture(tPrev, index: 0)
            encoder.setTexture(tCurr, index: 1)
            encoder.setTexture(tFlowFwd, index: 2)
            encoder.setTexture(outTex, index: 3)
            let tg = MTLSize(width: 16, height: 16, depth: 1)
            let ng = MTLSize(width: (w + tg.width - 1) / tg.width, height: (h + tg.height - 1) / tg.height, depth: 1)
            encoder.dispatchThreadgroups(ng, threadsPerThreadgroup: tg)
            encoder.endEncoding()
        }

        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()
        return outPBUnwrapped
    }

    func updatePostProcess(sharpen: Float) {
        self.sharpenAmount = max(0, sharpen)
    }

    func draw(in view: MTKView) {
        guard let drawable = view.currentDrawable, let textureCache = self.textureCache else { return }
        guard let pixelBuffer = self.pixelBuffer else {
            if let commandBuffer = commandQueue.makeCommandBuffer() {
                let rpd = MTLRenderPassDescriptor()
                rpd.colorAttachments[0].texture = drawable.texture
                rpd.colorAttachments[0].loadAction = .clear
                rpd.colorAttachments[0].storeAction = .store
                rpd.colorAttachments[0].clearColor = MTLClearColorMake(0, 0, 0, 1)
                if let enc = commandBuffer.makeRenderCommandEncoder(descriptor: rpd) {
                    enc.endEncoding()
                }
                commandBuffer.present(drawable)
                commandBuffer.commit()
            }
            return
        }

        let srcWidth = CVPixelBufferGetWidth(pixelBuffer)
        let srcHeight = CVPixelBufferGetHeight(pixelBuffer)

        var cvTexture: CVMetalTexture?
        CVMetalTextureCacheCreateTextureFromImage(kCFAllocatorDefault,
                                                  textureCache,
                                                  pixelBuffer,
                                                  nil,
                                                  .bgra8Unorm,
                                                  srcWidth,
                                                  srcHeight,
                                                  0,
                                                  &cvTexture)

        guard let sourceTexture = cvTexture.flatMap({ CVMetalTextureGetTexture($0) }) else {
            return
        }

        guard let commandBuffer = commandQueue.makeCommandBuffer() else { return }

        // Compute upscale (Integer/Bicubic/EASU) then optional RCAS sharpen, with optional half-float working chain
        let workingFormat: MTLPixelFormat = useHalfFloatPipeline ? .rgba16Float : drawable.texture.pixelFormat
        var upscaledTex: MTLTexture? = nil
        var blitOrigin = MTLOrigin(x: 0, y: 0, z: 0)
        var blitSize = MTLSize(width: drawable.texture.width, height: drawable.texture.height, depth: 1)

        switch scalingMode {
        case .integer:
            if let (scale, offX, offY) = integerScaleAndOffset(src: sourceTexture, dst: drawable.texture) {
                let usedW = Int(sourceTexture.width) * Int(scale)
                let usedH = Int(sourceTexture.height) * Int(scale)
                let scaleDesc = MTLTextureDescriptor.texture2DDescriptor(pixelFormat: workingFormat, width: usedW, height: usedH, mipmapped: false)
                scaleDesc.usage = [.shaderRead, .shaderWrite]
                if let scaled = device.makeTexture(descriptor: scaleDesc), let integerPipeline, let encoder = commandBuffer.makeComputeCommandEncoder() {
                    encoder.setComputePipelineState(integerPipeline)
                    encoder.setTexture(sourceTexture, index: 0)
                    encoder.setTexture(scaled, index: 1)
                    let tg = MTLSize(width: 16, height: 16, depth: 1)
                    let ng = MTLSize(width: (usedW + tg.width - 1) / tg.width, height: (usedH + tg.height - 1) / tg.height, depth: 1)
                    encoder.dispatchThreadgroups(ng, threadsPerThreadgroup: tg)
                    encoder.endEncoding()
                    upscaledTex = scaled
                    blitOrigin = MTLOrigin(x: Int(offX), y: Int(offY), z: 0)
                    blitSize = MTLSize(width: usedW, height: usedH, depth: 1)
                }
            }
        case .bicubic:
            let outDesc = MTLTextureDescriptor.texture2DDescriptor(pixelFormat: workingFormat, width: drawable.texture.width, height: drawable.texture.height, mipmapped: false)
            outDesc.usage = [.shaderRead, .shaderWrite]
            if let out = device.makeTexture(descriptor: outDesc), let encoder = commandBuffer.makeComputeCommandEncoder() {
                if let bicubicPipeline {
                    encoder.setComputePipelineState(bicubicPipeline)
                } else if let easuPipeline {
                    encoder.setComputePipelineState(easuPipeline)
                }
                encoder.setTexture(sourceTexture, index: 0)
                encoder.setTexture(out, index: 1)
                let tg = MTLSize(width: 16, height: 16, depth: 1)
                let ng = MTLSize(width: (out.width + tg.width - 1) / tg.width, height: (out.height + tg.height - 1) / tg.height, depth: 1)
                encoder.dispatchThreadgroups(ng, threadsPerThreadgroup: tg)
                encoder.endEncoding()
                upscaledTex = out
            }
        case .easu:
            let outDesc = MTLTextureDescriptor.texture2DDescriptor(pixelFormat: workingFormat, width: drawable.texture.width, height: drawable.texture.height, mipmapped: false)
            outDesc.usage = [.shaderRead, .shaderWrite]
            if let out = device.makeTexture(descriptor: outDesc), let easuPipeline, let encoder = commandBuffer.makeComputeCommandEncoder() {
                encoder.setComputePipelineState(easuPipeline)
                encoder.setTexture(sourceTexture, index: 0)
                encoder.setTexture(out, index: 1)
                let tg = MTLSize(width: 16, height: 16, depth: 1)
                let ng = MTLSize(width: (out.width + tg.width - 1) / tg.width, height: (out.height + tg.height - 1) / tg.height, depth: 1)
                encoder.dispatchThreadgroups(ng, threadsPerThreadgroup: tg)
                encoder.endEncoding()
                upscaledTex = out
            }
        }

        // Optional sharpening (RCAS-like) in working format
        var postTex = upscaledTex
        if let rcasPipeline, let inTex = postTex {
            let desc = MTLTextureDescriptor.texture2DDescriptor(pixelFormat: workingFormat, width: inTex.width, height: inTex.height, mipmapped: false)
            desc.usage = [.shaderRead, .shaderWrite]
            if let out = device.makeTexture(descriptor: desc), let encoder = commandBuffer.makeComputeCommandEncoder() {
                encoder.setComputePipelineState(rcasPipeline)
                encoder.setTexture(inTex, index: 0)
                encoder.setTexture(out, index: 1)
                var amount = self.sharpenAmount
                encoder.setBytes(&amount, length: MemoryLayout<Float>.size, index: 0)
                let tg = MTLSize(width: 16, height: 16, depth: 1)
                let ng = MTLSize(width: (out.width + tg.width - 1) / tg.width, height: (out.height + tg.height - 1) / tg.height, depth: 1)
                encoder.dispatchThreadgroups(ng, threadsPerThreadgroup: tg)
                encoder.endEncoding()
                postTex = out
            }
        }

        // Convert to drawable pixel format if using half-float
        var presentTex: MTLTexture? = postTex
        if useHalfFloatPipeline, let post = postTex {
            let presentDesc = MTLTextureDescriptor.texture2DDescriptor(pixelFormat: drawable.texture.pixelFormat, width: post.width, height: post.height, mipmapped: false)
            presentDesc.usage = [.shaderRead, .shaderWrite]
            if let out = device.makeTexture(descriptor: presentDesc), let convertPipeline, let encoder = commandBuffer.makeComputeCommandEncoder() {
                encoder.setComputePipelineState(convertPipeline)
                encoder.setTexture(post, index: 0)
                encoder.setTexture(out, index: 1)
                let tg = MTLSize(width: 16, height: 16, depth: 1)
                let ng = MTLSize(width: (out.width + tg.width - 1) / tg.width, height: (out.height + tg.height - 1) / tg.height, depth: 1)
                encoder.dispatchThreadgroups(ng, threadsPerThreadgroup: tg)
                encoder.endEncoding()
                presentTex = out
            }
        }

        if let finalTex = presentTex {
            if !drawable.texture.isFramebufferOnly, let blit = commandBuffer.makeBlitCommandEncoder() {
                blit.copy(from: finalTex,
                          sourceSlice: 0,
                          sourceLevel: 0,
                          sourceOrigin: .init(x: 0, y: 0, z: 0),
                          sourceSize: blitSize,
                          to: drawable.texture,
                          destinationSlice: 0,
                          destinationLevel: 0,
                          destinationOrigin: blitOrigin)
                blit.endEncoding()
            } else {
                // Fallback: cannot blit to framebufferOnly; no-op because view.framebufferOnly is disabled in init.
            }
        }

        commandBuffer.present(drawable)
        commandBuffer.commit()
    }

    func mtkView(_ view: MTKView, drawableSizeWillChange size: CGSize) { }
}

