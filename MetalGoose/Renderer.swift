import MetalKit
import MetalFX
import Vision
import CoreVideo
import QuartzCore
import AppKit
import ApplicationServices

final class Renderer: NSObject, MTKViewDelegate {
    let device: MTLDevice
    let commandQueue: MTLCommandQueue
    var textureCache: CVMetalTextureCache?
    
    // Pipelines
    var flowPipeline: MTLComputePipelineState?
    var integerPipeline: MTLComputePipelineState?
    var bicubicPipeline: MTLComputePipelineState?
    var renderPipeline: MTLRenderPipelineState?
    
    // Processor & Scaler
    var spatialScaler: MTLFXSpatialScaler?
    var spatialScalerDesc: MTLFXSpatialScalerDescriptor?
    
    // Vision
    let sequenceHandler = VNSequenceRequestHandler()
    
    // State
    var settings: CaptureSettings
    var prevTexture: MTLTexture?
    var prevBuffer: CVPixelBuffer?
    var privateOutputTexture: MTLTexture?
    weak var hostView: MTKView?
    private var fpsLayer: CATextLayer?
    private var lastFPSTimestamp: CFTimeInterval = CACurrentMediaTime()
    private var smoothedFPS: Double = 0.0
    
    private var trackedAXWindow: AXUIElement?
    private var trackedPID: pid_t = 0
    
    // Window Tracking
    var overlayWindow: NSWindow?
    var trackedWindowID: CGWindowID = 0
    var trackingTimer: Timer?
    
    init?(metalKitView: MTKView, settings: CaptureSettings) {
        guard let dev = MTLCreateSystemDefaultDevice(),
              let queue = dev.makeCommandQueue() else { return nil }
        
        self.device = dev
        self.commandQueue = queue
        self.settings = settings
        
        super.init()
        
        self.hostView = metalKitView
        
        metalKitView.device = dev
        metalKitView.isPaused = false
        metalKitView.enableSetNeedsDisplay = true
        metalKitView.preferredFramesPerSecond = settings.vsync ? 60 : 0
        metalKitView.framebufferOnly = false
        metalKitView.colorPixelFormat = .bgra8Unorm
        metalKitView.clearColor = MTLClearColor(red: 0, green: 0, blue: 0, alpha: 0)
        metalKitView.delegate = self
        metalKitView.layer?.isOpaque = false
        setupHUDLayerIfNeeded(on: metalKitView)
        
        CVMetalTextureCacheCreate(kCFAllocatorDefault, nil, device, nil, &textureCache)
        setupPipelines()
    }
    
    func setupPipelines() {
        guard let lib = try? device.makeDefaultLibrary(bundle: .main) else { return }
        
        if let fn = lib.makeFunction(name: "flow_warp") { flowPipeline = try? device.makeComputePipelineState(function: fn) }
        if let fn = lib.makeFunction(name: "integer_scale") { integerPipeline = try? device.makeComputePipelineState(function: fn) }
        if let fn = lib.makeFunction(name: "bicubic_scale") { bicubicPipeline = try? device.makeComputePipelineState(function: fn) }
        
        let pipeDesc = MTLRenderPipelineDescriptor()
        pipeDesc.vertexFunction = lib.makeFunction(name: "texture_vertex")
        pipeDesc.fragmentFunction = lib.makeFunction(name: "texture_fragment")
        pipeDesc.colorAttachments[0].pixelFormat = .bgra8Unorm
        pipeDesc.colorAttachments[0].isBlendingEnabled = true
        pipeDesc.colorAttachments[0].rgbBlendOperation = .add
        pipeDesc.colorAttachments[0].sourceRGBBlendFactor = .sourceAlpha
        pipeDesc.colorAttachments[0].destinationRGBBlendFactor = .oneMinusSourceAlpha
        renderPipeline = try? device.makeRenderPipelineState(descriptor: pipeDesc)
    }
    
    func processAndRender(buffer: CVPixelBuffer, view: MTKView) {
        setupHUDLayerIfNeeded(on: view)
        // 1. FRAME PROCESSOR STEP (VideoToolbox Scaling/Converting)
        let profile = settings.qualityMode.profile
        let processedBuffer = buffer
        
        guard let drawable = view.currentDrawable,
              let cmdBuffer = commandQueue.makeCommandBuffer(),
              let srcTex = createTexture(from: processedBuffer) else { return }
        
        var texToScale = srcTex
        
        // 2. FRAME GENERATION (Vision)
        if settings.frameGenMode != .off, let pTex = prevTexture, let pBuf = prevBuffer {
            // Ensure size/format match to avoid Vision assertions
            let w0 = CVPixelBufferGetWidth(processedBuffer)
            let h0 = CVPixelBufferGetHeight(processedBuffer)
            let w1 = CVPixelBufferGetWidth(pBuf)
            let h1 = CVPixelBufferGetHeight(pBuf)
            let f0 = CVPixelBufferGetPixelFormatType(processedBuffer)
            let f1 = CVPixelBufferGetPixelFormatType(pBuf)
            if w0 == w1, h0 == h1, f0 == f1 {
                let request = VNGenerateOpticalFlowRequest(targetedCVPixelBuffer: processedBuffer, options: [:])
                request.computationAccuracy = profile.flowAccuracy
                request.outputPixelFormat = kCVPixelFormatType_TwoComponent32Float

                let handler = VNSequenceRequestHandler()
                do {
                    try handler.perform([request], on: pBuf, orientation: .up)
                    if let res = request.results?.first, let flowTex = createTexture(from: res.pixelBuffer) {
                        let midDesc = MTLTextureDescriptor.texture2DDescriptor(pixelFormat: .bgra8Unorm, width: srcTex.width, height: srcTex.height, mipmapped: false)
                        midDesc.usage = [.shaderRead, .shaderWrite]

                        if let midTex = device.makeTexture(descriptor: midDesc),
                           let enc = cmdBuffer.makeComputeCommandEncoder(),
                           let pipe = flowPipeline {

                            enc.setComputePipelineState(pipe)
                            enc.setTexture(pTex, index: 0)
                            enc.setTexture(srcTex, index: 1)
                            enc.setTexture(flowTex, index: 2)
                            enc.setTexture(midTex, index: 3)
                            var t: Float = 0.5
                            enc.setBytes(&t, length: 4, index: 0)

                            let w = pipe.threadExecutionWidth
                            let h = pipe.maxTotalThreadsPerThreadgroup / w
                            let grids = MTLSize(width: (srcTex.width + w - 1)/w, height: (srcTex.height + h - 1)/h, depth: 1)
                            enc.dispatchThreadgroups(grids, threadsPerThreadgroup: MTLSize(width: w, height: h, depth: 1))
                            enc.endEncoding()

                            texToScale = midTex
                        }
                    }
                } catch {
                    // Skip MGFG on Vision error
                }
            } else {
                // Mismatch in size/format; skip MGFG for this frame
            }
        }
        
        prevTexture = srcTex
        prevBuffer = buffer
        
        // 3. UPSCALING (MetalFX or Shader)
        let dstTex = drawable.texture
        
        if settings.scalingType == .metalFX {
            setupMetalFX(src: texToScale, dst: dstTex, profile: profile)
            
            // Private Texture kontrolü
            if privateOutputTexture == nil || privateOutputTexture?.width != dstTex.width || privateOutputTexture?.height != dstTex.height {
                let desc = MTLTextureDescriptor.texture2DDescriptor(pixelFormat: dstTex.pixelFormat, width: dstTex.width, height: dstTex.height, mipmapped: false)
                desc.storageMode = .private
                desc.usage = [.shaderRead, .shaderWrite, .renderTarget]
                privateOutputTexture = device.makeTexture(descriptor: desc)
            }
            
            if let scaler = spatialScaler, let outTex = privateOutputTexture {
                scaler.colorTexture = texToScale
                scaler.outputTexture = outTex
                scaler.encode(commandBuffer: cmdBuffer)
                
                let blit = cmdBuffer.makeBlitCommandEncoder()
                blit?.copy(from: outTex, to: dstTex)
                blit?.endEncoding()
            }
        } else {
            // Integer / Bicubic via compute into intermediate texture, then blit to drawable
            // Ensure private output matches destination size
            if privateOutputTexture == nil || privateOutputTexture?.width != dstTex.width || privateOutputTexture?.height != dstTex.height || privateOutputTexture?.pixelFormat != dstTex.pixelFormat {
                let desc = MTLTextureDescriptor.texture2DDescriptor(pixelFormat: dstTex.pixelFormat, width: dstTex.width, height: dstTex.height, mipmapped: false)
                desc.storageMode = .private
                desc.usage = [.shaderRead, .shaderWrite, .renderTarget]
                privateOutputTexture = device.makeTexture(descriptor: desc)
            }
            guard let outTex = privateOutputTexture else { return }

            guard let enc = cmdBuffer.makeComputeCommandEncoder() else { return }
            let pipe = (settings.scalingType == .integer) ? integerPipeline : bicubicPipeline

            if let p = pipe {
                enc.setComputePipelineState(p)
                enc.setTexture(texToScale, index: 0)
                enc.setTexture(outTex, index: 1)

                if settings.scalingType == .integer {
                    var factor = settings.scaleFactor
                    enc.setBytes(&factor, length: 4, index: 0)
                }

                let w = p.threadExecutionWidth
                let h = p.maxTotalThreadsPerThreadgroup / w
                let grids = MTLSize(width: (outTex.width + w - 1)/w, height: (outTex.height + h - 1)/h, depth: 1)
                enc.dispatchThreadgroups(grids, threadsPerThreadgroup: MTLSize(width: w, height: h, depth: 1))
            }
            enc.endEncoding()

            if let blit = cmdBuffer.makeBlitCommandEncoder() {
                blit.copy(from: outTex, to: dstTex)
                blit.endEncoding()
            }
        }
        
        cmdBuffer.present(drawable)
        cmdBuffer.addCompletedHandler { [weak self] _ in
            guard let self = self else { return }
            let now = CACurrentMediaTime()
            let dt = now - self.lastFPSTimestamp
            self.lastFPSTimestamp = now
            let inst = dt > 0 ? (1.0 / dt) : 0.0
            self.smoothedFPS = self.smoothedFPS == 0 ? inst : (self.smoothedFPS * 0.85 + inst * 0.15)
            if self.settings.showFPS, let layer = self.fpsLayer {
                DispatchQueue.main.async {
                    layer.string = String(format: "FPS: %.0f", self.smoothedFPS)
                }
            }
        }
        cmdBuffer.commit()
    }
    
    func setupMetalFX(src: MTLTexture, dst: MTLTexture, profile: QualityProfile) {
        if spatialScaler == nil || spatialScalerDesc?.outputWidth != dst.width || spatialScalerDesc?.colorProcessingMode != profile.scalerMode {
            let desc = MTLFXSpatialScalerDescriptor()
            desc.inputWidth = src.width
            desc.inputHeight = src.height
            desc.outputWidth = dst.width
            desc.outputHeight = dst.height
            desc.colorTextureFormat = src.pixelFormat
            desc.outputTextureFormat = dst.pixelFormat
            desc.colorProcessingMode = .linear
            
            spatialScalerDesc = desc
            spatialScaler = desc.makeSpatialScaler(device: device)
        }
    }
    
    private func setupHUDLayerIfNeeded(on view: MTKView) {
        guard settings.showFPS else {
            fpsLayer?.removeFromSuperlayer()
            fpsLayer = nil
            return
        }
        if fpsLayer == nil {
            let layer = CATextLayer()
            layer.contentsScale = view.window?.backingScaleFactor ?? (view.layer?.contentsScale ?? 2.0)
            layer.font = NSFont.monospacedDigitSystemFont(ofSize: 12, weight: .regular)
            layer.fontSize = 12
            layer.alignmentMode = .left
            layer.foregroundColor = NSColor.white.cgColor
            layer.backgroundColor = NSColor.black.withAlphaComponent(0.55).cgColor
            layer.cornerRadius = 4
            layer.masksToBounds = true
            layer.string = "FPS: --"
            let h: CGFloat = 20
            layer.frame = CGRect(x: 8, y: (view.bounds.height - h - 8), width: 120, height: h)
            view.layer?.addSublayer(layer)
            fpsLayer = layer
        }
    }
    
    private func createTexture(from buffer: CVPixelBuffer) -> MTLTexture? {
        var cvTex: CVMetalTexture?
        let w = CVPixelBufferGetWidth(buffer)
        let h = CVPixelBufferGetHeight(buffer)
        guard let cache = textureCache else { return nil }
        let cvFormat = CVPixelBufferGetPixelFormatType(buffer)
        let metalFormat: MTLPixelFormat
        switch cvFormat {
        case kCVPixelFormatType_32BGRA:
            metalFormat = .bgra8Unorm
        case kCVPixelFormatType_TwoComponent32Float:
            metalFormat = .rg32Float
        default:
            metalFormat = .bgra8Unorm
        }
        CVMetalTextureCacheCreateTextureFromImage(kCFAllocatorDefault, cache, buffer, nil, metalFormat, w, h, 0, &cvTex)
        return cvTex.flatMap { CVMetalTextureGetTexture($0) }
    }
    
    // Tracking & Overlay
    func createOverlayWindow(targetFrame: CGRect) -> NSWindow {
        let panel = NSPanel(contentRect: targetFrame,
                            styleMask: [.nonactivatingPanel, .borderless],
                            backing: .buffered,
                            defer: false)
        panel.backgroundColor = .clear
        panel.isOpaque = false
        panel.hasShadow = false
        panel.level = .floating
        panel.ignoresMouseEvents = true
        panel.collectionBehavior = [.canJoinAllSpaces, .fullScreenAuxiliary, .ignoresCycle, .stationary]
        panel.hidesOnDeactivate = false
        panel.becomesKeyOnlyIfNeeded = false
        return panel
    }
    
    func startTracking(windowID: CGWindowID, pid: Int32, overlay: NSWindow) {
        self.trackedWindowID = windowID
        self.trackedPID = pid
        self.overlayWindow = overlay

        // Build AX window reference for faster tracking
        let appElement = AXUIElementCreateApplication(pid)
        var windowsRef: CFTypeRef?
        if AXUIElementCopyAttributeValue(appElement, kAXWindowsAttribute as CFString, &windowsRef) == .success,
           let windows = windowsRef as? [AXUIElement] {
            self.trackedAXWindow = windows.first
        } else {
            self.trackedAXWindow = nil
        }

        DispatchQueue.main.async {
            self.trackingTimer?.invalidate()
            self.trackingTimer = Timer.scheduledTimer(withTimeInterval: 0.016, repeats: true) { [weak self] _ in
                guard let self = self else { return }
                if self.trackedAXWindow != nil {
                    self.updateOverlayPositionAX()
                } else {
                    self.updateOverlayPosition()
                }
            }
        }
    }
    
    func stopTracking() {
        trackingTimer?.invalidate()
        trackingTimer = nil
    }
    
    deinit {
        trackingTimer?.invalidate()
    }
    
    private func updateOverlayPositionAX() {
        guard let overlay = overlayWindow, let axWin = trackedAXWindow else { return }
        var posValue: CFTypeRef?
        var sizeValue: CFTypeRef?
        AXUIElementCopyAttributeValue(axWin, kAXPositionAttribute as CFString, &posValue)
        AXUIElementCopyAttributeValue(axWin, kAXSizeAttribute as CFString, &sizeValue)

        var pos = CGPoint.zero
        var size = CGSize.zero
        if let p = posValue, CFGetTypeID(p) == AXValueGetTypeID() {
            AXValueGetValue(p as! AXValue, .cgPoint, &pos)
        }
        if let s = sizeValue, CFGetTypeID(s) == AXValueGetTypeID() {
            AXValueGetValue(s as! AXValue, .cgSize, &size)
        }
        guard size.width > 0, size.height > 0 else { return }

        // Convert to Cocoa coordinates using union of screens
        let union = NSScreen.screens.reduce(NSRect.null) { $0.union($1.frame) }
        let cocoaX = union.minX + pos.x
        let cocoaY = union.maxY - (pos.y + size.height)
        let newFrame = CGRect(x: cocoaX, y: cocoaY, width: size.width, height: size.height)
        if overlay.frame != newFrame {
            overlay.setFrame(newFrame, display: true, animate: false)
        }
    }
    
    private func updateOverlayPosition() {
        guard let overlay = overlayWindow else { return }
        guard let list = CGWindowListCopyWindowInfo([.optionIncludingWindow], trackedWindowID) as? [[String: Any]],
              let info = list.first,
              let bounds = info[kCGWindowBounds as String] as? [String: CGFloat] else {
            stopTracking()
            overlay.orderOut(nil)
            return
        }

        let x = bounds["X"] ?? 0
        let y = bounds["Y"] ?? 0
        let w = bounds["Width"] ?? 100
        let h = bounds["Height"] ?? 100

        // Compute union rect of all screens in Cocoa coordinates
        let union = NSScreen.screens.reduce(NSRect.null) { $0.union($1.frame) }
        let cocoaX = union.minX + x
        let cocoaY = union.maxY - (y + h)
        let newFrame = CGRect(x: cocoaX, y: cocoaY, width: w, height: h)

        if overlay.frame != newFrame {
            overlay.setFrame(newFrame, display: true, animate: false)
        }
    }
    
    // Resource cleanup for safe shutdown
    func cleanup() {
        stopTracking()
        prevTexture = nil
        prevBuffer = nil
        privateOutputTexture = nil
        spatialScaler = nil
        spatialScalerDesc = nil
    }
    
    // Delegate Stub
    func mtkView(_ view: MTKView, drawableSizeWillChange size: CGSize) {
        spatialScaler = nil
        privateOutputTexture = nil
        if let cache = textureCache { CVMetalTextureCacheFlush(cache, 0) }
    }
    func draw(in view: MTKView) {}
}

