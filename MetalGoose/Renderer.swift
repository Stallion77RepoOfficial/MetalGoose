import Foundation
import MetalKit
import MetalFX
import Vision
import VideoToolbox
import CoreVideo
import QuartzCore

final class Renderer: NSObject, MTKViewDelegate {
    let device: MTLDevice
    let commandQueue: MTLCommandQueue
    private weak var view: MTKView?
    var textureCache: CVMetalTextureCache?
    private let frameLock = NSLock()
    private var pendingBuffer: CVPixelBuffer?
    private let frameProcessor = FrameProcessor()
    
    // MetalFX
    var spatialScaler: MTLFXSpatialScaler?
    var spatialScalerDesc: MTLFXSpatialScalerDescriptor?
    
    // Vision / Frame Gen
    let sequenceHandler = VNSequenceRequestHandler()
    var flowPipeline: MTLComputePipelineState?
    
    // State
    var settings: CaptureSettings
    var prevTexture: MTLTexture?
    var prevBuffer: CVPixelBuffer?
    var privateOutputTexture: MTLTexture?
    private var fpsLayer: CATextLayer?
    private weak var overlayContentView: NSView?
    private var showFPSOverlay = false
    private var lastFrameTimestamp: CFAbsoluteTime?
    private var smoothedFPS: Double = 0
    private let maxFlowPixels: Int = 16_777_216 // 4096^2 guard for Vision
    
    // Window Tracking
    var overlayWindow: NSWindow?
    var trackedWindowID: CGWindowID = 0
    var trackingTimer: Timer?
    private var trackingFailureCount = 0
    
    init?(metalKitView: MTKView, settings: CaptureSettings) {
        guard let dev = MTLCreateSystemDefaultDevice(),
              let queue = dev.makeCommandQueue() else { return nil }
        
        self.device = dev
        self.commandQueue = queue
        self.settings = settings
        self.view = metalKitView
        
        super.init()
        
        metalKitView.device = dev
        metalKitView.framebufferOnly = false
        metalKitView.colorPixelFormat = .bgra8Unorm
        metalKitView.clearColor = MTLClearColor(red: 0, green: 0, blue: 0, alpha: 0)
        metalKitView.delegate = self
        metalKitView.layer?.isOpaque = false
        metalKitView.isPaused = true
        metalKitView.enableSetNeedsDisplay = true
        
        CVMetalTextureCacheCreate(kCFAllocatorDefault, nil, device, nil, &textureCache)
        
        setupPipelines()
    }

    func enqueue(buffer: CVPixelBuffer) {
        frameLock.lock()
        pendingBuffer = buffer
        frameLock.unlock()
        DispatchQueue.main.async { [weak self] in
            self?.view?.draw()
        }
    }

    private func nextBuffer() -> CVPixelBuffer? {
        frameLock.lock()
        defer { frameLock.unlock() }
        let buffer = pendingBuffer
        pendingBuffer = nil
        return buffer
    }
    
    func setupPipelines() {
        guard let lib = try? device.makeDefaultLibrary(bundle: .main) else { return }
        if let fn = lib.makeFunction(name: "flow_warp") {
            flowPipeline = try? device.makeComputePipelineState(function: fn)
        }
    }
    
    // MARK: - Window Tracking Logic
    func startTracking(windowID: CGWindowID, overlay: NSWindow) {
        self.trackedWindowID = windowID
        self.overlayWindow = overlay
        
        // Check window position 60 times a second
        DispatchQueue.main.async {
            self.trackingTimer = Timer.scheduledTimer(withTimeInterval: 1.0/60.0, repeats: true) { [weak self] _ in
                self?.updateOverlayPosition()
            }
        }
    }
    
    func stopTracking() {
        trackingTimer?.invalidate()
        trackingTimer = nil
    }
    
    private func updateOverlayPosition() {
        guard let overlay = overlayWindow else { return }
        
        // Get window info using CoreGraphics
        let list = CGWindowListCopyWindowInfo([.optionIncludingWindow], trackedWindowID) as? [[String: Any]]
        
        if let info = list?.first,
           let boundsDict = info[kCGWindowBounds as String] as? [String: CGFloat] {
            
            trackingFailureCount = 0
            
            let x = boundsDict["X"] ?? 0
            let y = boundsDict["Y"] ?? 0
            let w = boundsDict["Width"] ?? 100
            let h = boundsDict["Height"] ?? 100
            
            // Convert to macOS coordinates (Bottom-Left origin)
            let screenH = NSScreen.main?.frame.height ?? 1080
            let newY = screenH - (y + h)
            
            let newFrame = CGRect(x: x, y: newY, width: w, height: h)
            
            // Update position only if changed
            if overlay.frame != newFrame {
                overlay.setFrame(newFrame, display: true, animate: false)
            }
            
            // Ensure overlay stays visible and on top
            if !overlay.isVisible {
                overlay.orderFront(nil)
            }
            // Periodically bring to front to handle other overlays or focus changes
            overlay.orderFront(nil)
            
        } else {
            // Window might be closed or temporarily unavailable (e.g. during Alt-Tab or Space switch)
            // We allow a grace period before giving up
            trackingFailureCount += 1
            if trackingFailureCount > 300 { // ~5 seconds at 60Hz
                stopTracking()
                overlay.orderOut(nil)
            }
        }
    }
    
    // MARK: - Rendering
    func mtkView(_ view: MTKView, drawableSizeWillChange size: CGSize) {
        spatialScaler = nil
        privateOutputTexture = nil
    }
    
    func draw(in view: MTKView) {
        guard let buffer = nextBuffer() else { return }
          let profile = settings.qualityMode.profile
          let workingBuffer = frameProcessor.prepare(buffer: buffer,
                                       scaleFactor: settings.scaleFactor,
                                       scalingMode: profile.vtScalingMode)
          guard let drawable = view.currentDrawable,
              let cmdBuffer = commandQueue.makeCommandBuffer(),
              let srcTex = createTexture(from: workingBuffer) else { return }
        
        var texToScale = srcTex
        
        // 1. Frame Generation
          if let pTex = prevTexture,
              let pBuf = prevBuffer,
              shouldRunFrameGeneration(buffer: workingBuffer) {
            let request = VNGenerateOpticalFlowRequest(targetedCVPixelBuffer: workingBuffer, options: [:])
            request.computationAccuracy = profile.flowAccuracy
            request.outputPixelFormat = kCVPixelFormatType_TwoComponent32Float
            
            try? sequenceHandler.perform([request], on: pBuf, orientation: .up)
            
            if let res = request.results?.first,
               let flowTex = createTexture(from: res.pixelBuffer) {
                
                let midTexDesc = MTLTextureDescriptor.texture2DDescriptor(pixelFormat: .bgra8Unorm, width: srcTex.width, height: srcTex.height, mipmapped: false)
                midTexDesc.usage = [.shaderRead, .shaderWrite]
                
                if let midTex = device.makeTexture(descriptor: midTexDesc),
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
        }
        
        prevTexture = srcTex
        prevBuffer = workingBuffer
        
        // 2. MetalFX Upscaling
        let targetWidth = drawable.texture.width
        let targetHeight = drawable.texture.height
        
        if privateOutputTexture == nil ||
            privateOutputTexture?.width != targetWidth ||
            privateOutputTexture?.height != targetHeight {
            
            let desc = MTLTextureDescriptor.texture2DDescriptor(pixelFormat: drawable.texture.pixelFormat, width: targetWidth, height: targetHeight, mipmapped: false)
            desc.storageMode = .private
            desc.usage = [.shaderRead, .shaderWrite, .renderTarget]
            privateOutputTexture = device.makeTexture(descriptor: desc)
        }
        
        guard let outputTex = privateOutputTexture else { return }
        
        setupMetalFX(src: texToScale, dst: outputTex, profile: profile)
        
        if let scaler = spatialScaler {
            scaler.colorTexture = texToScale
            scaler.outputTexture = outputTex
            scaler.encode(commandBuffer: cmdBuffer)
            
            if let blit = cmdBuffer.makeBlitCommandEncoder() {
                blit.copy(from: outputTex, to: drawable.texture)
                blit.endEncoding()
            }
        } else {
            if let blit = cmdBuffer.makeBlitCommandEncoder() {
                blit.copy(from: texToScale, to: drawable.texture)
                blit.endEncoding()
            }
        }
        
        cmdBuffer.present(drawable)
        cmdBuffer.commit()
        recordFrameTiming()
    }
    
    func setupMetalFX(src: MTLTexture, dst: MTLTexture, profile: QualityProfile) {
        if spatialScaler == nil ||
            spatialScalerDesc?.inputWidth != src.width ||
            spatialScalerDesc?.outputWidth != dst.width ||
            spatialScalerDesc?.colorProcessingMode != profile.scalerMode {
            let desc = MTLFXSpatialScalerDescriptor()
            desc.inputWidth = src.width
            desc.inputHeight = src.height
            desc.outputWidth = dst.width
            desc.outputHeight = dst.height
            desc.colorTextureFormat = src.pixelFormat
            desc.outputTextureFormat = dst.pixelFormat
            desc.colorProcessingMode = profile.scalerMode
            
            spatialScalerDesc = desc
            spatialScaler = desc.makeSpatialScaler(device: device)
        }
    }
    
    private func createTexture(from buffer: CVPixelBuffer) -> MTLTexture? {
        var cvTex: CVMetalTexture?
        let w = CVPixelBufferGetWidth(buffer)
        let h = CVPixelBufferGetHeight(buffer)
        CVMetalTextureCacheCreateTextureFromImage(kCFAllocatorDefault, textureCache!, buffer, nil, .bgra8Unorm, w, h, 0, &cvTex)
        return cvTex.flatMap { CVMetalTextureGetTexture($0) }
    }
    
    func createOverlayWindow(targetFrame: CGRect) -> NSWindow {
        let win = NSWindow(contentRect: targetFrame, styleMask: .borderless, backing: .buffered, defer: false)
        win.backgroundColor = .clear
        win.isOpaque = false
        win.hasShadow = false
        win.level = .screenSaver
        win.ignoresMouseEvents = true
        win.collectionBehavior = [.canJoinAllSpaces, .fullScreenAuxiliary]
        let container = NSView(frame: CGRect(origin: .zero, size: targetFrame.size))
        container.translatesAutoresizingMaskIntoConstraints = true
        container.wantsLayer = true
        container.layer?.backgroundColor = NSColor.clear.cgColor
        if let metalView = view {
            metalView.frame = container.bounds
            metalView.autoresizingMask = [.width, .height]
            container.addSubview(metalView)
        }
        win.contentView = container
        overlayContentView = container
        fpsLayer?.removeFromSuperlayer()
        fpsLayer = nil
        return win
    }

    func setFPSOverlay(enabled: Bool) {
        showFPSOverlay = enabled
        DispatchQueue.main.async { [weak self] in
            guard let self = self else { return }
            self.ensureFPSLayer()?.isHidden = !enabled
        }
    }

    private func recordFrameTiming() {
        let now = CFAbsoluteTimeGetCurrent()
        if let last = lastFrameTimestamp {
            let delta = now - last
            guard delta > 0 else {
                lastFrameTimestamp = now
                return
            }
            let fps = 1.0 / delta
            smoothedFPS = smoothedFPS == 0 ? fps : (smoothedFPS * 0.85 + fps * 0.15)
            if showFPSOverlay {
                DispatchQueue.main.async { [weak self] in
                    guard let self = self else { return }
                    self.updateFPSOverlay(text: String(format: "MG FPS %.1f", self.smoothedFPS))
                }
            }
        }
        lastFrameTimestamp = now
    }

    private func ensureFPSLayer() -> CATextLayer? {
        guard let hostView = overlayContentView ?? view else { return nil }
        if hostView.layer == nil {
            hostView.wantsLayer = true
        }
        if let layer = fpsLayer {
            // Ensure layer is always on top
            if layer.superlayer != hostView.layer {
                layer.removeFromSuperlayer()
                hostView.layer?.addSublayer(layer)
            } else {
                // Re-add to bring to front
                layer.removeFromSuperlayer()
                hostView.layer?.addSublayer(layer)
            }
            return layer
        }
        let font = NSFont.monospacedSystemFont(ofSize: 13, weight: .bold)
        let layer = CATextLayer()
        layer.string = "MG FPS --"
        layer.font = font
        layer.fontSize = font.pointSize
        layer.foregroundColor = NSColor.green.cgColor
        layer.backgroundColor = NSColor.black.withAlphaComponent(0.75).cgColor
        layer.alignmentMode = .left
        layer.cornerRadius = 6
        layer.masksToBounds = true
        layer.contentsScale = hostView.window?.backingScaleFactor ?? NSScreen.main?.backingScaleFactor ?? 2.0
        layer.zPosition = 1000
        hostView.layer?.addSublayer(layer)
        fpsLayer = layer
        layoutFPSLayer(withText: "MG FPS --")
        return layer
    }

    private func updateFPSOverlay(text: String) {
        CATransaction.begin()
        CATransaction.setDisableActions(true)
        guard let layer = ensureFPSLayer() else { 
            CATransaction.commit()
            return 
        }
        layer.string = text
        layoutFPSLayer(withText: text)
        CATransaction.commit()
    }

    private func layoutFPSLayer(withText text: String) {
        guard let hostView = overlayContentView ?? view,
              let layer = fpsLayer else { return }
        let font = NSFont.monospacedSystemFont(ofSize: 13, weight: .bold)
        let attrs: [NSAttributedString.Key: Any] = [.font: font]
        let size = (text as NSString).size(withAttributes: attrs)
        let padding = NSEdgeInsets(top: 4, left: 8, bottom: 4, right: 8)
        let width = size.width + padding.left + padding.right
        let height = size.height + padding.top + padding.bottom
        let hostHeight = hostView.bounds.height
        let y = max(8, hostHeight - height - 12)
        layer.frame = CGRect(x: 12, y: y, width: width, height: height)
    }

    private func shouldRunFrameGeneration(buffer: CVPixelBuffer) -> Bool {
        let width = CVPixelBufferGetWidth(buffer)
        let height = CVPixelBufferGetHeight(buffer)
        guard width > 0, height > 0 else { return false }
        let pixels = width * height
        return pixels <= maxFlowPixels
    }
}

struct QualityProfile {
    let flowAccuracy: VNGenerateOpticalFlowRequest.ComputationAccuracy
    let scalerMode: MTLFXSpatialScalerColorProcessingMode
    let vtScalingMode: CFString
}

extension CaptureSettings.QualityMode {
    var profile: QualityProfile {
        switch self {
        case .performance:
            return QualityProfile(flowAccuracy: .low,
                                  scalerMode: .linear,
                                  vtScalingMode: kVTScalingMode_Normal)
        case .balanced:
            return QualityProfile(flowAccuracy: .medium,
                                  scalerMode: .perceptual,
                                  vtScalingMode: kVTScalingMode_Letterbox)
        case .quality:
            return QualityProfile(flowAccuracy: .high,
                                  scalerMode: .hdr,
                                  vtScalingMode: kVTScalingMode_Trim)
        }
    }
}
