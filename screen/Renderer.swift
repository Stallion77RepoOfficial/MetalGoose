import MetalKit
import MetalFX
import Vision
import CoreVideo

final class Renderer: NSObject, MTKViewDelegate {
    let device: MTLDevice
    let commandQueue: MTLCommandQueue
    var textureCache: CVMetalTextureCache?
    
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
        
        metalKitView.device = dev
        metalKitView.framebufferOnly = false
        metalKitView.colorPixelFormat = .bgra8Unorm
        metalKitView.clearColor = MTLClearColor(red: 0, green: 0, blue: 0, alpha: 0)
        metalKitView.delegate = self
        metalKitView.layer?.isOpaque = false
        
        CVMetalTextureCacheCreate(kCFAllocatorDefault, nil, device, nil, &textureCache)
        
        setupPipelines()
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
        } else {
            // Window might be closed
            stopTracking()
            overlay.orderOut(nil)
        }
    }
    
    // MARK: - Rendering
    func mtkView(_ view: MTKView, drawableSizeWillChange size: CGSize) {
        spatialScaler = nil
        privateOutputTexture = nil
    }
    
    func draw(in view: MTKView) { /* Driven by CaptureEngine */ }
    
    func processAndRender(buffer: CVPixelBuffer, view: MTKView) {
        guard let drawable = view.currentDrawable,
              let cmdBuffer = commandQueue.makeCommandBuffer(),
              let srcTex = createTexture(from: buffer) else { return }
        
        var texToScale = srcTex
        
        // 1. Frame Generation
        if let pTex = prevTexture, let pBuf = prevBuffer {
            let request = VNGenerateOpticalFlowRequest(targetedCVPixelBuffer: buffer, options: [:])
            request.computationAccuracy = .medium
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
        prevBuffer = buffer
        
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
        
        setupMetalFX(src: texToScale, dst: outputTex)
        
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
    }
    
    func setupMetalFX(src: MTLTexture, dst: MTLTexture) {
        if spatialScaler == nil || spatialScalerDesc?.inputWidth != src.width || spatialScalerDesc?.outputWidth != dst.width {
            let desc = MTLFXSpatialScalerDescriptor()
            desc.inputWidth = src.width
            desc.inputHeight = src.height
            desc.outputWidth = dst.width
            desc.outputHeight = dst.height
            desc.colorTextureFormat = src.pixelFormat
            desc.outputTextureFormat = dst.pixelFormat
            desc.colorProcessingMode = .perceptual
            
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
        win.level = .floating
        win.ignoresMouseEvents = true
        win.collectionBehavior = [.canJoinAllSpaces, .fullScreenAuxiliary]
        return win
    }
}
