import MetalKit
import CoreVideo
import AppKit
import ApplicationServices

final class Renderer: NSObject, MTKViewDelegate {
    private let device: MTLDevice
    private let commandQueue: MTLCommandQueue
    private var textureCache: CVMetalTextureCache?
    
    // Shader Pipelines
    private var bicubicPipeline: MTLComputePipelineState?
    private var integerPipeline: MTLComputePipelineState?
    private var rcasPipeline: MTLComputePipelineState?
    
    // Overlay Yönetimi
    private var overlayWindow: NSWindow?
    private weak var overlayView: MTKView?
    private var trackingTimer: Timer?
    private var trackedWindowID: CGWindowID?
    
    // Ayarlar
    var sharpenAmount: Float = 0.5
    var scalingMode: CaptureSettings.ScalingMode = .aspectFit
    
    init?(view: MTKView) {
        guard let device = MTLCreateSystemDefaultDevice(),
              let queue = device.makeCommandQueue() else { return nil }
        self.device = device
        self.commandQueue = queue
        super.init()
        
        view.device = device
        view.framebufferOnly = false
        view.autoResizeDrawable = true
        view.clearColor = MTLClearColor(red: 0, green: 0, blue: 0, alpha: 0)
        
        CVMetalTextureCacheCreate(kCFAllocatorDefault, nil, device, nil, &textureCache)
        
        buildPipelines()
    }
    
    private func buildPipelines() {
        guard let lib = try? device.makeDefaultLibrary(bundle: Bundle.main) else {
            print("Metal kütüphanesi yüklenemedi.")
            return
        }
        
        let makePipe = { (name: String) -> MTLComputePipelineState? in
            guard let fn = lib.makeFunction(name: name) else { return nil }
            return try? self.device.makeComputePipelineState(function: fn)
        }
        
        self.bicubicPipeline = makePipe("bicubic_upscale")
        self.integerPipeline = makePipe("integer_upscale")
        self.rcasPipeline = makePipe("rcas_sharpen")
    }
    
    // MARK: - Overlay (Pencere) Yönetimi
    
    func startOverlay(for windowID: CGWindowID, pid: pid_t) {
        self.trackedWindowID = windowID
        
        DispatchQueue.main.async {
            self.createWindow()
            self.startTracking()
        }
    }
    
    func stopOverlay() {
        trackingTimer?.invalidate()
        overlayWindow?.orderOut(nil)
        overlayWindow = nil
    }
    
    private func createWindow() {
        // Çerçevesiz, şeffaf pencere
        let win = NSWindow(contentRect: .zero, styleMask: .borderless, backing: .buffered, defer: false)
        win.isOpaque = false
        win.backgroundColor = .clear
        win.hasShadow = false
        
        // ÖNEMLİ DEĞİŞİKLİK: Pencere seviyesini bir tık düşürdük ki sistem kilitlenmesin
        win.level = .floating
        
        // KRİTİK ÇÖZÜM: Tıklamaları Yoksay (Click-Through)
        // Bu 'true' olduğunda, overlay sadece görsel olur, tıklamalar direkt arkadaki oyuna geçer.
        win.ignoresMouseEvents = true
        
        win.collectionBehavior = [.canJoinAllSpaces, .fullScreenAuxiliary, .stationary]
        
        let mtk = MTKView(frame: .zero, device: device)
        mtk.delegate = self
        mtk.framebufferOnly = false
        mtk.clearColor = MTLClearColor(red: 0, green: 0, blue: 0, alpha: 0)
        mtk.layer?.isOpaque = false
        
        win.contentView = mtk
        win.orderFrontRegardless()
        
        self.overlayWindow = win
        self.overlayView = mtk
        
        self.updateWindowPosition()
    }
    
    private func startTracking() {
        // Pencere konumunu hızlı takip et (120Hz civarı)
        trackingTimer = Timer.scheduledTimer(withTimeInterval: 0.008, repeats: true) { [weak self] _ in
            self?.updateWindowPosition()
        }
        RunLoop.current.add(trackingTimer!, forMode: .common)
    }
    
    private func updateWindowPosition() {
        guard let wid = trackedWindowID, let win = overlayWindow else { return }
        
        guard let info = CGWindowListCopyWindowInfo([.optionIncludingWindow], wid) as? [[String: Any]],
              let target = info.first,
              let bounds = target[kCGWindowBounds as String] as? [String: CGFloat] else { return }
        
        let x = bounds["X"] ?? 0
        let y = bounds["Y"] ?? 0
        // ÇÖKME ÖNLEYİCİ: Genişlik ve Yükseklik 0 olamaz
        let w = max(1, bounds["Width"] ?? 100)
        let h = max(1, bounds["Height"] ?? 100)
        
        // Koordinat düzeltmesi (macOS sol-alt köşe referansı)
        let screenH = NSScreen.main?.frame.height ?? 1080
        let nsY = screenH - (y + h)
        
        let newFrame = CGRect(x: x, y: nsY, width: w, height: h)
        
        // Sadece pozisyon değiştiyse güncelle
        if win.frame != newFrame {
            win.setFrame(newFrame, display: true, animate: false)
        }
    }
    
    // MARK: - Metal Çizim
    
    func draw(in view: MTKView) { }
    
    func mtkView(_ view: MTKView, drawableSizeWillChange size: CGSize) {}
    
    func update(with buffer: CVPixelBuffer) {
        guard let drawable = overlayView?.currentDrawable,
              let cmdBuffer = commandQueue.makeCommandBuffer() else { return }
        
        let w = CVPixelBufferGetWidth(buffer)
        let h = CVPixelBufferGetHeight(buffer)
        
        // GÜVENLİK: Geçersiz veya 0 boyutlu buffer gelirse çizim yapma
        if w <= 0 || h <= 0 || drawable.texture.width <= 0 || drawable.texture.height <= 0 {
            return
        }
        
        var cvTex: CVMetalTexture?
        CVMetalTextureCacheCreateTextureFromImage(kCFAllocatorDefault, textureCache!, buffer, nil, .bgra8Unorm, w, h, 0, &cvTex)
        guard let srcTex = CVMetalTextureGetTexture(cvTex!) else { return }
        
        let dstTex = drawable.texture
        
        // Ara texture oluştur
        let desc = MTLTextureDescriptor.texture2DDescriptor(pixelFormat: .bgra8Unorm, width: dstTex.width, height: dstTex.height, mipmapped: false)
        desc.usage = [.shaderRead, .shaderWrite]
        guard let interTex = device.makeTexture(descriptor: desc) else { return }
        
        // 1. Upscale
        if let encoder = cmdBuffer.makeComputeCommandEncoder() {
            let pipeline = (scalingMode == .integer) ? integerPipeline : bicubicPipeline
            if let pipe = pipeline {
                encoder.setComputePipelineState(pipe)
                encoder.setTexture(srcTex, index: 0)
                encoder.setTexture(interTex, index: 1)
                
                let tw = pipe.threadExecutionWidth
                let th = pipe.maxTotalThreadsPerThreadgroup / tw
                let threads = MTLSize(width: interTex.width, height: interTex.height, depth: 1)
                let groups = MTLSize(width: (threads.width + tw - 1) / tw, height: (threads.height + th - 1) / th, depth: 1)
                
                encoder.dispatchThreadgroups(groups, threadsPerThreadgroup: MTLSize(width: tw, height: th, depth: 1))
            }
            encoder.endEncoding()
        }
        
        // 2. RCAS Keskinleştirme
        if let encoder = cmdBuffer.makeComputeCommandEncoder(), let pipe = rcasPipeline {
            encoder.setComputePipelineState(pipe)
            encoder.setTexture(interTex, index: 0)
            encoder.setTexture(dstTex, index: 1)
            var sharp = self.sharpenAmount
            encoder.setBytes(&sharp, length: MemoryLayout<Float>.size, index: 0)
            
            let tw = pipe.threadExecutionWidth
            let th = pipe.maxTotalThreadsPerThreadgroup / tw
            let threads = MTLSize(width: dstTex.width, height: dstTex.height, depth: 1)
            let groups = MTLSize(width: (threads.width + tw - 1) / tw, height: (threads.height + th - 1) / th, depth: 1)
            
            encoder.dispatchThreadgroups(groups, threadsPerThreadgroup: MTLSize(width: tw, height: th, depth: 1))
            encoder.endEncoding()
        }
        
        cmdBuffer.present(drawable)
        cmdBuffer.commit()
    }
}
