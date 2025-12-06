import Foundation
import AppKit
import Metal
import MetalKit
import QuartzCore

struct OverlayWindowConfig {
    var targetScreen: NSScreen?
    var windowFrame: CGRect?
    var size: CGSize
    var refreshRate: Double
    var vsyncEnabled: Bool
    var adaptiveSyncEnabled: Bool
    var passThrough: Bool
    
    static var `default`: OverlayWindowConfig {
        OverlayWindowConfig(
            targetScreen: NSScreen.main,
            windowFrame: nil,
            size: NSScreen.main?.frame.size ?? CGSize(width: 1920, height: 1080),
            refreshRate: 120.0,
            vsyncEnabled: true,
            adaptiveSyncEnabled: true,
            passThrough: true
        )
    }
}

@available(macOS 26.0, *)
@MainActor
final class OverlayWindowManager: ObservableObject {
    @Published private(set) var isActive: Bool = false
    @Published private(set) var currentSize: CGSize = .zero
    @Published private(set) var currentFPS: Double = 0.0
    @Published private(set) var lastError: String?
    
    private let device: MTLDevice
    private let commandQueue: MTLCommandQueue
    private var textureCache: CVMetalTextureCache?
    private var overlayWindow: NSWindow?
    private var mtkView: MTKView?
    private var lastFrameTime: CFTimeInterval = 0
    private var frameCount: Int = 0
    private var fpsStartTime: CFTimeInterval = 0
    private var targetWindowID: CGWindowID = 0
    private var targetPID: pid_t = 0
    private var appObserver: Any?
    
    init?(device: MTLDevice? = nil) {
        guard let dev = device ?? MTLCreateSystemDefaultDevice() else {
            return nil
        }
        guard let queue = dev.makeCommandQueue() else {
            return nil
        }
        self.device = dev
        self.commandQueue = queue
        
        var cache: CVMetalTextureCache?
        let status = CVMetalTextureCacheCreate(kCFAllocatorDefault, nil, dev, nil, &cache)
        if status == kCVReturnSuccess {
            self.textureCache = cache
        }
    }
    
    deinit {
        if let observer = appObserver {
            NSWorkspace.shared.notificationCenter.removeObserver(observer)
        }
    }
    
    func createOverlay(config: OverlayWindowConfig = .default) -> Bool {
        destroyOverlay()
        
        let targetScreen = config.targetScreen ?? NSScreen.main ?? NSScreen.screens.first
        guard let screen = targetScreen else {
            lastError = "No screen available"
            return false
        }
        
        let frame = config.windowFrame ?? screen.frame
        currentSize = frame.size
        
        let window = NSWindow(
            contentRect: frame,
            styleMask: [.borderless],
            backing: .buffered,
            defer: false,
            screen: screen
        )
        
        window.isReleasedWhenClosed = false
        window.level = NSWindow.Level(rawValue: Int(CGWindowLevelForKey(.maximumWindow)))
        window.backgroundColor = .clear
        window.isOpaque = false
        window.hasShadow = false
        window.ignoresMouseEvents = config.passThrough
        window.acceptsMouseMovedEvents = !config.passThrough
        window.collectionBehavior = [.canJoinAllSpaces, .fullScreenAuxiliary, .fullScreenPrimary, .stationary]
        
        overlayWindow = window
        window.orderFrontRegardless()
        
        isActive = true
        lastError = nil
        return true
    }
    
    func setMTKView(_ view: MTKView) {
        guard let window = overlayWindow else { return }
        view.frame = window.contentView?.bounds ?? CGRect(origin: .zero, size: currentSize)
        window.contentView = view
        mtkView = view
        
        if let screen = window.screen ?? NSScreen.main {
            let scale = screen.backingScaleFactor
            view.drawableSize = CGSize(
                width: currentSize.width * scale,
                height: currentSize.height * scale
            )
        }
    }
    
    func destroyOverlay() {
        if let observer = appObserver {
            NSWorkspace.shared.notificationCenter.removeObserver(observer)
            appObserver = nil
        }
        mtkView = nil
        overlayWindow?.orderOut(nil)
        overlayWindow = nil
        isActive = false
        currentSize = .zero
    }
    
    func createTexture(from pixelBuffer: CVPixelBuffer) -> MTLTexture? {
        guard let cache = textureCache else { return nil }
        
        let width = CVPixelBufferGetWidth(pixelBuffer)
        let height = CVPixelBufferGetHeight(pixelBuffer)
        let pixelFormat = CVPixelBufferGetPixelFormatType(pixelBuffer)
        
        let mtlPixelFormat: MTLPixelFormat
        switch pixelFormat {
        case kCVPixelFormatType_32BGRA: mtlPixelFormat = .bgra8Unorm
        case kCVPixelFormatType_32RGBA: mtlPixelFormat = .rgba8Unorm
        case kCVPixelFormatType_64RGBAHalf: mtlPixelFormat = .rgba16Float
        default: mtlPixelFormat = .bgra8Unorm
        }
        
        var cvTexture: CVMetalTexture?
        let status = CVMetalTextureCacheCreateTextureFromImage(
            kCFAllocatorDefault, cache, pixelBuffer, nil, mtlPixelFormat, width, height, 0, &cvTexture
        )
        guard status == kCVReturnSuccess, let cvTex = cvTexture else { return nil }
        return CVMetalTextureGetTexture(cvTex)
    }
    
    func flushTextureCache() {
        if let cache = textureCache {
            CVMetalTextureCacheFlush(cache, 0)
        }
    }
    
    func setTargetWindow(_ windowID: CGWindowID, pid: pid_t) {
        targetWindowID = windowID
        targetPID = pid
        
        if let observer = appObserver {
            NSWorkspace.shared.notificationCenter.removeObserver(observer)
        }
        
        appObserver = NSWorkspace.shared.notificationCenter.addObserver(
            forName: NSWorkspace.didActivateApplicationNotification,
            object: nil,
            queue: .main
        ) { [weak self] notification in
            Task { @MainActor in
                guard let self = self else { return }
                guard self.targetPID != 0 else { return }
                guard let app = notification.userInfo?[NSWorkspace.applicationUserInfoKey] as? NSRunningApplication else { return }
                
                if app.processIdentifier == self.targetPID {
                    self.overlayWindow?.orderFrontRegardless()
                } else {
                    self.overlayWindow?.orderOut(nil)
                }
            }
        }
    }
    
    func updateWindowPosition() {
        guard targetWindowID != 0 else { return }
        guard let window = overlayWindow else { return }
        
        let opts: CGWindowListOption = [.optionIncludingWindow]
        guard let list = CGWindowListCopyWindowInfo(opts, targetWindowID) as? [[String: Any]],
              let info = list.first,
              let bounds = info[kCGWindowBounds as String] as? [String: CGFloat] else {
            return
        }
        
        let isOnScreen = info[kCGWindowIsOnscreen as String] as? Bool ?? true
        if !isOnScreen {
            window.orderOut(nil)
            return
        }
        
        let cgFrame = CGRect(
            x: bounds["X"] ?? 0,
            y: bounds["Y"] ?? 0,
            width: bounds["Width"] ?? 100,
            height: bounds["Height"] ?? 100
        )
        
        let screenH = NSScreen.main?.frame.height ?? 1080
        
        let nsFrame = CGRect(
            x: cgFrame.origin.x,
            y: screenH - cgFrame.origin.y - cgFrame.height,
            width: cgFrame.width,
            height: cgFrame.height
        )
        
        if !window.isVisible {
            window.orderFront(nil)
        }
        
        if window.frame != nsFrame {
            window.setFrame(nsFrame, display: false)
            
            if let view = mtkView, let screen = window.screen ?? NSScreen.main {
                view.frame = CGRect(origin: .zero, size: nsFrame.size)
                view.drawableSize = CGSize(
                    width: nsFrame.width * screen.backingScaleFactor,
                    height: nsFrame.height * screen.backingScaleFactor
                )
            }
            currentSize = nsFrame.size
        }
    }
    
    var metalDevice: MTLDevice { device }
    var metalCommandQueue: MTLCommandQueue { commandQueue }
    
    var drawableSize: CGSize {
        mtkView?.drawableSize ?? .zero
    }
    
    func setVisible(_ visible: Bool) {
        if visible { overlayWindow?.orderFrontRegardless() }
        else { overlayWindow?.orderOut(nil) }
    }
}

@available(macOS 26.0, *)
extension OverlayWindowManager {
    func createTexture(
        width: Int, height: Int,
        pixelFormat: MTLPixelFormat = .rgba16Float,
        usage: MTLTextureUsage = [.shaderRead, .shaderWrite]
    ) -> MTLTexture? {
        let desc = MTLTextureDescriptor.texture2DDescriptor(
            pixelFormat: pixelFormat, width: width, height: height, mipmapped: false
        )
        desc.usage = usage
        desc.storageMode = .private
        return device.makeTexture(descriptor: desc)
    }
    
    func createFlowTexture(width: Int, height: Int) -> MTLTexture? {
        createTexture(width: width, height: height, pixelFormat: .rg16Float, usage: [.shaderRead, .shaderWrite])
    }
    
    func createInterpolatedTexture(width: Int, height: Int) -> MTLTexture? {
        createTexture(width: width, height: height, pixelFormat: .rgba16Float, usage: [.shaderRead, .shaderWrite])
    }
    
    func createOutputTexture(width: Int, height: Int) -> MTLTexture? {
        createTexture(width: width, height: height, pixelFormat: .bgra8Unorm, usage: [.shaderRead, .shaderWrite, .renderTarget])
    }
}
