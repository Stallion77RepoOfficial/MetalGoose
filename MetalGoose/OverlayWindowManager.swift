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
    var scaleFactor: Float
    var captureCursor: Bool
}

class NonActivatingWindow: NSWindow {
    override var canBecomeKey: Bool { false }
    override var canBecomeMain: Bool { false }
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
    private var shouldCaptureCursor: Bool = false
    
    func setCaptureCursorEnabled(_ enabled: Bool) {
        self.shouldCaptureCursor = enabled
    }
    
    init?() {
        guard let dev = MTLCreateSystemDefaultDevice() else {
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
    
    func createOverlay(config: OverlayWindowConfig) -> Bool {
        destroyOverlay()
        
        self.shouldCaptureCursor = config.captureCursor
        
        guard let screen = config.targetScreen else {
            lastError = "Error Code: MG-OV-001 No screen available"
            return false
        }
        
        currentSize = config.size
        
        guard let frame = config.windowFrame else {
            lastError = "Error Code: MG-OV-002 Missing window frame"
            return false
        }
        
        let window = NonActivatingWindow(
            contentRect: CGRect(origin: frame.origin, size: config.size),
            styleMask: [.borderless],
            backing: .buffered,
            defer: false,
            screen: screen
        )
        
        window.isReleasedWhenClosed = false
        window.level = NSWindow.Level(rawValue: Int(CGWindowLevelForKey(.screenSaverWindow)) + 1)
        window.backgroundColor = .clear
        window.isOpaque = false
        window.hasShadow = false
        window.ignoresMouseEvents = config.passThrough
        window.acceptsMouseMovedEvents = !config.passThrough
        window.collectionBehavior = [.canJoinAllSpaces, .fullScreenAuxiliary, .stationary, .ignoresCycle]
        
        overlayWindow = window
        window.orderFrontRegardless()
        
        isActive = true
        lastError = nil
        return true
    }
    
    func setMTKView(_ view: MTKView) {
        guard let window = overlayWindow else { return }
        view.frame = CGRect(origin: .zero, size: currentSize)
        window.contentView = view
        mtkView = view
        
        guard let screen = window.screen else { return }
        let scale = screen.backingScaleFactor
        view.drawableSize = CGSize(
            width: currentSize.width * scale,
            height: currentSize.height * scale
        )
    }
    
    func destroyOverlay() {
        if let observer = appObserver {
            NSWorkspace.shared.notificationCenter.removeObserver(observer)
            appObserver = nil
        }
        MouseConstraintManager.shared.stopConstraining()
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
        default:
            lastError = "Error Code: MG-OV-003 Unsupported pixel format: \(pixelFormat)"
            return nil
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
        
        let isOnScreen = (info[kCGWindowIsOnscreen as String] as? Bool) == true
        if !isOnScreen {
            window.orderOut(nil)
            return
        }
        
        guard let boundX = bounds["X"],
              let boundY = bounds["Y"],
              let boundW = bounds["Width"],
              let boundH = bounds["Height"],
              let screen = window.screen else { return }
        
        let cgFrame = CGRect(x: boundX, y: boundY, width: boundW, height: boundH)
        
        let nsFrame = screen.frame
        
        if !window.isVisible {
            window.orderFront(nil)
        }
        
        if window.frame != nsFrame {
            window.setFrame(nsFrame, display: false)
            
            if let view = mtkView {
                view.frame = CGRect(origin: .zero, size: nsFrame.size)
                view.drawableSize = CGSize(
                    width: nsFrame.width * screen.backingScaleFactor,
                    height: nsFrame.height * screen.backingScaleFactor
                )
            }
            currentSize = nsFrame.size
        }
        
        if shouldCaptureCursor {
            MouseConstraintManager.shared.startConstraining(to: cgFrame)
        } else {
            MouseConstraintManager.shared.stopConstraining()
        }
    }
    
    var metalDevice: MTLDevice { device }
    var metalCommandQueue: MTLCommandQueue { commandQueue }
    
    var drawableSize: CGSize? {
        return mtkView?.drawableSize
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

@available(macOS 26.0, *)
class MouseConstraintManager {
    static let shared = MouseConstraintManager()
    
    private var eventTap: CFMachPort?
    private var runLoopSource: CFRunLoopSource?
    private var targetRect: CGRect = .zero
    private var isConstraining = false
    
    func startConstraining(to rect: CGRect) {
        self.targetRect = rect
        if isConstraining { return }
        
        CGDisplayHideCursor(CGMainDisplayID())
        
        let eventMask = (1 << CGEventType.mouseMoved.rawValue) |
                        (1 << CGEventType.leftMouseDragged.rawValue) |
                        (1 << CGEventType.rightMouseDragged.rawValue) |
                        (1 << CGEventType.otherMouseDragged.rawValue)
        
        let info = Unmanaged.passUnretained(self).toOpaque()
        let callback: CGEventTapCallBack = { (proxy, type, event, refcon) -> Unmanaged<CGEvent>? in
            guard let ref = refcon else { return Unmanaged.passRetained(event) }
            let manager = Unmanaged<MouseConstraintManager>.fromOpaque(ref).takeUnretainedValue()
            
            var location = event.location
            let bounds = manager.targetRect
            
            if !bounds.contains(location) {
                location.x = max(bounds.minX, min(bounds.maxX, location.x))
                location.y = max(bounds.minY, min(bounds.maxY, location.y))
                event.location = location
                CGWarpMouseCursorPosition(location)
            }
            
            return Unmanaged.passRetained(event)
        }
        
        eventTap = CGEvent.tapCreate(
            tap: .cgSessionEventTap,
            place: .headInsertEventTap,
            options: .defaultTap,
            eventsOfInterest: CGEventMask(eventMask),
            callback: callback,
            userInfo: info
        )
        
        if let tap = eventTap {
            runLoopSource = CFMachPortCreateRunLoopSource(kCFAllocatorDefault, tap, 0)
            CFRunLoopAddSource(CFRunLoopGetCurrent(), runLoopSource, .commonModes)
            CGEvent.tapEnable(tap: tap, enable: true)
            isConstraining = true
        }
    }
    
    func stopConstraining() {
        if !isConstraining { return }
        if let tap = eventTap {
            CGEvent.tapEnable(tap: tap, enable: false)
            CFMachPortInvalidate(tap)
        }
        if let source = runLoopSource {
            CFRunLoopRemoveSource(CFRunLoopGetCurrent(), source, .commonModes)
        }
        eventTap = nil
        runLoopSource = nil
        isConstraining = false
        
        CGDisplayShowCursor(CGMainDisplayID())
    }
}
