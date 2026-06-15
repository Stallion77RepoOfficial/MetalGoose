import Foundation
import AppKit
import MetalKit

struct OverlayWindowConfig {
    var targetScreen: NSScreen?
    var windowFrame: CGRect?
    var size: CGSize
    var captureCursor: Bool
    var displayBounds: CGRect
    var fullScreenOutput: Bool
}

class NonActivatingWindow: NSWindow {
    override var canBecomeKey: Bool { false }
    override var canBecomeMain: Bool { false }
}

@MainActor
final class OverlayWindowManager {
    private var currentSize: CGSize = .zero

    private var overlayWindow: NSWindow?
    private var mtkView: MTKView?
    private var targetWindowID: CGWindowID = 0
    private var targetPID: pid_t = 0
    nonisolated(unsafe) private var appObserver: NSObjectProtocol?

    private var shouldCaptureCursor: Bool = false
    private var fullScreenOutput: Bool = true
    private var sourceRectCG: CGRect = .zero
    private var displayBoundsCG: CGRect = .zero

    func setCaptureCursorEnabled(_ enabled: Bool) {
        self.shouldCaptureCursor = enabled
    }

    deinit {
        if let observer = appObserver {
            NSWorkspace.shared.notificationCenter.removeObserver(observer)
        }
    }

    func createOverlay(config: OverlayWindowConfig) -> Bool {
        destroyOverlay()

        self.shouldCaptureCursor = config.captureCursor
        self.fullScreenOutput = config.fullScreenOutput

        guard let screen = config.targetScreen else {
            return false
        }

        currentSize = config.size

        guard let frame = config.windowFrame else {
            return false
        }

        sourceRectCG = frame
        displayBoundsCG = config.displayBounds

        let window = NonActivatingWindow(
            contentRect: CGRect(origin: frame.origin, size: config.size),
            styleMask: [.borderless],
            backing: .buffered,
            defer: false,
            screen: screen
        )

        window.isReleasedWhenClosed = false
        window.level = NSWindow.Level(rawValue: Int(CGWindowLevelForKey(.maximumWindow)) + 1)
        window.backgroundColor = .clear
        window.isOpaque = false
        window.hasShadow = false
        window.ignoresMouseEvents = true
        window.acceptsMouseMovedEvents = false
        window.collectionBehavior = [.canJoinAllSpaces, .fullScreenAuxiliary, .stationary, .ignoresCycle]

        overlayWindow = window
        window.orderFrontRegardless()

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
        currentSize = .zero
        sourceRectCG = .zero
        displayBoundsCG = .zero
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
            let app = notification.userInfo?[NSWorkspace.applicationUserInfoKey] as? NSRunningApplication
            let pid = app?.processIdentifier
            Task { @MainActor in
                guard let self = self else { return }
                guard self.targetPID != 0, let pid else { return }

                if pid == self.targetPID {
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
        sourceRectCG = cgFrame

        let nsFrame: CGRect
        if fullScreenOutput {
            nsFrame = screen.frame
        } else {
            let primaryHeight = NSScreen.screens.first?.frame.height ?? 0
            nsFrame = CGRect(
                x: cgFrame.origin.x,
                y: primaryHeight - cgFrame.maxY,
                width: cgFrame.width,
                height: cgFrame.height
            )
        }

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

        if let displayID = screen.deviceDescription[NSDeviceDescriptionKey("NSScreenNumber")] as? CGDirectDisplayID {
            displayBoundsCG = CGDisplayBounds(displayID)
        }

        if shouldCaptureCursor {
            MouseConstraintManager.shared.startConstraining(sourceRect: cgFrame, displayBounds: displayBoundsCG)
        } else {
            MouseConstraintManager.shared.stopConstraining()
        }
    }

}

final class MouseConstraintManager: @unchecked Sendable {
    static let shared = MouseConstraintManager()

    private let lock = NSLock()
    private var eventTap: CFMachPort?
    private var runLoopSource: CFRunLoopSource?
    private var sourceRect: CGRect = .zero
    private var displayBounds: CGRect = .zero
    private var virtualPos: CGPoint = .zero
    private var lastMappedPoint: CGPoint = .zero
    private var isConstraining = false
    private var cursorHideTimer: Timer?
    private var cursorSpriteVisible = true
    private var cursorHideCount = 0

    func startConstraining(sourceRect: CGRect, displayBounds: CGRect) {
        lock.lock()
        self.sourceRect = sourceRect
        self.displayBounds = displayBounds
        let alreadyOn = isConstraining
        lock.unlock()

        if alreadyOn { return }

        lock.lock()
        virtualPos = CGPoint(x: displayBounds.midX, y: displayBounds.midY)
        lastMappedPoint = CGPoint(x: sourceRect.midX, y: sourceRect.midY)
        lock.unlock()

        let eventMask = (1 << CGEventType.mouseMoved.rawValue) |
                        (1 << CGEventType.leftMouseDragged.rawValue) |
                        (1 << CGEventType.rightMouseDragged.rawValue) |
                        (1 << CGEventType.otherMouseDragged.rawValue) |
                        (1 << CGEventType.leftMouseDown.rawValue) |
                        (1 << CGEventType.leftMouseUp.rawValue) |
                        (1 << CGEventType.rightMouseDown.rawValue) |
                        (1 << CGEventType.rightMouseUp.rawValue) |
                        (1 << CGEventType.otherMouseDown.rawValue) |
                        (1 << CGEventType.otherMouseUp.rawValue)

        let info = Unmanaged.passUnretained(self).toOpaque()
        let callback: CGEventTapCallBack = { (proxy, type, event, refcon) -> Unmanaged<CGEvent>? in
            guard let ref = refcon else { return Unmanaged.passUnretained(event) }
            let manager = Unmanaged<MouseConstraintManager>.fromOpaque(ref).takeUnretainedValue()
            manager.handle(type: type, event: event)
            return Unmanaged.passUnretained(event)
        }

        let tap = CGEvent.tapCreate(
            tap: .cgSessionEventTap,
            place: .headInsertEventTap,
            options: .defaultTap,
            eventsOfInterest: CGEventMask(eventMask),
            callback: callback,
            userInfo: info
        )

        guard let tap else { return }

        lock.lock()
        eventTap = tap
        runLoopSource = CFMachPortCreateRunLoopSource(kCFAllocatorDefault, tap, 0)
        CFRunLoopAddSource(CFRunLoopGetCurrent(), runLoopSource, .commonModes)
        CGEvent.tapEnable(tap: tap, enable: true)
        isConstraining = true
        lock.unlock()

        Self.enableBackgroundCursorControl()
        CGDisplayHideCursor(CGMainDisplayID())
        lock.lock()
        cursorHideCount += 1
        lock.unlock()

        let timer = Timer(timeInterval: 0.1, repeats: true) { [weak self] _ in
            CGDisplayHideCursor(CGMainDisplayID())
            self?.lock.lock()
            self?.cursorHideCount += 1
            self?.lock.unlock()
        }
        RunLoop.current.add(timer, forMode: .common)
        cursorHideTimer = timer
    }

    private static func enableBackgroundCursorControl() {
        typealias MainConnFn = @convention(c) () -> Int32
        typealias SetPropFn = @convention(c) (Int32, Int32, CFString, CFTypeRef) -> Int32
        guard let mainSym = dlsym(UnsafeMutableRawPointer(bitPattern: -2), "CGSMainConnectionID"),
              let setSym = dlsym(UnsafeMutableRawPointer(bitPattern: -2), "CGSSetConnectionProperty") else {
            return
        }
        let mainConn = unsafeBitCast(mainSym, to: MainConnFn.self)
        let setProp = unsafeBitCast(setSym, to: SetPropFn.self)
        let cid = mainConn()
        _ = setProp(cid, cid, "SetsCursorInBackground" as CFString, kCFBooleanTrue)
    }

    func currentCursorFraction() -> CGPoint? {
        lock.lock()
        defer { lock.unlock() }
        guard isConstraining, cursorSpriteVisible, sourceRect.width > 0, sourceRect.height > 0 else { return nil }
        let fx = (lastMappedPoint.x - sourceRect.minX) / sourceRect.width
        let fy = (lastMappedPoint.y - sourceRect.minY) / sourceRect.height
        return CGPoint(x: fx, y: fy)
    }

    func toggleCursorSpriteVisible() {
        lock.lock()
        cursorSpriteVisible.toggle()
        lock.unlock()
    }

    private func handle(type: CGEventType, event: CGEvent) {
        lock.lock()
        let src = sourceRect
        let disp = displayBounds
        guard src.width > 0, src.height > 0, disp.width > 0, disp.height > 0 else {
            lock.unlock()
            return
        }

        switch type {
        case .mouseMoved, .leftMouseDragged, .rightMouseDragged, .otherMouseDragged:
            let dx = CGFloat(event.getIntegerValueField(.mouseEventDeltaX))
            let dy = CGFloat(event.getIntegerValueField(.mouseEventDeltaY))
            virtualPos.x = min(max(disp.minX, virtualPos.x + dx), disp.maxX)
            virtualPos.y = min(max(disp.minY, virtualPos.y + dy), disp.maxY)

            let fx = (virtualPos.x - disp.minX) / disp.width
            let fy = (virtualPos.y - disp.minY) / disp.height
            let mapped = CGPoint(x: src.minX + fx * src.width,
                                 y: src.minY + fy * src.height)
            lastMappedPoint = mapped
            lock.unlock()

            event.location = mapped
            CGWarpMouseCursorPosition(mapped)

        case .leftMouseDown, .leftMouseUp,
             .rightMouseDown, .rightMouseUp,
             .otherMouseDown, .otherMouseUp:
            let mapped = lastMappedPoint
            lock.unlock()
            event.location = mapped

        default:
            lock.unlock()
        }
    }

    func stopConstraining() {
        lock.lock()
        let wasOn = isConstraining
        let tap = eventTap
        let source = runLoopSource
        let hideCount = cursorHideCount
        eventTap = nil
        runLoopSource = nil
        isConstraining = false
        cursorHideCount = 0
        cursorSpriteVisible = true
        lock.unlock()

        cursorHideTimer?.invalidate()
        cursorHideTimer = nil

        if !wasOn { return }

        if let tap {
            CGEvent.tapEnable(tap: tap, enable: false)
            CFMachPortInvalidate(tap)
        }
        if let source {
            CFRunLoopRemoveSource(CFRunLoopGetCurrent(), source, .commonModes)
        }

        for _ in 0...hideCount {
            CGDisplayShowCursor(CGMainDisplayID())
        }

        if let location = CGEvent(source: nil)?.location {
            CGWarpMouseCursorPosition(location)
        }
    }
}
