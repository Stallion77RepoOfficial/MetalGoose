import Foundation
import AppKit
import MetalKit

struct OverlayWindowConfig {
    var targetScreen: NSScreen?
    var windowFrame: CGRect?
    var captureCursor: Bool
    /// Requested magnification. 1.0 overlays the target window exactly; larger
    /// values grow around the window centre and stop at the screen edges.
    var outputScale: CGFloat
    /// Ignores `outputScale` and covers the whole display.
    var fillsScreen: Bool
}

class NonActivatingWindow: NSWindow {
    override var canBecomeKey: Bool { false }
    override var canBecomeMain: Bool { false }
}

@MainActor
final class OverlayWindowManager {
    private(set) var lastError: String?

    private var currentSize: CGSize = .zero

    private var overlayWindow: NSWindow?
    private var mtkView: MTKView?
    private var targetWindowID: CGWindowID = 0
    private var targetPID: pid_t = 0
    nonisolated(unsafe) private var appObserver: NSObjectProtocol?

    /// Whether the captured app is the one the user is currently in. The overlay
    /// draws the last frame it was given for as long as it is on screen, so while
    /// the user is somewhere else that is a still image of the game sitting on
    /// top of whatever they switched to — which reads as the game refusing to
    /// give up focus.
    private var targetIsFrontmost = true

    private var shouldCaptureCursor: Bool = false
    private var outputScale: CGFloat = 1.0
    private var fillsScreen: Bool = false
    private var displayBoundsCG: CGRect = .zero

    /// Scales the target window by `outputScale`, keeps the aspect ratio, caps
    /// the result at the screen, and keeps it centred on the window.
    private func outputFrame(forWindow cgFrame: CGRect, on screen: NSScreen) -> CGRect {
        if fillsScreen { return screen.frame }

        let primaryHeight = NSScreen.screens.first?.frame.height ?? 0
        let windowCocoa = CGRect(x: cgFrame.origin.x,
                                 y: primaryHeight - cgFrame.maxY,
                                 width: cgFrame.width,
                                 height: cgFrame.height)
        let bounds = screen.frame

        var width = cgFrame.width * outputScale
        var height = cgFrame.height * outputScale
        if width > 0, height > 0 {
            let fit = min(1.0, min(bounds.width / width, bounds.height / height))
            width *= fit
            height *= fit
        }

        var x = windowCocoa.midX - width / 2
        var y = windowCocoa.midY - height / 2
        x = min(max(x, bounds.minX), bounds.maxX - width)
        y = min(max(y, bounds.minY), bounds.maxY - height)
        return CGRect(x: x, y: y, width: width, height: height)
    }

    func setCaptureCursorEnabled(_ enabled: Bool) {
        self.shouldCaptureCursor = enabled
    }

    /// Applied on the next `updateWindowPosition`, so the overlay tracks a live
    /// change of Scale Factor without tearing the capture down.
    func setOutputScale(_ scale: CGFloat, fillsScreen: Bool) {
        self.outputScale = max(1.0, scale)
        self.fillsScreen = fillsScreen
    }

    deinit {
        if let observer = appObserver {
            NSWorkspace.shared.notificationCenter.removeObserver(observer)
        }
    }

    func createOverlay(config: OverlayWindowConfig) -> Bool {
        destroyOverlay()
        lastError = nil

        self.shouldCaptureCursor = config.captureCursor
        self.outputScale = max(1.0, config.outputScale)
        self.fillsScreen = config.fillsScreen

        guard let screen = config.targetScreen else {
            lastError = "Error Code: MG-OV-001 Target screen missing for overlay creation."
            return false
        }

        guard let frame = config.windowFrame else {
            lastError = "Error Code: MG-OV-002 Window frame missing for overlay creation."
            return false
        }

        // updateWindowPosition drives this from then on, and re-derives the
        // display bounds from the screen the overlay actually lands on.
        let initialFrame = outputFrame(forWindow: frame, on: screen)
        currentSize = initialFrame.size

        let window = NonActivatingWindow(
            contentRect: initialFrame,
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
        displayBoundsCG = .zero
    }

    func setTargetWindow(_ windowID: CGWindowID, pid: pid_t) {
        targetWindowID = windowID
        targetPID = pid
        // Seeded from the world rather than assumed, so the overlay does not show
        // itself over an app the user never left MetalGoose for. The observer only
        // fires on a change, and starting a capture from MetalGoose's own window
        // means the first change is the one that brings the target forward.
        targetIsFrontmost = NSWorkspace.shared.frontmostApplication?.processIdentifier == pid

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

                // The overlay follows the target app, and so does the pointer
                // constraint: both only make sense while the user is actually in
                // the captured window.
                self.targetIsFrontmost = (pid == self.targetPID)
                if self.targetIsFrontmost {
                    self.overlayWindow?.orderFrontRegardless()
                    self.mtkView?.isPaused = false
                    MouseConstraintManager.shared.setSuspended(false)
                } else {
                    self.overlayWindow?.orderOut(nil)
                    // Nothing to present into once it is hidden, and leaving it
                    // running only keeps pushing the stale frame.
                    self.mtkView?.isPaused = true
                    MouseConstraintManager.shared.setSuspended(true)
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

        // A window the user has switched away from is still "on screen" as far as
        // the window server is concerned, so this alone never notices that they
        // left. This runs on a timer, so re-showing the overlay here undid the
        // hide the activation observer had just performed: switching away flashed
        // the desktop and then put the overlay back on top, holding the last
        // frame it was handed. That still image of the game, over whatever the
        // user had switched to and not reacting to input, is what looked like the
        // game grabbing focus back.
        let isOnScreen = (info[kCGWindowIsOnscreen as String] as? Bool) == true
        if !isOnScreen || !targetIsFrontmost {
            window.orderOut(nil)
            return
        }

        guard let boundX = bounds["X"],
              let boundY = bounds["Y"],
              let boundW = bounds["Width"],
              let boundH = bounds["Height"],
              let screen = window.screen else { return }

        let cgFrame = CGRect(x: boundX, y: boundY, width: boundW, height: boundH)
        let nsFrame = outputFrame(forWindow: cgFrame, on: screen)

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

    func isTargetInUnreachableSpace() -> Bool {
        guard let window = overlayWindow, targetPID != 0 else { return false }
        guard let frontPID = NSWorkspace.shared.frontmostApplication?.processIdentifier,
              frontPID == targetPID else { return false }
        return !window.isOnActiveSpace
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

    /// Set while some app other than the capture target is frontmost. The
    /// constraint exists to hold the pointer inside the captured window while the
    /// user is playing, and both halves of it are hostile once they have switched
    /// away: the warp drags the pointer back into the game on the first mouse
    /// movement, and the hide leaves the whole session without a visible cursor.
    /// Between them, switching away with Cmd+Tab looked like the game was pulling
    /// focus back, and nothing else on screen could be used or even seen.
    private var isSuspended = false

    /// Fast enough that a cursor another process reveals is gone again within a
    /// frame at any refresh rate this app runs at.
    private static let cursorReassertInterval: TimeInterval = 0.1

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
        startCursorReassertTimer()
    }

    /// The hide is reference counted per connection, so re-asserting it has to
    /// release our own level first or the count climbs by ten a second and the
    /// teardown has to unwind thousands of levels to get the cursor back. The
    /// pair leaves our contribution at exactly one.
    private func startCursorReassertTimer() {
        cursorHideTimer?.invalidate()
        let timer = Timer(timeInterval: Self.cursorReassertInterval, repeats: true) { _ in
            CGDisplayShowCursor(CGMainDisplayID())
            CGDisplayHideCursor(CGMainDisplayID())
        }
        RunLoop.current.add(timer, forMode: .common)
        cursorHideTimer = timer
    }

    /// Suspending releases exactly the one hide level `startConstraining` took,
    /// so the cursor comes back for the rest of the system; resuming takes it
    /// again. Anything that leaves the count unbalanced either strands the user
    /// without a pointer or needs thousands of releases to recover.
    func setSuspended(_ suspended: Bool) {
        lock.lock()
        guard isConstraining, isSuspended != suspended else {
            lock.unlock()
            return
        }
        isSuspended = suspended
        lock.unlock()

        if suspended {
            cursorHideTimer?.invalidate()
            cursorHideTimer = nil
            CGDisplayShowCursor(CGMainDisplayID())
        } else {
            CGDisplayHideCursor(CGMainDisplayID())
            startCursorReassertTimer()
        }
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
        guard isConstraining, !isSuspended, cursorSpriteVisible,
              sourceRect.width > 0, sourceRect.height > 0 else { return nil }
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
        guard !isSuspended,
              src.width > 0, src.height > 0, disp.width > 0, disp.height > 0 else {
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
        // A suspended constraint has already given its hide level back, so
        // releasing again here would take the count below zero and hand a
        // permanent extra show to whoever hid the cursor next.
        let wasSuspended = isSuspended
        let tap = eventTap
        let source = runLoopSource
        eventTap = nil
        runLoopSource = nil
        isConstraining = false
        isSuspended = false
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

        // The re-assert timer balances itself, so exactly one level is ours.
        if !wasSuspended {
            CGDisplayShowCursor(CGMainDisplayID())
        }

        if let location = CGEvent(source: nil)?.location {
            CGWarpMouseCursorPosition(location)
        }
    }
}
