import AppKit
import CoreGraphics
import QuartzCore

/// A window that never takes focus: it exists to be looked at, and the app underneath it keeps
/// receiving the keyboard.
final class NonActivatingWindow: NSWindow {
    override var canBecomeKey: Bool { false }
    override var canBecomeMain: Bool { false }
}

/// The borderless window that sits over the captured window and shows the engine's output.
@MainActor
final class OverlayWindowManager {

    struct Configuration {
        let screen: NSScreen
        /// The target window's frame in CoreGraphics coordinates.
        let windowFrame: CGRect
        /// Whether the pointer is taken to where the picture shows the window, where the overlay is not over it point for
        /// point.
        let alignsPointer: Bool
        /// Requested magnification. 1.0 overlays the target window exactly; larger values grow around
        /// the window centre and stop at the screen edges.
        let outputScale: CGFloat
        /// Ignores `outputScale` and covers the whole display.
        let fillsScreen: Bool
    }

    /// Whether the overlay should currently be drawing. It follows the captured app: while the user is
    /// somewhere else the overlay would be a still image of the game sitting on top of whatever they
    /// switched to, which reads as the game refusing to give up focus.
    var onPresentingChange: ((Bool) -> Void)?
    /// The overlay's size or position changed, so what it holds has to be drawn again.
    var onGeometryChange: (() -> Void)?
    /// The captured window changed size or moved to a screen of a different density, so the stream
    /// has to be asked for it again.
    var onTargetResized: ((CaptureTarget) -> Void)?

    private var window: NonActivatingWindow?
    private var view: OverlayView?
    private var targetWindowID: CGWindowID = 0
    private var targetPID: pid_t = 0
    private var targetFrame: CGRect = .zero
    private var targetPixelSize: CGSize = .zero
    private var targetIsFrontmost = true
    private var isPresenting = false

    private var alignsPointer = true
    private var outputScale: CGFloat = 1
    private var fillsScreen = false

    private var activationObserver: NSObjectProtocol?
    private var pollTimer: Timer?

    /// How often the captured window is looked at again. Position and size come from the window
    /// server, one small synchronous call, and the overlay should follow a dragged or resized window
    /// within a few frames rather than a second behind it.
    private static let pollInterval: TimeInterval = 0.1

    // MARK: - Geometry

    /// Scales the target window by `outputScale`, keeps the aspect ratio, caps the result at the
    /// screen, and keeps it centred on the window.
    private func outputFrame(forWindow cgFrame: CGRect, on screen: NSScreen) -> CGRect {
        if fillsScreen { return screen.frame }

        let windowCocoa = ScreenGeometry.cocoaFrame(from: cgFrame)
        let bounds = screen.frame

        var width = cgFrame.width * outputScale
        var height = cgFrame.height * outputScale
        if width > 0, height > 0 {
            let fit = min(1.0, min(bounds.width / width, bounds.height / height))
            width *= fit
            height *= fit
        }

        let x = min(max(windowCocoa.midX - width / 2, bounds.minX), bounds.maxX - width)
        let y = min(max(windowCocoa.midY - height / 2, bounds.minY), bounds.maxY - height)
        return CGRect(x: x, y: y, width: width, height: height)
    }

    /// Applied on the next refresh, so the overlay tracks a live change of Scale Factor without tearing
    /// the capture down.
    func setOutputScale(_ scale: CGFloat, fillsScreen: Bool) {
        outputScale = max(1.0, scale)
        self.fillsScreen = fillsScreen
    }

    func setAlignsPointer(_ enabled: Bool) {
        alignsPointer = enabled
    }

    // MARK: - Lifecycle

    /// Creates the window and returns the layer the engine should present into.
    func createOverlay(_ configuration: Configuration) -> CAMetalLayer {
        destroyOverlay()

        alignsPointer = configuration.alignsPointer
        outputScale = max(1.0, configuration.outputScale)
        fillsScreen = configuration.fillsScreen
        targetFrame = configuration.windowFrame

        let frame = outputFrame(forWindow: configuration.windowFrame, on: configuration.screen)
        // `frame` is in global coordinates. Given a screen, the window would take it as relative to that screen's origin
        // and open off to the side on any screen but the primary one, until the first refresh moved it.
        let window = NonActivatingWindow(contentRect: frame, styleMask: [.borderless], backing: .buffered,
                                         defer: false, screen: nil)
        window.isReleasedWhenClosed = false
        window.level = NSWindow.Level(rawValue: Int(CGWindowLevelForKey(.maximumWindow)) + 1)
        window.backgroundColor = .clear
        window.isOpaque = false
        window.hasShadow = false
        window.ignoresMouseEvents = true
        window.acceptsMouseMovedEvents = false
        window.collectionBehavior = [.canJoinAllSpaces, .fullScreenAuxiliary, .stationary, .ignoresCycle]

        let view = OverlayView(frame: CGRect(origin: .zero, size: frame.size))
        window.contentView = view
        window.orderFrontRegardless()

        self.window = window
        self.view = view
        isPresenting = true

        MouseConstraintManager.shared.onPointerChange = { [weak view] fraction in view?.setPointer(fraction) }
        return view.metalLayer
    }

    func destroyOverlay() {
        if let observer = activationObserver {
            NSWorkspace.shared.notificationCenter.removeObserver(observer)
            activationObserver = nil
        }
        pollTimer?.invalidate()
        pollTimer = nil
        MouseConstraintManager.shared.stopConstraining()
        MouseConstraintManager.shared.onPointerChange = nil
        window?.orderOut(nil)
        window = nil
        view = nil
        targetWindowID = 0
        targetPID = 0
    }

    /// Starts following the captured window and the app that owns it.
    func setTarget(windowID: CGWindowID, pid: pid_t) {
        targetWindowID = windowID
        targetPID = pid
        // Seeded from the world rather than assumed, so the overlay does not show itself over an app
        // the user never left MetalGoose for. The observer only fires on a change, and starting a
        // capture from MetalGoose's own window means the first change is the one that brings the
        // target forward.
        targetIsFrontmost = NSWorkspace.shared.frontmostApplication?.processIdentifier == pid

        if let observer = activationObserver {
            NSWorkspace.shared.notificationCenter.removeObserver(observer)
        }
        activationObserver = NSWorkspace.shared.notificationCenter.addObserver(
            forName: NSWorkspace.didActivateApplicationNotification, object: nil, queue: .main
        ) { [weak self] notification in
            let app = notification.userInfo?[NSWorkspace.applicationUserInfoKey] as? NSRunningApplication
            let pid = app?.processIdentifier
            MainActor.assumeIsolated {
                guard let self, self.targetPID != 0, let pid else { return }
                self.targetIsFrontmost = pid == self.targetPID
                self.applyPresenting(self.targetIsFrontmost && self.isOnScreen())
            }
        }

        pollTimer?.invalidate()
        pollTimer = Timer.scheduledTimer(withTimeInterval: Self.pollInterval, repeats: true) { [weak self] _ in
            MainActor.assumeIsolated { self?.refresh() }
        }
        refresh()
    }

    // MARK: - Following the target

    private func windowInfo() -> [String: Any]? {
        guard targetWindowID != 0,
              let list = CGWindowListCopyWindowInfo([.optionIncludingWindow], targetWindowID) as? [[String: Any]] else { return nil }
        return list.first
    }

    private func isOnScreen() -> Bool {
        (windowInfo()?[kCGWindowIsOnscreen as String] as? Bool) == true
    }

    /// Hides or shows the overlay, and with it the pointer constraint: both only make sense while
    /// the user is actually in the captured window.
    private func applyPresenting(_ presenting: Bool) {
        guard presenting != isPresenting else { return }
        isPresenting = presenting
        if presenting {
            window?.orderFrontRegardless()
        } else {
            window?.orderOut(nil)
        }
        MouseConstraintManager.shared.setSuspended(!presenting)
        onPresentingChange?(presenting)
    }

    /// Re-reads the captured window and brings the overlay into line with it.
    func refresh() {
        guard let window, let view, let info = windowInfo(),
              let bounds = info[kCGWindowBounds as String] as? [String: CGFloat],
              let x = bounds["X"], let y = bounds["Y"], let width = bounds["Width"], let height = bounds["Height"] else { return }

        // A window the user has switched away from is still "on screen" as far as the window server
        // is concerned, so that alone never notices they left — frontmost status is what does.
        let onScreen = (info[kCGWindowIsOnscreen as String] as? Bool) == true
        applyPresenting(onScreen && targetIsFrontmost)
        guard isPresenting else { return }

        let cgFrame = CGRect(x: x, y: y, width: width, height: height)
        guard let screen = ScreenGeometry.screen(containing: cgFrame) else { return }

        let pixelSize = CGSize(width: cgFrame.width * screen.backingScaleFactor,
                               height: cgFrame.height * screen.backingScaleFactor)
        if pixelSize != targetPixelSize {
            if targetPixelSize != .zero {
                onTargetResized?(CaptureTarget(windowID: targetWindowID, size: cgFrame.size,
                                               backingScale: screen.backingScaleFactor))
            }
            targetPixelSize = pixelSize
        }
        targetFrame = cgFrame

        let frame = outputFrame(forWindow: cgFrame, on: screen)
        if window.frame != frame {
            window.setFrame(frame, display: false)
            view.frame = CGRect(origin: .zero, size: frame.size)
            onGeometryChange?()
        }

        // The pointer is taken to the picture where the picture is not where the window is. Where it is, the pointer is
        // where the user sees it already.
        let mouse = MouseConstraintManager.shared
        guard alignsPointer, !PointerMapping.isIdentity(window: cgFrame, overlay: ScreenGeometry.cgFrame(from: frame)) else {
            mouse.stopConstraining()
            return
        }
        let displays = ScreenGeometry.displayBounds
        if mouse.isConstraining {
            mouse.update(window: cgFrame, displays: displays)
        } else {
            mouse.startConstraining(window: cgFrame, displays: displays)
        }
    }

    /// The target is the frontmost app but the overlay is not on the active Space: the target went
    /// fullscreen, which gets a Space of its own that the overlay cannot join.
    func isTargetInUnreachableSpace() -> Bool {
        guard let window, targetPID != 0,
              NSWorkspace.shared.frontmostApplication?.processIdentifier == targetPID else { return false }
        return !window.isOnActiveSpace
    }
}
