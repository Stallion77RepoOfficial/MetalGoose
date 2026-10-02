import AppKit
import Carbon.HIToolbox
import Combine
import CoreGraphics
import SwiftUI

/// Runs a scaling session: picks the window the user is in, captures it, shows the engine's output
/// over it, and tears it all down again. The view only reads its state and calls `toggle`.
@MainActor
final class ScalingController: ObservableObject {

    enum Phase: Equatable {
        case idle
        case countingDown(Int)
        case active
    }

    @Published private(set) var phase: Phase = .idle
    /// A message for the user, shown as an alert until dismissed.
    @Published var alertMessage: String?

    private let settings: CaptureSettings
    private let permissions: PermissionManager

    private var engine: GooseEngine?
    private let capture = WindowCaptureManager()
    private let overlay = OverlayWindowManager()
    private let hud = HUDWindowController()

    private var engineCreationFailed = false
    private var isTransitioning = false
    private var activeScreen: NSScreen?
    private var targetPID: pid_t = 0

    private var statsTimer: Timer?
    private var countdownTimer: Timer?
    private var settingsObserver: AnyCancellable?

    private var lastHotkeyTime: CFTimeInterval = 0
    private var lastCursorHotkeyTime: CFTimeInterval = 0

    /// Set by the view once it has a window, so the controller can hide and restore it without
    /// guessing which window is the right one from its title.
    weak var mainWindow: NSWindow?

    /// The target window has to hold the overlay's Space for this many consecutive checks before the
    /// session is ended: a Space switch in progress looks the same as a fullscreen target for a moment.
    private var fullscreenStrikes = 0
    private static let fullscreenStrikeLimit = 3

    init(settings: CaptureSettings, permissions: PermissionManager) {
        self.settings = settings
        self.permissions = permissions

        settingsObserver = settings.objectWillChange
            .receive(on: DispatchQueue.main)
            .sink { [weak self] _ in self?.settingsChanged() }

        // Both hotkeys are global and deliberately outlive the window. Closing it with Cmd+W leaves
        // the overlay and the capture running, and tearing the hotkeys down with the window left no
        // way to stop either: no window to click Stop in, and Cmd+Shift+T dead. They cost nothing
        // while the process lives and die with it.
        //
        // A shortcut another app already holds cannot be shared, and failing silently would leave a
        // key that does nothing and no way to learn why, so it is reported.
        var unavailable: [String] = []
        if !GlobalHotkeyManager.shared.register(keyCode: UInt32(kVK_ANSI_T), modifiers: UInt32(cmdKey | shiftKey), handler: { [weak self] in
            Task { @MainActor in self?.hotkeyToggle() }
        }) { unavailable.append("⌘⇧T") }
        if !GlobalHotkeyManager.shared.register(keyCode: UInt32(kVK_ANSI_C), modifiers: UInt32(cmdKey | shiftKey), handler: { [weak self] in
            Task { @MainActor in self?.hotkeyToggleCursor() }
        }) { unavailable.append("⌘⇧C") }
        if !unavailable.isEmpty { alertMessage = MGError.shortcutUnavailable(unavailable).message }

        // The pointer is hidden system-wide while a constraint is on, and a hidden cursor outlives the
        // process, so that part is undone synchronously on the way out; the rest of the teardown is
        // idempotent.
        NotificationCenter.default.addObserver(forName: NSApplication.willTerminateNotification, object: nil,
                                               queue: .main) { [weak self] _ in
            MainActor.assumeIsolated {
                MouseConstraintManager.shared.stopConstraining()
                self?.stop()
            }
        }

        overlay.onPresentingChange = { [weak self] presenting in self?.engine?.setPresenting(presenting) }
        overlay.onGeometryChange = { [weak self] in self?.engine?.requestRedraw() }
        overlay.onTargetResized = { [weak self] target in
            guard let self else { return }
            Task { await self.capture.reconfigure(window: target) }
        }
        capture.onStop = { [weak self] error in
            Task { @MainActor in self?.captureEnded(error) }
        }
    }

    var isActive: Bool { phase == .active }

    // MARK: - Engine

    /// Created on first use, so a Mac without Metal shows the reason when the user asks for scaling
    /// instead of failing at launch.
    private func makeEngineIfNeeded() -> GooseEngine? {
        if let engine { return engine }
        switch GooseEngine.make() {
        case .success(let created):
            engine = created
            created.apply(settings.engineConfig)
            return created
        case .failure(let error):
            if !engineCreationFailed { alertMessage = error.message }
            engineCreationFailed = true
            return nil
        }
    }

    // MARK: - Starting

    func hotkeyToggle() {
        let now = CACurrentMediaTime()
        guard now - lastHotkeyTime >= 0.4 else { return }
        lastHotkeyTime = now
        toggle()
    }

    private func hotkeyToggleCursor() {
        let now = CACurrentMediaTime()
        guard now - lastCursorHotkeyTime >= 0.3 else { return }
        lastCursorHotkeyTime = now
        MouseConstraintManager.shared.toggleCursorSpriteVisible()
    }

    func toggle() {
        guard permissionsAllowScaling, !isTransitioning else { return }
        if isActive { stop() } else { start() }
    }

    /// Screen Recording is always needed. Accessibility only is while the pointer is being remapped:
    /// without it the event tap cannot be made, and the overlay would show a pointer that is not
    /// where the game thinks it is.
    var permissionsAllowScaling: Bool {
        permissions.screenRecordingGranted && (permissions.accessibilityGranted || !settings.captureCursor)
    }

    func startCountdown() {
        guard phase == .idle else { return }
        phase = .countingDown(5)
        countdownTimer?.invalidate()
        countdownTimer = Timer.scheduledTimer(withTimeInterval: 1.0, repeats: true) { [weak self] _ in
            MainActor.assumeIsolated {
                guard let self, case .countingDown(let remaining) = self.phase else { return }
                if remaining > 1 {
                    self.phase = .countingDown(remaining - 1)
                } else {
                    self.countdownTimer?.invalidate()
                    self.countdownTimer = nil
                    self.phase = .idle
                    self.mainWindow?.orderOut(nil)
                    self.start()
                }
            }
        }
    }

    func cancelCountdown() {
        countdownTimer?.invalidate()
        countdownTimer = nil
        if case .countingDown = phase { phase = .idle }
    }

    private func start() {
        guard !isTransitioning else { return }
        isTransitioning = true
        Task {
            defer { isTransitioning = false }
            await startSession()
        }
    }

    private func fail(_ error: MGError) {
        alertMessage = error.message
    }

    private func startSession() async {
        guard let app = NSWorkspace.shared.frontmostApplication,
              app.processIdentifier != ProcessInfo.processInfo.processIdentifier else {
            fail(.frontmostIsSelf)
            return
        }
        guard let target = TargetWindowFinder.topWindow(ofProcess: app.processIdentifier) else {
            fail(.targetWindowNotFound)
            return
        }
        guard let engine = makeEngineIfNeeded() else { return }
        guard let screen = ScreenGeometry.screen(containing: target.frame) else {
            fail(.noDisplay)
            return
        }
        guard let displayID = screen.deviceDescription[NSDeviceDescriptionKey("NSScreenNumber")] as? CGDirectDisplayID else {
            fail(.displayIDUnavailable)
            return
        }

        // Two independent readings of the same panel rather than a hardcoded fallback: AppKit reports
        // the mode's rate, CoreGraphics reports the active display mode. Some virtual and captured
        // displays leave one of the two at zero, and no real display leaves both there.
        let modeRate = CGDisplayCopyDisplayMode(displayID)?.refreshRate ?? 0
        let maximumRate = max(screen.maximumFramesPerSecond, Int(modeRate.rounded()))
        guard maximumRate > 0 else {
            fail(.refreshRateUnavailable)
            return
        }
        // The panel's own floor. On a fixed-refresh display it equals the ceiling, which is what tells
        // the engine there is no variable range.
        let longestInterval = screen.maximumRefreshInterval
        let minimumRate = longestInterval > 0 ? Int((1.0 / longestInterval).rounded()) : maximumRate
        let displayRate = DisplayRate(maximum: maximumRate, minimum: minimumRate)

        let upscaling = settings.isUpscaling
        let captureTarget = CaptureTarget(windowID: target.id, size: target.frame.size,
                                          backingScale: screen.backingScaleFactor)

        engine.apply(settings.engineConfig)
        engine.beginSession()
        // Wired before the stream starts: frames arrive the moment it does.
        capture.onFrame = { [engine] frame in engine.receive(frame) }

        guard await capture.startCapture(target: captureTarget, maxFPS: maximumRate, showsCursor: false,
                                         renderScale: upscaling ? settings.renderScale.multiplier : 1.0,
                                         queueDepth: settings.bufferCount) else {
            alertMessage = (capture.lastError ?? MGError("MG-CAP-002", "Unknown capture error.")).message
            await capture.stopCapture()
            capture.onFrame = nil
            return
        }

        let configuration = OverlayWindowManager.Configuration(
            screen: screen, windowFrame: target.frame, captureCursor: settings.captureCursor,
            outputScale: upscaling ? CGFloat(settings.scaleFactor.value) : 1.0,
            fillsScreen: upscaling && settings.scaleFactor.fillsScreen)
        engine.attach(to: overlay.createOverlay(configuration), displayRate: displayRate)

        overlay.setTarget(windowID: target.id, pid: app.processIdentifier)
        activeScreen = screen
        targetPID = app.processIdentifier
        fullscreenStrikes = 0
        phase = .active

        if settings.showMGHUD { showHUD(on: screen, engine: engine) }
        startStatsTimer()

        // The target gave up focus to MetalGoose's window while the session started; hand it back.
        let pid = app.processIdentifier
        try? await Task.sleep(for: .milliseconds(300))
        NSRunningApplication(processIdentifier: pid)?.activate()
    }

    // MARK: - Stopping

    func stop() {
        MouseConstraintManager.shared.stopConstraining()
        cancelCountdown()
        guard isActive, !isTransitioning else { return }
        isTransitioning = true
        Task {
            defer { isTransitioning = false }
            await endSession()
        }
    }

    /// Everything is torn down in the order that leaves nothing presenting into a window that is
    /// already gone: the engine stops presenting, then the overlay goes, then the capture.
    private func endSession() async {
        statsTimer?.invalidate()
        statsTimer = nil

        engine?.detach()
        overlay.destroyOverlay()
        activeScreen = nil
        engine?.endSession()
        capture.onFrame = nil
        await capture.stopCapture()

        phase = .idle
        targetPID = 0
        hud.hide()
    }

    /// The stream ended without being asked to — the captured window closed, or the system stopped it.
    private func captureEnded(_ error: MGError?) {
        guard isActive, !isTransitioning else { return }
        isTransitioning = true
        Task {
            defer { isTransitioning = false }
            await endSession()
            if let error { alertMessage = error.message }
        }
    }

    // MARK: - While running

    private func settingsChanged() {
        engine?.apply(settings.engineConfig)

        let upscaling = settings.isUpscaling
        overlay.setOutputScale(upscaling ? CGFloat(settings.scaleFactor.value) : 1.0,
                               fillsScreen: upscaling && settings.scaleFactor.fillsScreen)
        overlay.setCaptureCursor(settings.captureCursor)
        guard isActive else { return }

        Task { await capture.reconfigure(renderScale: upscaling ? settings.renderScale.multiplier : 1.0) }

        if settings.showMGHUD, !hud.isShowing, let screen = activeScreen, let engine {
            showHUD(on: screen, engine: engine)
        } else if !settings.showMGHUD {
            hud.hide()
        }
    }

    private func showHUD(on screen: NSScreen, engine: GooseEngine) {
        hud.show(on: screen)
        pushHUD(engine: engine)
    }

    private func startStatsTimer() {
        statsTimer?.invalidate()
        statsTimer = Timer.scheduledTimer(withTimeInterval: 1.0, repeats: true) { [weak self] _ in
            MainActor.assumeIsolated { self?.tick() }
        }
    }

    private func tick() {
        guard isActive, let engine else { return }
        pushHUD(engine: engine)
        if let error = engine.takeError() { alertMessage = error.message }

        // Fullscreen gets a Space of its own, which the overlay cannot join; the target is then
        // frontmost but the overlay is nowhere to be seen. Checked a few times in a row because a
        // Space switch in progress looks the same.
        if overlay.isTargetInUnreachableSpace() {
            fullscreenStrikes += 1
            if fullscreenStrikes >= Self.fullscreenStrikeLimit {
                fullscreenStrikes = 0
                isTransitioning = true
                Task {
                    defer { isTransitioning = false }
                    await endSession()
                    bringToFront()
                    alertMessage = MGError.targetEnteredFullscreen.message
                }
            }
        } else {
            fullscreenStrikes = 0
        }
    }

    private func pushHUD(engine: GooseEngine) {
        guard hud.isShowing else { return }
        hud.update(stats: engine.stats, info: hudInfo(engine))
    }

    private func hudInfo(_ engine: GooseEngine) -> HUDInfo {
        let size = capture.capturePixelSize
        let generation: String
        switch settings.frameGenMode {
        case .off:           generation = "Off"
        case .interpolation:
            // What was asked for is a preference; the engine reports what is actually in use.
            generation = "Interp (\(engine.interpolationSteps)x) · \(String(localized: engine.activeInterpolationEngine.title))"
        case .extrapolation: generation = "Extrap (\(settings.effectiveMultiplier)x)"
        }
        return HUDInfo(deviceName: engine.deviceName,
                       pid: targetPID,
                       captureResolution: size == .zero ? "-" : "\(Int(size.width))x\(Int(size.height))",
                       upscale: "\(String(localized: settings.scalingMethod.title)) \(settings.scaleFactor.rawValue)",
                       renderScale: String(localized: settings.renderScale.title),
                       frameGeneration: generation,
                       antiAliasing: String(localized: settings.aaMode.title),
                       vsync: settings.vsync ? "On" : "Off")
    }

    func bringToFront() {
        mainWindow?.makeKeyAndOrderFront(nil)
        NSApp.activate(ignoringOtherApps: true)
    }
}
