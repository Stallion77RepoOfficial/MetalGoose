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
    @Published var alertMessage: String? {
        didSet {
            // Opening an alert ends target selection. A timer must never start a
            // capture behind the dialog or replace its message before acknowledgement.
            if alertMessage != nil { cancelCountdown() }
        }
    }
    @Published private(set) var isTransitioning = false

    private let settings: CaptureSettings
    private let permissions: PermissionManager

    private var engine: GooseEngine?
    private let capture = WindowCaptureManager()
    private let overlay = OverlayWindowManager()
    private let hud = HUDWindowController()

    private var engineCreationFailed = false
    private var activeScreen: NSScreen?
    private var targetPID: pid_t = 0

    private var statsTimer: Timer?
    private var countdownTimer: Timer?
    private var settingsObserver: AnyCancellable?
    private var updateObserver: AnyCancellable?

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
        updateObserver = AutoUpdater.shared.$state.sink { [weak self] state in
            if state.blocksScalingStart { self?.cancelCountdown() }
        }

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
        overlay.onTargetChanged = { [weak self] target, covered in
            self?.capture.follow(target, includesAppWindows: covered)
        }
        capture.onStop = { [weak self] error in
            Task { @MainActor in self?.captureEnded(error) }
        }
        overlay.onDisplayChanged = { [weak self] screen, rate in
            guard let self, let engine = self.engine else { return }
            self.activeScreen = screen
            engine.updateDisplayRate(rate)
            Task { await self.capture.reconfigure(maxFPS: rate.maximum) }
            if self.settings.showMGHUD { self.showHUD(on: screen, engine: engine) }
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
            if !engineCreationFailed { fail(error) }
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
        // A dialog or a revoked permission blocks starting, but the global
        // shortcut must still be able to stop an existing session.
        guard !isTransitioning else { return }
        if isActive { stop() } else { start() }
    }

    /// Both permissions are required before a session starts.
    var permissionsAllowScaling: Bool {
        permissions.screenRecordingGranted && permissions.accessibilityGranted
    }

    private var hasBlockingPresentation: Bool {
        alertMessage != nil || AutoUpdater.shared.state.blocksScalingStart
            || NSApp.modalWindow != nil || mainWindow?.attachedSheet != nil
    }

    func startCountdown() {
        guard phase == .idle, permissionsAllowScaling, !isTransitioning, !hasBlockingPresentation else { return }
        phase = .countingDown(5)
        countdownTimer = Timer.scheduledTimer(withTimeInterval: 1.0, repeats: true) { [weak self] _ in
            MainActor.assumeIsolated {
                guard let self, case .countingDown(let remaining) = self.phase else { return }
                guard !self.hasBlockingPresentation else {
                    self.cancelCountdown()
                    return
                }
                if remaining > 1 {
                    self.phase = .countingDown(remaining - 1)
                } else {
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
        cancelCountdown()
        guard !isActive, permissionsAllowScaling, !isTransitioning, !hasBlockingPresentation else { return }
        isTransitioning = true
        Task {
            defer { isTransitioning = false }
            await startSession()
        }
    }

    private func fail(_ error: MGError) {
        bringToFront()
        alertMessage = error.message
    }

    private func startSession() async {
        guard permissionsAllowScaling, !hasBlockingPresentation else { return }
        // Resolve the target before hiding our window. Ordering it out first can
        // activate an unrelated app and make it look like the user's chosen target.
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
        guard ScreenGeometry.displayID(of: screen) != nil else {
            fail(.displayIDUnavailable)
            return
        }

        // Read the active panel's ceiling and variable-refresh floor through the same
        // helper used when the target moves to another display.
        guard let displayRate = ScreenGeometry.displayRate(of: screen) else {
            fail(.refreshRateUnavailable)
            return
        }

        let upscaling = settings.isUpscaling
        let captureTarget = CaptureTarget(windowID: target.id, frame: target.frame,
                                          backingScale: screen.backingScaleFactor)

        engine.apply(settings.engineConfig)
        engine.beginSession()
        // Wired before the stream starts: frames arrive the moment it does.
        capture.onFrame = { [engine] frame in engine.receive(frame) }

        guard await capture.startCapture(target: captureTarget, maxFPS: displayRate.maximum, showsCursor: false,
                                         renderScale: upscaling ? settings.renderScale.multiplier : 1.0,
                                         pipelineDepth: settings.bufferCount) else {
            let error = capture.lastError ?? MGError("MG-CAP-002", String(localized: "Unknown capture error."))
            await capture.stopCapture()
            capture.onFrame = nil
            engine.endSession()
            fail(error)
            return
        }

        mainWindow?.orderOut(nil)
        let configuration = OverlayWindowManager.Configuration(
            screen: screen, windowFrame: target.frame, alignsPointer: settings.alignPointer,
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
        overlay.setAlignsPointer(settings.alignPointer)
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
        // Leave queued engine errors pending while a user-facing alert is open.
        if alertMessage == nil, let error = engine.takeError() { alertMessage = error.message }

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
        // What was asked for is a preference; the engine reports what is actually in use.
        let off = settings.frameGenMode == .off
        let choice = engine.generation
        return HUDInfo(deviceName: engine.deviceName,
                       pid: targetPID,
                       captureResolution: size == .zero ? "-" : "\(Int(size.width))x\(Int(size.height))",
                       upscale: "\(String(localized: settings.scalingMethod.title)) \(String(localized: settings.scaleFactor.title))",
                       renderScale: String(localized: settings.renderScale.title),
                       frameGeneration: off ? String(localized: "Off") : Self.describeMode(settings.frameGenMode, choice),
                       generatesFrames: !off,
                       generationEngine: off ? "-" : Self.describeEngine(choice),
                       antiAliasing: String(localized: settings.aaMode.title),
                       vsync: settings.vsync ? String(localized: "On") : String(localized: "Off"))
    }

    /// What the Frame Gen row says: the mode, and how many images it delivers a capture where an engine is making them.
    private static func describeMode(_ mode: FrameGenMode, _ choice: GenerationChoice) -> String {
        let name = String(localized: mode.title)
        let multiplier = String(localized: "Frame multiplier", defaultValue: "\(choice.multiplier)×")
        return choice.engine == nil ? name : "\(name) \(multiplier)"
    }

    /// What the Engine row says: the engine making the images, and the size the Neural Engine works at where that is not the
    /// capture's own; or why no engine is making any.
    private static func describeEngine(_ choice: GenerationChoice) -> String {
        guard let engine = choice.engine else {
            // No engine: the panel could not show more than the captures, the Neural Engine's session is being built, or
            // nothing can make its images in time.
            if choice.limitedByPanel { return String(localized: "Display-limited") }
            // Literal keys, one to a call, so that Xcode's string extraction sees every one of them.
            if choice.neuralRung != nil { return String(localized: "Starting") }
            return String(localized: "Not keeping up")
        }
        let name = String(localized: engine.title)
        guard let size = choice.neuralSize else { return name }
        return "\(name) (\(size.width)x\(size.height))"
    }

    func bringToFront() {
        mainWindow?.makeKeyAndOrderFront(nil)
        NSApp.activate(ignoringOtherApps: true)
    }
}
