// Goose HUB integration for MetalGoose. Distributed under GPL-3.0; see LICENSE.
import AppKit
import Carbon.HIToolbox
import Combine
import CoreGraphics
import Darwin
import Foundation
@preconcurrency import Metal

/// MetalGoose's `ScalingController` with the hub in place of its window: the hub names the game to
/// scale instead of a countdown taking the frontmost app, and errors go back to the hub instead of
/// into an alert. The rest is MetalGoose's own: its settings (`CaptureSettings`), its HUD, its two
/// shortcuts and the order a session is set up and torn down in.
@MainActor
final class BackendSession {

    enum Phase: String {
        case idle, starting, active, stopping
    }

    let output: BackendOutput
    let permissions = PermissionManager()
    private let settings = CaptureSettings.shared

    private var engine: GooseEngine?
    private let capture = WindowCaptureManager()
    private let overlay = OverlayWindowManager()
    private let hud = HUDController()

    private(set) var phase: Phase = .idle
    private var isTransitioning = false
    private var activeScreen: NSScreen?
    private var targetPID: pid_t = 0
    private var targetWindowID: CGWindowID = 0

    private var statsTimer: Timer?
    private var settingsObserver: AnyCancellable?

    private var lastHotkeyTime: CFTimeInterval = 0
    private var lastCursorHotkeyTime: CFTimeInterval = 0

    /// The target window has to hold the overlay's Space for this many consecutive checks before the
    /// session is ended: a Space switch in progress looks the same as a fullscreen target for a moment.
    private var fullscreenStrikes = 0
    private static let fullscreenStrikeLimit = 3

    /// The hub that started this backend. Its window is never a target, as MetalGoose's own never is.
    private let hubPID = getppid()

    init(output: BackendOutput) {
        self.output = output

        settingsObserver = settings.objectWillChange
            .receive(on: DispatchQueue.main)
            .sink { [weak self] _ in self?.settingsChanged() }

        overlay.onPresentingChange = { [weak self] presenting in
            self?.engine?.setPresenting(presenting)
        }
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
            self.drawPointer(on: screen, engine: engine)
            if self.settings.showMGHUD { self.showHUD(on: screen, engine: engine) }
        }
    }

    var isActive: Bool { phase == .active }

    /// MetalGoose's shortcuts: ⌘⇧T stops scaling, or starts it on the window the user is in, and ⌘⇧C
    /// shows or hides the pointer. Both are global, and live as long as the backend does. A shortcut
    /// another app already holds cannot be shared, and failing silently would leave a key that does
    /// nothing and no way to learn why, so it is reported.
    func registerHotkeys() {
        var unavailable: [String] = []
        if !GlobalHotkeyManager.shared.register(keyCode: UInt32(kVK_ANSI_T), modifiers: UInt32(cmdKey | shiftKey), handler: { [weak self] in
            Task { @MainActor in self?.hotkeyToggle() }
        }) { unavailable.append("⌘⇧T") }
        if !GlobalHotkeyManager.shared.register(keyCode: UInt32(kVK_ANSI_C), modifiers: UInt32(cmdKey | shiftKey), handler: { [weak self] in
            Task { @MainActor in self?.hotkeyToggleCursor() }
        }) { unavailable.append("⌘⇧C") }
        if !unavailable.isEmpty { output.send(.failure(MGError.shortcutUnavailable(unavailable))) }
    }

    func status() -> BackendStatus {
        permissions.refresh()
        let presentation = overlay.presentationState
        return BackendStatus(screenRecordingGranted: permissions.screenRecordingGranted,
                             accessibilityGranted: permissions.accessibilityGranted,
                             shaderLibrary: GPUContext.shaderLibraryURL?.path,
                             device: engine?.deviceName ?? MTLCreateSystemDefaultDevice()?.name,
                             engineInitialized: engine != nil, phase: phase.rawValue,
                             presenting: presentation.presenting, frontmostPID: presentation.frontmostPID,
                             targetOnScreen: presentation.onScreen, suspensionReason: presentation.reason,
                             pid: targetPID == 0 ? nil : targetPID,
                             windowID: targetWindowID == 0 ? nil : targetWindowID,
                             configuration: BackendConfiguration(settings),
                             metrics: isActive ? engine.map { BackendMetrics($0.stats, generation: $0.generation) } : nil)
    }

    func configure(_ configuration: BackendConfiguration) throws {
        try configuration.apply(to: settings, sessionRunning: phase != .idle)
    }

    // MARK: - Engine

    /// Created on first use, so a Mac without Metal reports the reason when scaling is asked for
    /// instead of failing at launch.
    private func makeEngineIfNeeded() throws -> GooseEngine {
        if let engine { return engine }
        switch GooseEngine.make() {
        case .success(let created):
            engine = created
            created.apply(settings.engineConfig)
            return created
        case .failure(let error):
            throw error
        }
    }

    // MARK: - Starting

    private func hotkeyToggle() {
        let now = CACurrentMediaTime()
        guard now - lastHotkeyTime >= 0.4 else { return }
        lastHotkeyTime = now
        guard !isTransitioning else { return }
        if isActive {
            Task {
                await stop()
                output.send(BackendResponse(event: "stopped", reason: "shortcut", status: status()))
            }
        } else if phase == .idle, let app = NSWorkspace.shared.frontmostApplication {
            Task {
                do {
                    try await start(pid: app.processIdentifier, windowID: nil)
                    output.send(BackendResponse(event: "started", reason: "shortcut", status: status()))
                } catch {
                    output.send(.failure(error))
                }
            }
        }
    }

    private func hotkeyToggleCursor() {
        let now = CACurrentMediaTime()
        guard now - lastCursorHotkeyTime >= 0.3 else { return }
        lastCursorHotkeyTime = now
        MouseConstraintManager.shared.toggleCursorSpriteVisible()
    }

    /// Starts scaling the process's window: the one named, which must be the process's own, or else
    /// its topmost ordinary window, the one MetalGoose would pick.
    func start(pid: pid_t, windowID: CGWindowID?) async throws {
        guard phase == .idle, !isTransitioning else {
            throw MGError("HUB-MG-BUSY", "MetalGoose is already scaling a window.")
        }
        isTransitioning = true
        phase = .starting
        defer {
            isTransitioning = false
            if phase == .starting { phase = .idle }
        }
        try await startSession(pid: pid, windowID: windowID)
    }

    private func startSession(pid: pid_t, windowID: CGWindowID?) async throws {
        permissions.refresh()
        guard permissions.screenRecordingGranted && permissions.accessibilityGranted else { throw MGError.permissionsRequired }
        guard pid > 0, pid != ProcessInfo.processInfo.processIdentifier, pid != hubPID else { throw MGError.frontmostIsSelf }
        guard let target = targetWindow(pid: pid, windowID: windowID) else { throw MGError.targetWindowNotFound }
        let engine = try makeEngineIfNeeded()
        guard let screen = ScreenGeometry.screen(containing: target.frame) else { throw MGError.noDisplay }
        guard ScreenGeometry.displayID(of: screen) != nil else { throw MGError.displayIDUnavailable }

        // Read the active panel's ceiling and variable-refresh floor through the same
        // helper used when the target moves to another display.
        guard let displayRate = ScreenGeometry.displayRate(of: screen) else { throw MGError.refreshRateUnavailable }

        let upscaling = settings.isUpscaling
        let captureTarget = CaptureTarget(windowID: target.id, frame: target.frame,
                                          backingScale: screen.backingScaleFactor, ownerPID: pid)

        engine.apply(settings.engineConfig)
        engine.beginSession()
        // Wired before the stream starts: frames arrive the moment it does.
        capture.onFrame = { [engine] frame in engine.receive(frame) }

        guard await capture.startCapture(target: captureTarget, maxFPS: displayRate.maximum,
                                         renderScale: upscaling ? settings.renderScale.multiplier : 1.0,
                                         pipelineDepth: settings.bufferCount) else {
            let error = capture.lastError ?? MGError("MG-CAP-002", String(localized: "Unknown capture error."))
            await capture.stopCapture()
            capture.onFrame = nil
            engine.endSession()
            throw error
        }

        let configuration = OverlayWindowManager.Configuration(
            screen: screen, windowFrame: target.frame, alignsPointer: settings.alignPointer,
            outputScale: upscaling ? CGFloat(settings.scaleFactor.value) : 1.0,
            fillsScreen: upscaling && settings.scaleFactor.fillsScreen, drawsPointer: !settings.vrr)
        engine.attach(to: overlay.createOverlay(configuration), displayRate: displayRate)
        drawPointer(on: screen, engine: engine)

        overlay.setTarget(windowID: target.id, pid: pid)
        activeScreen = screen
        targetPID = pid
        targetWindowID = target.id
        fullscreenStrikes = 0
        phase = .active

        if settings.showMGHUD { showHUD(on: screen, engine: engine) }
        startStatsTimer()

        // The game can have given up focus to the hub while the session started; hand it back.
        try? await Task.sleep(for: .milliseconds(300))
        NSRunningApplication(processIdentifier: pid)?.activate()
    }

    private func targetWindow(pid: pid_t, windowID: CGWindowID?) -> TargetWindow? {
        guard let windowID else { return TargetWindowFinder.topWindow(ofProcess: pid) }
        guard let info = (CGWindowListCopyWindowInfo([.optionIncludingWindow], windowID) as? [[String: Any]])?
                .first(where: { ($0[kCGWindowNumber as String] as? CGWindowID) == windowID }),
              (info[kCGWindowOwnerPID as String] as? Int32) == pid,
              (info[kCGWindowIsOnscreen as String] as? Bool) == true,
              let bounds = info[kCGWindowBounds as String] as? [String: CGFloat],
              let x = bounds["X"], let y = bounds["Y"],
              let width = bounds["Width"], let height = bounds["Height"] else { return nil }
        return TargetWindow(id: windowID, frame: CGRect(x: x, y: y, width: width, height: height))
    }

    // MARK: - Stopping

    func stop() async {
        MouseConstraintManager.shared.stopConstraining()
        guard isActive, !isTransitioning else { return }
        isTransitioning = true
        defer { isTransitioning = false }
        await endSession()
    }

    /// Everything is torn down in the order that leaves nothing presenting into a window that is
    /// already gone: the engine stops presenting, then the overlay goes, then the capture.
    private func endSession() async {
        statsTimer?.invalidate()
        statsTimer = nil
        phase = .stopping

        engine?.detach()
        overlay.destroyOverlay()
        activeScreen = nil
        engine?.endSession()
        capture.onFrame = nil
        await capture.stopCapture()

        phase = .idle
        targetPID = 0
        targetWindowID = 0
        hud.hide()
    }

    /// The stream ended or could not apply a configuration.
    private func captureEnded(_ error: MGError?) {
        guard isActive, !isTransitioning else { return }
        end(reason: "capture", error: error)
    }

    private func end(reason: String, error: MGError?) {
        isTransitioning = true
        Task {
            defer { isTransitioning = false }
            await endSession()
            if let error { output.send(.failure(error)) }
            output.send(BackendResponse(event: "stopped", reason: reason, status: status()))
        }
    }

    /// This must also run synchronously on process termination to restore a hidden system cursor.
    func restorePointer() { MouseConstraintManager.shared.stopConstraining() }

    // MARK: - While running

    private func settingsChanged() {
        engine?.apply(settings.engineConfig)

        let upscaling = settings.isUpscaling
        overlay.setOutputScale(upscaling ? CGFloat(settings.scaleFactor.value) : 1.0,
                               fillsScreen: upscaling && settings.scaleFactor.fillsScreen)
        overlay.setAlignsPointer(settings.alignPointer)
        overlay.setDrawsPointer(!settings.vrr)
        guard isActive else { return }

        Task { await capture.reconfigure(renderScale: upscaling ? settings.renderScale.multiplier : 1.0) }

        if settings.showMGHUD, !hud.isShowing, let screen = activeScreen, let engine {
            showHUD(on: screen, engine: engine)
        } else if !settings.showMGHUD {
            hud.hide()
        }
        if let engine { followContentResolution(engine: engine) }
    }

    /// The pointer the engine draws into the picture with VRR on, at the pixels a point the overlay is drawn at.
    private func drawPointer(on screen: NSScreen, engine: GooseEngine) {
        let pointer = OverlayView.pointerImage(scale: overlay.drawableScale ?? screen.backingScaleFactor)
        engine.setPointer(pointer?.image, hotSpot: pointer?.hotSpot ?? .zero, position: { MouseConstraintManager.shared.pointer })
    }

    private func showHUD(on screen: NSScreen, engine: GooseEngine) {
        hud.show(scale: overlay.drawableScale ?? screen.backingScaleFactor, engine: engine)
        pushHUD(engine: engine)
    }

    /// The overlay at the content's resolution once the capture has found the content drawn at one pixel a point and
    /// nothing scales it up (`OverlayView.contentMagnification`), and the HUD and the pointer the engine draws into the
    /// picture at it too.
    private func followContentResolution(engine: GooseEngine) {
        let before = overlay.drawableScale
        overlay.setContentMagnification(settings.isUpscaling ? nil : capture.contentMagnification)
        guard overlay.drawableScale != before, let screen = activeScreen else { return }
        drawPointer(on: screen, engine: engine)
        if settings.showMGHUD { showHUD(on: screen, engine: engine) }
    }

    private func startStatsTimer() {
        statsTimer?.invalidate()
        statsTimer = Timer.scheduledTimer(withTimeInterval: 1.0, repeats: true) { [weak self] _ in
            MainActor.assumeIsolated { self?.tick() }
        }
    }

    private func tick() {
        guard isActive, !isTransitioning, let engine else { return }
        followContentResolution(engine: engine)
        pushHUD(engine: engine)
        if let error = engine.takeError() { output.send(.failure(error)) }

        // A Wine process need not be registered with LaunchServices, so the PID is what says the game
        // is gone.
        if Darwin.kill(targetPID, 0) != 0, errno == ESRCH {
            end(reason: "target-exited", error: nil)
            return
        }

        // Fullscreen gets a Space of its own, which the overlay cannot join; the target is then
        // frontmost but the overlay is nowhere to be seen. Checked a few times in a row because a
        // Space switch in progress looks the same.
        if overlay.isTargetInUnreachableSpace() {
            fullscreenStrikes += 1
            if fullscreenStrikes >= Self.fullscreenStrikeLimit {
                fullscreenStrikes = 0
                end(reason: "fullscreen", error: MGError.targetEnteredFullscreen)
                return
            }
        } else {
            fullscreenStrikes = 0
        }
        output.send(BackendResponse(event: "stats", status: status()))
    }

    private func pushHUD(engine: GooseEngine) {
        guard hud.isShowing else { return }
        hud.update(stats: engine.stats, info: hudInfo(engine))
    }

    private func hudInfo(_ engine: GooseEngine) -> HUDInfo {
        let size = capture.contentPixelSize
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
                       vsync: settings.vsync ? String(localized: "On") : String(localized: "Off"),
                       vrr: settings.vrr ? String(localized: "On") : String(localized: "Off"))
    }

    /// What the Frame Gen row says: the mode, and how many images it delivers a capture where an engine is making them.
    private static func describeMode(_ mode: FrameGenMode, _ choice: GenerationChoice) -> String {
        let name = String(localized: mode.title)
        let multiplier = String(localized: "Frame multiplier", defaultValue: "\(choice.multiplier)×")
        return choice.engine == nil ? name : "\(name) \(multiplier)"
    }

    /// What the Engine row says: the engine making the images, and the size the Neural Engine works at where that is not the
    /// capture's own; or why no engine is making them.
    private static func describeEngine(_ choice: GenerationChoice) -> String {
        guard let engine = choice.engine else {
            // No engine: the Neural Engine's session is being built, or nothing can make the steps asked for just now.
            // Literal keys, one to a call, so that Xcode's string extraction sees every one of them.
            if choice.unavailable { return String(localized: "Unavailable") }
            return String(localized: "Starting")
        }
        let name = String(localized: engine.title)
        guard let size = choice.neuralSize else { return name }
        return "\(name) (\(size.width)x\(size.height))"
    }
}
