import AppKit
import Combine
import CoreGraphics
import Darwin
import Foundation

/// Owns the active capture/engine/overlay/HUD pipeline shared by the GUI app (`ScalingController`)
/// and the headless Goose HUB service (`BackendSession`).
@MainActor
final class ScalingSession {

    private let settings: CaptureSettings
    private var engine: GooseEngine?
    private let capture = WindowCaptureManager()
    private let overlay = OverlayWindowManager()
    private let hud = HUDController()

    private(set) var activeScreen: NSScreen?
    private(set) var targetPID: pid_t = 0
    private(set) var targetWindowID: CGWindowID = 0

    private var statsTimer: Timer?
    private var settingsObserver: AnyCancellable?
    private var activationTask: Task<Void, Never>?

    /// The target window has to hold the overlay's Space for this many consecutive checks before the
    /// session is ended: a Space switch in progress looks the same as a fullscreen target for a moment.
    private var fullscreenStrikes = 0
    private static let fullscreenStrikeLimit = 3

    var onEngineError: ((MGError) -> Void)?
    var onUnexpectedEnd: ((_ reason: String, _ error: MGError?) -> Void)?
    var onTick: (() -> Void)?

    init(settings: CaptureSettings) {
        self.settings = settings

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
            Task { @MainActor in self?.onUnexpectedEnd?("capture", error) }
        }
        overlay.onDisplayChanged = { [weak self] screen, rate in
            guard let self, let engine = self.engine else { return }
            self.activeScreen = screen
            engine.updateDisplayRate(rate, colorSpace: ScreenGeometry.metalColorSpace(of: screen))
            Task { await self.capture.reconfigure(maxFPS: rate.maximum) }
            self.drawPointer(on: screen, engine: engine)
            if self.settings.showMGHUD { self.showHUD(on: screen, engine: engine) }
        }
    }

    var isActive: Bool { targetPID != 0 }
    var isEngineInitialized: Bool { engine != nil }
    var deviceName: String? { engine?.deviceName }
    var stats: PipelineStats? { engine?.stats }
    var generation: GenerationChoice? { engine?.generation }
    var presentationState: (presenting: Bool, frontmostPID: Int32?, onScreen: Bool?, reason: String?) {
        overlay.presentationState
    }

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

    /// Starts capturing and presenting `target` owned by `pid`.
    ///
    /// - Parameter reactivateTarget: when true, schedules a non-blocking 300 ms activation check
    ///   (cancelled on `endSession`) in case hiding the host window shifted focus away from the game.
    func start(target: TargetWindow, pid: pid_t, reactivateTarget: Bool,
               beforeOverlay: (() -> Void)? = nil) async throws {
        let engine = try makeEngineIfNeeded()
        guard let screen = ScreenGeometry.screen(containing: target.frame) else { throw MGError.noDisplay }
        guard ScreenGeometry.displayID(of: screen) != nil else { throw MGError.displayIDUnavailable }
        guard let displayRate = ScreenGeometry.displayRate(of: screen) else { throw MGError.refreshRateUnavailable }

        let upscaling = settings.isUpscaling
        let captureTarget = CaptureTarget(windowID: target.id, frame: target.frame,
                                          backingScale: screen.backingScaleFactor,
                                          ownerPID: pid,
                                          colorSpaceName: ScreenGeometry.captureColorSpaceName(of: screen))

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

        beforeOverlay?()

        let configuration = OverlayWindowManager.Configuration(
            screen: screen, windowFrame: target.frame, alignsPointer: settings.alignPointer,
            outputScale: upscaling ? CGFloat(settings.scaleFactor.value) : 1.0,
            fillsScreen: upscaling && settings.scaleFactor.fillsScreen, drawsPointer: !settings.vrr)
        engine.attach(to: overlay.createOverlay(configuration), displayRate: displayRate,
                      colorSpace: ScreenGeometry.metalColorSpace(of: screen))
        drawPointer(on: screen, engine: engine)

        overlay.setTarget(windowID: target.id, pid: pid)
        activeScreen = screen
        targetPID = pid
        targetWindowID = target.id
        fullscreenStrikes = 0

        if settings.showMGHUD { showHUD(on: screen, engine: engine) }
        startStatsTimer()

        if reactivateTarget {
            activationTask?.cancel()
            activationTask = Task { [weak self] in
                try? await Task.sleep(for: .milliseconds(300))
                guard !Task.isCancelled, let self, self.targetPID == pid else { return }
                NSRunningApplication(processIdentifier: pid)?.activate()
            }
        }
    }

    /// Everything is torn down in the order that leaves nothing presenting into a window that is
    /// already gone: the engine stops presenting, then the overlay goes, then the capture.
    func endSession() async {
        activationTask?.cancel()
        activationTask = nil
        statsTimer?.invalidate()
        statsTimer = nil

        engine?.detach()
        overlay.destroyOverlay()
        activeScreen = nil
        engine?.endSession()
        capture.onFrame = nil
        await capture.stopCapture()

        targetPID = 0
        targetWindowID = 0
        hud.hide()
    }

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
        guard isActive, let engine else { return }
        followContentResolution(engine: engine)
        pushHUD(engine: engine)
        if let error = engine.takeError() { onEngineError?(error) }

        // A Wine process need not be registered with LaunchServices, so the PID is what says the target
        // has exited.
        if Darwin.kill(targetPID, 0) != 0, errno == ESRCH {
            onUnexpectedEnd?("target-exited", nil)
            return
        }

        // Fullscreen gets a Space of its own, which the overlay cannot join; the target is then
        // frontmost but the overlay is nowhere to be seen. Checked a few times in a row because a
        // Space switch in progress looks the same.
        if overlay.isTargetInUnreachableSpace() {
            fullscreenStrikes += 1
            if fullscreenStrikes >= Self.fullscreenStrikeLimit {
                fullscreenStrikes = 0
                onUnexpectedEnd?("fullscreen", .targetEnteredFullscreen)
                return
            }
        } else {
            fullscreenStrikes = 0
        }
        onTick?()
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
