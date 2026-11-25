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

    private let permissions: PermissionManager
    private let session: ScalingSession

    private var engineCreationFailed = false
    private var countdownTimer: Timer?
    private var updateObserver: AnyCancellable?

    private var lastHotkeyTime: CFTimeInterval = 0
    private var lastCursorHotkeyTime: CFTimeInterval = 0

    /// Set by the view once it has a window, so the controller can hide and restore it without
    /// guessing which window is the right one from its title.
    weak var mainWindow: NSWindow?

    /// The main window was put away when a session started, so that it is not left over the captured app; it comes back
    /// when the user comes to MetalGoose itself (`showMainWindowIfPutAway`). A window the user closed stays closed.
    private var mainWindowPutAway = false

    init(settings: CaptureSettings, permissions: PermissionManager) {
        self.permissions = permissions
        self.session = ScalingSession(settings: settings)

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

        // Command-Tab to MetalGoose, or a click on its Dock icon, during a session found no window to stop it from: the
        // one with Stop Scaling had been put away.
        NotificationCenter.default.addObserver(forName: NSApplication.didBecomeActiveNotification, object: nil,
                                               queue: .main) { [weak self] _ in
            MainActor.assumeIsolated { self?.showMainWindowIfPutAway() }
        }

        session.onEngineError = { [weak self] error in
            guard let self, self.alertMessage == nil else { return }
            self.alertMessage = error.message
        }
        session.onUnexpectedEnd = { [weak self] reason, error in
            self?.handleUnexpectedEnd(reason: reason, error: error)
        }
    }

    var isActive: Bool { phase == .active }

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

        let willPutAway = mainWindow?.isVisible == true
        do {
            try await session.start(target: target, pid: app.processIdentifier, reactivateTarget: willPutAway) { [weak self] in
                guard let self, let mainWindow = self.mainWindow, mainWindow.isVisible else { return }
                mainWindow.orderOut(nil)
                self.mainWindowPutAway = true
            }
            phase = .active
        } catch let error as MGError {
            if error.code.hasPrefix("MG-ENG-") && !session.isEngineInitialized {
                if !engineCreationFailed { fail(error) }
                engineCreationFailed = true
            } else {
                fail(error)
            }
        } catch {
            fail(MGError("MG-CAP-002", error.localizedDescription))
        }
    }

    // MARK: - Stopping

    func stop() {
        MouseConstraintManager.shared.stopConstraining()
        cancelCountdown()
        guard isActive, !isTransitioning else { return }
        isTransitioning = true
        Task {
            defer { isTransitioning = false }
            await session.endSession()
            phase = .idle
        }
    }

    private func handleUnexpectedEnd(reason: String, error: MGError?) {
        guard isActive, !isTransitioning else { return }
        isTransitioning = true
        Task {
            defer { isTransitioning = false }
            await session.endSession()
            phase = .idle
            if reason == "fullscreen" { bringToFront() }
            if let error { alertMessage = error.message }
        }
    }

    func bringToFront() {
        mainWindowPutAway = false
        mainWindow?.makeKeyAndOrderFront(nil)
        NSApp.activate(ignoringOtherApps: true)
    }

    private func showMainWindowIfPutAway() {
        guard mainWindowPutAway, let mainWindow else { return }
        mainWindowPutAway = false
        mainWindow.makeKeyAndOrderFront(nil)
    }
}
