// Goose HUB integration for MetalGoose. Distributed under GPL-3.0; see LICENSE.
import AppKit
import Carbon.HIToolbox
import CoreGraphics
import Foundation
@preconcurrency import Metal

/// MetalGoose's `ScalingController` with the hub in place of its window: the hub names the game to
/// scale instead of a countdown taking the frontmost app, and errors go back to the hub instead of
/// into an alert. The rest is MetalGoose's own (`ScalingSession`): its settings, its HUD, its two
/// shortcuts and the order a session is set up and torn down in.
@MainActor
final class BackendSession {

    enum Phase: String {
        case idle, starting, active, stopping
    }

    let output: BackendOutput
    let permissions = PermissionManager()
    private let settings = CaptureSettings.shared
    private let session = ScalingSession(settings: CaptureSettings.shared)

    private(set) var phase: Phase = .idle
    private var isTransitioning = false

    private var lastHotkeyTime: CFTimeInterval = 0
    private var lastCursorHotkeyTime: CFTimeInterval = 0

    /// The hub that started this backend. Its window is never a target, as MetalGoose's own never is.
    private let hubPID = getppid()

    init(output: BackendOutput) {
        self.output = output

        session.onEngineError = { [weak self] error in
            self?.output.send(.failure(error))
        }
        session.onUnexpectedEnd = { [weak self] reason, error in
            guard let self, self.isActive, !self.isTransitioning else { return }
            self.end(reason: reason, error: error)
        }
        session.onTick = { [weak self] in
            guard let self, self.isActive, !self.isTransitioning else { return }
            self.output.send(BackendResponse(event: "stats", status: self.status()))
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
        let presentation = session.presentationState
        let metrics = isActive ? session.stats.flatMap { stats in
            session.generation.map { BackendMetrics(stats, generation: $0) }
        } : nil
        return BackendStatus(screenRecordingGranted: permissions.screenRecordingGranted,
                             accessibilityGranted: permissions.accessibilityGranted,
                             shaderLibrary: GPUContext.shaderLibraryURL?.path,
                             device: session.deviceName ?? MTLCreateSystemDefaultDevice()?.name,
                             engineInitialized: session.isEngineInitialized, phase: phase.rawValue,
                             presenting: presentation.presenting, frontmostPID: presentation.frontmostPID,
                             targetOnScreen: presentation.onScreen, suspensionReason: presentation.reason,
                             pid: session.targetPID == 0 ? nil : session.targetPID,
                             windowID: session.targetWindowID == 0 ? nil : session.targetWindowID,
                             configuration: BackendConfiguration(settings),
                             metrics: metrics)
    }

    func configure(_ configuration: BackendConfiguration) throws {
        try configuration.apply(to: settings, sessionRunning: phase != .idle)
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

        try await session.start(target: target, pid: pid, reactivateTarget: true)
        phase = .active
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
        phase = .stopping
        await session.endSession()
        phase = .idle
    }

    private func end(reason: String, error: MGError?) {
        isTransitioning = true
        Task {
            defer { isTransitioning = false }
            phase = .stopping
            await session.endSession()
            phase = .idle
            if let error { output.send(.failure(error)) }
            output.send(BackendResponse(event: "stopped", reason: reason, status: status()))
        }
    }

    /// This must also run synchronously on process termination to restore a hidden system cursor.
    func restorePointer() { MouseConstraintManager.shared.stopConstraining() }
}
