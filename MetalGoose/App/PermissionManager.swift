import AppKit
import ApplicationServices
import CoreGraphics

/// Whether MetalGoose may capture the screen and remap the mouse, and the ways of getting from
/// "no" to "yes".
@MainActor
final class PermissionManager: ObservableObject {

    @Published private(set) var accessibilityGranted = AXIsProcessTrusted()
    @Published private(set) var screenRecordingGranted = CGPreflightScreenCaptureAccess()

    private var timer: Timer?

    init() {
        // Permissions are granted in System Settings, so the likeliest moment for one to have changed
        // is when the user comes back to this app.
        NotificationCenter.default.addObserver(forName: NSApplication.didBecomeActiveNotification, object: nil,
                                               queue: .main) { [weak self] _ in
            MainActor.assumeIsolated {
                guard let self else { return }
                self.refresh()
                if !(self.accessibilityGranted && self.screenRecordingGranted) { self.startMonitoring() }
            }
        }
    }

    // MARK: - State

    /// Re-reads both permissions. Each check is a round trip to tccd, so they run on demand and on a
    /// timer only while something is missing.
    func refresh() {
        accessibilityGranted = AXIsProcessTrusted()
        screenRecordingGranted = CGPreflightScreenCaptureAccess()
    }

    /// Polls while a permission is missing: it is granted in another app, which does not notify this one.
    func startMonitoring() {
        timer?.invalidate()
        refresh()
        timer = Timer.scheduledTimer(withTimeInterval: 1.0, repeats: true) { [weak self] _ in
            MainActor.assumeIsolated {
                guard let self else { return }
                self.refresh()
                if self.accessibilityGranted && self.screenRecordingGranted { self.timer?.invalidate() }
            }
        }
    }

    func stopMonitoring() {
        timer?.invalidate()
        timer = nil
    }

    // MARK: - Requesting

    func requestAccessibility() {
        guard !AXIsProcessTrusted() else { return }
        // Raw value of kAXTrustedCheckOptionPrompt: the imported constant is a global var, which
        // Swift 6 concurrency checking rejects.
        _ = AXIsProcessTrustedWithOptions(["AXTrustedCheckOptionPrompt": true] as CFDictionary)
        startMonitoring()
    }

    func requestScreenRecording() {
        guard !CGPreflightScreenCaptureAccess() else {
            refresh()
            return
        }
        _ = CGRequestScreenCaptureAccess()
        startMonitoring()
    }

    func openScreenRecordingSettings() {
        open("x-apple.systempreferences:com.apple.preference.security?Privacy_ScreenCapture")
        startMonitoring()
    }

    func openAccessibilitySettings() {
        open("x-apple.systempreferences:com.apple.preference.security?Privacy_Accessibility")
        startMonitoring()
    }

    private func open(_ url: String) {
        if let url = URL(string: url) { NSWorkspace.shared.open(url) }
    }
}
