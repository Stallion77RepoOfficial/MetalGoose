import SwiftUI

@main
struct MetalGooseApp: App {
    @NSApplicationDelegateAdaptor(MetalGooseAppDelegate.self) private var appDelegate
    @StateObject private var settings = CaptureSettings.shared
    @StateObject private var permissions: PermissionManager
    /// Owned here rather than by the window: the session, the overlay and the global hotkeys all
    /// have to outlive it. Closing the window with Cmd+W must not tear down a running capture.
    @StateObject private var controller: ScalingController

    init() {
        let permissions = PermissionManager()
        _permissions = StateObject(wrappedValue: permissions)
        _controller = StateObject(wrappedValue: ScalingController(settings: .shared, permissions: permissions))
    }

    var body: some Scene {
        // One session and one alert state need one presentation owner. WindowGroup
        // creates additional hosts for the same alerts, including restored windows.
        Window("MetalGoose", id: "main") {
            ContentView(settings: settings, permissions: permissions, controller: controller)
        }
        .windowStyle(.hiddenTitleBar)
        .defaultSize(width: 900, height: 600)
    }
}

@MainActor
final class MetalGooseAppDelegate: NSObject, NSApplicationDelegate {
    // A singleton Window otherwise quits the app when closed. The capture and global
    // shortcuts must keep running after Cmd+W, just as they did with WindowGroup.
    func applicationShouldTerminateAfterLastWindowClosed(_ sender: NSApplication) -> Bool { false }

    /// One copy of MetalGoose at a time. A copy at another path — a build beside the installed app — registers the same
    /// global shortcuts, and the system hands every press to both: measured with two copies, each Command-Shift-T toggled
    /// both, so with one scaling and the other not, every press stopped one and started the other on the same window, and
    /// scaling could not be turned off from the captured app. The copy launched last is the one asked for, so the others
    /// are asked to quit, and made to if they have not within a few seconds.
    func applicationWillFinishLaunching(_ notification: Notification) {
        guard let identifier = Bundle.main.bundleIdentifier else { return }
        let others = NSRunningApplication.runningApplications(withBundleIdentifier: identifier)
            .filter { $0.processIdentifier != ProcessInfo.processInfo.processIdentifier }
        guard !others.isEmpty else { return }
        for other in others { other.terminate() }
        DispatchQueue.main.asyncAfter(deadline: .now() + Self.quitGrace) {
            for other in others where !other.isTerminated { other.forceTerminate() }
        }
    }

    private static let quitGrace: TimeInterval = 5

    /// The Neural Engine's models — the ten shapes VideoProcessing ships and the sizes used before — are built while nothing is
    /// being scaled yet, where the system does not have them: a session that waits for one begins without its images
    /// (`NeuralModelCache`).
    func applicationDidFinishLaunching(_ notification: Notification) {
        NeuralModelCache.buildAtLaunch()
    }
}
