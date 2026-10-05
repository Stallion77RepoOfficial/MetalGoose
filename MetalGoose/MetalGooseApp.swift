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
}
