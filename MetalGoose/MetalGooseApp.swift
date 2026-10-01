import SwiftUI

@main
struct MetalGooseApp: App {
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
        WindowGroup {
            ContentView(settings: settings, permissions: permissions, controller: controller)
        }
        .windowStyle(.hiddenTitleBar)
        .defaultSize(width: 900, height: 600)
    }
}
