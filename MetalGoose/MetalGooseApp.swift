import SwiftUI

@main
struct MetalGooseApp: App {
    var body: some Scene {
        WindowGroup {
            if AXIsProcessTrusted() && CGPreflightScreenCaptureAccess() {
                ContentView()
            } else {
                PermissionsView()
            }
        }
        .windowStyle(.hiddenTitleBar)
        .defaultSize(width: 900, height: 600)
    }
}
