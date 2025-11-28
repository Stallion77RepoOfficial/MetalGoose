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
        .defaultSize(width: 450, height: 500)
        .windowResizability(.contentSize)
        .windowStyle(.hiddenTitleBar)
    }
}
