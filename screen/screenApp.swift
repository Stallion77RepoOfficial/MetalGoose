import SwiftUI

@main
struct ScreenApp: App {
    var body: some Scene {
        WindowGroup {
            if AXIsProcessTrusted() && CGPreflightScreenCaptureAccess() {
                ContentView()
            } else {
                PermissionsView()
            }
        }
        .windowStyle(.hiddenTitleBar)
    }
}
