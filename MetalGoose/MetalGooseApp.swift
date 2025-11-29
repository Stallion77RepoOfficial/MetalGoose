import SwiftUI

@main
struct MetalGooseApp: App {
    var body: some Scene {
        WindowGroup {
            if AXIsProcessTrusted() && CGPreflightScreenCaptureAccess() {
                if #available(macOS 26.0, *) {
                    ContentView()
                } else {
                    Text("MetalGoose requires macOS 26.0 or newer")
                        .padding()
                }
            } else {
                PermissionsView()
            }
        }
        .windowStyle(.hiddenTitleBar)
        .defaultSize(width: 900, height: 600)
    }
}
