import SwiftUI
@main
struct MetalGooseApp: App {
    init() {}
    var body: some Scene {
        WindowGroup {
            if #available(macOS 26.0, *) {
                ContentView()
            } else {
                Text("MetalGoose requires macOS 26.0 or newer")
                    .padding()
            }
        }
        .windowStyle(.hiddenTitleBar)
        .defaultSize(width: 900, height: 600)
    }
}