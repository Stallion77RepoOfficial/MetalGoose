import SwiftUI

@main
struct ScreenApp: App {
    var body: some Scene {
        WindowGroup {
            // Ana görünümü başlat
            RootView()
        }
        .windowStyle(.hiddenTitleBar) // Başlık çubuğunu gizle
        .windowResizability(.contentSize) // Pencere içeriğe göre boyutlansın
    }
}
