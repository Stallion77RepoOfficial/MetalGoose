// swift-tools-version: 6.0
import PackageDescription

// The app is built by MetalGoose.xcodeproj. This package exists to run the pure logic — frame
// scheduling, filters, version ordering, where the pointer is held — under `swift test` without a GPU, a window or a capture
// permission: the files below are the ones with no dependency on Metal, AppKit or the engine.
let package = Package(
    name: "MetalGoose",
    // The pure-logic test package supports older CI hosts; the app still requires 27.
    platforms: [.macOS(.v15)],
    targets: [
        .target(
            name: "MetalGooseCore",
            path: "MetalGoose",
            exclude: ["App", "Assets.xcassets", "Capture", "Engine", "HUD", "Shaders", "Views",
                      "Localizable.xcstrings", "InfoPlist.xcstrings", "MetalGoose-Bridging-Header.h", "MetalGoose.entitlements", "MetalGooseApp.swift",
                      "Settings/CaptureSettings.swift", "Overlay/MouseConstraintManager.swift", "Overlay/OverlayView.swift",
                      "Overlay/OverlayWindowManager.swift", "Overlay/ScreenGeometry.swift"],
            sources: ["Core", "Settings/Options.swift", "Overlay/PointerMapping.swift", "Overlay/PointerTracker.swift"]
        ),
        .testTarget(
            name: "MetalGooseCoreTests",
            dependencies: ["MetalGooseCore"],
            path: "Tests/MetalGooseCoreTests"
        ),
    ]
)
