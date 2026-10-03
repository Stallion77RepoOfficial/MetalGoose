// swift-tools-version: 6.0
//
// The engine's decisions that need no Metal, no window server and no capture — the frame schedule, the engine choice,
// the sizes the Neural Engine takes, the stability blend's rule, where the pointer is held — built on their own so that
// `swift test --package-path LogicTests` checks them in seconds, on macOS or Linux. The sources are the app's own, linked
// in; the app is built from MetalGoose.xcodeproj, which this does not touch.

import PackageDescription

let package = Package(
    name: "MetalGooseLogic",
    platforms: [.macOS(.v14)],
    targets: [
        .target(name: "MetalGooseLogic", path: "Sources/MetalGooseLogic"),
        .testTarget(name: "MetalGooseLogicTests", dependencies: ["MetalGooseLogic"], path: "Tests/MetalGooseLogicTests"),
    ]
)
