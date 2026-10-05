import Foundation
import CoreGraphics
import CoreVideo
import IOSurface

/// One captured frame on its way into the engine.
struct CapturedFrame: @unchecked Sendable {
    let pixelBuffer: CVPixelBuffer
    let surface: IOSurfaceRef
    /// ScreenCaptureKit's presentation time of the frame, on the host clock.
    let captureTime: CFTimeInterval
    /// The picture was replaced rather than moved.
    let isSceneCut: Bool
    /// The target window's size in pixels before render scale was applied. Frame
    /// generation works at this size, so that lowering render scale cannot degrade it.
    let nativePixelSize: CGSize
}

/// The window being captured, resolved by the caller. Screens are an AppKit matter, so the
/// pixel density is decided on the main thread rather than here.
struct CaptureTarget: Sendable, Equatable {
    let windowID: CGWindowID
    /// The window's frame in CoreGraphics coordinates: points, origin at the top-left of the primary display.
    let frame: CGRect
    /// Pixels per point on the display the window sits on.
    let backingScale: CGFloat

    var pixelSize: CGSize { CGSize(width: frame.width * backingScale, height: frame.height * backingScale) }
}
