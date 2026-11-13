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
    /// The target window's size in pixels before render scale was applied, or its content's where that is drawn at half of
    /// them. Frame generation works at this size, so that lowering render scale cannot degrade it.
    let nativePixelSize: CGSize
    /// How the frame's content, drawn at half the window's pixels, was magnified to them; the pipeline takes it back to the
    /// content's own size first (`ContentResolution`). Nil for a frame that is the content as it is.
    var magnification: ContentResolution.Magnification?
}

/// The window being captured, resolved by the caller. Screens are an AppKit matter, so the
/// pixel density is decided on the main thread rather than here.
struct CaptureTarget: Sendable, Equatable {
    let windowID: CGWindowID
    /// The window's frame in CoreGraphics coordinates: points, origin at the top-left of the primary display.
    let frame: CGRect
    /// Pixels per point on the display the window sits on.
    let backingScale: CGFloat
    /// An explicit caller binds a window to its owning process at ScreenCaptureKit selection too.
    var ownerPID: Int32? = nil

    var pixelSize: CGSize { CGSize(width: frame.width * backingScale, height: frame.height * backingScale) }
}
