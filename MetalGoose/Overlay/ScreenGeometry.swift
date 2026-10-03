import Foundation
import AppKit
import CoreGraphics

/// CoreGraphics window bounds and AppKit screen frames disagree about which way
/// Y runs: `kCGWindowBounds` measures down from the top of the origin display,
/// `NSScreen.frame` measures up from its bottom.
///
/// Three call sites derived that flip independently from the same window rect —
/// the capture manager to pick a backing scale, the UI to pick an output screen,
/// and the overlay to place itself. Identical input could therefore resolve to
/// three different rectangles the moment one of them was corrected and the
/// others were not: the overlay lands off the window, at the wrong pixel
/// density, and nothing crashes to say so.
enum ScreenGeometry {

    /// The display the global coordinate origin belongs to. `NSScreen.screens`
    /// is an ordered list, not an anchored one, so the origin display is
    /// identified by sitting at the origin rather than by coming first.
    static var originScreen: NSScreen? {
        NSScreen.screens.first { $0.frame.origin == .zero } ?? NSScreen.screens.first
    }

    /// Converts a CoreGraphics rect into AppKit's coordinate space.
    static func cocoaFrame(from cgFrame: CGRect) -> CGRect {
        let originHeight = originScreen?.frame.height ?? 0
        return CGRect(x: cgFrame.origin.x,
                      y: originHeight - cgFrame.maxY,
                      width: cgFrame.width,
                      height: cgFrame.height)
    }

    /// The screen a CoreGraphics window rect sits on: the one holding most of it,
    /// as AppKit decides a window's screen, so a window that reaches a little onto
    /// a second display is not drawn on that one at its density. `NSScreen.main` is
    /// the fallback rather than an error: a window that intersects no screen has
    /// nowhere to be captured from or drawn to, and the key screen is the only
    /// neutral answer available.
    static func screen(containing cgFrame: CGRect) -> NSScreen? {
        let cocoa = cocoaFrame(from: cgFrame)
        func area(_ screen: NSScreen) -> CGFloat {
            let shared = screen.frame.intersection(cocoa)
            return shared.isNull ? 0 : shared.width * shared.height
        }
        guard let best = NSScreen.screens.max(by: { area($0) < area($1) }), area(best) > 0 else { return NSScreen.main }
        return best
    }

    /// Every display, in CoreGraphics coordinates: where the cursor can be.
    static var displayBounds: [CGRect] {
        NSScreen.screens.compactMap {
            ($0.deviceDescription[NSDeviceDescriptionKey("NSScreenNumber")] as? CGDirectDisplayID).map(CGDisplayBounds)
        }
    }
}
