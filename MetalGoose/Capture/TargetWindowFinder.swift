import AppKit
import CoreGraphics

/// A window on screen, as the window server describes it.
struct TargetWindow: Sendable {
    let id: CGWindowID
    /// In CoreGraphics coordinates: points, origin at the top-left of the primary display.
    let frame: CGRect
}

enum TargetWindowFinder {

    /// Anything smaller is a tooltip, a badge or a helper rather than something worth magnifying.
    private static let minimumDimension: CGFloat = 64

    /// The window of an app that the user is looking at: the topmost ordinary window.
    ///
    /// The window list is ordered front to back, and an app owns more windows than the one on
    /// screen — floating palettes, popups, invisible helpers. Only layer 0 holds the app's own
    /// windows, so everything above it is skipped, as is anything nearly transparent or too small
    /// to be the thing being captured.
    static func topWindow(ofProcess pid: pid_t) -> TargetWindow? {
        let options: CGWindowListOption = [.optionOnScreenOnly, .excludeDesktopElements]
        guard let list = CGWindowListCopyWindowInfo(options, kCGNullWindowID) as? [[String: Any]] else { return nil }

        for info in list {
            guard (info[kCGWindowOwnerPID as String] as? Int32) == pid,
                  (info[kCGWindowLayer as String] as? Int) == 0,
                  ((info[kCGWindowAlpha as String] as? Double) ?? 1) > 0.01,
                  let id = info[kCGWindowNumber as String] as? CGWindowID,
                  let bounds = info[kCGWindowBounds as String] as? [String: CGFloat],
                  let x = bounds["X"], let y = bounds["Y"],
                  let width = bounds["Width"], let height = bounds["Height"],
                  width >= minimumDimension, height >= minimumDimension else { continue }
            return TargetWindow(id: id, frame: CGRect(x: x, y: y, width: width, height: height))
        }
        return nil
    }
}
