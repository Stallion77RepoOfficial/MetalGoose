import CoreGraphics
import Foundation

/// Where the pointer may be while the overlay magnifies a window, and where on the overlay it is drawn.
///
/// The system cursor stays in the captured window and moves there exactly as it always does, at the system's own speed
/// and acceleration. Every app then reads one and the same position however it reads it: the location an event carries,
/// the deltas it adds up from the events, or the cursor it polls. The overlay draws the pointer where that position
/// appears in the magnified image, so what the user points at is what the window is clicked at.
///
/// This replaced a pointer of MetalGoose's own that crossed the whole display at the system's speed and was mapped down
/// into the window, with every event's location rewritten and the cursor warped after it. That agreed with an app that
/// read the event's location or polled the cursor, and with nothing else: the events still carried the deltas of the
/// undivided motion, so an app that added those up (a pointer the game draws itself, Wine while the cursor is on an edge of
/// where it may go) moved further than the pointer that was drawn, by the overlay's stretch — 18% across and 31% down for a
/// 1280x748 window on a 1512x982 display in Fullscreen — and its clicks drifted off the drawn pointer, most of all up and
/// down. Modelled, that was 85 points on average at a click and up to 237; it is now none for all
/// three ways of reading the mouse. A window that reached past the display was a second way to miss: the pointer could be
/// drawn over its part off the display, where the cursor could not follow. The cost of following the system is that the
/// pointer crosses the overlay in the time it crosses the window, which is how the game moves it at that size anyway, and
/// at 1.0x is exactly the system's pointer.
///
/// The only thing done to the cursor is to keep it in the window, on a display: outside the window a click would go to
/// whatever is under the overlay there, and outside the displays the system pins the cursor where no click could land.
struct PointerMapping: Equatable, Sendable {

    /// The captured window, in CoreGraphics coordinates: points, origin at the top-left of the primary display.
    var window: CGRect
    /// The displays the cursor can be on, in the same coordinates.
    var displays: [CGRect]

    /// A cursor held at the far edge of the window is held this far inside it, so that an app that rounds the position
    /// rather than truncating it does not land on the first point past the window. On a 2x display this is the window's
    /// last pixel.
    static let edgeInset: CGFloat = 0.5

    /// Whether the overlay stands over the window point for point: 1.0x on a window the display's edge does not hold back.
    /// The cursor is where the user sees it then, and there is nothing to map. Half a point either way is the same place.
    static func isIdentity(window: CGRect, overlay: CGRect) -> Bool {
        abs(window.minX - overlay.minX) <= edgeInset && abs(window.minY - overlay.minY) <= edgeInset
            && abs(window.maxX - overlay.maxX) <= edgeInset && abs(window.maxY - overlay.maxY) <= edgeInset
    }

    /// The parts of the window the cursor can be on: the window on each display it overlaps.
    var reachable: [CGRect] {
        displays.map { window.intersection($0) }.filter { !$0.isNull && $0.width > Self.edgeInset && $0.height > Self.edgeInset }
    }

    /// Where a cursor at `point` is held: `point` itself where it is in the window and on a display, otherwise the nearest
    /// point that is. Nil where the window is on no display at all, and there is nowhere to hold it.
    func constrain(_ point: CGPoint) -> CGPoint? {
        let regions = reachable
        guard !regions.isEmpty else { return nil }
        // Half-open, as the window server tests it: a window spans [minX, maxX).
        if regions.contains(where: { point.x >= $0.minX && point.x < $0.maxX && point.y >= $0.minY && point.y < $0.maxY }) {
            return point
        }
        var best = point
        var bestDistance = CGFloat.infinity
        for region in regions {
            let held = CGPoint(x: min(max(point.x, region.minX), region.maxX - Self.edgeInset),
                               y: min(max(point.y, region.minY), region.maxY - Self.edgeInset))
            let distance = (held.x - point.x) * (held.x - point.x) + (held.y - point.y) * (held.y - point.y)
            if distance < bestDistance {
                best = held
                bestDistance = distance
            }
        }
        return best
    }

    /// Whether `point` is within a point of the border of the part of the window the cursor can be on. A mouse that pushes on
    /// from there can leave the cursor still because there is nowhere further for it to go.
    func isAtBorder(_ point: CGPoint) -> Bool {
        let reach = 2 * Self.edgeInset
        return reachable.contains {
            point.x <= $0.minX + reach || point.x >= $0.maxX - reach || point.y <= $0.minY + reach || point.y >= $0.maxY - reach
        }
    }

    /// Where `point` is as a fraction of the window, measured from its top-left corner and kept to it: what the overlay
    /// draws the pointer at, scaled to its own size. Nil for a window with no area.
    func fraction(of point: CGPoint) -> CGPoint? {
        guard window.width > 0, window.height > 0 else { return nil }
        return CGPoint(x: min(max((point.x - window.minX) / window.width, 0), 1),
                       y: min(max((point.y - window.minY) / window.height, 0), 1))
    }
}
