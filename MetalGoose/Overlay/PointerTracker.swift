import CoreGraphics
import Foundation

/// Where the pointer is held and whether there is one to draw, as the mouse's events say.
///
/// An app that reads the mouse as motion alone — a game that looks around with it — keeps the cursor still: it cuts the
/// cursor off from the mouse (`CGAssociateMouseAndMouseCursorPosition`), or puts it back at a point after every event. The
/// events keep carrying the motion, and say the cursor has not gone anywhere. There is no pointer to show then, and one
/// drawn would sit over the middle of the picture, where the app has put the cursor away.
struct PointerTracker: Sendable {

    /// The window the cursor is held in, and the displays it can be on.
    var mapping: PointerMapping

    /// Where the cursor is, as the last event held it.
    private(set) var pointer: CGPoint = .zero

    /// The app has taken the mouse for itself: its movements have left the cursor where it was, a number of times in a row.
    private(set) var isGrabbed = false

    private var stillEvents = 0

    /// Movements in a row that leave the cursor where it was, before the mouse is taken to be the app's own. One can be a
    /// movement too small for the cursor to show; three in a row are not.
    static let stillEventsToGrab = 3

    init(mapping: PointerMapping) {
        self.mapping = mapping
    }

    /// Takes the cursor to be at `point`, which is where it is when the hold begins or goes on after a break, and holds it in
    /// the window. Nil where the window is on no display and there is nowhere to hold it.
    mutating func place(_ point: CGPoint) -> CGPoint? {
        stillEvents = 0
        isGrabbed = false
        guard let held = mapping.constrain(point) else { return nil }
        pointer = held
        return held
    }

    /// A mouse event at `incoming`. `moves` is set where it carries movement of the mouse: a click does not, and says nothing
    /// about whether the cursor follows the mouse. Returns where the cursor is to be held, nil where there is nowhere.
    ///
    /// A cursor held at the border of the window gets an event outside it each time the mouse pushes on: the system moves it
    /// on from where it was put back, so that is a movement of the cursor, not a still one. Where the border is the edge of
    /// a display the system keeps the cursor there itself, and a mouse pushing on from it leaves it still: that is not the
    /// app taking the mouse either.
    mutating func event(at incoming: CGPoint, moves: Bool) -> CGPoint? {
        guard let held = mapping.constrain(incoming) else { return nil }
        if moves {
            if incoming == pointer, !mapping.isAtBorder(pointer) {
                stillEvents += 1
                if stillEvents >= Self.stillEventsToGrab { isGrabbed = true }
            } else {
                stillEvents = 0
                isGrabbed = false
            }
        }
        pointer = held
        return held
    }
}
