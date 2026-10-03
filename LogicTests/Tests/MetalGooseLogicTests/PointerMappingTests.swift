import Foundation
import XCTest
@testable import MetalGooseLogic

final class PointerMappingTests: XCTestCase {

    private let display = CGRect(x: 0, y: 0, width: 1512, height: 982)
    private let window = CGRect(x: 116, y: 117, width: 1280, height: 748)

    // MARK: - Holding the cursor

    func testAPointInTheWindowIsLeftWhereItIs() {
        let mapping = PointerMapping(window: window, displays: [display])
        for point in [CGPoint(x: 116, y: 117), CGPoint(x: 700.25, y: 400.5), CGPoint(x: 1395.9, y: 864.9)] {
            XCTAssertEqual(mapping.constrain(point), point)
        }
    }

    func testAPointOutsideIsHeldAtTheNearestEdgeInside() {
        let mapping = PointerMapping(window: window, displays: [display])
        XCTAssertEqual(mapping.constrain(CGPoint(x: 50, y: 400)), CGPoint(x: 116, y: 400))
        XCTAssertEqual(mapping.constrain(CGPoint(x: 700, y: 10)), CGPoint(x: 700, y: 117))
        // The far edges are open: the window ends before maxX, so the cursor is held just inside.
        XCTAssertEqual(mapping.constrain(CGPoint(x: 1396, y: 400)), CGPoint(x: 1396 - PointerMapping.edgeInset, y: 400))
        XCTAssertEqual(mapping.constrain(CGPoint(x: 2000, y: 2000)),
                       CGPoint(x: 1396 - PointerMapping.edgeInset, y: 865 - PointerMapping.edgeInset))
    }

    func testAWindowPastTheDisplayIsHeldToWhatIsOnIt() {
        // A window the size of the display, pushed down under the menu bar: its last 65 points are off the display, where the
        // system would pin the cursor and no click could land.
        let tall = CGRect(x: 0, y: 37, width: 1512, height: 1010)
        let mapping = PointerMapping(window: tall, displays: [display])
        XCTAssertEqual(mapping.constrain(CGPoint(x: 400, y: 1000)), CGPoint(x: 400, y: 982 - PointerMapping.edgeInset))
        XCTAssertEqual(mapping.constrain(CGPoint(x: 400, y: 20)), CGPoint(x: 400, y: 37))
    }

    func testAWindowAcrossTwoDisplaysCanBeReachedOnBoth() {
        let right = CGRect(x: 1512, y: 0, width: 1920, height: 1080)
        let spanning = CGRect(x: 1200, y: 100, width: 800, height: 600)
        let mapping = PointerMapping(window: spanning, displays: [display, right])
        XCTAssertEqual(mapping.constrain(CGPoint(x: 1300, y: 300)), CGPoint(x: 1300, y: 300))
        XCTAssertEqual(mapping.constrain(CGPoint(x: 1800, y: 300)), CGPoint(x: 1800, y: 300))
        // Below the shorter display there is no window to be on: the cursor is held at the nearest point that is.
        let tallSpanning = CGRect(x: 1200, y: 100, width: 800, height: 950)
        let held = PointerMapping(window: tallSpanning, displays: [display, right]).constrain(CGPoint(x: 1300, y: 1000))
        XCTAssertEqual(held, CGPoint(x: 1300, y: 982 - PointerMapping.edgeInset))
        XCTAssertEqual(PointerMapping(window: tallSpanning, displays: [display, right]).constrain(CGPoint(x: 1600, y: 1040)),
                       CGPoint(x: 1600, y: 1040))
    }

    func testAWindowOnNoDisplayHoldsNothing() {
        let mapping = PointerMapping(window: CGRect(x: 5000, y: 5000, width: 100, height: 100), displays: [display])
        XCTAssertNil(mapping.constrain(CGPoint(x: 10, y: 10)))
    }

    // MARK: - Drawing the pointer

    func testTheFractionRunsFromTheTopLeftCorner() {
        let mapping = PointerMapping(window: window, displays: [display])
        XCTAssertEqual(mapping.fraction(of: CGPoint(x: 116, y: 117)), CGPoint(x: 0, y: 0))
        XCTAssertEqual(mapping.fraction(of: CGPoint(x: 756, y: 491)), CGPoint(x: 0.5, y: 0.5))
        XCTAssertEqual(mapping.fraction(of: CGPoint(x: 1396, y: 865)), CGPoint(x: 1, y: 1))
        XCTAssertEqual(mapping.fraction(of: CGPoint(x: 0, y: 2000)), CGPoint(x: 0, y: 1))
        XCTAssertNil(PointerMapping(window: .zero, displays: [display]).fraction(of: .zero))
    }

    // MARK: - Where a click lands

    /// A model of the window server and of the three ways an app reads the mouse, driven by a hand that moves at every speed
    /// from a crawl to a flick and clicks every so often. Where each app thinks a click is, shown on the overlay, must be
    /// where the overlay draws the pointer, to within a point:
    ///
    /// - the event's location (AppKit's `locationInWindow`, SDL, GLFW, Unity; Wine away from an edge);
    /// - the cursor polled after the event (`NSEvent.mouseLocation`, Wine's `GetCursorPos`);
    /// - the event's deltas added up and held to the window (a pointer the game draws itself; Wine at an edge).
    ///
    /// The window server moves the cursor by each event's delta, pinned to the displays, and stamps the event with it; the
    /// tap holds it in the window as `MouseConstraintManager` does. The pointer MetalGoose used to keep agreed with the
    /// first two and drifted from the third by the magnification: 85 points on average at a click, and up to 237, for this
    /// window in Fullscreen.
    func testAClickLandsWhereThePointerIsDrawnHoweverTheAppReadsTheMouse() {
        let scenarios: [(window: CGRect, overlay: CGRect)] = [
            (window, display),                                                       // Fullscreen
            (CGRect(x: 276, y: 207, width: 960, height: 568), display),              // a smaller window in Fullscreen
            (window, window),                                                         // 1.0x
            (CGRect(x: 436, y: 297, width: 640, height: 388),
             CGRect(x: 116, y: 103, width: 1280, height: 776)),                       // 2.0x
        ]
        for scenario in scenarios {
            let worst = Self.worstClickError(window: scenario.window, overlay: scenario.overlay, display: display)
            XCTAssertLessThan(worst.location, 1, "\(scenario)")
            XCTAssertLessThan(worst.polled, 1, "\(scenario)")
            XCTAssertLessThan(worst.deltas, 1, "\(scenario)")
        }
    }

    private struct Worst {
        var location = 0.0
        var polled = 0.0
        var deltas = 0.0
    }

    private static func worstClickError(window: CGRect, overlay: CGRect, display: CGRect) -> Worst {
        var random = SplitMix(seed: 7)
        let mapping = PointerMapping(window: window, displays: [display])
        func pinned(_ p: CGPoint, _ r: CGRect, inset: CGFloat) -> CGPoint {
            CGPoint(x: min(max(p.x, r.minX), r.maxX - inset), y: min(max(p.y, r.minY), r.maxY - inset))
        }
        func onOverlay(_ p: CGPoint) -> CGPoint {
            CGPoint(x: overlay.minX + (p.x - window.minX) / window.width * overlay.width,
                    y: overlay.minY + (p.y - window.minY) / window.height * overlay.height)
        }
        func distance(_ a: CGPoint, _ b: CGPoint) -> Double { Double(hypot(a.x - b.x, a.y - b.y)) }

        var cursor = CGPoint(x: window.midX, y: window.midY)
        var pointer = cursor
        var summed = cursor
        var worst = Worst()
        for event in 0..<20_000 {
            if event % 37 == 36 {
                // A click: the window server stamps it with the cursor; the tap holds it in the window.
                let location = mapping.constrain(cursor) ?? cursor
                cursor = location
                pointer = location
                let fraction = mapping.fraction(of: pointer)!
                let drawn = CGPoint(x: overlay.minX + fraction.x * overlay.width, y: overlay.minY + fraction.y * overlay.height)
                worst.location = max(worst.location, distance(drawn, onOverlay(location)))
                worst.polled = max(worst.polled, distance(drawn, onOverlay(cursor)))
                worst.deltas = max(worst.deltas, distance(drawn, onOverlay(summed)))
                continue
            }
            let speed = [0.3, 1.0, 3.0, 9.0][Int(random.next() * 4)]
            let delta = CGPoint(x: random.normal() * speed, y: random.normal() * speed)
            let incoming = pinned(CGPoint(x: cursor.x + delta.x, y: cursor.y + delta.y), display, inset: 0.0001)
            cursor = mapping.constrain(incoming) ?? incoming
            pointer = cursor
            summed = pinned(CGPoint(x: summed.x + delta.x, y: summed.y + delta.y), window, inset: PointerMapping.edgeInset)
        }
        return worst
    }
}

/// A small seeded generator, so that the model runs the same way every time.
struct SplitMix {
    private var state: UInt64
    init(seed: UInt64) { state = seed }

    mutating func next() -> Double {
        state &+= 0x9E37_79B9_7F4A_7C15
        var z = state
        z = (z ^ (z >> 30)) &* 0xBF58_476D_1CE4_E5B9
        z = (z ^ (z >> 27)) &* 0x94D0_49BB_1331_11EB
        z ^= z >> 31
        return Double(z >> 11) / Double(1 << 53)
    }

    mutating func normal() -> Double {
        let u = max(next(), 1e-12)
        let v = next()
        return (-2 * log(u)).squareRoot() * cos(2 * .pi * v)
    }
}
