import Foundation
import XCTest
@testable import MetalGooseLogic

final class GenerationSelectorTests: XCTestCase {

    /// A 1080p window: the Neural Engine's ladder for it, finest first, with the session serving at 960x540 and timed.
    private func inputs(capturesPerSecond: Double, refreshRate: Int, requested: Int = 4) -> GenerationSelector.Inputs {
        GenerationSelector.Inputs(
            requested: requested, captureInterval: 1 / capturesPerSecond, shortestInterval: 1 / capturesPerSecond,
            refreshRate: refreshRate, framePixels: 1920 * 1080, neuralRungs: [1920 * 1080, 1280 * 720, 960 * 540],
            neuralActive: 2, neuralMidpointTime: 0.005, neuralQuartersTime: 0.0135,
            neuralLatency: 0.016, metalFXLatency: 0)
    }

    /// Runs the selector for a while at one setting, as the capture path does once a capture, and returns where it settled.
    private func settle(_ selector: inout GenerationSelector, _ i: GenerationSelector.Inputs, from start: CFTimeInterval,
                        seconds: Double = 15) -> (choice: GenerationChoice, end: CFTimeInterval) {
        var now = start
        var choice = GenerationChoice.nothing
        while now < start + seconds {
            choice = selector.choose(i, now: now)
            now += i.captureInterval
        }
        return (choice, now)
    }

    func testFourStepsAt40CapturesASecondOn120Hz() {
        // Three refreshes a capture: not every quarter is seen, but the motion is more even than with the midpoint alone.
        var selector = GenerationSelector()
        let settled = settle(&selector, inputs(capturesPerSecond: 40, refreshRate: 120), from: 100).choice
        XCTAssertEqual(settled.engine, .neuralEngine)
        XCTAssertEqual(settled.multiplier, 4)
    }

    func testFourStepsAt30CapturesASecondOn120Hz() {
        var selector = GenerationSelector()
        let settled = settle(&selector, inputs(capturesPerSecond: 30, refreshRate: 120), from: 100).choice
        XCTAssertEqual(settled.multiplier, 4)
    }

    func testTwoStepsWhereThePanelShowsTwoImagesACapture() {
        for (rate, refresh) in [(60.0, 120), (30.0, 60)] {
            var selector = GenerationSelector()
            let settled = settle(&selector, inputs(capturesPerSecond: rate, refreshRate: refresh), from: 100).choice
            XCTAssertEqual(settled.engine, .neuralEngine, "\(rate) on \(refresh)")
            XCTAssertEqual(settled.multiplier, 2, "\(rate) on \(refresh)")
        }
    }

    func testFourStepsAreNotTakenAt50ButAreKeptDownTo48() {
        var selector = GenerationSelector()
        XCTAssertEqual(settle(&selector, inputs(capturesPerSecond: 50, refreshRate: 120), from: 100).choice.multiplier, 2)

        var keeping = GenerationSelector()
        let entered = settle(&keeping, inputs(capturesPerSecond: 45, refreshRate: 120), from: 100)
        XCTAssertEqual(entered.choice.multiplier, 4)
        let kept = settle(&keeping, inputs(capturesPerSecond: 48, refreshRate: 120), from: entered.end, seconds: 5)
        XCTAssertEqual(kept.choice.multiplier, 4)
        let left = settle(&keeping, inputs(capturesPerSecond: 50, refreshRate: 120), from: kept.end, seconds: 5)
        XCTAssertEqual(left.choice.multiplier, 2)
    }

    func testFourStepsWaitForACallThatFits() {
        // The panel has room for them, but three images in one call do not fit in a 40 a second interval.
        var i = inputs(capturesPerSecond: 40, refreshRate: 120)
        i.neuralQuartersTime = 0.024
        i.neuralMidpointTime = 0.009
        var selector = GenerationSelector()
        XCTAssertEqual(settle(&selector, i, from: 100).choice.multiplier, 2)
    }
}
