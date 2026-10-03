import Foundation
import XCTest
@testable import MetalGooseLogic

final class ScheduleAlignmentTests: XCTestCase {

    // MARK: - Where the samples land

    func testSamplesOnTheImagesAreAsClearAsTheyCanBe() {
        // 30 captures a second at 2x on 60 Hz: a step is a refresh, so a delay of whole refreshes puts every sample on an image.
        let landing = ScheduleAlignment.landing(delay: 3 / 60.0, phase: 0, refreshPeriod: 1 / 60.0, step: 1 / 60.0)
        XCTAssertEqual(landing.clearance, 0.5, accuracy: 1e-9)
        XCTAssertEqual(landing.firstPast, 0.5, accuracy: 1e-9)
    }

    func testSamplesHalfARefreshOffAreOnTheHalfSteps() {
        let landing = ScheduleAlignment.landing(delay: 3.5 / 60.0, phase: 0, refreshPeriod: 1 / 60.0, step: 1 / 60.0)
        XCTAssertEqual(landing.clearance, 0, accuracy: 1e-9)
    }

    func testTwoSamplesAStepAreBestAQuarterStepEitherSide() {
        // 30 captures a second at 2x on 120 Hz: two refreshes a step, so the best a delay can do is a quarter step from both.
        let period = 1 / 120.0
        let best = (0..<48).map { ScheduleAlignment.landing(delay: 0.04 + Double($0) * period / 48, phase: 0,
                                                             refreshPeriod: period, step: 2 * period).clearance }.max()!
        XCTAssertEqual(best, 0.25, accuracy: 0.011)
    }

    // MARK: - The delay

    func testTheDelayLeavesTheFirstImageItsMarginAndKeepsTheSamplesClear() {
        let period = 1 / 60.0
        for latency in stride(from: 0.004, through: 0.030, by: 0.0005) {
            guard let delay = ScheduleAlignment.delay(captureInterval: 2 * period, generationLatency: latency, steps: 2,
                                                      phase: 0, refreshPeriod: period) else { return XCTFail() }
            let landing = ScheduleAlignment.landing(delay: delay, phase: 0, refreshPeriod: period, step: period)
            let wanted = delay + (0.5 + landing.firstPast) * period
            let margin = ScheduleAlignment.margin(captureInterval: 2 * period, steps: 2, generationLatency: latency)
            XCTAssertGreaterThanOrEqual(wanted - (2 * period + latency), margin - 1e-9, "latency \(latency)")
            XCTAssertGreaterThanOrEqual(landing.clearance, ScheduleAlignment.clearance * 0.99, "latency \(latency)")
            // No longer than the worst-case delay by more than a refresh: the most a grid can cost.
            let worstCase = 1.5 * period + latency
            XCTAssertLessThanOrEqual(delay, worstCase + period + 1e-9, "latency \(latency)")
        }
    }

    func testTheDelayInUseIsKeptWhileItStillWorks() {
        let period = 1 / 60.0
        let first = ScheduleAlignment.delay(captureInterval: 2 * period, generationLatency: 0.011, steps: 2, phase: 0,
                                            refreshPeriod: period)!
        // A latency that has come down a little is no reason to move: a move is a step in the motion.
        let kept = ScheduleAlignment.delay(captureInterval: 2 * period, generationLatency: 0.0105, steps: 2, phase: 0,
                                           refreshPeriod: period, current: first)
        XCTAssertEqual(kept, first)
        // One that has come down by a whole refresh is.
        let shorter = ScheduleAlignment.delay(captureInterval: 2 * period, generationLatency: 0.011 - period, steps: 2,
                                              phase: 0, refreshPeriod: period, current: first)!
        XCTAssertLessThan(shorter, first)
        // One that has gone up past it moves it at once, or the images would be wanted before they exist.
        let longer = ScheduleAlignment.delay(captureInterval: 2 * period, generationLatency: 0.011 + period / 2, steps: 2,
                                             phase: 0, refreshPeriod: period, current: first)!
        XCTAssertGreaterThan(longer, first)
    }

    func testWhatIsNotAScheduleHasNoDelay() {
        XCTAssertNil(ScheduleAlignment.delay(captureInterval: 0, generationLatency: 0.01, steps: 2, phase: 0, refreshPeriod: 1 / 60.0))
        XCTAssertNil(ScheduleAlignment.delay(captureInterval: 1 / 30.0, generationLatency: 0.01, steps: 1, phase: 0, refreshPeriod: 1 / 60.0))
        XCTAssertNil(ScheduleAlignment.delay(captureInterval: 1 / 30.0, generationLatency: 0.01, steps: 2, phase: 0, refreshPeriod: 0))
    }

    // MARK: - The schedule as the planner runs it

    /// 30 captures a second from a game that presents in step with a 60 Hz panel, interpolated at 2x: with the delay chosen
    /// where the samples land, every refresh shows the next image — a capture, then the midpoint after it — and the content on
    /// screen keeps an even pace, whatever the time the images take. With the worst-case delay alone, a latency that put the
    /// samples on the half-steps lost up to a fifth of the midpoints and left the pace 8 ms off.
    func testEveryImageIsShownOnTimeWhateverTheLatency() {
        for latency in stride(from: 0.008, through: 0.030, by: 0.001) {
            let run = ScheduleModel.run(fps: 30, refresh: 60, steps: 2, latency: latency, aligned: true)
            XCTAssertGreaterThan(run.imagesPerSecond, 59.5, "latency \(latency)")
            XCTAssertLessThan(run.judder, 0.0005, "latency \(latency)")
        }
    }

    func testQuartersOnA120HzPanelAreShownOnTimeToo() {
        for latency in stride(from: 0.012, through: 0.030, by: 0.002) {
            let run = ScheduleModel.run(fps: 30, refresh: 120, steps: 4, latency: latency, aligned: true)
            XCTAssertGreaterThan(run.imagesPerSecond, 119, "latency \(latency)")
            XCTAssertLessThan(run.judder, 0.0005, "latency \(latency)")
        }
    }

    func testTheWorstCaseDelayLosesImagesWhereTheSamplesFallOnTheHalfSteps() {
        // The case the alignment is for, kept as a record of it: at this latency the worst-case delay puts the samples a hair
        // past the half-step, and the midpoint is wanted the moment its median arrival comes. The aligned delay shows the
        // image before instead, a refresh later.
        let worstCase = ScheduleModel.run(fps: 30, refresh: 60, steps: 2, latency: 0.0165, aligned: false)
        let aligned = ScheduleModel.run(fps: 30, refresh: 60, steps: 2, latency: 0.0165, aligned: true)
        XCTAssertLessThan(worstCase.imagesPerSecond, 55)
        XCTAssertGreaterThan(worstCase.judder, 0.004)
        XCTAssertEqual(aligned.imagesPerSecond, 60, accuracy: 0.5)
        XCTAssertEqual(aligned.latency - worstCase.latency, 1 / 60.0, accuracy: 0.004)
    }

    func testOutsideTheBandsTheAlignedDelayShowsTheSameImagesAtTheSameLatency() {
        for latency in [0.008, 0.010, 0.012, 0.019, 0.022, 0.025] {
            let worstCase = ScheduleModel.run(fps: 30, refresh: 60, steps: 2, latency: latency, aligned: false)
            let aligned = ScheduleModel.run(fps: 30, refresh: 60, steps: 2, latency: latency, aligned: true)
            XCTAssertEqual(aligned.latency, worstCase.latency, accuracy: 0.0005, "latency \(latency)")
            XCTAssertEqual(aligned.imagesPerSecond, worstCase.imagesPerSecond, accuracy: 0.5, "latency \(latency)")
        }
    }
}

final class CaptureCadenceTests: XCTestCase {

    func testCapturesOnEveryOtherRefreshHaveThatBeat() {
        var cadence = CaptureCadence()
        for i in 0...30 { cadence.add(100 + Double(i) * 2 / 60) }
        XCTAssertEqual(cadence.beat(refreshPeriod: 1 / 60.0)!, 2 / 60.0, accuracy: 1e-12)
    }

    func testAFreeRunningSourceOnTheCompositorsGridHasNone() {
        // 60 a second, ready at no particular moment, shown on a 120 Hz grid: 8.3, 16.7 and 25 ms apart.
        var cadence = CaptureCadence()
        var random = SplitMix(seed: 3)
        var content = 100.0
        var last = 0.0
        for _ in 0..<60 {
            content += 1 / 60.0
            let shown = ((content + 0.004 + random.next() * 0.008) * 120).rounded(.up) / 120
            if shown > last { cadence.add(shown); last = shown }
        }
        XCTAssertNil(cadence.beat(refreshPeriod: 1 / 120.0))
    }

    func testAnOddCaptureIsTolerated() {
        var cadence = CaptureCadence()
        var time = 100.0
        for i in 0..<30 {
            time += i == 20 ? 3 / 60.0 : 2 / 60.0
            cadence.add(time)
        }
        XCTAssertNotNil(cadence.beat(refreshPeriod: 1 / 60.0))
    }

    func testAPauseStartsOver() {
        var cadence = CaptureCadence()
        for i in 0...30 { cadence.add(100 + Double(i) / 30) }
        cadence.add(105)
        XCTAssertNil(cadence.beat(refreshPeriod: 1 / 60.0))
    }
}

final class GridPhaseTests: XCTestCase {

    func testTimesEitherSideOfAGridLineAverageToTheLine() {
        var phase = GridPhase()
        let period = 1 / 120.0
        for i in 0..<40 {
            phase.add(Double(i) * period + (i.isMultiple(of: 2) ? 0.0002 : -0.0002), period: period)
        }
        let offset = phase.offset(period: period)!
        XCTAssertLessThan(min(offset, period - offset), 0.00001)
    }

    func testAConstantOffsetIsFound() {
        var phase = GridPhase()
        let period = 1 / 60.0
        for i in 0..<40 { phase.add(1000 + Double(i) * period + 0.004, period: period) }
        XCTAssertEqual(phase.offset(period: period)!, 0.004, accuracy: 1e-6)
    }

    func testTimesThatDoNotAgreeHaveNoPhase() {
        var phase = GridPhase()
        var random = SplitMix(seed: 9)
        for _ in 0..<40 { phase.add(random.next(), period: 1 / 60.0) }
        XCTAssertNil(phase.offset(period: 1 / 60.0))
    }
}

/// A game presenting in step with the display, captured on the compositor's grid, each pair's images ready `latency` after its
/// capture give or take a tenth (the latency the schedule is given is the median), and the display link calling back a
/// refresh ahead with a little jitter: the render thread's schedule as `RenderPipeline` runs it, through the app's own
/// planner and governor.
enum ScheduleModel {

    struct Frame: TimedFrame {
        let timestamp: CFTimeInterval
        let isSceneCut = false
    }

    static func run(fps: Double, refresh: Double, steps: Int, latency: CFTimeInterval, aligned: Bool,
                    seconds: Double = 8) -> (imagesPerSecond: Double, judder: Double, latency: Double) {
        let period = 1 / refresh
        let interval = (refresh / fps).rounded() * period
        var random = SplitMix(seed: 11)
        var governor = DelayGovernor()
        var cadence = CaptureCadence()
        var samplePhase = GridPhase()
        var stampPhase = GridPhase()
        var current: CFTimeInterval?
        var captures: [CFTimeInterval] = []
        var ready: [CFTimeInterval: CFTimeInterval] = [:]
        var nextCapture = 1.0
        var shown: PresentedImage?
        var shownContent = 0.0
        var newImages = 0
        var errors: [Double] = []

        var target = 1.0
        while target < seconds {
            let now = target - period + random.next() * 0.0004
            while nextCapture <= now {
                captures.append(nextCapture)
                ready[nextCapture] = nextCapture + latency * (1 + 0.1 * random.normal())
                cadence.add(nextCapture)
                stampPhase.add(nextCapture, period: period)
                nextCapture += interval
            }
            samplePhase.add(now, period: period)
            let ring = captures.suffix(4).map { Frame(timestamp: $0) }
            let needed = FramePlanner.interpolationDelay(captureInterval: interval, generationLatency: latency, steps: steps)
            var wanted = needed
            if aligned, let beat = cadence.beat(refreshPeriod: period), let s = samplePhase.offset(period: period),
               let t = stampPhase.offset(period: period) {
                current = ScheduleAlignment.delay(captureInterval: beat, generationLatency: latency, steps: steps,
                                                  phase: s - t, refreshPeriod: period, current: current)
                wanted = current ?? needed
            }
            let delay = governor.apply(wanted, now: now)
            let plan = FramePlanner.plan(ring, PlanningInput(multiplier: steps, sampleTime: now, delay: delay))
            var image: PresentedImage?
            var content = shownContent
            switch plan {
            case .nothing:
                break
            case .captured(let i):
                image = .captured(ring[i].timestamp)
                content = ring[i].timestamp
            case .interpolated(let p, let n, let step, let total):
                if ready[ring[n].timestamp]! <= now {
                    let phase = Double(step) / Double(total)
                    image = .interpolated(previous: ring[p].timestamp, next: ring[n].timestamp, phase: phase)
                    content = ring[p].timestamp + phase * (ring[n].timestamp - ring[p].timestamp)
                } else {
                    image = .captured(ring[p].timestamp)
                    content = ring[p].timestamp
                }
            }
            if let image, image != shown {
                shown = image
                shownContent = content
                if target > 3 { newImages += 1 }
            }
            if target > 3 { errors.append(target - shownContent) }
            target += period
        }
        // Judder is how far the content on screen is off an even pace: its lag behind the clock, about the mean lag of the
        // quarter second either side, so that a one-off change of the delay counts once rather than for the rest of the run.
        let mean = errors.reduce(0, +) / Double(errors.count)
        let half = Int((0.25 / period).rounded())
        var squares = 0.0
        for k in errors.indices {
            let window = errors[max(0, k - half)..<min(errors.count, k + half + 1)]
            let local = window.reduce(0, +) / Double(window.count)
            squares += (errors[k] - local) * (errors[k] - local)
        }
        return (Double(newImages) / (seconds - 3), (squares / Double(errors.count)).squareRoot(), mean)
    }
}
