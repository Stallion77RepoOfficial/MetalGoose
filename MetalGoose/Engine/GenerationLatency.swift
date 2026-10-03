import Foundation
import QuartzCore
import os

/// How long after a capture arrives the first image generated from the pair it completes is ready to be
/// shown, for one engine.
///
/// The frame schedule is built on it: interpolation has to sample far enough behind the newest capture
/// that the image exists by the time it is wanted, and how far that is depends on how long the motion
/// takes to measure, how long the GPU or the Neural Engine takes to make the image, and how busy the
/// machine is — none of which can be assumed. So it is measured, at the one place that knows: where the
/// image is actually finished, not where its work was handed off.
///
/// Each engine has its own, and keeps it while the other is in use, so that taking an engine back does not start
/// from nothing; but a figure that was not renewed for a few seconds is not what the machine is doing now, and is not
/// reported.
final class GenerationLatency: Sendable {

    private struct State {
        var filter = IntervalFilter()
        var updated: CFTimeInterval = 0
    }
    private let state = OSAllocatedUnfairLock(initialState: State())

    /// How long a measurement stands in for the present: past it the engine is predicted again.
    private static let freshness: CFTimeInterval = 3

    /// The images of this pair have just become available.
    func record(previous: CFTimeInterval, next: CFTimeInterval) {
        let now = CACurrentMediaTime()
        let latency = now - next
        // One sample per pair, so each stands for the time between the pair's frames.
        state.withLock {
            $0.filter.add(latency, window: EngineShared.measurementWindow, elapsed: next - previous)
            $0.updated = now
        }
    }

    /// 0 until enough pairs have been measured, and again once the measurement is old.
    var value: CFTimeInterval {
        state.withLock { CACurrentMediaTime() - $0.updated <= Self.freshness ? $0.filter.value : 0 }
    }

    func reset() {
        state.withLock { $0 = State() }
    }
}
