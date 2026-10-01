import Foundation
import QuartzCore
import os

/// How long after a capture arrives the midpoint of the pair it completes is ready to be shown.
///
/// The frame schedule is built on it: interpolation has to sample far enough behind the newest capture
/// that the midpoint exists by the time it is wanted, and how far that is depends on how long the motion
/// takes to measure, how long the GPU or the Neural Engine takes to make the image, and how busy the
/// machine is — none of which can be assumed. So it is measured, at the one place that knows: where the
/// image is actually finished, not where its work was handed off.
final class GenerationLatency: Sendable {

    private let filter = OSAllocatedUnfairLock(initialState: IntervalFilter())

    /// The midpoint of this pair has just become available.
    func record(previous: CFTimeInterval, next: CFTimeInterval) {
        let latency = CACurrentMediaTime() - next
        // One sample per pair, so each stands for the time between the pair's frames.
        filter.withLock { $0.add(latency, window: EngineShared.measurementWindow, elapsed: next - previous) }
    }

    /// 0 until enough pairs have been measured.
    var value: CFTimeInterval { filter.withLock { $0.value } }

    func reset() {
        filter.withLock { $0.reset() }
    }
}
