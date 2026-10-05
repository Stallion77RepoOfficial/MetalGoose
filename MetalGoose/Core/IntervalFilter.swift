import Foundation

/// An estimate of a steady quantity — the interval between captures, the time a generated image
/// takes to arrive — that one odd sample does not move.
///
/// The value follows the median of the last few samples, so a pause cannot enter it (ScreenCaptureKit
/// sends nothing while a window is still, so a gap of seconds is an ordinary sample). The first
/// estimate waits for three samples for the same reason.
///
/// The median is smoothed by a first-order filter whose time constant is a span of wall-clock time:
/// the weight of a sample is the time it stands for as a share of the window, so the estimate settles
/// in the same time at 5 fps as at 240.
struct IntervalFilter {
    private(set) var value: Double = 0

    private var recent: [Double] = []
    private static let medianSpan = 5
    private static let minimumSamples = 3

    /// - Parameter elapsed: the time this sample stands for, when that is not the sample itself. An
    ///   interval stands for its own length; a latency measured once per capture stands for the
    ///   capture interval.
    mutating func add(_ sample: Double, window: Double, elapsed: Double? = nil) {
        guard sample > 0 else { return }

        recent.append(sample)
        if recent.count > Self.medianSpan { recent.removeFirst() }
        guard recent.count >= Self.minimumSamples else { return }

        let typical = recent.sorted()[recent.count / 2]
        // The first estimate is the median itself.
        guard value > 0 else { value = typical; return }

        let weight = min(1, (elapsed ?? sample) / window)
        value += (typical - value) * weight
    }

    mutating func reset() {
        value = 0
        recent.removeAll(keepingCapacity: true)
    }
}
