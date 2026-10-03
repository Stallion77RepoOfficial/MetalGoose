import Foundation

/// How close together captures come, as they have come over the last several seconds: not their average spacing but their short one.
///
/// ScreenCaptureKit hands a source's frames over on the compositor's clock, so a 60 fps source on a 120 Hz panel arrives 8.3, 16.7
/// and 25 ms apart rather than 16.7 every time; measured, a tenth of the intervals were 8.8 ms or less against a median of 17.4.
/// What an engine can keep up with is decided by the short ones, since a pair that follows its predecessor closely arrives while
/// the engine is still on it, whereas the average says there is room. A tenth of the intervals is the figure: a call that fits
/// it leaves nine pairs in ten to find the engine free.
struct IntervalSpread {

    /// How far back the intervals are kept, and how few make an estimate. Long enough that the tenth percentile of a source
    /// whose captures come in bursts does not wander by a tenth from one second to the next, which moved a choice made on it
    /// back and forth every dozen seconds.
    static let span: CFTimeInterval = 8
    static let minimumSamples = 24

    /// The share of intervals that are shorter than the figure.
    static let share = 0.1

    /// A gap this long is a pause in the picture, not an interval between its frames.
    static let longestInterval: CFTimeInterval = 1

    private var recent: [(arrival: CFTimeInterval, interval: CFTimeInterval)] = []
    private var lastArrival: CFTimeInterval = 0
    private(set) var shortest: CFTimeInterval = 0

    /// A capture arrived at `time`.
    mutating func add(arrival time: CFTimeInterval) {
        defer { lastArrival = time }
        guard lastArrival > 0, time > lastArrival else { return }
        let interval = time - lastArrival
        guard interval <= Self.longestInterval else {
            // A pause: what came before it says nothing about what comes after.
            recent.removeAll(keepingCapacity: true)
            shortest = 0
            return
        }
        recent.append((time, interval))
        if let stale = recent.firstIndex(where: { time - $0.arrival <= Self.span }), stale > 0 { recent.removeFirst(stale) }
        guard recent.count >= Self.minimumSamples else {
            shortest = 0
            return
        }
        let sorted = recent.map(\.interval).sorted()
        shortest = sorted[Int(Double(sorted.count - 1) * Self.share)]
    }

    mutating func reset() {
        recent.removeAll(keepingCapacity: true)
        lastArrival = 0
        shortest = 0
    }
}
