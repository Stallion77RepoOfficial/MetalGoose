import Foundation

/// Whether the captures come on a steady beat of the display's refreshes, and how long the beat is.
///
/// A game that presents in step with the display — the usual way on macOS, at 30, 60 or 120 a second — is captured on that
/// beat exactly: the compositor's times for its frames are a whole number of refreshes apart. Its frames' content is then
/// evenly spaced too, and the schedule can put the display's samples where they fall clear of every change of image
/// (`ScheduleAlignment`). A game that presents when it is ready lands on the compositor's grid unevenly — a 60 fps one on a
/// 120 Hz panel 8.3, 16.7 and 25 ms apart — and there is no one place for the samples to be; that is left to the
/// worst-case schedule.
struct CaptureCadence {

    /// Intervals weighed: a little under a second at 30 a second.
    static let span = 24
    /// The share of them that have to be within `tolerance` of the beat.
    static let steadyShare = 0.9
    /// How far from the beat an interval may be, as a share of a refresh.
    static let tolerance = 0.25
    /// How far the beat may be from a whole number of refreshes, as a share of one.
    static let onGrid = 0.1
    /// A gap this long is a pause, after which the beat is learnt again.
    static let longestInterval: CFTimeInterval = 0.5

    private var intervals: [CFTimeInterval] = []
    private var last: CFTimeInterval?

    /// A capture's time, on the compositor's clock; in order.
    mutating func add(_ time: CFTimeInterval) {
        defer { last = time }
        guard let last, time > last else { return }
        let interval = time - last
        guard interval <= Self.longestInterval else {
            intervals.removeAll(keepingCapacity: true)
            return
        }
        intervals.append(interval)
        if intervals.count > Self.span { intervals.removeFirst(intervals.count - Self.span) }
    }

    /// The beat, a whole number of refreshes, where the captures keep to one; nil where they do not, or have not yet for long
    /// enough to say.
    func beat(refreshPeriod: CFTimeInterval) -> CFTimeInterval? {
        guard refreshPeriod > 0, intervals.count >= Self.span else { return nil }
        let median = intervals.sorted()[intervals.count / 2]
        let refreshes = (median / refreshPeriod).rounded()
        guard refreshes >= 1, abs(median / refreshPeriod - refreshes) <= Self.onGrid else { return nil }
        let beat = refreshes * refreshPeriod
        let onBeat = intervals.filter { abs($0 - beat) <= Self.tolerance * refreshPeriod }.count
        return Double(onBeat) >= Self.steadyShare * Double(intervals.count) ? beat : nil
    }

    mutating func reset() {
        intervals.removeAll(keepingCapacity: true)
        last = nil
    }
}

/// Where a series of times falls on a grid of one period, on average: a mean taken around the circle, so that times either
/// side of a grid line average to the line and not to half a period away.
struct GridPhase {

    /// How much each time weighs: about the last twenty.
    static let weight = 0.05
    /// How tightly the times have to agree for their average to mean anything: the length of the mean of their unit vectors.
    static let agreement = 0.9

    private var x = 0.0
    private var y = 0.0
    private var count = 0

    mutating func add(_ time: CFTimeInterval, period: CFTimeInterval) {
        guard period > 0 else { return }
        let angle = 2 * Double.pi * (time / period - (time / period).rounded(.down))
        let weight = count < 20 ? 1 / Double(count + 1) : Self.weight
        x += (cos(angle) - x) * weight
        y += (sin(angle) - y) * weight
        count += 1
    }

    /// The average offset from the grid, in [0, period); nil until there are times enough and while they do not agree.
    func offset(period: CFTimeInterval) -> CFTimeInterval? {
        guard count >= 8, (x * x + y * y).squareRoot() >= Self.agreement else { return nil }
        var angle = atan2(y, x)
        if angle < 0 { angle += 2 * .pi }
        return angle / (2 * .pi) * period
    }

    mutating func reset() {
        x = 0
        y = 0
        count = 0
    }
}
