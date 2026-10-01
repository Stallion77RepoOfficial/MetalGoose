import Foundation

/// How evenly images reach the screen. Smooth motion needs the gaps between presented images to
/// be equal, whatever their length: a steady 60 images a second is better than an average of 60
/// made of bursts and stalls.
struct PacingTracker {
    private var intervals: [Double] = []
    private var lastTime: CFTimeInterval = 0

    /// Records an image reaching the screen at `time`. The window holds `capacity` intervals —
    /// sized by the caller to span a fixed stretch of wall-clock time at the rate being driven,
    /// not a fixed count that would average over a quarter of a second at one rate and two
    /// seconds at another.
    mutating func record(_ time: CFTimeInterval, capacity: Int) {
        defer { lastTime = time }
        guard lastTime > 0, time > lastTime else { return }
        intervals.append(time - lastTime)
        if intervals.count > max(2, capacity) { intervals.removeFirst(intervals.count - max(2, capacity)) }
    }

    mutating func reset() {
        intervals.removeAll()
        lastTime = 0
    }

    /// Mean interval and a 0...100 score, or `nil` until there is a pair of intervals to compare.
    /// The score is how much each interval differs from the one before it, relative to the mean.
    var summary: (averageInterval: Double, score: Double)? {
        guard intervals.count >= 2 else { return nil }
        let average = intervals.reduce(0, +) / Double(intervals.count)
        var jitter = 0.0
        for i in 1..<intervals.count { jitter += abs(intervals[i] - intervals[i - 1]) }
        jitter /= Double(intervals.count - 1)
        let score = average > 0 ? max(0, 100 * (1 - min(1, jitter / average))) : 100
        return (average, score)
    }
}
