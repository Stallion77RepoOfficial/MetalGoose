import Foundation

/// The frame schedule's delay, chosen where the display's samples of the capture timeline fall rather than for the worst
/// place they could fall.
///
/// At each display callback the planner shows the image of the pair nearest to `now - delay`: within a pair cut into
/// `steps` steps it rounds the position to a step, so its choice changes at the half-steps. The callbacks come a refresh
/// apart and the captures' times sit on the same display's refresh grid, so where the samples fall within each step is set
/// by the delay alone, modulo a refresh.
///
/// `FramePlanner.interpolationDelay` does not know where that is, so it takes the worst: the first image of a pair is
/// wanted right at the half-step, and it allows the median time the image takes to arrive. Where the samples do fall at a
/// half-step the choice flips from one callback to the next with the smallest unevenness in the timing, and the image
/// wanted there is missing half the time; where they fall past it, the delay is longer than it had to be. Which of the two
/// a session got moved with the measured latency, in bands a refresh apart. Modelled at 30 captures a second on a 60 Hz
/// panel, latencies of 14 to 17 ms and 28 to 30 ms lost up to a sixth of the midpoints, and the
/// content on screen was up to 6 ms off an even pace, where it can be exact.
///
/// So the delay is chosen from where the samples land. Every delay over a span of a refresh is tried; one is possible when
/// the first sample to want a pair's first image comes at least `margin` after the image is expected, and of those the
/// shortest is taken that keeps every sample `clearance` from a half-step. Outside the bands that is the delay the worst
/// case gave, give or take where in the step the samples fall, and the same images are shown; inside them it is the next
/// image back — a refresh more latency where the worst case was dropping images to stay a refresh ahead.
///
/// Only for captures on a steady beat of the refreshes (`CaptureCadence`): a game that presents when it is ready lands on
/// the compositor's grid unevenly, and there is no one place for the samples to be.
///
/// In a fuller model of the pipeline — the compositor, ScreenCaptureKit's delivery, the capture queue, the Neural Engine one
/// call at a time with the newest pair waiting, the display link's callbacks, all with jitter — swept over the time a call
/// takes at 30 a second on 60 Hz, at 4x on 120 Hz and at 60 a second on 120 Hz (26 settings, 12 runs each), the content on
/// screen went from 1.9 ms off an even pace to 0.45 on average, and from 92 images a second to 96 of the 97 there were to
/// show; it was worse at none of them. The latency rose by 5.6 ms on average, where it took the image a step back. Sources
/// that do not keep a beat were not changed at all.
enum ScheduleAlignment {

    /// Delays tried in each refresh. They are on a fixed grid rather than counted from the least delay, so that the one chosen
    /// stays put while the latency it was chosen for drifts, and moves only when it stops being possible or best.
    static let candidates = 48
    /// Samples whose positions are weighed. Captures on a steady beat of the refreshes (`CaptureCadence`) repeat their pattern
    /// of positions every few refreshes, well inside this.
    static let samples = 48
    /// How far from a half-step every sample has to be for its choice to stand, in steps and at the least in time: further
    /// than the display link's callbacks and the captures' times wander. Where no delay keeps them this clear, the clearest
    /// one does.
    static let clearance = 0.25
    static let leastClearance: CFTimeInterval = 0.002

    /// The share of `clearance` a delay in use may keep before it is chosen again.
    static let keptClearance = 0.6

    /// How long after it is expected an image is first wanted, at the least: the expectation is the median, and the margin
    /// is what a slower than usual image has. It grows with the latency, as the latency's spread does: a quarter of it covers
    /// images that come a tenth later or earlier than the median at random. This is the one number that trades latency
    /// against images shown — it decides, near the point where a shorter delay would do, which of the two is taken.
    static let latencyShare = 0.25
    static func margin(captureInterval: CFTimeInterval, steps: Int, generationLatency: CFTimeInterval) -> CFTimeInterval {
        max(0.001, 0.1 * captureInterval / Double(steps), latencyShare * generationLatency)
    }

    /// How much more room than `margin` a shorter delay has to give before it replaces the one in use, as a share of a step.
    /// Without it, a latency that wanders about the point where a shorter delay becomes possible moves the schedule back and
    /// forth, and each move is a step in the motion on the screen.
    static let hysteresisShare = 0.15
    static func hysteresis(captureInterval: CFTimeInterval, steps: Int) -> CFTimeInterval {
        max(0.0015, hysteresisShare * captureInterval / Double(steps))
    }

    /// The delay to run the schedule at.
    ///
    /// - Parameters:
    ///   - captureInterval: the time between captures.
    ///   - generationLatency: how long after a capture arrives the first image of its pair is expected (the median).
    ///   - steps: what each pair is cut into, 2 or 4.
    ///   - phase: how far past the refresh grid the samples are, less how far past it the captures' times are, in seconds;
    ///     any value, taken modulo a refresh.
    ///   - refreshPeriod: the time between the display's refreshes.
    ///   - current: the delay this chose last time, which is kept while it is possible and no shorter one is clearly so.
    /// - Returns: nil where the inputs do not describe a schedule, for the caller to fall back to the worst-case delay.
    static func delay(captureInterval: CFTimeInterval, generationLatency: CFTimeInterval, steps: Int,
                      phase: CFTimeInterval, refreshPeriod: CFTimeInterval, current: CFTimeInterval? = nil) -> CFTimeInterval? {
        guard captureInterval > 0, refreshPeriod > 0, steps >= 2 else { return nil }
        let step = captureInterval / Double(steps)
        let margin = Self.margin(captureInterval: captureInterval, steps: steps, generationLatency: generationLatency)
        let due = captureInterval + generationLatency

        // How a delay does: how clear of the half-steps its samples are, and how long after the first image of a pair is
        // expected that image is first wanted. The first sample past the half-step into the first image comes `firstPast`
        // steps after it, so the image is wanted half a step and that much into the pair.
        func judge(_ delay: CFTimeInterval) -> (clearance: Double, room: CFTimeInterval) {
            let landing = Self.landing(delay: delay, phase: phase, refreshPeriod: refreshPeriod, step: step)
            return (landing.clearance, delay + (0.5 + landing.firstPast) * step - due)
        }

        // The least any delay could be is a sample exactly on the first image, wanted a whole step into the pair.
        let lowest = captureInterval - step + generationLatency + margin
        let span = max(refreshPeriod, step)
        let grain = refreshPeriod / Double(candidates)
        let first = Int((lowest / grain).rounded(.up))
        let last = Int(((lowest + span) / grain).rounded(.up))
        let tried = (first...last).map { candidate -> (delay: CFTimeInterval, clearance: Double, room: CFTimeInterval) in
            let delay = Double(candidate) * grain
            let verdict = judge(delay)
            return (delay, verdict.clearance, verdict.room)
        }
        guard let clearest = tried.filter({ $0.room >= margin }).map(\.clearance).max() else { return nil }
        // Clear enough is `clearance`, or the clearest there is where nothing is that clear: more only costs latency.
        let enough = min(clearest, max(Self.clearance, Self.leastClearance / step)) * 0.999
        func shortest(withRoom room: CFTimeInterval) -> CFTimeInterval? {
            tried.first { $0.room >= room && $0.clearance >= enough }?.delay
        }

        if let current {
            // Kept at a little less clearance than it was taken at: it was taken at the least that is enough, and the samples'
            // phase wanders by a fraction of a millisecond.
            let verdict = judge(current)
            if verdict.room >= margin, verdict.clearance >= enough * Self.keptClearance {
                let hysteresis = Self.hysteresis(captureInterval: captureInterval, steps: steps)
                if let shorter = shortest(withRoom: margin + hysteresis), shorter <= current - hysteresis {
                    return shorter
                }
                return current
            }
        }
        return shortest(withRoom: margin)
    }

    /// Where the samples land within the steps, at this delay, in steps: `clearance` is how far the nearest one is from a
    /// half-step (0.5 when every sample is on an image, 0 when one is on a half-step), and `firstPast` how little past a
    /// half-step a sample can be.
    static func landing(delay: CFTimeInterval, phase: CFTimeInterval, refreshPeriod: CFTimeInterval,
                        step: CFTimeInterval) -> (clearance: Double, firstPast: Double) {
        var clearance = 0.5
        var firstPast = 1.0
        for n in 0..<samples {
            let position = (Double(n) * refreshPeriod + phase - delay) / step
            let past = (position - 0.5) - (position - 0.5).rounded(.down)
            firstPast = min(firstPast, past)
            clearance = min(clearance, min(past, 1 - past))
        }
        return (clearance, firstPast)
    }
}
