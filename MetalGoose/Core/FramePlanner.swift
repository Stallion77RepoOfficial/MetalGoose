import Foundation

/// What the planner needs to know about a captured frame. The engine's frames carry
/// textures too, but scheduling is decided by arrival time and scene cuts alone — which
/// is what keeps this logic free of Metal and testable.
protocol TimedFrame {
    /// Arrival time of the capture, on the `CACurrentMediaTime()` clock.
    var timestamp: CFTimeInterval { get }
    /// The frame starts a new shot, so nothing may be generated across it.
    var isSceneCut: Bool { get }
}

/// What the display callback should put on screen. Indices refer to the frames the
/// plan was made from, oldest first.
enum PresentationPlan: Equatable {
    /// Nothing has been captured yet.
    case nothing
    /// A captured frame, exactly as captured.
    case captured(Int)
    /// The midpoint between two captures.
    case interpolated(previous: Int, next: Int)
    /// The newest capture warped `step` of `steps` of the way into the next interval.
    case extrapolated(source: Int, step: Int, steps: Int)
}

/// What a presented image is, identified without reference to how it was produced.
/// Two plans with the same identity put the same pixels on screen, so the second one
/// does not need to be drawn.
enum PresentedImage: Equatable {
    case captured(CFTimeInterval)
    case interpolated(previous: CFTimeInterval, next: CFTimeInterval)
    case extrapolated(source: CFTimeInterval, step: Int)
}

struct PlanningInput {
    var mode: FrameGenMode
    /// How many images each capture interval should carry. Only extrapolation uses it:
    /// MetalFX synthesises one image per pair, the midpoint, so interpolation is 2x.
    var multiplier: Int
    /// The time at which to sample the capture timeline, on the same clock as the frame timestamps: the
    /// moment of the display callback.
    ///
    /// Not the time its image will reach the screen. Every image passes through the same display
    /// pipeline, so that latency delays all of them equally and drops out — but only if the schedule is
    /// measured from the callback. An interval is cut into steps by how old the newest capture is *now*,
    /// and a clock that already includes the pipeline's latency starts every interval most of the way
    /// through it.
    var sampleTime: CFTimeInterval
    /// Smoothed interval between captures; 0 until it has been measured.
    var captureInterval: CFTimeInterval
    /// How long after a capture arrives the midpoint of the pair it completes can be shown; 0 until it has
    /// been measured. Only interpolation reads it.
    var generationLatency: CFTimeInterval
    /// The newest capture has already been shown as captured.
    var newestWasPresented: Bool
    /// Timestamp of the capture the latest motion field was measured against, if there is one.
    var motionTimestamp: CFTimeInterval?
}

enum FramePlanner {

    private static let motionFreshness: CFTimeInterval = 1.0 / 30.0

    static func plan<F: TimedFrame>(_ frames: [F], _ input: PlanningInput) -> PresentationPlan {
        guard let newestIndex = frames.indices.last else { return .nothing }

        switch input.mode {
        case .extrapolation:
            return extrapolate(frames, newestIndex: newestIndex, input)
        case .interpolation:
            guard frames.count >= 2 else { return .captured(newestIndex) }
            return interpolate(frames, input)
        case .off:
            return .captured(newestIndex)
        }
    }

    /// The identity of what a plan shows, for the frames it was made from.
    static func image<F: TimedFrame>(of plan: PresentationPlan, in frames: [F]) -> PresentedImage? {
        switch plan {
        case .nothing:
            return nil
        case .captured(let i):
            return .captured(frames[i].timestamp)
        case .interpolated(let previous, let next):
            return .interpolated(previous: frames[previous].timestamp, next: frames[next].timestamp)
        case .extrapolated(let source, let step, _):
            return .extrapolated(source: frames[source].timestamp, step: step)
        }
    }

    // MARK: - Extrapolation

    /// Every capture is shown as captured, once. Only the gaps between captures are
    /// generated — warping real frames as well destroys the image for no benefit.
    ///
    /// The multiplier is how many images the gap should carry, so the gap is cut into
    /// that many slots: slot 0 is the capture itself and each later slot is one warp
    /// phase.
    private static func extrapolate<F: TimedFrame>(_ frames: [F], newestIndex: Int,
                                                   _ input: PlanningInput) -> PresentationPlan {
        let newest = frames[newestIndex]
        guard input.newestWasPresented else { return .captured(newestIndex) }

        let interval = input.captureInterval
        let steps = max(1, input.multiplier)
        let elapsed = interval > 0
            ? min(max((input.sampleTime - newest.timestamp) / interval, 0), 1)
            : 0
        let step = min(steps - 1, Int(elapsed * Double(steps)))

        // A field much older than the capture it is being applied to describes a velocity
        // the scene has already left. Warping on it is what turned a flick of the
        // mouse into a violent throw and back. The newest capture's own field is not ready
        // in the first slots after it arrives, and those slots use the one before: two
        // intervals of the rate actually being measured are tolerated, because refusing it
        // — as one interval's tolerance did, by a hair, whenever arrivals jittered —
        // switched the warp off and on within a single interval. How long a field takes
        // does not shrink with the capture rate, so at a high one it is several intervals
        // old by the time it exists; a thirtieth of a second is tolerated whatever the rate.
        guard step > 0, !newest.isSceneCut, newestIndex >= 1,
              let motionTimestamp = input.motionTimestamp,
              interval <= 0 || newest.timestamp - motionTimestamp <= max(2 * interval, motionFreshness) else {
            return .captured(newestIndex)
        }
        return .extrapolated(source: newestIndex, step: step, steps: steps)
    }

    // MARK: - Interpolation

    /// A pair offers three images: `previous` at position 0, the generated midpoint at 0.5 and `next` at
    /// 1. Each position is served by whichever is nearest, which puts the midpoint's share at a quarter
    /// to three quarters of the way through — the same at any capture rate and any refresh rate. (A
    /// tuned snap window in output-interval units did the same job only while the output rate happened
    /// to match.)
    private static let midpointStart = 0.25

    /// How far behind real time the frame clock samples.
    ///
    /// Interpolation blends between a pair that has already arrived, so the clock runs behind the
    /// newest capture. How far is set by the midpoint: it is first wanted when the clock is a quarter
    /// of the way from `previous` to `next`, which with a delay of `d` happens `d` minus three quarters
    /// of an interval after `next` arrived — and it cannot be shown before it has been made. So the
    /// delay is three quarters of an interval plus the time the midpoint takes to arrive. Any less and
    /// it is asked for before it exists, the screen holds the previous capture instead, and the pair
    /// goes by without its midpoint; any more is latency for nothing.
    ///
    /// Ring timestamps are arrival times, so with this delay the sample sweeps from `previous` to `next`
    /// as the gap fills in, and every phase in the bracket is reachable.
    static func interpolationDelay(captureInterval: CFTimeInterval, generationLatency: CFTimeInterval) -> CFTimeInterval {
        (1 - midpointStart) * captureInterval + generationLatency
    }

    private static func interpolate<F: TimedFrame>(_ frames: [F], _ input: PlanningInput) -> PresentationPlan {
        let delay = interpolationDelay(captureInterval: input.captureInterval, generationLatency: input.generationLatency)
        let targetTime = input.sampleTime - delay
        let (previousIndex, nextIndex) = bracket(frames, around: targetTime)
        let previous = frames[previousIndex]
        let next = frames[nextIndex]

        let duration = next.timestamp - previous.timestamp
        guard duration > 0 else { return .captured(previousIndex) }
        let position = min(max((targetTime - previous.timestamp) / duration, 0), 1)

        if position < midpointStart { return .captured(previousIndex) }
        // Nothing may be generated across a cut: the new shot is shown as soon as it is wanted.
        if position >= 1 - midpointStart || next.isSceneCut { return .captured(nextIndex) }
        return .interpolated(previous: previousIndex, next: nextIndex)
    }

    /// The two adjacent frames whose timestamps bracket `time`. A time past the newest
    /// frame takes the newest pair; one before the oldest takes the oldest pair.
    /// Needs at least two frames.
    static func bracket<F: TimedFrame>(_ frames: [F], around time: CFTimeInterval) -> (Int, Int) {
        for i in 0..<(frames.count - 1) where time >= frames[i].timestamp && time <= frames[i + 1].timestamp {
            return (i, i + 1)
        }
        if time > frames[frames.count - 1].timestamp {
            return (frames.count - 2, frames.count - 1)
        }
        return (0, 1)
    }
}
