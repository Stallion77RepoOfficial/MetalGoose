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
    /// The image `step` of `steps` of the way from one capture to the next: the midpoint of a pair at 2 steps,
    /// its quarters at 4.
    case interpolated(previous: Int, next: Int, step: Int, steps: Int)
}

/// What a presented image is, identified without reference to how it was produced.
/// Two plans with the same identity put the same pixels on screen, so the second one
/// does not need to be drawn.
enum PresentedImage: Equatable {
    case captured(CFTimeInterval)
    /// `phase` is how far from `previous` to `next` the image sits: the midpoint is 0.5 whether the pair was cut
    /// into two steps or four, so the same image is the same image either way.
    case interpolated(previous: CFTimeInterval, next: CFTimeInterval, phase: Double)
}

struct PlanningInput {
    /// How many images each capture interval carries: a pair is cut into 2 steps, which is its midpoint, or into 4,
    /// which is its quarters. Less than 2 generates nothing, and the newest capture is shown as it arrives.
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
    /// How far behind the newest capture's arrival the schedule runs (`FramePlanner.interpolationDelay`), as the
    /// caller smooths it.
    var delay: CFTimeInterval
}

enum FramePlanner {

    static func plan<F: TimedFrame>(_ frames: [F], _ input: PlanningInput) -> PresentationPlan {
        guard let newestIndex = frames.indices.last else { return .nothing }
        guard input.multiplier >= InterpolationSteps.halves, frames.count >= 2 else { return .captured(newestIndex) }
        return interpolate(frames, input)
    }

    /// The identity of what a plan shows, for the frames it was made from.
    static func image<F: TimedFrame>(of plan: PresentationPlan, in frames: [F]) -> PresentedImage? {
        switch plan {
        case .nothing:
            return nil
        case .captured(let i):
            return .captured(frames[i].timestamp)
        case .interpolated(let previous, let next, let step, let steps):
            return .interpolated(previous: frames[previous].timestamp, next: frames[next].timestamp,
                                 phase: Double(step) / Double(steps))
        }
    }

    // MARK: - Interpolation

    /// A pair offers `steps + 1` images: `previous` at position 0, the generated ones at k/steps and `next` at
    /// 1. Each position is served by whichever is nearest, so the first generated image's share starts half a
    /// step in — a quarter of the way through a pair at 2 steps, an eighth at 4 — the same at any capture
    /// rate and any refresh rate. (A tuned snap window in output-interval units did the same job only while
    /// the output rate happened to match.)
    private static func firstImageStart(steps: Int) -> Double { 0.5 / Double(steps) }

    /// How far behind real time the frame clock samples.
    ///
    /// Interpolation blends between a pair that has already arrived, so the clock runs behind the
    /// newest capture. How far is set by the first generated image: it is first wanted when the clock is
    /// `firstImageStart` of the way from `previous` to `next`, which with a delay of `d` happens `d` minus the
    /// rest of an interval after `next` arrived — and it cannot be shown before it has been made. So the
    /// delay is that rest of an interval plus the time the image takes to arrive. Any less and it is asked
    /// for before it exists, the screen holds the previous capture instead, and the pair goes by without it;
    /// any more is latency for nothing.
    ///
    /// The later images of a pair are wanted later than the first and come out of the same call, so the first
    /// sets the delay and the rest are in time.
    ///
    /// Ring timestamps are arrival times, so with this delay the sample sweeps from `previous` to `next`
    /// as the gap fills in, and every phase in the bracket is reachable.
    static func interpolationDelay(captureInterval: CFTimeInterval, generationLatency: CFTimeInterval,
                                   steps: Int = 2) -> CFTimeInterval {
        (1 - firstImageStart(steps: steps)) * captureInterval + generationLatency
    }

    /// When the first image of a pair is first wanted on the screen: when the clock, `delay` behind real time, is
    /// `firstImageStart` of the way from `previous` to `next`. For a pair a capture interval long it is the delay's generation
    /// latency after `next` arrived; for a pair shorter than the interval, later than that, and for a longer one, sooner.
    static func firstImageWanted(previous: CFTimeInterval, next: CFTimeInterval, delay: CFTimeInterval, steps: Int) -> CFTimeInterval {
        previous + firstImageStart(steps: steps) * (next - previous) + delay
    }

    private static func interpolate<F: TimedFrame>(_ frames: [F], _ input: PlanningInput) -> PresentationPlan {
        let steps = InterpolationSteps.steps(for: input.multiplier)
        let targetTime = input.sampleTime - input.delay
        let (previousIndex, nextIndex) = bracket(frames, around: targetTime)
        let previous = frames[previousIndex]
        let next = frames[nextIndex]

        let duration = next.timestamp - previous.timestamp
        guard duration > 0 else { return .captured(previousIndex) }
        let position = min(max((targetTime - previous.timestamp) / duration, 0), 1)

        let step = Int((position * Double(steps)).rounded())
        if step <= 0 { return .captured(previousIndex) }
        // Nothing may be generated across a cut: the new shot is shown as soon as it is wanted.
        if step >= steps || next.isSceneCut { return .captured(nextIndex) }
        return .interpolated(previous: previousIndex, next: nextIndex, step: step, steps: steps)
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
