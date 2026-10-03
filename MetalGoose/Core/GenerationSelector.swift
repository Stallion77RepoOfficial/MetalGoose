import Foundation

/// What makes the images between two captures.
enum GenerationEngine: Sendable, Equatable {
    /// The Neural Engine, through VideoToolbox's low-latency frame interpolation. The GPU only converts pixel
    /// formats and blends the captures back in where they barely differ, which leaves it to the captured app. Makes the
    /// midpoint of each pair or its quarters, at the sizes of `NeuralSizes`.
    case neuralEngine
    /// MetalFX on the GPU, along the media engine's motion field. Not limited in size, but it makes only the
    /// midpoint and takes GPU time from whatever is being captured.
    case metalFX

    var title: LocalizedStringResource {
        switch self {
        case .neuralEngine: return "Neural Engine"
        case .metalFX:      return "GPU (MetalFX)"
        }
    }
}

/// An engine, and the images per capture interval it makes, 2 or 4; or neither, when nothing can make them in the
/// time there is, and every capture is shown as it arrives, with nothing held back.
struct GenerationChoice: Sendable, Equatable {
    var engine: GenerationEngine?
    var multiplier: Int
    /// The rung of the Neural Engine's ladder (`NeuralSizes`, finest first) that its session is wanted at. Present
    /// whenever the Neural Engine is wanted, also while another engine stands in for it until its session has started.
    var neuralRung: Int?
    /// The size the Neural Engine is working at, when it is the engine and that is not the size of the frames it was
    /// given; filled in by whoever owns the sessions.
    var neuralSize: NeuralSizes.Size?
    /// How long after a capture arrives the first image of its pair is expected to be ready: what the engine has been
    /// measured to take, or what its size predicts until it has been. The frame schedule is built on it.
    var latency: CFTimeInterval
    /// Nothing is made because the panel shows too few images in a capture interval for them to be seen: captures that come
    /// as fast as the panel refreshes have no room between them.
    var limitedByPanel = false

    /// Nothing is made, as before the first choice.
    static let nothing = GenerationChoice(engine: nil, multiplier: 1, neuralRung: nil, neuralSize: nil, latency: 0)

    /// The same engine making the same number of images, whatever the size or the estimate.
    func isSameWork(as other: GenerationChoice) -> Bool {
        engine == other.engine && multiplier == other.multiplier
    }
}

/// Which engine makes the images between captures, and how many, from what the window and the machine allow.
///
/// MGFG-1 interpolates, with two engines and an order of preference. The Neural Engine is the cheapest on the GPU, and
/// its images, with the captures blended back in where nothing moved, are good: it makes the midpoint or the quarters,
/// about three times as long for the quarters, at the largest size it takes or at a coarser one when that is more than
/// the rate leaves time for. MetalFX is the more faithful, takes any size, but makes only the midpoint and costs the GPU
/// what the captured app wants. Both hold the newest capture back by most of a capture interval, which at 30 captures a
/// second is some 40 to 50 ms. So, best first, for 4 images an interval:
///
///     Neural Engine making 4, Neural Engine making 2, MetalFX making 2, none
///
/// and for 2 the same without the first. None is a capture shown as it arrives, which is what is left when no engine
/// can make its images before the next capture is due.
///
/// The choice is made so that no engine is asked for more than it has time for — a pair that arrives while the last is
/// still being made goes by without its images — and so that it does not flap:
///
/// - The best engine that keeps up. A Neural Engine session is chosen by its size as well: the finest rung of the
///   ladder whose calls take their share of an interval, which a faster or slower capture rate moves up or down.
/// - An engine is used once it is ready. The Neural Engine's session for a size nobody has used before takes a second
///   or three to compile, and until it has started another engine makes the images — MetalFX, after a short wait that
///   spares it a start-up when the session was compiled before and takes a tenth of a second — and the Neural Engine takes
///   over the moment it is ready. A session that is serving while a better-sized one is built keeps serving while it is
///   not far behind.
/// - A multiplier the panel cannot show is not made: it shows about as many images as its refresh rate gives in a
///   capture interval, so 4 at 30 captures a second on 60 Hz is 2. Where it shows fewer than two — 60 captures a second on
///   60 Hz, 120 on 120 — there is no room between the captures for an image to be seen, and nothing is made or held back.
/// - An engine that stops keeping up is left at once, and not tried again for a while. A better one is taken only
///   when it would keep up with room to spare and the choice has stood for a while, so a rate near a limit does not
///   move the engine back and forth: every move shifts the schedule by what interpolation holds back.
/// - Until an engine has been timed, what it would take is predicted from the size, and held to the stricter share
///   that is asked to take it, not the looser one that is allowed to keep it.
/// - Taking more — four images, a finer size, a better engine — is judged by the shortest interval between captures of
///   the last few seconds, and keeping what is taken by the one now. A game that stutters for a second makes its captures
///   come further apart, and an option that fits only a rate it will be back from in a moment is not worth moving to.
///
/// The Neural Engine and the media engine are separate units and do not slow each other (measured), so nothing waits
/// on a unit that the engine chosen does not use: only what the chosen engine needs is run.
struct GenerationSelector {

    /// What a choice is made from.
    struct Inputs {
        /// The multiplier asked for.
        var requested: Int
        /// The smoothed time between captures; 0 until it has been measured.
        var captureInterval: CFTimeInterval
        /// How short the intervals between captures have been, a tenth of them being shorter (`IntervalSpread`); 0 until that
        /// has been measured.
        var shortestInterval: CFTimeInterval = 0
        /// The panel's, in Hz; 0 when it is not known.
        var refreshRate: Int
        /// The pixels of the frames MetalFX works on.
        var framePixels: Int
        /// The pixels of each size the Neural Engine can work at for these frames, finest first; empty when it cannot
        /// be used at all — it has failed, or the frames are not ones it takes.
        var neuralRungs: [Int]
        /// The rung whose session has started and is taking frames, if any.
        var neuralActive: Int?
        /// How long a call at the active rung takes for the midpoint and for the three quarters; 0 until they have been
        /// timed.
        var neuralMidpointTime: CFTimeInterval
        var neuralQuartersTime: CFTimeInterval
        /// How long after a capture arrives its pair's first image has been ready, for each engine; 0 until it has been
        /// measured, or when it is too old to be.
        var neuralLatency: CFTimeInterval
        var metalFXLatency: CFTimeInterval
    }

    // MARK: Numbers

    /// What a call for the midpoint costs by size, with the Neural Engine idle between calls as it is between captures, in
    /// the pipeline at 30 captures a second: 3.5 ms at 640x360, 5 at 960x540, 7 at 1280x720 — and then the call gets dearer by
    /// a step, to 13 at 1312x738, 17 at 1440x810 and 18 at 1920x1080. A size of up to 0.92 megapixels is one side of the
    /// step (`NeuralSizes`). Stands in until the engine has timed a call, and says how a measurement at one size carries to
    /// another; a machine that is busier than this one was measured on scales it up, and what is measured takes its place.
    static func predictedNeuralMidpoint(pixels: Int) -> CFTimeInterval {
        pixels <= 921_600 ? 0.0020 + 6.2e-9 * Double(pixels) : 0.0100 + 3.6e-9 * Double(pixels)
    }

    /// What the GPU adds either side of the call: the conversion of the capture, which has to be on the GPU and back
    /// before the call can start, and the hops between the threads that take it there.
    static func predictedNeuralOverhead(pixels: Int) -> CFTimeInterval { 0.0022 + 0.3e-9 * Double(pixels) }

    /// From a capture arriving to MetalFX's image of its pair being ready, by size: 7.6 to 9.9 ms at 1280x720 and 12 to
    /// 13 ms at 1920x1080, with the media engine's field and the interpolator both in it.
    static func predictedMetalFX(pixels: Int) -> CFTimeInterval { 0.0034 + 4.4e-9 * Double(pixels) }

    /// The share of a capture interval that a call for the midpoint may take, to be taken and to be kept. A pair
    /// that misses loses one image, where one that misses at four steps loses three, so these are looser than the
    /// shares for the quarters (`InterpolationSteps`).
    private static let midpointEnterShare = 0.7
    private static let midpointKeepShare = 0.85

    /// The same for MetalFX, whose time is the whole way from the capture to its image.
    private static let metalFXEnterShare = 0.65
    private static let metalFXKeepShare = 0.8

    /// How much of an interval a session that is being replaced by a better-sized one may take and still go on
    /// serving until that one is ready: it misses pairs, but fewer than an engine that has to be started would.
    private static let limpShare = 1.0

    /// A finer rung is taken when its call is expected to need this much less of the interval than a rung is taken at.
    private static let climbMargin = 0.15

    /// How a call compares with the shortest intervals between captures (`IntervalSpread`), to be taken and to be kept: a pair
    /// that arrives sooner than a call takes finds the engine still on the last, and goes by. A 60 fps source on a 120 Hz
    /// panel comes at 8.7 ms or less a tenth of the time; measured at the same settings, a call of 10.9 ms (1.25 times that)
    /// left 72 to 80 pairs in 100 their images, and one of 6.7 ms (0.77) 96 to 98, with 17 points better pacing and 4 ms
    /// less latency. A source that is steady gives the share of the interval above all the room it needs.
    private static let closeEnter = 0.85
    private static let closeKeep = 1.15

    /// How long a choice stands before a better one may replace it, how long an engine that could not keep up is left
    /// alone, and how long MetalFX waits for a Neural Engine session that has not started before standing in for it.
    private static let dwell: CFTimeInterval = 10
    private static let block: CFTimeInterval = 30
    private static let grace: CFTimeInterval = 0.4

    /// How long a choice of nothing stands before an engine replaces it: nothing is held back while nothing is made, so
    /// the picture is the captures' own, and getting images between them is worth more than the quiet of a long wait.
    private static let idleDwell: CFTimeInterval = 5

    /// A multiplier of 4 is made while the panel can show about that many images in a capture interval, and kept down
    /// to a little less.
    private static let enterSlack = 0.4
    private static let keepSlack = 0.7

    /// Images the panel shows in a capture interval for two to be worth making, to be taken and to be kept: below these the
    /// generated image would be on the screen for less than a refresh, or for none.
    private static let enterImages = 1.7
    private static let keepImages = 1.4

    // MARK: State

    private(set) var choice = GenerationChoice.nothing
    private var decided = false
    private var changedAt: CFTimeInterval = 0
    private var startedAt: CFTimeInterval?
    private var neuralBlockedUntil: CFTimeInterval = 0
    private var metalFXBlockedUntil: CFTimeInterval = 0
    private var neuralRung: Int?
    private var neuralRungChangedAt: CFTimeInterval = 0
    private var asked: Int?

    /// The smoothed interval between captures as it has been over the last few seconds, for `enteringInterval`.
    private var recentIntervals: [(time: CFTimeInterval, interval: CFTimeInterval)] = []
    private static let recentSpan: CFTimeInterval = 4

    /// The choice stands in for the Neural Engine, whose session has not started: anything better replaces it as soon as
    /// it can be had, without waiting for the choice to have stood.
    private var waitingForNeuralEngine = false

    /// The choice is nothing because the panel cannot show more; an engine replaces it the moment it can.
    private var holdingForPanel = false

    /// What each rung took the last time it was serving, when, for the midpoint and for the quarters. A size that was left
    /// because its calls took too long is not taken again on a prediction, or on a quick moment at another size: the prediction
    /// for 1238x742 was 7.7 ms where it measured 10.9, and the selector climbed back to it from 928x556 every dozen seconds
    /// on a call of 4.4 ms that the smaller size had once.
    private var timedRungs: [Int: (at: CFTimeInterval, midpoint: CFTimeInterval, quarters: CFTimeInterval)] = [:]
    private static let rungMemory: CFTimeInterval = 120

    /// The time of the choice being made, for what is remembered of the rungs.
    private var clock: CFTimeInterval = 0

    private struct Candidate: Equatable {
        var engine: GenerationEngine?
        var steps: Int
    }

    /// Forgets what has been chosen and what has been blocked, so that the next choice is made as if from the start.
    mutating func reset() {
        decided = false
        changedAt = 0
        startedAt = nil
        neuralBlockedUntil = 0
        metalFXBlockedUntil = 0
        neuralRung = nil
        neuralRungChangedAt = 0
        asked = nil
        waitingForNeuralEngine = false
        holdingForPanel = false
        recentIntervals.removeAll()
        timedRungs.removeAll()
    }

    // MARK: Choosing

    mutating func choose(_ i: Inputs, now: CFTimeInterval) -> GenerationChoice {
        // Something the user changed is a new question, not a drift to be smoothed over.
        let requested = InterpolationSteps.steps(for: i.requested)
        if requested != asked {
            let kept = (choice, timedRungs)
            reset()
            choice = kept.0
            timedRungs = kept.1
            asked = requested
        }
        if startedAt == nil { startedAt = now }
        clock = now
        noteInterval(i.captureInterval, now: now)
        if let active = i.neuralActive, i.neuralMidpointTime > 0 || i.neuralQuartersTime > 0 {
            timedRungs[active] = (now, neuralMeasured(steps: InterpolationSteps.halves, i),
                                  neuralMeasured(steps: InterpolationSteps.quarters, i))
        }

        let steps = deliverable(i, requested: requested)
        let panelLimited = limitedByPanel(i)
        let candidates = panelLimited ? [Candidate(engine: nil, steps: 1)] : Self.candidates(making: steps)

        // What the Neural Engine is prepared for: the size its most ambitious candidate that has one wants.
        var target: (steps: Int, rung: Int)?
        for candidate in candidates where candidate.engine == .neuralEngine {
            if let rung = neuralRung(for: candidate.steps, i, now: now) {
                target = (candidate.steps, rung)
                break
            }
        }
        // Wanted at a size that has not started: another engine stands in for it, if it is not serving.
        let pending = target != nil && i.neuralActive != target?.rung

        let best = candidates.first { viable($0, keeping: false, target: target?.rung, pending: pending, i, now: now) }
            ?? Candidate(engine: nil, steps: 1)

        guard decided else {
            holdingForPanel = panelLimited
            return adopt(best, target: target?.rung, waiting: pending && best != candidates[0], i, now: now, limited: panelLimited)
        }

        let current = Candidate(engine: choice.engine, steps: choice.multiplier)
        if let rank = candidates.firstIndex(of: current),
           viable(current, keeping: true, target: target?.rung, pending: pending, i, now: now) {
            // Still keeping up. A better engine waits until it would keep up with room to spare and this one has stood
            // for a while — unless this one is only standing in for the Neural Engine until it is ready, or is nothing
            // because the panel could not show more, which it can now.
            if let better = candidates.firstIndex(of: best), better < rank,
               now - changedAt >= (current.engine == nil ? Self.idleDwell : Self.dwell) || waitingForNeuralEngine || holdingForPanel {
                holdingForPanel = false
                return adopt(best, target: target?.rung, waiting: pending && best != candidates[0], i, now: now)
            }
            waitingForNeuralEngine = waitingForNeuralEngine && pending
            return refreshed(target: target?.rung, i, now: now)
        }

        // Not keeping up, or not the work asked for any more: leave now. What was timed and could not keep up is left
        // alone for a while; what was only predicted not to is simply not taken. The Neural Engine is left alone only
        // where no size of its ladder would keep up: where one would, it is being built, and another engine stands in.
        if !panelLimited, current.engine == .neuralEngine, best.engine != .neuralEngine, target == nil, i.neuralActive != nil,
           neuralMeasured(steps: current.steps, i) > 0 {
            neuralBlockedUntil = now + Self.block
        }
        if !panelLimited, current.engine == .metalFX, best.engine != .metalFX, i.metalFXLatency > 0 {
            metalFXBlockedUntil = now + Self.block
        }
        holdingForPanel = panelLimited
        return adopt(best, target: target?.rung, waiting: pending && best != candidates[0], i, now: now, limited: panelLimited)
    }

    /// The candidates for `steps` images an interval, best first. None is always there at the end.
    private static func candidates(making steps: Int) -> [Candidate] {
        let none = Candidate(engine: nil, steps: 1)
        if steps == InterpolationSteps.quarters {
            return [Candidate(engine: .neuralEngine, steps: InterpolationSteps.quarters),
                    Candidate(engine: .neuralEngine, steps: InterpolationSteps.halves),
                    Candidate(engine: .metalFX, steps: InterpolationSteps.halves), none]
        }
        return [Candidate(engine: .neuralEngine, steps: InterpolationSteps.halves),
                Candidate(engine: .metalFX, steps: InterpolationSteps.halves), none]
    }

    /// Whether the panel shows too few images in a capture interval for generated ones to be seen: taken at a stricter share
    /// than it is kept, and judged by the shortest recent interval to be taken (a stutter makes no room that will last).
    private func limitedByPanel(_ i: Inputs) -> Bool {
        guard i.refreshRate > 0, i.captureInterval > 0 else { return false }
        let keeping = decided && choice.engine != nil
        let images = Double(i.refreshRate) * (keeping ? i.captureInterval : enteringInterval(i))
        return images < (keeping ? Self.keepImages : Self.enterImages)
    }

    /// The steps asked for, or 2 where the panel cannot show four images in a capture interval.
    private func deliverable(_ i: Inputs, requested: Int) -> Int {
        guard requested == InterpolationSteps.quarters, i.refreshRate > 0, i.captureInterval > 0 else { return requested }
        let keeping = decided && choice.multiplier == InterpolationSteps.quarters
        let images = Double(i.refreshRate) * (keeping ? i.captureInterval : enteringInterval(i))
        let slack = keeping ? Self.keepSlack : Self.enterSlack
        return images >= Double(InterpolationSteps.quarters) - slack ? InterpolationSteps.quarters : InterpolationSteps.halves
    }

    // MARK: The interval

    private mutating func noteInterval(_ interval: CFTimeInterval, now: CFTimeInterval) {
        guard interval > 0 else { return }
        recentIntervals.removeAll { now - $0.time > Self.recentSpan }
        // Ten samples a second are plenty: the interval is smoothed over half a second already.
        if recentIntervals.last.map({ now - $0.time >= 0.1 }) ?? true { recentIntervals.append((now, interval)) }
    }

    /// The interval to judge taking more by: the shortest of the last few seconds, or the one now if that is shorter.
    private func enteringInterval(_ i: Inputs) -> CFTimeInterval {
        guard i.captureInterval > 0 else { return i.captureInterval }
        return min(i.captureInterval, recentIntervals.map(\.interval).min() ?? i.captureInterval)
    }

    // MARK: The Neural Engine's sizes

    private static func shares(steps: Int) -> (enter: Double, keep: Double) {
        steps == InterpolationSteps.quarters
            ? (InterpolationSteps.enterShare, InterpolationSteps.keepShare)
            : (midpointEnterShare, midpointKeepShare)
    }

    /// What the active session has been seen to take for `steps`, or 0 where it has not been timed. Of the two figures
    /// one is made from the other where only that one has been seen.
    private func neuralMeasured(steps: Int, _ i: Inputs) -> CFTimeInterval {
        if steps == InterpolationSteps.quarters {
            return i.neuralQuartersTime > 0 ? i.neuralQuartersTime : i.neuralMidpointTime * InterpolationSteps.quartersCostRatio
        }
        return i.neuralMidpointTime > 0 ? i.neuralMidpointTime : i.neuralQuartersTime / InterpolationSteps.quartersCostRatio
    }

    /// How long a call for `steps` takes at `rung`: what the active session has been seen to take, and for another rung
    /// what it took the last time it served, or else the prediction for its size scaled by how the active session compares
    /// with its own. Scaled up, never down: a machine that is busier than the one the prediction was measured on is what the
    /// scale is for, and a quick moment of the active session (4.4 ms against a usual 6.7) is not a quicker machine; taken for
    /// one it had the selector climb to a size that then took 10.9 ms.
    private func neuralTime(rung: Int, steps: Int, _ i: Inputs) -> CFTimeInterval {
        func predicted(_ rung: Int) -> CFTimeInterval {
            let midpoint = Self.predictedNeuralMidpoint(pixels: i.neuralRungs[rung])
            return steps == InterpolationSteps.quarters ? midpoint * InterpolationSteps.quartersCostRatio : midpoint
        }
        let measured = neuralMeasured(steps: steps, i)
        guard let active = i.neuralActive, active < i.neuralRungs.count, measured > 0 else { return predicted(rung) }
        if rung == active { return measured }
        if let seen = timedRungs[rung], clock - seen.at <= Self.rungMemory {
            return steps == InterpolationSteps.quarters ? seen.quarters : seen.midpoint
        }
        return predicted(rung) * min(3, max(1, measured / predicted(active)))
    }

    /// The rung of the ladder to run `steps` at: the finest whose call takes its share of the interval, the rung already
    /// wanted kept while it does, and a finer one taken when it would need clearly less and the rung has stood for a while.
    /// Nil where none does, or the Neural Engine is not to be used.
    private func neuralRung(for steps: Int, _ i: Inputs, now: CFTimeInterval) -> Int? {
        let count = i.neuralRungs.count
        guard count > 0, now >= neuralBlockedUntil else { return nil }
        guard i.captureInterval > 0 else { return min(neuralRung ?? i.neuralActive ?? 0, count - 1) }

        let (enter, keep) = Self.shares(steps: steps)
        let entering = enteringInterval(i)
        func fits(_ rung: Int, _ share: Double, within interval: CFTimeInterval) -> Bool {
            neuralTime(rung: rung, steps: steps, i) <= share * interval
        }
        // Whether a call at this rung is over before the closest captures of the last seconds bring the next pair.
        func clear(_ rung: Int, _ factor: Double) -> Bool {
            i.shortestInterval <= 0 || neuralTime(rung: rung, steps: steps, i) <= factor * i.shortestInterval
        }
        // The finest rung from `start` that fits its share and clears the closest captures; where none clears them, the coarsest
        // that fits, which leaves the fewest pairs to go by.
        func finest(from start: Int) -> Int? {
            let fitting = (start..<count).filter { fits($0, enter, within: entering) }
            return fitting.first { clear($0, Self.closeEnter) } ?? fitting.last
        }

        guard let wanted = neuralRung, wanted < count else { return finest(from: 0) }
        // The looser share to keep applies only to a session that has been timed, and is judged by the interval now.
        let timed = wanted == i.neuralActive && neuralMeasured(steps: steps, i) > 0
        if !fits(wanted, timed ? keep : enter, within: timed ? i.captureInterval : entering) { return finest(from: wanted + 1) }
        // It keeps up on the whole, but the pairs that follow closely find it still busy: a coarser rung that leaves them room.
        if !clear(wanted, timed ? Self.closeKeep : Self.closeEnter), let coarser = finest(from: wanted + 1) { return coarser }
        if wanted > 0, now - neuralRungChangedAt >= Self.dwell,
           fits(wanted - 1, enter - Self.climbMargin, within: entering), clear(wanted - 1, Self.closeEnter) { return wanted - 1 }
        return wanted
    }

    // MARK: Whether an engine keeps up

    /// Whether `candidate` can make its images in the time there is: to be taken, or, when it is already the choice, to be
    /// kept. The looser share to keep applies only to what has been timed.
    private func viable(_ candidate: Candidate, keeping: Bool, target: Int?, pending: Bool, _ i: Inputs,
                        now: CFTimeInterval) -> Bool {
        switch candidate.engine {
        case nil:
            return true

        case .neuralEngine?:
            guard let active = i.neuralActive, active < i.neuralRungs.count,
                  neuralRung(for: candidate.steps, i, now: now) != nil else { return false }
            guard i.captureInterval > 0 else { return true }
            let (enter, keep) = Self.shares(steps: candidate.steps)
            var share = keeping && neuralMeasured(steps: candidate.steps, i) > 0 ? keep : enter
            // A better-sized session is being built: this one goes on while it is not far behind.
            if let target, target != active { share = max(share, Self.limpShare) }
            return neuralTime(rung: active, steps: candidate.steps, i) <= share * (keeping ? i.captureInterval : enteringInterval(i))

        case .metalFX?:
            guard now >= metalFXBlockedUntil else { return false }
            // The Neural Engine is about to be ready, probably: not worth a start of MetalFX, which takes three pairs to begin.
            if pending, i.neuralActive == nil, !decided || waitingForNeuralEngine, let startedAt, now - startedAt < Self.grace {
                return false
            }
            guard i.captureInterval > 0 else { return true }
            let measured = i.metalFXLatency > 0
            let time = measured ? i.metalFXLatency : Self.predictedMetalFX(pixels: i.framePixels)
            let share = keeping && measured ? Self.metalFXKeepShare : Self.metalFXEnterShare
            return time <= share * (keeping ? i.captureInterval : enteringInterval(i))
        }
    }

    // MARK: Adopting

    /// The choice as it stands, with the size and the latency estimate brought up to date.
    private mutating func refreshed(target: Int?, _ i: Inputs, now: CFTimeInterval) -> GenerationChoice {
        let candidate = Candidate(engine: choice.engine, steps: choice.multiplier)
        return adopt(candidate, target: target, waiting: waitingForNeuralEngine, i, now: now, limited: choice.limitedByPanel)
    }

    private mutating func adopt(_ candidate: Candidate, target: Int?, waiting: Bool, _ i: Inputs,
                                now: CFTimeInterval, limited: Bool = false) -> GenerationChoice {
        if target != neuralRung { neuralRungChangedAt = now }
        neuralRung = target

        let next = GenerationChoice(engine: candidate.engine, multiplier: candidate.steps, neuralRung: target,
                                    neuralSize: nil, latency: latency(of: candidate, target: target, i), limitedByPanel: limited)
        if !decided || !next.isSameWork(as: choice) { changedAt = now }
        decided = true
        choice = next
        waitingForNeuralEngine = waiting
        return next
    }

    /// How long after a capture arrives the first image of its pair is expected to be ready.
    private func latency(of candidate: Candidate, target: Int?, _ i: Inputs) -> CFTimeInterval {
        switch candidate.engine {
        case nil:
            return 0
        case .neuralEngine?:
            if i.neuralLatency > 0 { return i.neuralLatency }
            guard let rung = i.neuralActive ?? target, rung < i.neuralRungs.count else { return 0 }
            return neuralTime(rung: rung, steps: candidate.steps, i) + Self.predictedNeuralOverhead(pixels: i.neuralRungs[rung])
        case .metalFX?:
            return i.metalFXLatency > 0 ? i.metalFXLatency : Self.predictedMetalFX(pixels: i.framePixels)
        }
    }
}
