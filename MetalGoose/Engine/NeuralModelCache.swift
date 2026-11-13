import Foundation
@preconcurrency import VideoToolbox
import QuartzCore
import os

/// Builds the Neural Engine's models before a session needs them.
///
/// A frame-interpolation session builds the model for its size when it starts: 0.5 to 1.9 s for a size this app has not had
/// built, 0.03 to 0.1 s for one it has. The system keeps the models it built for an app, by the app's signing identifier, so
/// from one launch and one build of the app to the next (after 24 new sizes in a row, the nine built before them that were
/// checked were all still kept); but every size has its own, and what the system keeps is up to it. A session waiting for its
/// model leaves MetalFX making the midpoint alone: measured with captures at 29 a second on a 120 Hz panel, a session at sizes
/// not built before began as MetalFX 2x and reached the Neural Engine's 4x at 2.5 and 2.85 s.
///
/// So the models are built ahead of time. As the app launches: all ten shapes VideoProcessing ships the network in
/// (`NeuralSizes.shipped`), the ladders' 16:9 and 4:3 rungs and the larger shapes first, and the sizes of the user's own
/// frames used before (remembered sizes that had not been built, built at launch, gave a session started 6 s later its 4x at
/// 0.25 s). And every rung of a frame size's ladder once frames of that size arrive, after the rung a session is being
/// started at, so that moving to another rung does not wait; while one is being built, `GenerationSelector` runs a rung
/// that is. Building one is starting a session at that size and ending it. One is built at a time, in the background, and
/// none while a session is being started for real: two sizes built at once took about twice as long each, while a session
/// started at the size being built here waits for that build alone (two starts of one new size took 1.69 s together, one
/// 1.63 s).
enum NeuralModelCache {

    /// How many sizes of the user's frames are remembered for the next launch, most recent first, besides the shipped shapes,
    /// which are always built: the rungs of the last few frame sizes that are not shipped shapes.
    private static let rememberedCount = 8
    private static let rememberedKey = "NeuralEngineModelSizes"

    /// How long the other rungs wait for the one a session is being started at, before they are built anyway.
    private static let servingWait: CFTimeInterval = 10

    private struct State {
        var queue: [NeuralSizes.Size] = []
        /// Built, or started by a session, in this run.
        var built: Set<NeuralSizes.Size> = []
        var working = false
        var startingSessions = 0
        var awaited: NeuralSizes.Size?
        var awaitedUntil: CFTimeInterval = 0
    }

    private static let state = OSAllocatedUnfairLock(initialState: State())
    private static let worker = DispatchQueue(label: "com.metalgoose.neural.models", qos: .utility)

    /// The configuration every session is started with, here and in `NeuralInterpolator`: a model is built for a
    /// configuration, not only a size.
    static func configuration(for size: NeuralSizes.Size) -> VTLowLatencyFrameInterpolationConfiguration? {
        // Configured for the quarters, which a call for the midpoint alone takes as well and at the same cost,
        // so that going from two steps to four does not mean a new session.
        VTLowLatencyFrameInterpolationConfiguration(frameWidth: size.width, frameHeight: size.height, numberOfInterpolatedFrames: 2)
    }

    /// Builds the models of all ten shipped shapes and of the sizes used before. At launch.
    static func buildAtLaunch() {
        guard let limits = NeuralInterpolator.limits else { return }
        let rungShapes = NeuralSizes.shipped.filter { $0.width >= 832 || $0.height >= 832 }
        let sizes = rungShapes + remembered() + NeuralSizes.shipped.filter { !rungShapes.contains($0) }
        enqueue(sizes.filter { NeuralSizes.accepts(width: $0.width, height: $0.height, limits: limits) }, after: nil)
    }

    /// Whether a session at this size starts at once: its model has been built, or a session started at it, in this run.
    static func isBuilt(_ size: NeuralSizes.Size) -> Bool {
        state.withLock { $0.built.contains(size) }
    }

    /// Builds the models of a frame size's ladder: `serving`, the rung a session is being started at, by that session, and
    /// the others after it. The ladder is remembered for the next launch.
    static func prepare(_ ladder: [NeuralSizes.Size], serving: NeuralSizes.Size?) {
        remember(ladder)
        enqueue(ladder.filter { $0 != serving }, after: serving)
    }

    /// A session's start, which builds the model unless it has been built, with nothing else being built meanwhile.
    static func starting<T>(_ size: NeuralSizes.Size, _ start: () throws -> T) rethrows -> T {
        state.withLock { $0.startingSessions += 1 }
        defer { state.withLock { $0.startingSessions -= 1 } }
        let result = try start()
        state.withLock { _ = $0.built.insert(size) }
        return result
    }

    // MARK: - Building

    private static func enqueue(_ sizes: [NeuralSizes.Size], after serving: NeuralSizes.Size?) {
        let starts = state.withLock { state -> Bool in
            let fresh = sizes.filter { !state.built.contains($0) }
            state.queue = fresh + state.queue.filter { !fresh.contains($0) }
            if let serving {
                state.awaited = serving
                state.awaitedUntil = CACurrentMediaTime() + servingWait
            }
            guard !state.working, !state.queue.isEmpty else { return false }
            state.working = true
            return true
        }
        if starts { worker.async { while let size = next() { build(size) } } }
    }

    /// The next size to build, once no session is being started and the rung one is being started at has been; nil, which
    /// ends the worker, where there is none.
    private static func next() -> NeuralSizes.Size? {
        while true {
            let (size, waits) = state.withLock { state -> (NeuralSizes.Size?, Bool) in
                if state.startingSessions > 0 { return (nil, true) }
                if let awaited = state.awaited {
                    if !state.built.contains(awaited), CACurrentMediaTime() < state.awaitedUntil { return (nil, true) }
                    state.awaited = nil
                }
                while !state.queue.isEmpty {
                    let size = state.queue.removeFirst()
                    if !state.built.contains(size) { return (size, false) }
                }
                state.working = false
                return (nil, false)
            }
            guard waits else { return size }
            Thread.sleep(forTimeInterval: 0.05)
        }
    }

    /// Starts a session at the size and ends it: only a successful start marks the model as built.
    private static func build(_ size: NeuralSizes.Size) {
        guard let configuration = configuration(for: size) else { return }
        let session = VTFrameProcessor()
        do { try session.startSession(configuration: configuration) } catch { return }
        state.withLock { _ = $0.built.insert(size) }
        session.endSession()
    }

    // MARK: - Remembering

    private static func remembered() -> [NeuralSizes.Size] {
        let pairs = UserDefaults.standard.array(forKey: rememberedKey) as? [[Int]] ?? []
        return pairs.compactMap { $0.count == 2 ? NeuralSizes.Size(width: $0[0], height: $0[1]) : nil }
    }

    private static func remember(_ ladder: [NeuralSizes.Size]) {
        let own = ladder.filter { !NeuralSizes.shipped.contains($0) }
        let sizes = own + remembered().filter { !own.contains($0) }
        UserDefaults.standard.set(sizes.prefix(rememberedCount).map { [$0.width, $0.height] }, forKey: rememberedKey)
    }
}
