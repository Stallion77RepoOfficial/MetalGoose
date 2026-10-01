import Foundation
@preconcurrency import Metal
import QuartzCore

/// Turns captured frames into what the overlay shows.
///
/// Two pipelines do the work and share almost nothing. The capture pipeline processes each frame
/// as it arrives, on a serial queue of its own; the render pipeline answers each display callback,
/// on a thread of its own. Frames pass between them through a small ring, and everything else they
/// both touch is a handful of locked values (`EngineShared`). This type owns both, wires them to
/// the outside world, and is the only thing the UI talks to.
final class GooseEngine: @unchecked Sendable {

    /// Capture command buffers that may be on the GPU at once, which is also the deepest a drawable
    /// queue goes: past it, frames wait longer than they are worth.
    static let maxInFlight = 3

    let deviceName: String

    private let shared: EngineShared
    private let capture: CapturePipeline

    /// Exists while the overlay does. Main thread only.
    private struct Presentation {
        let pipeline: RenderPipeline
        let driver: RenderDriver
        let layer: CAMetalLayer
    }
    private var presentation: Presentation?

    // MARK: - Creation

    static func make(libraryURL: URL? = nil) -> Result<GooseEngine, MGError> {
        GPUContext.make(libraryURL: libraryURL).map(GooseEngine.init)
    }

    private init(gpu: GPUContext) {
        deviceName = gpu.device.name
        shared = EngineShared(gpu: gpu)
        capture = CapturePipeline(shared: shared)
    }

    // MARK: - Observation

    var stats: PipelineStats { shared.stats.withLock { $0 } }

    /// What is producing midpoint frames right now. The setting is a preference — a window too large for
    /// the Neural Engine, or a processor that failed, is interpolated by MetalFX whatever it says.
    var activeInterpolationEngine: InterpolationEngine { shared.interpolationBackend.withLock { $0 } }

    /// The next error raised off the main thread, if any.
    func takeError() -> MGError? { shared.errors.take() }

    // MARK: - Settings

    /// Applies a new configuration. Only the changes that alter what the pipelines hold rebuild
    /// anything; Scale Factor lives in the overlay's geometry and the sharpening profile is read per
    /// frame, so neither needs more than the new value.
    @MainActor
    func apply(_ config: EngineConfig) {
        let previous = shared.config.withLock { current -> EngineConfig in
            defer { current = config }
            return current
        }

        // The stages that change which textures the capture path writes need the pool rebuilt.
        if config.upscaling != previous.upscaling || config.antiAliasing != previous.antiAliasing {
            capture.reset(clearFrames: true)
        } else if config.motionSource != previous.motionSource {
            capture.reset(clearFrames: false)
        }

        if config.bufferDepth != previous.bufferDepth {
            capture.applyBufferDepth(config.bufferDepth)
        }
        if let presentation, config.vsync != previous.vsync || config.bufferDepth != previous.bufferDepth {
            presentation.driver.configure(layer: presentation.layer, config: config)
        }

        // The cumulative counters describe one mode's behaviour. Carrying them across a mode switch
        // mixes two schedules into one set of totals, and the generated/passthrough split stops
        // meaning anything. A multiplier change reshapes the same split, so it invalidates the
        // totals exactly the way a mode change does.
        if config.frameGeneration != previous.frameGeneration || config.multiplier != previous.multiplier
            || config.interpolationEngine != previous.interpolationEngine {
            shared.stats.withLock {
                $0.resetCumulativeCounters()
                $0.frameCount = 0
            }
            shared.renderResetRequested.withLock { $0 = true }
        }
    }

    // MARK: - A capture session

    /// Clears what the previous session left behind.
    func beginSession() {
        // A session that ended with the overlay hidden must not begin with it still hidden.
        shared.isPresenting.withLock { $0 = true }
        capture.reset(clearFrames: true)
        shared.errors.reset()
        shared.captureInterval.withLock { $0.reset() }
        shared.stats.withLock { stats in
            // The panel does not change between sessions.
            let refresh = stats.screenRefreshRate
            let variable = stats.isProMotion
            stats = PipelineStats()
            stats.screenRefreshRate = refresh
            stats.isProMotion = variable
        }
        capture.applyBufferDepth(shared.config.withLock { $0.bufferDepth })
        capture.resetTiming()
        shared.renderResetRequested.withLock { $0 = true }
    }

    /// Drops what the session held. The ring keeps one texture from the pool for every entry, so
    /// leaving them there would keep the whole pool alive until the next capture overwrote it.
    func endSession() {
        capture.reset(clearFrames: true)
        capture.resetTiming()
    }

    /// The entry point for captured frames, from whichever thread they arrive on.
    func receive(_ frame: CapturedFrame) {
        capture.receive(frame)
    }

    // MARK: - Presentation

    /// Starts presenting into `layer`, which belongs to the overlay.
    @MainActor
    func attach(to layer: CAMetalLayer, displayRate: DisplayRate) {
        detach()
        shared.stats.withLock {
            $0.screenRefreshRate = displayRate.maximum
            $0.isProMotion = displayRate.isVariable
        }
        let pipeline = RenderPipeline(shared: shared, motion: capture.motion)
        let driver = RenderDriver(shared: shared, pipeline: pipeline, displayRate: displayRate)
        presentation = Presentation(pipeline: pipeline, driver: driver, layer: layer)
        driver.start(layer: layer)
    }

    @MainActor
    func detach() {
        guard let presentation else { return }
        presentation.driver.stop()
        self.presentation = nil
    }

    /// Stops the pipeline while the overlay is hidden: there is nowhere to present into, and leaving
    /// the callbacks or the capture path running only keeps producing frames nobody can see.
    @MainActor
    func setPresenting(_ presenting: Bool) {
        capture.setPresenting(presenting)
        presentation?.driver.setPaused(!presenting)
        if presenting { presentation?.pipeline.requestRedraw() }
    }

    /// The overlay changed shape, so the drawable has to be filled again even though no new image
    /// has arrived.
    @MainActor
    func requestRedraw() {
        presentation?.pipeline.requestRedraw()
    }
}
