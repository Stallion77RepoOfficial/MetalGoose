import Foundation
@preconcurrency import Metal
import os

/// What the capture path and the render path both read and write, each piece behind its own
/// lock. Everything else in either pipeline belongs to exactly one thread and is touched by no
/// other — which is why those pieces need no lock at all.
final class EngineShared: @unchecked Sendable {
    let gpu: GPUContext
    let ring = FrameRing()
    let errors = ErrorLog()
    let stats = OSAllocatedUnfairLock(initialState: PipelineStats())
    let config = OSAllocatedUnfairLock(initialState: EngineConfig())

    /// Smoothed interval between captures. Written on the processing queue and read by the render
    /// thread for every frame — the frame schedule is built on it — so it cannot be a bare property.
    let captureInterval = OSAllocatedUnfairLock(initialState: IntervalFilter())

    /// How close together captures come (`IntervalSpread`): the engine has to keep up with the short intervals, not the average.
    let captureSpread = OSAllocatedUnfairLock(initialState: IntervalSpread())

    /// The processing queue asks the render thread to drop what it holds. The render thread does it at the top of its
    /// next frame, so a texture is never created on one thread and released on another mid-encode.
    let renderResetRequested = OSAllocatedUnfairLock(initialState: false)

    /// Whether the overlay is on screen. While it is not, nothing can be seen, so the capture path
    /// holds the newest frame and does no work on it.
    let isPresenting = OSAllocatedUnfairLock(initialState: true)

    /// What is producing the in-between images right now, and how many a capture interval carries. The capture
    /// path decides, per frame, with the settings, which sizes the Neural Engine can work at, and how long each engine
    /// takes (`GenerationSelector`); the render thread plans with it.
    let generation = OSAllocatedUnfairLock(initialState: GenerationChoice.nothing)

    let neural: NeuralInterpolator
    let metalFX: MetalFXInterpolator

    /// The images both engines make, which the render thread picks from.
    let images = GeneratedImages()

    /// How long after a capture arrives the first image of its pair is ready, for each engine. The interpolation
    /// schedule sits that far behind the newest capture.
    let neuralLatency = GenerationLatency()
    let metalFXLatency = GenerationLatency()

    /// The engine's only tuned number. Capture rate, frame schedule and rate preference are all
    /// first-order filters, and each derives its own coefficient from this window and the rate it
    /// is actually running at — so none of them carries a constant that assumes a particular
    /// display or capture speed.
    static let measurementWindow: Double = 0.5

    init(gpu: GPUContext) {
        self.gpu = gpu
        neural = NeuralInterpolator(gpu: gpu, errors: errors, latency: neuralLatency, images: images)
        metalFX = MetalFXInterpolator(gpu: gpu, errors: errors, latency: metalFXLatency, images: images)
    }
}
