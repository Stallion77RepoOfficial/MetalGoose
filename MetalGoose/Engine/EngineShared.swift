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

    /// The processing queue asks the render thread to drop the extrapolation state it owns. The render
    /// thread does it at the top of its next frame, so a texture is never created on one thread and
    /// released on another mid-encode.
    let renderResetRequested = OSAllocatedUnfairLock(initialState: false)

    /// Whether the overlay is on screen. While it is not, nothing can be seen, so the capture path
    /// holds the newest frame and does no work on it.
    let isPresenting = OSAllocatedUnfairLock(initialState: true)

    /// What is producing the in-between images right now. The capture path decides, per frame, from whether
    /// the Neural Engine can take this size and has not failed; the render thread reads it to know where to
    /// look for the result.
    let interpolationBackend = OSAllocatedUnfairLock(initialState: InterpolationEngine.metalFX)
    let neural: NeuralInterpolator
    let metalFX: MetalFXInterpolator

    /// The steps each pair is cut into right now, 2 or 4. The capture path decides with the backend: MetalFX
    /// makes the midpoint alone, and the Neural Engine makes quarters only while it can keep up. The render
    /// thread plans with it.
    let interpolationSteps = OSAllocatedUnfairLock(initialState: InterpolationSteps.halves)

    /// How long after a capture arrives the first image of its pair is ready, whichever engine makes it. The
    /// interpolation schedule sits that far behind the newest capture.
    let generationLatency = GenerationLatency()

    /// The engine's only tuned number. Capture rate, frame schedule and rate preference are all
    /// first-order filters, and each derives its own coefficient from this window and the rate it
    /// is actually running at — so none of them carries a constant that assumes a particular
    /// display or capture speed.
    static let measurementWindow: Double = 0.5

    init(gpu: GPUContext) {
        self.gpu = gpu
        neural = NeuralInterpolator(gpu: gpu, errors: errors, latency: generationLatency)
        metalFX = MetalFXInterpolator(gpu: gpu, errors: errors, latency: generationLatency)
    }

    /// The image `phase` of the way from `previous` to `next`, from whichever engine is making them, once it
    /// exists. Called from the render thread.
    func interpolated(previous: CFTimeInterval, next: CFTimeInterval, phase: Double) -> MTLTexture? {
        switch interpolationBackend.withLock({ $0 }) {
        case .neuralEngine: neural.texture(previous: previous, next: next, phase: phase)
        case .metalFX: phase == 0.5 ? metalFX.texture(previous: previous, next: next) : nil
        }
    }
}
