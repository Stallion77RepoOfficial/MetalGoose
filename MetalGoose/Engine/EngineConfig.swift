import Foundation

/// Everything the pipeline reads from the settings, as one immutable value. Every consumer
/// takes a single snapshot at the top of its unit of work, so a concurrent change cannot
/// tear the pipeline mid-frame.
struct EngineConfig: Equatable, Sendable {
    var upscaling = false
    var antiAliasing: AAMode = .off
    var frameGeneration: FrameGenMode = .off
    /// Presented images per captured frame the pipeline aims for. Interpolation is pinned to
    /// 2 by what the generators produce; extrapolation uses it to decide how many warp phases
    /// to sample in each gap.
    var multiplier = 1
    var vsync = true
    var profile = Sharpening.balanced.profile
    var bufferDepth = GooseEngine.maxInFlight
    var motionSource: MotionSource = .mediaEngine
    var interpolationEngine: InterpolationEngine = .neuralEngine

    var generatesFrames: Bool { frameGeneration != .off }

    /// Sharpening is part of the upscale; with scaling off the image goes through untouched.
    var sharpness: Float { upscaling ? profile.sharpnessScale : 0 }
}
