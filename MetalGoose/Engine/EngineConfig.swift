import Foundation

/// Everything the pipeline reads from the settings, as one immutable value. Every consumer
/// takes a single snapshot at the top of its unit of work, so a concurrent change cannot
/// tear the pipeline mid-frame.
struct EngineConfig: Equatable, Sendable {
    var upscaling = false
    var antiAliasing: AAMode = .off
    var frameGeneration: FrameGenMode = .off
    /// Presented images per captured frame the pipeline aims for: 2, the midpoint of each pair, or 4, its quarters. What
    /// is delivered is never more than this (`GenerationSelector`): the Neural Engine makes 4 while it can keep up, and
    /// MetalFX, which makes the midpoint alone, never does.
    var multiplier = 1
    var vsync = true
    var profile = Sharpening.balanced.profile
    var bufferDepth = GooseEngine.maxInFlight

    var generatesFrames: Bool { frameGeneration != .off }

    /// Sharpening is part of the upscale; with scaling off the image goes through untouched.
    var sharpness: Float { upscaling ? profile.sharpnessScale : 0 }
}
