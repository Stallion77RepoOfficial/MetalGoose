import Foundation
import CoreGraphics

/// What the HUD reads. `outputFrameCount` is `generatedFrameCount + passthroughFrameCount`:
/// every presented image is either a captured frame shown as captured or one the generator
/// synthesised, and no image is presented twice.
struct PipelineStats: Sendable, Equatable {
    var captureFPS: Float = 0
    /// Images presented per second. The panel's refresh rate is not a measure of this — it
    /// repeats whatever it was last given — so it is the number of *new* images that counts.
    var outputFPS: Float = 0
    /// Synthesised images presented per second.
    var generatedFPS: Float = 0

    var frameTime: Float = 0
    var gpuTime: Float = 0
    /// Completed command-buffer duration / wall time: a workload budget estimate,
    /// not hardware utilisation. Preemption and overlap can inflate it.
    var gpuLoad: Float = 0
    var captureGPUTime: Float = 0
    var captureLatency: Float = 0
    var presentLatency: Float = 0
    var endToEndLatency: Float = 0
    var avgFrameTime: Float = 0
    var framePacingScore: Float = 100

    var frameCount: UInt64 = 0
    var outputFrameCount: UInt64 = 0
    var droppedFrames: UInt64 = 0
    var generatedFrameCount: UInt64 = 0
    var passthroughFrameCount: UInt64 = 0
    var counterEpoch = 0

    var gpuMemoryUsed: UInt64 = 0
    var gpuMemoryTotal: UInt64 = 0
    var processMemoryUsed: UInt64 = 0
    var cpuUsage: Float = 0

    var outputResolution: CGSize = .zero
    var screenRefreshRate: Int = 0
    var isProMotion = false
    /// Images per second the screen should be given: the capture rate times the multiplier in use.
    var targetOutputFPS: Int = 0

    /// The counters that describe one mode's behaviour. Carrying them across a mode switch
    /// mixes two schedules into one set of totals, and the generated/passthrough split stops
    /// meaning anything.
    mutating func resetCumulativeCounters() {
        counterEpoch &+= 1
        droppedFrames = 0
        outputFrameCount = 0
        generatedFrameCount = 0
        passthroughFrameCount = 0
    }
}
