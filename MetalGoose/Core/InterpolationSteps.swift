import Foundation

/// How many steps a pair of captures is cut into by interpolation: 2 gives the midpoint, 4 its quarters.
///
/// Four steps are three images per pair instead of one, and the Neural Engine takes about three times as
/// long for the three as for the one, so what is delivered is a budget: the time the three take against the
/// time between captures, and against how many images the panel can show in that time at all. Both are
/// measured; neither rate is assumed. `GenerationSelector` weighs them, and falls back to the midpoint when four steps
/// cannot be kept up with — a pair the engine is still busy with goes by without its images.
enum InterpolationSteps {
    static let halves = 2
    static let quarters = 4

    /// The steps a multiplier asks for. Nothing but the midpoint and the quarters can be made, so any other
    /// multiplier takes the nearest below.
    static func steps(for multiplier: Int) -> Int {
        multiplier >= quarters ? quarters : halves
    }

    /// What a call for the three quarters costs against a call for the midpoint, measured from 640x360 to 1920x1080, idle in
    /// between and not, in the processor alone and inside the pipeline: 2.2 to 3.8, mostly 2.3 to 2.9. It stands in until
    /// the quarters have been timed.
    static let quartersCostRatio = 2.7

    /// The share of a capture interval that a call for the quarters may take: four steps are taken when it is
    /// expected to need no more than `enterShare` of it, and kept until it needs more than `keepShare`. A call
    /// that takes most of the interval leaves nothing for the variation in how long it takes, and a pair that
    /// arrives while the last is still being made is a pair that goes by without its images: measured, four in
    /// five pairs got them when a call took 85% of the interval.
    static let enterShare = 0.65
    static let keepShare = 0.8
}
