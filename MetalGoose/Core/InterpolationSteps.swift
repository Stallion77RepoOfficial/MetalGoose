import Foundation

/// How many steps a pair of captures is cut into by interpolation: 2 gives the midpoint, 4 its quarters.
///
/// Four steps are three images per pair instead of one, and the Neural Engine takes about three times as
/// long for the three as for the one, so what is delivered is a budget: the time the three take against the
/// time between captures, and against how many images the panel can show in that time at all. Both are
/// measured; neither rate is assumed. What is delivered is never more than what was asked for, and it starts
/// from the midpoint, which is what the three are costed from. It falls back to it when four steps cannot be kept
/// up with — a pair the engine is still busy with goes by without its images — and returns once they can, with
/// room to spare, so that a rate near the limit does not flip back and forth.
enum InterpolationSteps {
    static let halves = 2
    static let quarters = 4

    /// The steps a multiplier asks for. Nothing but the midpoint and the quarters can be made, so any other
    /// multiplier takes the nearest below.
    static func steps(for multiplier: Int) -> Int {
        multiplier >= quarters ? quarters : halves
    }

    /// What a call for the three quarters costs against a call for the midpoint, measured at 720p, 1080p and
    /// 1280x1016, idle in between and not, in the processor alone and inside the pipeline: 2.2 to 3.8. It stands
    /// in until the quarters have been timed.
    static let quartersCostRatio = 3.0

    /// The share of a capture interval that a call for the quarters may take: four steps are taken when it is
    /// expected to need no more than `enterShare` of it, and kept until it needs more than `keepShare`. A call
    /// that takes most of the interval leaves nothing for the variation in how long it takes, and a pair that
    /// arrives while the last is still being made is a pair that goes by without its images: measured, four in
    /// five pairs got them when a call took 85% of the interval.
    private static let enterShare = 0.65
    private static let keepShare = 0.8

    /// Images the panel shows in one capture interval: four steps pay only if it can show about four.
    private static let enterImages = 3.6
    private static let keepImages = 3.3

    /// - Parameters:
    ///   - requested: the multiplier the user chose.
    ///   - current: the steps in use.
    ///   - midpointTime: how long a call for the midpoint takes; 0 until it has been measured.
    ///   - quartersTime: how long a call for the three quarters takes; 0 until it has been measured.
    ///   - captureInterval: the smoothed time between captures; 0 until it has been measured.
    ///   - refreshRate: the panel's, in Hz; 0 when it is not known.
    ///   - mayEnter: false while the engine has recently failed to keep up with four steps, so that it is not
    ///     asked to try again at once.
    static func choose(requested: Int, current: Int, midpointTime: CFTimeInterval, quartersTime: CFTimeInterval,
                       captureInterval: CFTimeInterval, refreshRate: Int, mayEnter: Bool = true) -> Int {
        guard steps(for: requested) == quarters else { return halves }
        let keeping = current == quarters
        if !keeping && !mayEnter { return halves }
        // Four steps start from two: until the rates and the time a call takes are known there is nothing to say
        // whether three images fit, and a pair that does not get them goes by without.
        guard captureInterval > 0 else { return halves }

        if refreshRate > 0, Double(refreshRate) * captureInterval < (keeping ? keepImages : enterImages) { return halves }

        // Once the quarters have been timed their own time is what counts. Before that, and while they are not
        // being made, it is what the midpoint's time says they will cost.
        let needed = keeping && quartersTime > 0 ? quartersTime : midpointTime * quartersCostRatio
        guard needed > 0, needed <= (keeping ? keepShare : enterShare) * captureInterval else { return halves }
        return quarters
    }
}
