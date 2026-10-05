import Foundation

/// Decides, frame by frame, whether the picture has been replaced rather than moved.
///
/// A cut is an outlier in how much the frame changed, so it is detected as one: the mean
/// absolute luma change over a sample grid is tracked with a running mean and variance, and
/// a frame is a cut when it sits more than `cutSigma` standard deviations above that.
struct SceneCutDetector {
    static let cutSigma = 6.0

    /// A change smaller than this is not a cut whatever the running statistics say. A cut
    /// replaces the image, so it moves a large fraction of the grid a long way; ordinary
    /// camera motion does not reach a mean absolute luma change of an eighth of full range.
    /// Without a floor, an estimator whose variance has not grown yet puts the threshold at
    /// the mean and calls every above-average frame a cut.
    static let minimumChange = 0.125

    /// Frames of history collected before any cut may be reported, so the spread is
    /// measured rather than assumed.
    static let warmupFrames = 30

    private var mean: Double?
    private var variance = 0.0
    private var warmupRemaining = 0

    mutating func reset() {
        mean = nil
        variance = 0
        warmupRemaining = 0
    }

    /// `alpha` is how much one frame weighs in the baseline: roughly one second of capture
    /// whatever the frame rate is, instead of a fixed number of frames that means half a
    /// second at 20 fps and a tenth of one at 120.
    mutating func isCut(previous: [UInt8]?, current: [UInt8], alpha: Double) -> Bool {
        guard let previous, previous.count == current.count, !current.isEmpty else { return false }

        var total = 0
        for i in current.indices {
            total += abs(Int(current[i]) - Int(previous[i]))
        }
        // Normalised to 0...1 against the full luma range, so the statistic does not depend
        // on grid size or bit depth.
        let change = Double(total) / Double(current.count * Int(UInt8.max))

        guard let mean else {
            self.mean = change
            variance = 0
            warmupRemaining = Self.warmupFrames
            return false
        }

        let threshold = max(Self.minimumChange, mean + Self.cutSigma * variance.squareRoot())
        let isCut = warmupRemaining == 0 && change > threshold
        if warmupRemaining > 0 { warmupRemaining -= 1 }

        // Every frame updates the baseline, but a cut enters it held at the threshold rather
        // than at its own value: it must not pull the mean up behind it, or the frames after
        // a transition inherit its statistics and the next cut is missed.
        //
        // Skipping the update entirely censors the sample down to the frames that passed the
        // test. The spread then gets measured from the quiet half alone, so it collapses
        // toward zero, the threshold collapses onto the mean, and from there every frame that
        // moves more than average is a cut. In gameplay that is about half of them.
        let observed = min(change, threshold)
        let deviation = observed - mean
        variance += (deviation * deviation - variance) * alpha
        self.mean = mean + alpha * deviation

        return isCut
    }

    /// The baseline spans roughly one second of capture at the rate the stream is asked for.
    static func alpha(forFrameRate fps: Int) -> Double {
        1.0 / Double(max(8, fps))
    }
}
