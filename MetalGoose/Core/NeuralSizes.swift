import Foundation

/// The sizes the Neural Engine works at, for frames of a given size.
///
/// VideoToolbox's frame interpolator takes at most 1920 pixels on a side and 2.07 megapixels, and builds a model for the
/// size it is given. How long a call takes does not grow smoothly with the size: measured in the pipeline at 30 captures
/// a second, it is 4 ms at 640x360, 7 at 960x540, 10 at 1280x720, 13 at 1312x738, 17 at 1440x810 and 18 to 20 from there
/// to 1920x1080. A frame of up to 1280x720's 0.92 megapixels takes about half as long as one a little over it, which is
/// nearly as long as the largest there is. So the Neural Engine is not given every frame as it is:
///
/// - A frame it takes is given whole. Its odd pixel, if it has one, is left out, because the chroma planes are half-size.
/// - A larger frame is shrunk to the largest size it takes, in the same proportions, and the images that come back
///   are brought up to the frame's size again (`blendTowardCapturesFromYUV...` does that as it blends them with the captures).
/// - Where that size is more than the rate the frames arrive at leaves time for, a coarser one is used rather than none,
///   and the coarser sizes are those where the call gets cheaper: 0.92 megapixels (1280x720) and 0.52 (960x540). Measured
///   against the real in-between frame on 1080p clips, the images made at 1280x720 and blended with the captures were
///   37.3 dB, 3 below the full size, and those made at 960x540 36.3: coarser, but with the structure of the picture kept
///   (SSIM 0.978 and 0.970 against 0.965 for a plain mix of the two captures), and cheaper on the GPU than any other engine.
///
/// The sizes are a ladder, finest first, that depends on the frame alone. Which rung is in use is decided by
/// `GenerationSelector` from how long a call takes at each, and moving between rungs does not touch the frames the
/// pipeline holds, only the session the Neural Engine runs them through.
enum NeuralSizes {

    struct Size: Equatable, Sendable {
        let width: Int
        let height: Int
        var pixels: Int { width * height }
    }

    /// What VideoToolbox's frame interpolator takes, as it reports it.
    struct Limits: Equatable, Sendable {
        let maximumDimension: Int
        let maximumPixels: Int
    }

    /// The processor takes nothing smaller on either side.
    static let minimumDimension = 64

    /// The pixels the rungs below the finest are given: the largest size before the call gets dearer, and one more. A
    /// rung is offered only if it has clearly fewer pixels than the one above it, and its longer side is at least
    /// `coarsestLongSide`. Below 960x540 the images were no closer to the real frame than a plain mix of the two captures
    /// (at 640x360 from a 720p capture: 36.2 dB against 38.5 for the mix, and no better by structure), so there is no rung there.
    static let budgets = [921_600, 518_400]
    static let smallestShare = 0.85

    /// A coarser rung is not offered where its longer side would be shorter than this: the images the Neural Engine
    /// makes at less are not worth showing next to the captures.
    static let coarsestLongSide = 640

    /// The ladder for frames of this size, finest first; empty where the Neural Engine cannot take them at all.
    static func ladder(frameWidth: Int, frameHeight: Int, limits: Limits) -> [Size] {
        guard frameWidth >= minimumDimension, frameHeight >= minimumDimension,
              limits.maximumDimension > 0, limits.maximumPixels > 0 else { return [] }

        let share = min(1.0,
                        Double(limits.maximumDimension) / Double(max(frameWidth, frameHeight)),
                        (Double(limits.maximumPixels) / Double(frameWidth * frameHeight)).squareRoot())
        func even(_ value: Double) -> Int { Int(value.rounded(.down)) & ~1 }

        let finest = Size(width: even(Double(frameWidth) * share), height: even(Double(frameHeight) * share))
        guard finest.width >= minimumDimension, finest.height >= minimumDimension,
              max(finest.width, finest.height) <= limits.maximumDimension, finest.pixels <= limits.maximumPixels else { return [] }

        var rungs = [finest]
        for budget in budgets where Double(budget) <= Double(rungs[rungs.count - 1].pixels) * smallestShare {
            let share = (Double(budget) / Double(finest.pixels)).squareRoot()
            let size = Size(width: even(Double(finest.width) * share), height: even(Double(finest.height) * share))
            guard size.width >= minimumDimension, size.height >= minimumDimension,
                  max(size.width, size.height) >= coarsestLongSide else { break }
            rungs.append(size)
        }
        return rungs
    }
}
