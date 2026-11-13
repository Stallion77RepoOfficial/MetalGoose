import Foundation

/// The sizes the Neural Engine works at, for frames of a given size.
///
/// VideoToolbox's frame interpolator takes at most 1920 pixels on a side and 2.07 megapixels, and builds a model for the
/// size it is given. It runs its network in two ways, by size. Up to 1280x720's 921,600 pixels it runs a fast way, and above
/// it a slower one that is much closer to the real in-between frame: measured on frames scaled from 4K clips, the midpoint
/// against the real one, 37.6 to 39.0 dB on a camera clip and 41.4 to 45.1 on a rendered one at every size from 640x360 to
/// 1280x720, against 41.1 to 43.1 and 47.8 to 49.6 at every size from 1280x722 to 1920x1080; and 3.3 ms for a call at
/// 1280x720 against 6.3 at 1280x722. So:
///
/// - A frame of more than 921,600 pixels that the processor takes is given whole. Its odd pixel, if it has one, is left
///   out, because the chroma planes are half-size. A larger one is shrunk to the largest size it takes, in the same
///   proportions, and the images that come back are brought up to the frame's size again (`blendTowardCapturesFromYUV...`
///   does that as it blends them with the captures).
/// - A frame of up to 921,600 pixels is enlarged into the slower way: to the shipped shape of its proportions (below), or
///   to just over 921,600 pixels in its own. Measured on the same clips: 1280x720 frames went from 37.9 / 44.1 dB to 41.3 /
///   48.8 at 1920x1080 and 40.7 / 47.6 at 1284x722, 960x540 from 38.0 / 42.9 to 42.3 / 49.7, 800x600 from 37.7 / 43.7 to
///   42.5 / 49.7 at 1248x936.
/// - Where the rate leaves no time for those, coarser sizes in the fast way: 0.92 megapixels and 0.52, the shipped shapes
///   where the frame's proportions are near enough to theirs (no worse than its own proportions at those sizes: 36.73 /
///   44.18 dB at 1280x720 against 36.84 / 44.33 at 1214x758 for 1512x944 frames). Below 960x540 the images were no closer to
///   the real frame than a plain mix of the two captures (at 640x360 from a 720p capture: 36.2 dB against 38.5 for the mix),
///   so there is no rung there.
///
/// VideoProcessing ships the network in ten shapes (`shipped`); all ten are built at launch (`NeuralModelCache`). Where the
/// frame's proportions are no family's, the nearest shape is added last if it is close enough. The sizes are a ladder,
/// finest first, that depends on the frame alone. Which rung is in use is decided by `GenerationSelector` from how long a
/// call takes at each, and moving between rungs does not touch the frames the pipeline holds, only the session the Neural
/// Engine runs them through.
enum NeuralSizes {

    struct Size: Equatable, Hashable, Sendable {
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

    /// Above this many pixels the processor runs its network the slower, closer way; at or below it, the fast one.
    static let fastModePixels = 921_600

    /// The shapes VideoProcessing ships its interpolation network in (`mini-film-ifr-nano_<width>x<height>p`): one network,
    /// one set of weights, ten input shapes.
    static let shipped = [Size(width: 1920, height: 1080), Size(width: 1080, height: 1920), Size(width: 1248, height: 936),
                          Size(width: 1280, height: 720), Size(width: 960, height: 540), Size(width: 832, height: 624),
                          Size(width: 640, height: 480), Size(width: 640, height: 360), Size(width: 416, height: 312),
                          Size(width: 384, height: 320)]

    /// The shipped shapes in families of one proportion: the one past the fast way, and those within it that are rungs.
    private struct Family {
        let aspect: Double
        let slow: Size
        let fast: [Size]
    }
    private static let families = [
        Family(aspect: 16.0 / 9.0, slow: Size(width: 1920, height: 1080),
               fast: [Size(width: 1280, height: 720), Size(width: 960, height: 540)]),
        Family(aspect: 4.0 / 3.0, slow: Size(width: 1248, height: 936), fast: [Size(width: 832, height: 624)]),
        Family(aspect: 9.0 / 16.0, slow: Size(width: 1080, height: 1920), fast: []),
    ]

    /// How far a frame's proportions may be from a family's for its shapes to be used, as a ratio. A frame of 16:9 with a
    /// title bar on top is 3 to 4% narrower; 16:10 is 10%; and between 4:3 and 16:9 there is no frame that is further than
    /// this from both.
    static let familyTolerance = 1.16

    /// Whether the processor takes frames of this size. What it reports — 1920 px on a side and 2,073,600 pixels — is not the
    /// whole of it: it starts a session for 1874x1106 (2.07 MP) and then answers every call with "Processor is not
    /// initialized", where 1920x1080 and 1708x1016 work. Over 196 sizes from 1.2 to 2.16 MP, with the sides rounded up to a
    /// multiple of 64 and of 32 the area must not pass the limit, or the size must be the largest one, 1920x1080, exactly;
    /// no size that rule takes was refused in those measurements, and two it does not take were (1808x1102 and 1872x1072).
    /// This is a candidate filter: macOS 27 still rejects 1296x1600 at exactly the reported pixel limit, while 1440x1440
    /// works. Every session is tried with a call before it serves, so a reported-valid size that fails is rejected there.
    static func accepts(width: Int, height: Int, limits: Limits) -> Bool {
        guard width >= minimumDimension, height >= minimumDimension, width % 2 == 0, height % 2 == 0,
              max(width, height) <= limits.maximumDimension, width * height <= limits.maximumPixels else { return false }
        if width * height == limits.maximumPixels { return true }
        let padded = ((width + 63) / 64 * 64) * ((height + 31) / 32 * 32)
        return padded <= limits.maximumPixels
    }

    /// The pixels of the rungs in the fast way where the frame's proportions are no shipped family's: the largest size before
    /// the call gets dearer, and one more. A rung is offered only if it has clearly fewer pixels than the one above it.
    static let budgets = [921_600, 518_400]
    static let smallestShare = 0.85

    /// No rung is offered whose longer side is shorter than this: the images the Neural Engine makes at less are not worth
    /// showing next to the captures.
    static let coarsestLongSide = 640

    /// The ladder for frames of this size, finest first; empty where the Neural Engine cannot take them at all.
    static func ladder(frameWidth: Int, frameHeight: Int, limits: Limits) -> [Size] {
        guard frameWidth >= minimumDimension, frameHeight >= minimumDimension,
              limits.maximumDimension > 0, limits.maximumPixels > 0 else { return [] }
        func even(_ value: Double) -> Int { Int(value.rounded(.down)) & ~1 }
        func takes(_ size: Size) -> Bool { accepts(width: size.width, height: size.height, limits: limits) }

        // The largest size, in the frame's proportions, that the processor takes: a frame that is taken is kept as it is,
        // and one that is not is shrunk a little at a time until it is.
        var share = min(1.0,
                        Double(limits.maximumDimension) / Double(max(frameWidth, frameHeight)),
                        (Double(limits.maximumPixels) / Double(frameWidth * frameHeight)).squareRoot())
        var own = Size(width: even(Double(frameWidth) * share), height: even(Double(frameHeight) * share))
        while !takes(own), min(own.width, own.height) >= minimumDimension {
            share *= 0.99
            own = Size(width: even(Double(frameWidth) * share), height: even(Double(frameHeight) * share))
        }
        guard takes(own) else { return [] }

        let aspect = Double(own.width) / Double(own.height)
        let family = families.first { max($0.aspect / aspect, aspect / $0.aspect) <= familyTolerance }

        var rungs: [Size] = []
        if own.pixels > fastModePixels {
            rungs.append(own)
        } else {
            // Enlarged into the slower way: the family's shape, then just over the line in the frame's own proportions.
            if let slow = family?.slow, takes(slow) { rungs.append(slow) }
            var scale = (Double(fastModePixels + 9_400) / Double(own.pixels)).squareRoot()
            var over = Size(width: even(Double(own.width) * scale), height: even(Double(own.height) * scale))
            while over.pixels <= fastModePixels {
                scale *= 1.003
                over = Size(width: even(Double(own.width) * scale), height: even(Double(own.height) * scale))
            }
            if takes(over), over.pixels > fastModePixels { rungs.append(over) }
        }

        // The fast way, for rates that leave no time for the slower one: 0.92 and 0.52 megapixels, each the family's shape of
        // that size where it has one and the frame's proportions where not; and the frame itself where it is in the fast way
        // and larger than those.
        var fast = budgets.map { budget -> Size in
            if let shape = family?.fast.first(where: { abs($0.pixels - budget) * 10 < budget }) { return shape }
            let scale = (Double(min(budget, own.pixels)) / Double(own.pixels)).squareRoot()
            return Size(width: even(Double(own.width) * scale), height: even(Double(own.height) * scale))
        }
        if own.pixels <= fastModePixels, !fast.contains(own), fast.allSatisfy({ $0.pixels < own.pixels }) {
            fast.insert(own, at: 0)
        }
        for size in fast where takes(size) && max(size.width, size.height) >= coarsestLongSide && !rungs.contains(size) {
            // A rung in the same way as the one above it is offered only if it has clearly fewer pixels; one in the fast way
            // below one in the slower way is a step of its own, whatever its pixels.
            if let last = rungs.last, (last.pixels > fastModePixels) == (size.pixels > fastModePixels),
               Double(size.pixels) > Double(last.pixels) * smallestShare { continue }
            rungs.append(size)
        }

        // Where no rung is a shipped shape, the nearest one in proportions of the fast ones the families use is added last,
        // so that a session can start at once while the others are built (`GenerationSelector`); the smaller of two alike,
        // and none where even the nearest is more than half as wide again or as narrow (a portrait frame).
        let shapes = families.flatMap(\.fast).filter(takes)
        func distance(_ size: Size) -> Double { abs(log(Double(size.width) / Double(size.height) / aspect)) }
        if !rungs.contains(where: { shipped.contains($0) }),
           let nearest = shapes.min(by: { (distance($0), $0.pixels) < (distance($1), $1.pixels) }),
           distance(nearest) <= log(1.5) {
            rungs.append(nearest)
        }
        return rungs
    }
}
