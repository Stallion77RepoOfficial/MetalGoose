import Foundation

/// How a frame is searched for motion.
///
/// The media engine's time grows with the pixels it searches — 3 ms at 1280x720, 6 ms at 1920x1080 and 11 ms at
/// 2560x1440 in 4x4 blocks on a frame averaged down by two, against 1 to 3 ms in 16x16 blocks at full size — and a
/// field that arrives after the next capture is due has nothing current to follow. So a frame is searched as finely
/// as fits in a share of the capture interval. A 120 Hz capture gets a coarser field than a 30 Hz one, and a slower
/// media engine (a slower chip, a GPU shared with a game, a video decoder in the captured window sharing the
/// engine) a coarser one than a fast one.
///
/// 4x4 blocks at full size are not on the ladder, though they are the most accurate on video. A block matches the
/// pixels it holds, and on thin periodic lines — scrolling text, gridlines, a texture of small cells — the best match
/// along a line is arbitrary. Measured against the real next frame, they lost 3 to 7 dB to the default search on
/// scrolling text, a grid and a tiled texture, while gaining 1 to 2 dB on video. The same blocks on a frame averaged
/// down by two cover 8x8 pixels and lost nothing (at most 0.8 dB), keeping most of the gain.
struct MotionAnalysis: Equatable {
    /// The frame is averaged down by this whole factor before it is searched, and the vectors multiplied back.
    let divisor: Int
    /// Pixels of the searched image that one vector stands for.
    let blockSize: Int
    /// The searched image.
    let width: Int
    let height: Int

    /// Frame pixels that one vector stands for.
    var span: Int { divisor * blockSize }

    /// The vectors that cover the image. The media engine pads its field past them, up to a multiple of four
    /// vectors, and the padding is not part of the frame.
    var vectorWidth: Int { (width + blockSize - 1) / blockSize }
    var vectorHeight: Int { (height + blockSize - 1) / blockSize }

    init(rung: Int, frameWidth: Int, frameHeight: Int) {
        let rung = Self.rung(rung)
        divisor = rung.divisor
        blockSize = rung.blockSize
        width = frameWidth / rung.divisor
        height = frameHeight / rung.divisor
    }

    // MARK: - The ladder

    /// One way of analysing a frame, and what the media engine takes in proportion to the pixels it searches,
    /// measured on an M4.
    private struct Rung {
        let divisor: Int
        let blockSize: Int
        let nanosecondsPerPixel: Double
    }

    /// Finest first; each costs a fraction of the one before it.
    private static let ladder = [
        Rung(divisor: 2, blockSize: 4, nanosecondsPerPixel: 11.9),
        Rung(divisor: 1, blockSize: 16, nanosecondsPerPixel: 0.8),
        Rung(divisor: 2, blockSize: 16, nanosecondsPerPixel: 0.8)
    ]

    static var coarsest: Int { ladder.count - 1 }

    private static func rung(_ index: Int) -> Rung { ladder[min(max(index, 0), coarsest)] }

    /// Seconds from a frame reaching the motion pipeline to the vectors of its pair coming back, by those
    /// measurements: 1.7 ms and 0.65 ns for each pixel of the frame — the wait behind the capture's own GPU
    /// work, and the luma conversion — then the media engine's time.
    static func cost(rung: Int, frameWidth: Int, frameHeight: Int) -> CFTimeInterval {
        let rung = Self.rung(rung)
        let searched = (frameWidth / rung.divisor) * (frameHeight / rung.divisor)
        return 0.0017 + 0.65e-9 * Double(frameWidth * frameHeight) + rung.nanosecondsPerPixel * 1e-9 * Double(searched)
    }

    /// A rung is taken when its vectors would be back within this share of a capture interval, which leaves
    /// them ready before the next pair is due. It is kept until it needs more than `keepShare`, and left for a
    /// finer one only when that would need no more than `climbShare`, so an interval that wanders about a limit
    /// does not rebuild the session each time.
    private static let fitShare = 0.6
    private static let keepShare = 0.75
    private static let climbShare = 0.5

    /// The interval the search is planned for is never longer than this. A capture interval that grows because
    /// the machine cannot keep up is not time to spend on a finer search, which would be more work for a machine
    /// that cannot keep up; and a source slower than this is not helped much by the finest search at a size where
    /// it would not fit at 24 fps.
    private static let longestInterval: CFTimeInterval = 1.0 / 24

    /// The finest rung that fits, given the rung in use if there is one.
    ///
    /// - Parameters:
    ///   - slowdown: how much longer than the measurements say the vectors have been taking, 0 while that is
    ///     not known.
    ///   - refused: rungs the media engine would not build for this frame, which are not offered again.
    static func choose(current: Int?, frameWidth: Int, frameHeight: Int, interval: CFTimeInterval,
                       slowdown: Double, refused: Set<Int> = []) -> Int {
        let interval = min(interval, longestInterval)
        let slower = min(8, max(0.5, slowdown > 0 ? slowdown : 1))
        func time(_ rung: Int) -> CFTimeInterval {
            refused.contains(rung) ? .infinity : cost(rung: rung, frameWidth: frameWidth, frameHeight: frameHeight) * slower
        }

        guard let current else {
            return ladder.indices.first { time($0) <= fitShare * interval } ?? coarsest
        }
        if time(current) > keepShare * interval {
            return ladder.indices.first { $0 > current && time($0) <= fitShare * interval } ?? coarsest
        }
        if current > 0, time(current - 1) <= climbShare * interval { return current - 1 }
        return current
    }
}
