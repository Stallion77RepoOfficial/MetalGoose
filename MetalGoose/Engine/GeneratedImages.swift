import Foundation
@preconcurrency import Metal
import os

/// The images the engines have made, by the pair of captures and the phase between them that each stands for, whichever
/// engine made them.
///
/// Both engines publish here and the render thread reads here, so a pair that one engine was still on when the other
/// took over is shown all the same, and an image is blended with the tolerance of the engine that made it, not the one
/// that is current. Stored ANE frames retain their buffer leases; the presentation pass
/// retains the same lease until its GPU read finishes.
final class GeneratedImages: @unchecked Sendable {

    /// What an image is kept as: the colour MetalFX made, or the planes the Neural Engine wrote, which are turned into
    /// colour as the image is shown (`blendTowardCapturesFromYUV`).
    enum Source {
        case colour(MTLTexture)
        case planes(YUVFrame)
    }

    struct Image {
        let previous: CFTimeInterval
        let next: CFTimeInterval
        /// How far from `previous` to `next` the image sits.
        let phase: Double
        let source: Source
        let engine: GenerationEngine
    }

    /// How many images of an engine are kept: two pairs of the Neural Engine's quarters, and one pair per capture the
    /// ring holds, less the one that is the newest, of MetalFX's midpoints.
    static func capacity(of engine: GenerationEngine) -> Int {
        switch engine {
        case .neuralEngine: return 2 * NeuralInterpolator.maximumImages
        case .metalFX:      return FrameRing.capacity - 1
        }
    }

    private let images = OSAllocatedUnfairLock<[Image]>(uncheckedState: [])

    /// Makes `new` available to the render thread, oldest first, and lets go of what its engine no longer needs.
    func publish(_ new: [Image]) {
        guard let engine = new.first?.engine else { return }
        images.withLockUnchecked { stored in
            stored.append(contentsOf: new)
            var surplus = stored.reduce(0) { $0 + ($1.engine == engine ? 1 : 0) } - Self.capacity(of: engine)
            guard surplus > 0 else { return }
            stored.removeAll { image in
                guard surplus > 0, image.engine == engine else { return false }
                surplus -= 1
                return true
            }
        }
    }

    /// The image `phase` of the way from `previous` to `next`, once it has been made. Called from the render thread.
    func image(previous: CFTimeInterval, next: CFTimeInterval, phase: Double) -> Image? {
        images.withLockUnchecked { stored in
            stored.last { $0.previous == previous && $0.next == next && $0.phase == phase }
        }
    }

    func reset() {
        images.withLockUnchecked { $0.removeAll() }
    }
}
