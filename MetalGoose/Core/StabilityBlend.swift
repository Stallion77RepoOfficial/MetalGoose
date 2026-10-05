import Foundation

/// How far a generated image is trusted where the two captures it was made from barely differ.
///
/// The Neural Engine's images are lossy where nothing moved, so there the captures' own mix at the image's phase is
/// closer to the real frame in between than the engine's image is: an interface drawn identically in every frame came out
/// at 21 dB from it, and a still scene was further from the truth than repeating a capture. MetalFX's, along the media
/// engine's field, sit closer to the captures where nothing moved, but an interface still gains from it. Where the content
/// moved, the engine's image is the one that is right, and the captures' mix would be a ghost.
///
/// What tells the two apart is how much the captures differ compared with how much there is to differ: a texture that
/// moved by a pixel changes by about its own contrast, one that moved by a tenth of a pixel by a tenth of it, and noise
/// by the same small amount whatever the texture. So the weight of the captures' mix falls with the ratio of the largest
/// change between them to the largest contrast around, both within two pixels, which is about how far in pixels the
/// content moved, up to one. A fixed tolerance on the change alone, which this replaces, could not tell a low contrast
/// texture in motion from a still one with noise: it gained 4 dB on slow video and lost 0.3 to 0.7 on a scene turning at
/// 60 captures a second, and cost MetalFX up to 0.7 dB there.
///
/// The blend is the shaders `blendTowardCaptures...`; this is the numbers they take and the rule they apply, which is kept
/// here so that it can be tested.
enum StabilityBlend {

    /// How far, by that ratio, the content may have moved for the Neural Engine's image to stand alone. Chosen against the
    /// real frame in between on 18 real-clip cases at 720p and 1080p and 54 rendered scenes in motion from 720p to 4K, 1, 2
    /// and 4 frames apart (60, 30 and 15 captures a second): from 0.5 to 1.2 the scenes were within 0.1 dB of each other, and
    /// the clips gained with it up to 0.9, where the structure of the picture (SSIM) started to give it back.
    static let neuralEngineMotion: Float = 0.9

    /// The same for MetalFX, whose images are closer to the captures to begin with: it gains 0.2 dB on the scenes from 0.5
    /// to 0.9 and loses a little on the clips, less the smaller it is, so it is held to 0.7.
    static let metalFXMotion: Float = 0.7

    /// A change of this share of the full range is noise: 4 levels of 255.
    static let noiseFloor: Float = 4.0 / 255.0

    /// Where the captures are the same, to within this, over the three pixels around, they are the picture whatever the
    /// contrast around: an interface drawn the same, and the edge of one against a moving scene.
    static let exactChange: Float = 1.0 / 255.0

    /// How far the content may have moved for an engine's image to stand alone.
    static func motion(for engine: GenerationEngine) -> Float {
        switch engine {
        case .neuralEngine: return neuralEngineMotion
        case .metalFX:      return metalFXMotion
        }
    }

    /// The share of the captures' mix to put in at a pixel.
    ///
    /// - Parameters:
    ///   - largestChange: the largest difference of any channel between the captures within two pixels, as a share of
    ///     the full range.
    ///   - nearestChange: the same within one pixel.
    ///   - contrast: the largest difference of any channel to a neighbour, in either capture, within two pixels.
    static func weight(largestChange: Float, nearestChange: Float, contrast: Float, motion: Float,
                       noiseFloor: Float = StabilityBlend.noiseFloor) -> Float {
        if nearestChange <= exactChange { return 1 }
        let remaining = min(max(1 - (largestChange / (contrast + noiseFloor)) / motion, 0), 1)
        return remaining * remaining
    }
}
