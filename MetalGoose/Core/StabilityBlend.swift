import Foundation

/// How far a generated image is trusted where the two captures it was made from barely differ.
///
/// The Neural Engine's images are lossy where nothing moved, so there the captures' own mix at the image's phase is
/// closer to the real frame in between than the engine's image is. MetalFX's, along the media engine's field, already
/// sit close to the captures where nothing moved, so it gets a smaller share. The blend is the shader
/// `blendTowardCaptures`; this is the numbers it takes and the rule it applies, which is kept here so that it can be
/// tested.
enum StabilityBlend {

    /// The change between the captures, as a share of the full range, at which the Neural Engine's image stands alone.
    /// Chosen against the real frame in between on six clips, with and without an interface drawn over them, 1, 2 and 4
    /// captures apart: from 48 to 255 a larger value scored higher by PSNR and a smaller one by SSIM on the most moving
    /// pairs, and 64 keeps nearly all of the first without paying for it in the second.
    static let neuralEngineTolerance: Float = 64.0 / 255.0

    /// The same for MetalFX, on the same clips: 16 to 96 levels scored within half a dB of each other and of MetalFX
    /// alone, except for the interface, which came out closer with every step up, and the most moving pairs, which
    /// came out further. 32 is where the interface gains most for what the rest gives up.
    static let metalFXTolerance: Float = 32.0 / 255.0

    /// The tolerance an engine's images are blended with.
    static func tolerance(for engine: GenerationEngine) -> Float {
        switch engine {
        case .neuralEngine: return neuralEngineTolerance
        case .metalFX:      return metalFXTolerance
        }
    }

    /// The share of the captures' mix to put in, given the largest change between the captures within two pixels of
    /// the one being drawn: all of it where nothing changed, none once the change reaches `tolerance`, a line between.
    static func weight(largestChange: Float, tolerance: Float) -> Float {
        min(max(1 - largestChange / tolerance, 0), 1)
    }
}
