import Foundation

/// What synthesises the in-between images in interpolation mode. Not a setting: the Neural Engine is
/// used while it can take the frames and has not failed, and MetalFX is what is left.
enum InterpolationEngine: Sendable {
    /// The Neural Engine, through VideoToolbox's low-latency frame interpolation. The GPU only converts
    /// pixel formats, which leaves it to the captured app. Makes the midpoint of each pair or its quarters,
    /// for frames up to 1920 pixels and 2.07 megapixels.
    case neuralEngine
    /// MetalFX on the GPU. Not limited in size, but it makes only the midpoint and takes GPU time from
    /// whatever is being captured.
    case metalFX

    var title: LocalizedStringResource {
        switch self {
        case .neuralEngine: return "Neural Engine"
        case .metalFX:      return "GPU (MetalFX)"
        }
    }
}
