import Foundation

/// A setting with a fixed set of choices. The raw value is what gets persisted, so it
/// must never change once shipped; `title` is what the user reads and is free to be
/// reworded and translated.
protocol SettingOption: CaseIterable, Identifiable, Hashable, RawRepresentable, Sendable where RawValue == String {
    var title: LocalizedStringResource { get }
}

extension SettingOption {
    var id: String { rawValue }
}

enum ScalingMethod: String, SettingOption {
    case off = "Off"
    case mgup1 = "MGUP-1"

    var title: LocalizedStringResource {
        switch self {
        case .off:   return "Off"
        case .mgup1: return "MGUP-1"
        }
    }
}

enum RenderScale: String, SettingOption {
    case native = "Native (100%)"
    case p75 = "75%"
    case p67 = "67%"
    case p50 = "50%"
    case p33 = "33%"

    var multiplier: Float {
        switch self {
        case .native: return 1.0
        case .p75:    return 0.75
        case .p67:    return 0.67
        case .p50:    return 0.50
        case .p33:    return 0.33
        }
    }

    var title: LocalizedStringResource {
        switch self {
        case .native: return "Native (100%)"
        case .p75:    return "75%"
        case .p67:    return "67%"
        case .p50:    return "50%"
        case .p33:    return "33%"
        }
    }
}

enum ScaleFactor: String, SettingOption {
    case x1 = "1.0x"
    case x1_5 = "1.5x"
    case x2 = "2.0x"
    case x2_5 = "2.5x"
    case x3 = "3.0x"
    case x4 = "4.0x"
    case x5 = "5.0x"
    case x6 = "6.0x"
    case x8 = "8.0x"
    case x10 = "10.0x"
    case fullscreen = "Fullscreen"

    /// Fills the display exactly instead of scaling the window by a factor, so the
    /// aspect ratio follows the screen rather than the source window.
    var fillsScreen: Bool { self == .fullscreen }

    var value: Float {
        switch self {
        case .fullscreen, .x1: return 1.0
        case .x1_5: return 1.5
        case .x2:   return 2.0
        case .x2_5: return 2.5
        case .x3:   return 3.0
        case .x4:   return 4.0
        case .x5:   return 5.0
        case .x6:   return 6.0
        case .x8:   return 8.0
        case .x10:  return 10.0
        }
    }

    var title: LocalizedStringResource {
        switch self {
        case .fullscreen: return "Fullscreen"
        default:          return LocalizedStringResource(stringLiteral: rawValue)
        }
    }
}

enum FrameGenMode: String, SettingOption {
    case off = "Off"
    /// Blends between two captured frames. Highest quality, but the newest frame has
    /// to be held back: the schedule runs behind real time by three quarters of a
    /// capture interval plus the time the midpoint takes to make, because the midpoint
    /// cannot be shown before it exists and a clock any closer to real time asks for it
    /// too early. The generators synthesise one image per pair — the midpoint — so the
    /// unique image rate is twice the capture rate.
    case interpolation = "MGFG-1-Interpolation"
    /// Warps the newest frame forward along its motion. Lower quality around
    /// disocclusions, but nothing is held back, so latency is unchanged, and the warp
    /// phase is continuous — the gap can be sampled at as many points as the
    /// multiplier asks for.
    case extrapolation = "MGFG-1-Extrapolation"

    var title: LocalizedStringResource {
        switch self {
        case .off:           return "Off"
        case .interpolation: return "MGFG-1-Interpolation"
        case .extrapolation: return "MGFG-1-Extrapolation"
        }
    }

    /// Interpolation synthesises exactly one image per frame pair — the midpoint — so
    /// it is 2x and cannot be anything else. The warp used by extrapolation takes a
    /// continuous phase, so it can be sampled at as many points in the gap as asked for.
    var multiplierRange: ClosedRange<Int> {
        switch self {
        case .off, .interpolation: return 2...2
        case .extrapolation:       return 2...4
        }
    }
}

enum AAMode: String, SettingOption {
    case off = "Off"
    case fxaa = "FXAA"
    case smaa = "SMAA"

    var title: LocalizedStringResource {
        switch self {
        case .off:  return "Off"
        case .fxaa: return "FXAA"
        case .smaa: return "SMAA"
        }
    }
}

/// How strongly the image is sharpened and anti-aliased. The upscale itself has no
/// quality knob — MetalFX exposes none beyond `colorProcessingMode` — so this is the
/// only thing the picker has ever controlled. The raw values predate that being said
/// plainly and stay as they are so stored settings keep working.
enum Sharpening: String, SettingOption {
    case light = "Performance"
    case balanced = "Balanced"
    case strong = "Ultra"

    var title: LocalizedStringResource {
        switch self {
        case .light:    return "Light"
        case .balanced: return "Balanced"
        case .strong:   return "Strong"
        }
    }

    var profile: QualityProfile {
        switch self {
        case .light:    return QualityProfile(sharpnessScale: 0.8, aaThreshold: 0.18, smaaSearchSteps: 8)
        case .balanced: return QualityProfile(sharpnessScale: 1.0, aaThreshold: 0.12, smaaSearchSteps: 12)
        case .strong:   return QualityProfile(sharpnessScale: 1.2, aaThreshold: 0.08, smaaSearchSteps: 16)
        }
    }
}

struct QualityProfile: Equatable, Sendable {
    let sharpnessScale: Float
    let aaThreshold: Float
    let smaaSearchSteps: Int
}

/// What synthesises the midpoint frame in interpolation mode.
enum InterpolationEngine: String, SettingOption {
    /// The Neural Engine, through VideoToolbox's low-latency frame interpolation. The
    /// GPU only converts pixel formats, which leaves it to the captured app.
    case neuralEngine
    /// MetalFX on the GPU. Faster per frame and not limited in size, but it takes GPU
    /// time from whatever is being captured.
    case metalFX

    var title: LocalizedStringResource {
        switch self {
        case .neuralEngine: return "Neural Engine"
        case .metalFX:      return "GPU (MetalFX)"
        }
    }
}
