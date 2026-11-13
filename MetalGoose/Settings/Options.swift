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
    /// MGFG-1 makes the images between captures by interpolation: the Neural Engine, and MetalFX on the GPU where the
    /// Neural Engine cannot (`GenerationSelector`). It blends between two captured frames, so the newest one has to be
    /// held back: the schedule runs behind real time by most of a capture interval plus the time the first generated
    /// image takes to make, because an image cannot be shown before it exists and a clock any closer to real time asks
    /// for it too early.
    case mgfg1 = "MGFG-1"

    var title: LocalizedStringResource {
        switch self {
        case .off:   return "Off"
        case .mgfg1: return "MGFG-1"
        }
    }

    /// The multipliers the mode can deliver, lowest first. A pair of captures is cut into two steps, the midpoint, or
    /// four, its quarters, and nothing in between: the Neural Engine rounds any other phase to the nearest eighth, and
    /// takes five times as long for thirds.
    var multipliers: [Int] {
        switch self {
        case .off:   return [1]
        case .mgfg1: return [2, 4]
        }
    }

    /// Versions before MGFG-1 stored the mode that interpolated or the one that extrapolated. There is one mode now, and
    /// it interpolates, so both are MGFG-1; a value that was never a mode is none.
    init?(storedValue: String) {
        switch storedValue {
        case "MGFG-1-Interpolation", "MGFG-1-Extrapolation": self = .mgfg1
        default:                                              self.init(rawValue: storedValue)
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
