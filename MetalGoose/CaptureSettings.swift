import SwiftUI

final class CaptureSettings: ObservableObject {
    nonisolated(unsafe) static let shared = CaptureSettings()

    enum RenderScale: String, CaseIterable, Identifiable {
        case native = "Native (100%)"
        case p75 = "75%"
        case p67 = "67%"
        case p50 = "50%"
        case p33 = "33%"

        var id: String { rawValue }

        var multiplier: Float {
            switch self {
            case .native: return 1.0
            case .p75: return 0.75
            case .p67: return 0.67
            case .p50: return 0.50
            case .p33: return 0.33
            }
        }
    }

    enum ScalingType: String, CaseIterable, Identifiable {
        case off = "Off"
        case mgup1 = "MGUP-1"

        var id: String { rawValue }
    }

    enum QualityMode: String, CaseIterable, Identifiable {
        case performance = "Performance"
        case balanced = "Balanced"
        case ultra = "Ultra"

        var id: String { rawValue }
    }

    enum ScaleFactorOption: String, CaseIterable, Identifiable {
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

        var id: String { rawValue }

        /// Fills the display exactly instead of scaling the window by a factor,
        /// so the aspect ratio follows the screen rather than the source window.
        var fillsScreen: Bool { self == .fullscreen }

        var floatValue: Float {
            switch self {
            case .fullscreen: return 1.0
            case .x1: return 1.0
            case .x1_5: return 1.5
            case .x2: return 2.0
            case .x2_5: return 2.5
            case .x3: return 3.0
            case .x4: return 4.0
            case .x5: return 5.0
            case .x6: return 6.0
            case .x8: return 8.0
            case .x10: return 10.0
            }
        }
    }

    enum FrameGenMode: String, CaseIterable, Identifiable {
        case off = "Off"
        /// Blends between two captured frames. Highest quality, but the newest
        /// frame has to be held back, so it costs a full capture interval of
        /// latency — the phase has to sweep the whole gap between the pair, and
        /// a clock any closer to real time runs past the newest frame for part
        /// of every capture period. MetalFX synthesises one image per pair — the
        /// midpoint — so the unique image rate is twice the capture rate however
        /// many times each image is presented.
        case interpolation = "MGFG-1-Interpolation"
        /// Warps the newest frame forward along its motion. Lower quality around
        /// disocclusions, but nothing is held back, so latency is unchanged, and
        /// the warp phase is continuous — the gap can be sampled at as many
        /// points as the multiplier asks for.
        case extrapolation = "MGFG-1-Extrapolation"

        var id: String { rawValue }
    }

    enum AAMode: String, CaseIterable, Identifiable {
        case off = "Off"
        case fxaa = "FXAA"
        case smaa = "SMAA"

        var id: String { rawValue }
    }

    @Published var renderScale: RenderScale = .native            { didSet { store(renderScale, "renderScale") } }
    @Published var scalingType: ScalingType = .off               { didSet { store(scalingType, "scalingType") } }
    @Published var qualityMode: QualityMode = .ultra             { didSet { store(qualityMode, "qualityMode") } }
    @Published var scaleFactor: ScaleFactorOption = .x1          { didSet { store(scaleFactor, "scaleFactor") } }
    @Published var frameGenMode: FrameGenMode = .off             { didSet { store(frameGenMode, "frameGenMode") } }
    /// How many presented images the pipeline should produce per captured frame.
    /// Stored raw and clamped on read, so switching to interpolation and back
    /// does not destroy an extrapolation setting the user chose.
    @Published var frameGenMultiplier: Int = 2                   { didSet { store(frameGenMultiplier, "frameGenMultiplier") } }
    @Published var aaMode: AAMode = .off                         { didSet { store(aaMode, "aaMode") } }

    static let minFrameGenMultiplier = 2

    /// MetalFX interpolation synthesises exactly one image per frame pair — the
    /// midpoint — so interpolation is 2x and cannot be anything else. The warp
    /// used by extrapolation takes a continuous phase, so it can be sampled at
    /// as many points in the gap as asked for.
    static func maxFrameGenMultiplier(for mode: FrameGenMode) -> Int {
        switch mode {
        case .off:           return minFrameGenMultiplier
        case .interpolation: return 2
        case .extrapolation: return 4
        }
    }

    /// The multiplier the pipeline actually runs at: the stored value clamped to
    /// what the selected mode can deliver. `off` generates nothing, so 1.
    var effectiveFrameGenMultiplier: Int {
        guard frameGenMode != .off else { return 1 }
        let upper = Self.maxFrameGenMultiplier(for: frameGenMode)
        return min(upper, max(Self.minFrameGenMultiplier, frameGenMultiplier))
    }

    @Published var captureCursor: Bool = true                    { didSet { store(captureCursor, "captureCursor") } }
    @Published var showMGHUD: Bool = true                        { didSet { store(showMGHUD, "showMGHUD") } }
    @Published var vsync: Bool = true                            { didSet { store(vsync, "vsync") } }

    /// Double vs triple buffering. Stored as a flag because the pipeline only
    /// ever supported those two depths.
    @Published var tripleBuffering: Bool = true                  { didSet { store(tripleBuffering, "tripleBuffering") } }

    var bufferCount: Int { tripleBuffering ? 3 : 2 }

    // MARK: - Persistence

    private let defaults = UserDefaults.standard

    /// Set while `init` is assigning loaded values, so restoring a setting does
    /// not immediately write it back.
    private var isRestoring = false

    private init() {
        isRestoring = true
        renderScale     = restore("renderScale", renderScale)
        scalingType     = restore("scalingType", scalingType)
        qualityMode     = restore("qualityMode", qualityMode)
        scaleFactor     = restore("scaleFactor", scaleFactor)
        frameGenMode    = restore("frameGenMode", frameGenMode)
        frameGenMultiplier = restore("frameGenMultiplier", frameGenMultiplier)
        aaMode          = restore("aaMode", aaMode)
        captureCursor   = restore("captureCursor", captureCursor)
        showMGHUD       = restore("showMGHUD", showMGHUD)
        vsync           = restore("vsync", vsync)
        tripleBuffering = restore("tripleBuffering", tripleBuffering)
        isRestoring = false
    }

    private func store<T: RawRepresentable>(_ value: T, _ key: String) where T.RawValue == String {
        guard !isRestoring else { return }
        defaults.set(value.rawValue, forKey: Self.prefix + key)
    }

    private func store(_ value: Bool, _ key: String) {
        guard !isRestoring else { return }
        defaults.set(value, forKey: Self.prefix + key)
    }

    private func store(_ value: Int, _ key: String) {
        guard !isRestoring else { return }
        defaults.set(value, forKey: Self.prefix + key)
    }

    /// An unrecognised stored value means the option was renamed or removed, so
    /// the current default stands rather than a forced fallback elsewhere.
    private func restore<T: RawRepresentable>(_ key: String, _ fallback: T) -> T where T.RawValue == String {
        guard let raw = defaults.string(forKey: Self.prefix + key),
              let value = T(rawValue: raw) else { return fallback }
        return value
    }

    private func restore(_ key: String, _ fallback: Bool) -> Bool {
        guard defaults.object(forKey: Self.prefix + key) != nil else { return fallback }
        return defaults.bool(forKey: Self.prefix + key)
    }

    private func restore(_ key: String, _ fallback: Int) -> Int {
        guard defaults.object(forKey: Self.prefix + key) != nil else { return fallback }
        return defaults.integer(forKey: Self.prefix + key)
    }

    private static let prefix = "MetalGoose."
}

struct QualityProfile {
    let sharpnessScale: Float
    let aaThreshold: Float
    let smaaSearchSteps: Int
}

extension CaptureSettings.QualityMode {
    var profile: QualityProfile {
        switch self {
        case .performance:
            return QualityProfile(
                sharpnessScale: 0.8,
                aaThreshold: 0.18,
                smaaSearchSteps: 8
            )
        case .balanced:
            return QualityProfile(
                sharpnessScale: 1.0,
                aaThreshold: 0.12,
                smaaSearchSteps: 12
            )
        case .ultra:
            return QualityProfile(
                sharpnessScale: 1.2,
                aaThreshold: 0.08,
                smaaSearchSteps: 16
            )
        }
    }
}
