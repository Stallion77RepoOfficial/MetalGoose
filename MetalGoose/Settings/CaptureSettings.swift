import SwiftUI

/// What the user has chosen, persisted across launches. The pipeline never sees this
/// object: it reads an immutable `EngineConfig` snapshot instead, so a change made
/// while a frame is in flight cannot tear that frame.
@MainActor
final class CaptureSettings: ObservableObject {
    static let shared = CaptureSettings()

    @Published var scalingMethod: ScalingMethod = .off          { didSet { save(scalingMethod.rawValue, .scalingMethod) } }
    @Published var scaleFactor: ScaleFactor = .x1               { didSet { save(scaleFactor.rawValue, .scaleFactor) } }
    @Published var renderScale: RenderScale = .native           { didSet { save(renderScale.rawValue, .renderScale) } }
    @Published var sharpening: Sharpening = .strong             { didSet { save(sharpening.rawValue, .sharpening) } }
    @Published var frameGenMode: FrameGenMode = .off            { didSet { save(frameGenMode.rawValue, .frameGenMode) } }
    /// How many presented images the pipeline aims for per captured frame. Stored raw and
    /// read as the nearest value the mode can deliver, so switching between modes does not
    /// destroy a setting the user chose.
    @Published var frameGenMultiplier: Int = 2                  { didSet { save(frameGenMultiplier, .frameGenMultiplier) } }
    @Published var aaMode: AAMode = .off                        { didSet { save(aaMode.rawValue, .aaMode) } }
    @Published var captureCursor: Bool = true                   { didSet { save(captureCursor, .captureCursor) } }
    @Published var showMGHUD: Bool = true                       { didSet { save(showMGHUD, .showMGHUD) } }
    @Published var vsync: Bool = true                           { didSet { save(vsync, .vsync) } }
    /// Double versus triple buffering. Stored as a flag because the pipeline only ever
    /// supported those two depths.
    @Published var tripleBuffering: Bool = true                 { didSet { save(tripleBuffering, .tripleBuffering) } }

    var bufferCount: Int { tripleBuffering ? 3 : 2 }

    /// Whether scaling is active at all. Scale Factor and Render Scale only mean
    /// something while it is.
    var isUpscaling: Bool { scalingMethod != .off }

    /// The multiplier the pipeline is asked to run at: the highest one the selected mode offers that
    /// the stored value reaches, and the lowest if it reaches none. `off` generates nothing, so 1.
    var effectiveMultiplier: Int {
        let offered = frameGenMode.multipliers
        return offered.last { $0 <= frameGenMultiplier } ?? offered[0]
    }

    var engineConfig: EngineConfig {
        EngineConfig(upscaling: isUpscaling,
                     antiAliasing: aaMode,
                     frameGeneration: frameGenMode,
                     multiplier: effectiveMultiplier,
                     vsync: vsync,
                     profile: sharpening.profile,
                     bufferDepth: bufferCount)
    }

    // MARK: - Persistence

    private enum Key: String {
        case scalingMethod = "scalingType"   // the key predates the rename and is kept for stored values
        case scaleFactor, renderScale, frameGenMode, frameGenMultiplier
        case sharpening = "qualityMode"      // ditto
        case aaMode, captureCursor, showMGHUD, vsync, tripleBuffering

        var path: String { "MetalGoose." + rawValue }
    }

    private let defaults: UserDefaults

    private init(defaults: UserDefaults = .standard) {
        self.defaults = defaults
        // Observers do not fire for assignments made from the initializer, so restoring a
        // value does not write it straight back.
        scalingMethod       = restore(.scalingMethod, scalingMethod)
        scaleFactor         = restore(.scaleFactor, scaleFactor)
        renderScale         = restore(.renderScale, renderScale)
        sharpening          = restore(.sharpening, sharpening)
        frameGenMode        = restoreFrameGeneration()
        frameGenMultiplier  = restore(.frameGenMultiplier, frameGenMultiplier)
        aaMode              = restore(.aaMode, aaMode)
        captureCursor       = restore(.captureCursor, captureCursor)
        showMGHUD           = restore(.showMGHUD, showMGHUD)
        vsync               = restore(.vsync, vsync)
        tripleBuffering     = restore(.tripleBuffering, tripleBuffering)
    }

    /// The stored frame generation mode. Versions before MGFG-1 stored the mode that interpolated or the one that
    /// extrapolated, and both are MGFG-1 now (`FrameGenMode.init(storedValue:)`); the value is written back at once, so
    /// that what is stored is the current one from then on.
    private func restoreFrameGeneration() -> FrameGenMode {
        guard let raw = defaults.string(forKey: Key.frameGenMode.path),
              let mode = FrameGenMode(storedValue: raw) else { return frameGenMode }
        if mode.rawValue != raw { save(mode.rawValue, .frameGenMode) }
        return mode
    }

    private func save(_ value: Any, _ key: Key) {
        defaults.set(value, forKey: key.path)
    }

    /// An unrecognised stored value means the option was renamed or removed, so the
    /// current default stands rather than a forced fallback elsewhere.
    private func restore<T: RawRepresentable>(_ key: Key, _ fallback: T) -> T where T.RawValue == String {
        guard let raw = defaults.string(forKey: key.path), let value = T(rawValue: raw) else { return fallback }
        return value
    }

    private func restore(_ key: Key, _ fallback: Bool) -> Bool {
        defaults.object(forKey: key.path) as? Bool ?? fallback
    }

    private func restore(_ key: Key, _ fallback: Int) -> Int {
        defaults.object(forKey: key.path) as? Int ?? fallback
    }
}
