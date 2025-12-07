
import SwiftUI
import MetalFX


@available(macOS 26.0, *)
final class CaptureSettings: ObservableObject {
    static let shared = CaptureSettings()
    
    @Published var selectedProfile: String = "Default"
    @Published var profiles: [String] = ["Default", "Performance", "Quality", "Ultra"]
    
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
        
        var description: String {
            switch self {
            case .native: return "Full resolution capture - highest quality, highest GPU usage"
            case .p75: return "75% resolution - slight quality loss, improved performance"
            case .p67: return "67% resolution - good balance for MGUP-1"
            case .p50: return "50% resolution - best for 2x upscaling"
            case .p33: return "33% resolution - maximum performance, for 3x upscaling"
            }
        }
    }
    
    enum ScalingType: String, CaseIterable, Identifiable {
        case off = "Off"
        case mgup1 = "MGUP-1"
        case mgup1Fast = "MGUP-1 Fast"
        case mgup1Quality = "MGUP-1 Quality"
        
        var id: String { rawValue }
        
        var description: String {
            switch self {
            case .off: return "No upscaling - original resolution passthrough"
            case .mgup1: return "MetalGoose Upscaler - Balanced MetalFX AI upscaling"
            case .mgup1Fast: return "MetalGoose Fast - Bilinear + Adaptive Sharpening"
            case .mgup1Quality: return "MetalGoose Quality - MetalFX Spatial + CAS"
            }
        }
        
        var usesMetalFX: Bool {
            switch self {
            case .off, .mgup1Fast: return false
            case .mgup1, .mgup1Quality: return true
            }
        }
    }
    
    enum QualityMode: String, CaseIterable, Identifiable {
        case performance = "Performance"
        case balanced = "Balanced"
        case ultra = "Ultra"
        
        var id: String { rawValue }
        
        var scalerMode: MTLFXSpatialScalerColorProcessingMode {
            switch self {
            case .performance: return .linear
            case .balanced: return .perceptual
            case .ultra: return .hdr
            }
        }
        
        var description: String {
            switch self {
            case .performance: return "Linear processing - fastest, slight color shift"
            case .balanced: return "Perceptual processing - best balance"
            case .ultra: return "HDR processing - highest quality, wider gamut"
            }
        }
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
        
        var id: String { rawValue }
        
        var floatValue: Float {
            switch self {
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
        
        var supportsTemporalMode: Bool { floatValue <= 3.0 }
        
        var description: String {
            switch self {
            case .x1: return "1.0x - No upscale, passthrough"
            case .x1_5: return "1.5x - Minimal upscale, best quality"
            case .x2: return "2.0x - Standard upscale, recommended"
            case .x2_5: return "2.5x - Good for 1080p→2.7K"
            case .x3: return "3.0x - Good for 720p→2160p"
            case .x4: return "4.0x - Maximum for MetalFX Temporal"
            case .x5: return "5.0x - High upscale ratio"
            case .x6: return "6.0x - Very high ratio"
            case .x8: return "8.0x - Extreme upscale"
            case .x10: return "10.0x - Maximum ratio, requires powerful GPU"
            }
        }
    }
    
    enum FrameGenMode: String, CaseIterable, Identifiable {
        case off = "Off"
        case mgfg1 = "MGFG-1"
        
        var id: String { rawValue }
        
        var description: String {
            switch self {
            case .off: return "No frame generation - lowest latency"
            case .mgfg1: return "MetalGoose Frame Generation - Optical flow interpolation"
            }
        }
    }
    
    enum FrameGenType: String, CaseIterable, Identifiable {
        case adaptive = "Adaptive"
        case fixed = "Fixed"
        
        var id: String { rawValue }
        
        var description: String {
            switch self {
            case .adaptive: return "Auto-adjust to reach target FPS"
            case .fixed: return "Fixed frame multiplication ratio"
            }
        }
    }
    
    enum TargetFPS: String, CaseIterable, Identifiable {
        case fps60 = "60 FPS"
        case fps90 = "90 FPS"
        case fps120 = "120 FPS"
        case fps144 = "144 FPS"
        case fps165 = "165 FPS"
        case fps180 = "180 FPS"
        case fps240 = "240 FPS"
        case fps360 = "360 FPS"
        
        var id: String { rawValue }
        
        var intValue: Int {
            switch self {
            case .fps60: return 60
            case .fps90: return 90
            case .fps120: return 120
            case .fps144: return 144
            case .fps165: return 165
            case .fps180: return 180
            case .fps240: return 240
            case .fps360: return 360
            }
        }
        
        var description: String {
            switch self {
            case .fps60: return "60 FPS - Standard smooth"
            case .fps90: return "90 FPS - VR/High refresh"
            case .fps120: return "120 FPS - High refresh displays"
            case .fps144: return "144 FPS - Gaming monitors"
            case .fps165: return "165 FPS - Premium gaming"
            case .fps180: return "180 FPS - Ultra high refresh"
            case .fps240: return "240 FPS - Competitive/Pro"
            case .fps360: return "360 FPS - Extreme/Esports"
            }
        }
    }
    
    enum FrameGenMultiplier: String, CaseIterable, Identifiable {
        case x2 = "2x (Double FPS)"
        case x3 = "3x (Triple FPS)"
        case x4 = "4x (Quadruple FPS)"
        
        var id: String { rawValue }
        
        var intValue: Int {
            switch self {
            case .x2: return 2
            case .x3: return 3
            case .x4: return 4
            }
        }
        
        var description: String {
            switch self {
            case .x2: return "30→60 or 60→120 FPS"
            case .x3: return "30→90 or 40→120 FPS"
            case .x4: return "30→120 or 60→240 FPS"
            }
        }
    }
    
    enum AAMode: String, CaseIterable, Identifiable {
        case off = "Off"
        case fxaa = "FXAA"
        case smaa = "SMAA"
        case msaa = "MSAA"
        case taa = "TAA"
        
        var id: String { rawValue }
        
        var description: String {
            switch self {
            case .off: return "No anti-aliasing - sharpest but aliased"
            case .fxaa: return "Fast Approximate AA - quick, slight blur"
            case .smaa: return "Subpixel Morphological AA - high quality edges"
            case .msaa: return "Multisample AA - hardware-like, clean edges"
            case .taa: return "Temporal AA - motion-compensated, best quality"
            }
        }
        
        var isTemporal: Bool {
            return self == .taa
        }
    }
    
    
    @Published var renderScale: RenderScale = .native
    
    @Published var scalingType: ScalingType = .off
    @Published var qualityMode: QualityMode = .ultra
    @Published var scaleFactor: ScaleFactorOption = .x1
    
    @Published var frameGenMode: FrameGenMode = .off
    @Published var frameGenType: FrameGenType = .adaptive
    @Published var targetFPS: TargetFPS = .fps120
    @Published var frameGenMultiplier: FrameGenMultiplier = .x2
    
    @Published var aaMode: AAMode = .off
    
    @Published var captureCursor: Bool = true
    @Published var reduceLatency: Bool = true
    @Published var adaptiveSync: Bool = true
    @Published var showMGHUD: Bool = true
    @Published var vsync: Bool = true
    
    @Published var sharpening: Float = 0.5
    @Published var temporalBlend: Float = 0.1
    @Published var motionScale: Float = 1.0
    
    
    var effectiveUpscaleFactor: Float {
        guard scalingType != .off else { return 1.0 }
        return scaleFactor.floatValue / renderScale.multiplier
    }
    
    var isUpscalingEnabled: Bool {
        return scalingType != .off
    }
    
    var isFrameGenEnabled: Bool {
        return frameGenMode != .off
    }
    
    var isTemporalAAEnabled: Bool {
        return aaMode == .taa
    }
    
    var outputMultiplier: Float {
        guard isUpscalingEnabled else { return 1.0 }
        return scaleFactor.floatValue
    }
    
    var interpolatedFrameCount: Int {
        guard isFrameGenEnabled else { return 1 }
        return frameGenMultiplier.intValue
    }
    
    var effectiveTargetFPS: Int {
        guard isFrameGenEnabled else { return 60 }
        switch frameGenType {
        case .adaptive:
            return targetFPS.intValue
        case .fixed:
            return 60 * frameGenMultiplier.intValue
        }
    }
    
    func calculateAdaptiveMultiplier(sourceFPS: Float) -> Int {
        guard frameGenType == .adaptive, sourceFPS > 0 else {
            return frameGenMultiplier.intValue
        }
        let needed = Float(targetFPS.intValue) / sourceFPS
        return max(1, min(4, Int(ceil(needed))))
    }
    
    
    func saveProfile(_ name: String) {
        let defaults = UserDefaults.standard
        let prefix = "MetalGoose.Profile.\(name)."
        
        defaults.set(renderScale.rawValue, forKey: prefix + "renderScale")
        defaults.set(scalingType.rawValue, forKey: prefix + "scalingType")
        defaults.set(qualityMode.rawValue, forKey: prefix + "qualityMode")
        defaults.set(scaleFactor.rawValue, forKey: prefix + "scaleFactor")
        defaults.set(frameGenMode.rawValue, forKey: prefix + "frameGenMode")
        defaults.set(frameGenType.rawValue, forKey: prefix + "frameGenType")
        defaults.set(targetFPS.rawValue, forKey: prefix + "targetFPS")
        defaults.set(frameGenMultiplier.rawValue, forKey: prefix + "frameGenMultiplier")
        defaults.set(aaMode.rawValue, forKey: prefix + "aaMode")
        defaults.set(captureCursor, forKey: prefix + "captureCursor")
        defaults.set(reduceLatency, forKey: prefix + "reduceLatency")
        defaults.set(adaptiveSync, forKey: prefix + "adaptiveSync")
        defaults.set(showMGHUD, forKey: prefix + "showMGHUD")
        defaults.set(vsync, forKey: prefix + "vsync")
        defaults.set(sharpening, forKey: prefix + "sharpening")
        
        if !profiles.contains(name) {
            profiles.append(name)
            defaults.set(profiles, forKey: "MetalGoose.Profiles")
        }
    }
    
    func loadProfile(_ name: String) {
        let defaults = UserDefaults.standard
        let prefix = "MetalGoose.Profile.\(name)."
        
        if let rs = defaults.string(forKey: prefix + "renderScale"),
           let renderScaleValue = RenderScale(rawValue: rs) {
            renderScale = renderScaleValue
        }
        if let st = defaults.string(forKey: prefix + "scalingType"),
           let scalingTypeValue = ScalingType(rawValue: st) {
            scalingType = scalingTypeValue
        }
        if let qm = defaults.string(forKey: prefix + "qualityMode"),
           let qualityModeValue = QualityMode(rawValue: qm) {
            qualityMode = qualityModeValue
        }
        if let sf = defaults.string(forKey: prefix + "scaleFactor"),
           let scaleFactorValue = ScaleFactorOption(rawValue: sf) {
            scaleFactor = scaleFactorValue
        }
        if let fg = defaults.string(forKey: prefix + "frameGenMode"),
           let frameGenModeValue = FrameGenMode(rawValue: fg) {
            frameGenMode = frameGenModeValue
        }
        if let ft = defaults.string(forKey: prefix + "frameGenType"),
           let frameGenTypeValue = FrameGenType(rawValue: ft) {
            frameGenType = frameGenTypeValue
        }
        if let tfps = defaults.string(forKey: prefix + "targetFPS"),
           let targetFPSValue = TargetFPS(rawValue: tfps) {
            targetFPS = targetFPSValue
        }
        if let fm = defaults.string(forKey: prefix + "frameGenMultiplier"),
           let frameGenMultiplierValue = FrameGenMultiplier(rawValue: fm) {
            frameGenMultiplier = frameGenMultiplierValue
        }
        if let aa = defaults.string(forKey: prefix + "aaMode"),
           let aaModeValue = AAMode(rawValue: aa) {
            aaMode = aaModeValue
        }
        
        if defaults.object(forKey: prefix + "captureCursor") != nil {
            captureCursor = defaults.bool(forKey: prefix + "captureCursor")
        }
        if defaults.object(forKey: prefix + "reduceLatency") != nil {
            reduceLatency = defaults.bool(forKey: prefix + "reduceLatency")
        }
        if defaults.object(forKey: prefix + "adaptiveSync") != nil {
            adaptiveSync = defaults.bool(forKey: prefix + "adaptiveSync")
        }
        if defaults.object(forKey: prefix + "showMGHUD") != nil {
            showMGHUD = defaults.bool(forKey: prefix + "showMGHUD")
        }
        if defaults.object(forKey: prefix + "vsync") != nil {
            vsync = defaults.bool(forKey: prefix + "vsync")
        }
        if defaults.object(forKey: prefix + "sharpening") != nil {
            sharpening = defaults.float(forKey: prefix + "sharpening")
        }
        
        selectedProfile = name
    }
    
    func deleteProfile(_ name: String) {
        guard name != "Default" else { return }
        profiles.removeAll { $0 == name }
        
        let defaults = UserDefaults.standard
        defaults.set(profiles, forKey: "MetalGoose.Profiles")
        
        let prefix = "MetalGoose.Profile.\(name)."
        let keys = ["renderScale", "scalingType", "qualityMode", "scaleFactor",
                    "frameGenMode", "frameGenType", "targetFPS", "frameGenMultiplier", "aaMode",
                    "captureCursor", "reduceLatency", "adaptiveSync",
                    "showMGHUD", "vsync", "sharpening"]
        for key in keys {
            defaults.removeObject(forKey: prefix + key)
        }
    }
    
    
    init() {
        if let savedProfiles = UserDefaults.standard.array(forKey: "MetalGoose.Profiles") as? [String] {
            profiles = savedProfiles
        }
        loadProfile("Default")
    }
}


struct QualityProfile {
    let scalerMode: MTLFXSpatialScalerColorProcessingMode
    let sharpness: Float
    let temporalBlend: Float
}

@available(macOS 26.0, *)
extension CaptureSettings.QualityMode {
    var profile: QualityProfile {
        switch self {
        case .performance:
            return QualityProfile(scalerMode: .linear, sharpness: 0.3, temporalBlend: 0.15)
        case .balanced:
            return QualityProfile(scalerMode: .perceptual, sharpness: 0.5, temporalBlend: 0.1)
        case .ultra:
            return QualityProfile(scalerMode: .hdr, sharpness: 0.7, temporalBlend: 0.08)
        }
    }
}
