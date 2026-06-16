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
    }
    
    enum FrameGenMode: String, CaseIterable, Identifiable {
        case off = "Off"
        case mgfg1 = "MGFG-1"

        var id: String { rawValue }
    }

    enum AAMode: String, CaseIterable, Identifiable {
        case off = "Off"
        case fxaa = "FXAA"
        case smaa = "SMAA"

        var id: String { rawValue }
    }
    
    @Published var renderScale: RenderScale = .native
    @Published var scalingType: ScalingType = .off
    @Published var qualityMode: QualityMode = .ultra
    @Published var scaleFactor: ScaleFactorOption = .x1
    
    @Published var frameGenMode: FrameGenMode = .off

    @Published var frameGenMultiplier: Int = 2

    static let minFrameGenMultiplier = 2
    static let maxFrameGenMultiplier = 4

    @Published var aaMode: AAMode = .off

    @Published var captureCursor: Bool = true
    @Published var showMGHUD: Bool = true
    @Published var vsync: Bool = true

    @Published var bufferCount: Int = 3

    static let minBufferCount = 2
    static let maxBufferCount = 3

    var isFrameGenEnabled: Bool {
        return frameGenMode != .off
    }

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
