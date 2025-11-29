import SwiftUI
import Vision
import MetalFX
import VideoToolbox

class CaptureSettings: ObservableObject {
    static let shared = CaptureSettings()
    
    // --- PROFILES ---
    @Published var selectedProfile: String = "Default"
    
    // --- SCALING OPTIONS ---
    enum ScalingType: String, CaseIterable, Identifiable {
        case metalFX = "MetalFX (AI Upscale)"
        case integer = "Integer (Pixel Art)"
        case bicubic = "Bicubic (Soft)"
        var id: String { rawValue }
    }
    
    enum MetalFXMode: String, CaseIterable, Identifiable {
        case spatial = "Spatial"
        case temporal = "Temporal"
        case temporalDenoised = "Temporal Denoised"
        var id: String { rawValue }
        var requiresMotionVectors: Bool { self != .spatial }
    }
    
    enum QualityMode: String, CaseIterable, Identifiable {
        case performance = "Performance"
        case balanced = "Balanced"
        case quality = "Quality"
        var id: String { rawValue }
    }
    
    @Published var scalingType: ScalingType = .metalFX
    @Published var metalFXMode: MetalFXMode = .spatial
    @Published var qualityMode: QualityMode = .quality
    @Published var scaleFactor: Float = 2.0
    
    // --- FRAME GENERATION ---
    enum FrameGenMode: String, CaseIterable, Identifiable {
        case off = "Off"
        case x2 = "MGFG 2.0 (x2)"
        case x3 = "MGFG 3.0 (x3)"
        var id: String { rawValue }
    }
    
    enum FrameGenBackend: String, CaseIterable, Identifiable {
        case vision = "Vision Optical Flow"
        case metalFX = "MetalFX Frame Interpolator"
        var id: String { rawValue }
    }
    
    @Published var frameGenMode: FrameGenMode = .x2
    @Published var frameGenBackend: FrameGenBackend = .vision
    
    // --- CURSOR & HUD ---
    @Published var captureCursor: Bool = true
    @Published var clipCursor: Bool = false
    @Published var showFPS: Bool = true
    @Published var vsync: Bool = true
    @Published var maxLatency: Int = 1
}

// Quality Profile structure
struct QualityProfile {
    let flowAccuracy: VNGenerateOpticalFlowRequest.ComputationAccuracy
    let scalerMode: MTLFXSpatialScalerColorProcessingMode
    let vtScalingMode: CFString
}

extension CaptureSettings.QualityMode {
    var profile: QualityProfile {
        switch self {
        case .performance:
            return QualityProfile(flowAccuracy: .low,
                                  scalerMode: .linear,
                                  vtScalingMode: kVTScalingMode_Normal)
        case .balanced:
            return QualityProfile(flowAccuracy: .medium,
                                  scalerMode: .perceptual,
                                  vtScalingMode: kVTScalingMode_Letterbox)
        case .quality:
            return QualityProfile(flowAccuracy: .high,
                                  scalerMode: .hdr,
                                  vtScalingMode: kVTScalingMode_Trim)
        }
    }
}
