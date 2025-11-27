import SwiftUI
import ScreenCaptureKit

final class CaptureSettings: ObservableObject {
    enum ScalingMode: String, CaseIterable, Identifiable {
        case fit
        case fill
        case integer
        
        var id: String { rawValue }
    }
    
    @Published var selectedDisplay: SCDisplay?
    @Published var selectedWindow: SCWindow?
    
    @Published var scalingMode: ScalingMode = .fit
    @Published var integerScale: Int = 2
    @Published var enableFrameGeneration: Bool = false
    @Published var frameGenerationRatio: Double = 0.0 // represents +1.0 in UI label (0..2)
    @Published var showFullscreenOverlayOnRun: Bool = true
    @Published var downscaleFactor: Double = 1.0 // 0.5..1.0 in UI
    @Published var sharpenAmount: Double = 0.0 // 0..1
    @Published var showFPSCounter: Bool = false
    
    func clamp() {
        // Placeholder for future clamping logic if needed
    }
}

