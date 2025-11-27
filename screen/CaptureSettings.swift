import SwiftUI
import ScreenCaptureKit

final class CaptureSettings: ObservableObject {
    enum ScalingMode: String, CaseIterable, Identifiable {
        case aspectFit = "Aspect Fit"
        case integer = "Integer Scaling"
        case fullscreen = "Fullscreen Stretch"
        
        var id: String { rawValue }
    }
    
    // UI Bindings
    @Published var scalingMode: ScalingMode = .aspectFit
    @Published var sharpenAmount: Double = 0.5 // Default sharpness
    @Published var targetFPS: Int = 60
    @Published var showFPS: Bool = true
    
    // Internal States
    @Published var selectedWindowID: CGWindowID?
    @Published var selectedDisplayID: CGDirectDisplayID?
}
