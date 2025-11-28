import SwiftUI

class CaptureSettings: ObservableObject {
    @Published var scaleFactor: Float = 2.0
    @Published var qualityMode: QualityMode = .quality
    @Published var showFPSOverlay: Bool = false
    
    enum QualityMode: String, CaseIterable, Identifiable {
        case performance = "PERFORMANCE"
        case balanced = "BALANCED"
        case quality = "QUALITY"
        var id: String { rawValue }
    }
    
    @Published var frameGenRatio: Int = 2
    
    static let shared = CaptureSettings()
}
