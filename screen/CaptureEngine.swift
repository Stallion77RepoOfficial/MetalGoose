import ScreenCaptureKit
import CoreVideo

@MainActor
final class CaptureEngine: NSObject, ObservableObject, SCStreamOutput {
    @Published var currentFrame: CVPixelBuffer?
    private var stream: SCStream?
    private let videoOutputQueue = DispatchQueue(label: "com.metalgoose.video", qos: .userInteractive)
    
    func startCapture(targetWindowID: CGWindowID, scaleFactor: Float) async throws {
        let content = try await SCShareableContent.excludingDesktopWindows(false, onScreenWindowsOnly: true)
        
        guard let window = content.windows.first(where: { $0.windowID == targetWindowID }) else {
            throw NSError(domain: "Capture", code: 404, userInfo: [NSLocalizedDescriptionKey: "TARGET_WINDOW_LOST"])
        }
        
        let filter = SCContentFilter(desktopIndependentWindow: window)
        
        let config = SCStreamConfiguration()
        config.width = Int(window.frame.width)
        config.height = Int(window.frame.height)
        config.pixelFormat = kCVPixelFormatType_32BGRA
        config.minimumFrameInterval = CMTime(value: 1, timescale: 120)
        config.queueDepth = 6
        config.showsCursor = true
        
        stream = SCStream(filter: filter, configuration: config, delegate: nil)
        try stream?.addStreamOutput(self, type: .screen, sampleHandlerQueue: videoOutputQueue)
        try await stream?.startCapture()
    }
    
    func stopCapture() {
        stream?.stopCapture()
        stream = nil
        currentFrame = nil
    }
    
    nonisolated func stream(_ stream: SCStream, didOutputSampleBuffer sampleBuffer: CMSampleBuffer, of type: SCStreamOutputType) {
        guard type == .screen, let buffer = sampleBuffer.imageBuffer else { return }
        
        // UI Threading hatasını önlemek için MainActor kullanımı
        Task { @MainActor in
            self.currentFrame = buffer
        }
    }
}
