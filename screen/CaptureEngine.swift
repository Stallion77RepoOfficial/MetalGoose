import ScreenCaptureKit
import CoreVideo
import Foundation

@MainActor
final class CaptureEngine: NSObject, ObservableObject, SCStreamOutput {
    @Published var isCapturing: Bool = false
    @Published var currentFrame: CVPixelBuffer?
    
    private var stream: SCStream?
    private let videoOutputQueue = DispatchQueue(label: "com.metalgoose.video", qos: .userInteractive)
    
    func startCapture(windowID: CGWindowID, displayID: CGDirectDisplayID) async {
        guard !isCapturing else { return }
        
        do {
            let content = try await SCShareableContent.excludingDesktopWindows(false, onScreenWindowsOnly: true)
            
            guard let window = content.windows.first(where: { $0.windowID == windowID }) else {
                print("Window not found!")
                return
            }
            
            // Filter: Capture specific window
            let filter = SCContentFilter(desktopIndependentWindow: window)
            
            // Configuration: High Performance
            let config = SCStreamConfiguration()
            config.width = Int(window.frame.width * 2) // Capture at higher res for better downsampling
            config.height = Int(window.frame.height * 2)
            config.pixelFormat = kCVPixelFormatType_32BGRA
            config.minimumFrameInterval = CMTime(value: 1, timescale: 120) // Target high FPS
            config.queueDepth = 5
            config.showsCursor = true
            
            stream = SCStream(filter: filter, configuration: config, delegate: nil)
            try stream?.addStreamOutput(self, type: .screen, sampleHandlerQueue: videoOutputQueue)
            try await stream?.startCapture()
            
            isCapturing = true
            
        } catch {
            print("Capture failed: \(error)")
            isCapturing = false
        }
    }
    
    func stopCapture() {
        stream?.stopCapture()
        stream = nil
        isCapturing = false
        currentFrame = nil
    }
    
    nonisolated func stream(_ stream: SCStream, didOutputSampleBuffer sampleBuffer: CMSampleBuffer, of type: SCStreamOutputType) {
        guard type == .screen, let buffer = sampleBuffer.imageBuffer else { return }
        
        Task { @MainActor in
            self.currentFrame = buffer
        }
    }
}
