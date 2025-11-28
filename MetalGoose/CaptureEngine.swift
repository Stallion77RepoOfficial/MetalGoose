import ScreenCaptureKit
import CoreVideo
import CoreMedia

@MainActor
final class CaptureEngine: NSObject, ObservableObject, SCStreamOutput {
    @Published var currentFrame: CVPixelBuffer?
    private var stream: SCStream?
    private let videoOutputQueue = DispatchQueue(label: "com.metalgoose.video", qos: .userInteractive)
    
    func startCapture(targetWindowID: CGWindowID?, displayID: CGDirectDisplayID?, captureCursor: Bool, queueDepth: Int) async throws {
        let content = try await SCShareableContent.excludingDesktopWindows(false, onScreenWindowsOnly: true)
        var filter: SCContentFilter
        var captureSize: CGSize?

        if let winID = targetWindowID, let window = content.windows.first(where: { $0.windowID == winID }) {
            filter = SCContentFilter(desktopIndependentWindow: window)
            captureSize = window.frame.size
        } else if let dispID = displayID, let display = content.displays.first(where: { $0.displayID == dispID }) {
            filter = SCContentFilter(display: display, excludingWindows: [])
            captureSize = display.frame.size
        } else {
            throw NSError(domain: "Capture", code: 404, userInfo: [NSLocalizedDescriptionKey: "Target window/display not found"])
        }

        let config = SCStreamConfiguration()
        if let size = captureSize {
            config.width = Int(size.width)
            config.height = Int(size.height)
        }
        config.pixelFormat = kCVPixelFormatType_32BGRA
        config.minimumFrameInterval = CMTime(value: 1, timescale: 120)
        config.queueDepth = queueDepth
        config.showsCursor = captureCursor

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
        Task { @MainActor in
            self.currentFrame = buffer
        }
    }
}

