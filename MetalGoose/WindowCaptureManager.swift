import Foundation
import ScreenCaptureKit
import CoreGraphics
import CoreMedia
import IOSurface
import os

@available(macOS 26.0, *)
final class WindowCaptureManager: NSObject, ObservableObject, SCStreamDelegate, SCStreamOutput, @unchecked Sendable {
    
    private let captureQueue = DispatchQueue(label: "com.metalgoose.capture", qos: .userInteractive)
    
    @Published private(set) var isActive: Bool = false
    @Published private(set) var lastError: String?
    @Published private(set) var currentResolution: CGSize = .zero
    
    private var stream: SCStream?
    
    var onFrameReceived: ((_ surface: IOSurfaceRef, _ timestamp: Double) -> Void)?
    
    func startCapture(windowID: CGWindowID, refreshRate: Int) async -> Bool {
        await stopCapture()
        
        do {
            let availableContent = try await SCShareableContent.excludingDesktopWindows(false, onScreenWindowsOnly: true)
            
            guard let targetWindow = availableContent.windows.first(where: { $0.windowID == windowID }) else {
                lastError = "Error Code: MG-CAP-001 Target window not found."
                return false
            }
            
            currentResolution = targetWindow.frame.size
            
            let filter = SCContentFilter(desktopIndependentWindow: targetWindow)
            
            let config = SCStreamConfiguration()
            config.width = Int(targetWindow.frame.width)
            config.height = Int(targetWindow.frame.height)
            config.minimumFrameInterval = CMTime(value: 1, timescale: CMTimeScale(refreshRate))
            config.pixelFormat = kCVPixelFormatType_32BGRA
            config.showsCursor = false
            
            // Critical settings for minimum latency and zero-copy mapping
            config.queueDepth = 2
            config.captureResolution = .nominal
            config.shouldBeOpaque = false
            config.backgroundColor = .clear
            
            let captureStream = SCStream(filter: filter, configuration: config, delegate: self)
            try captureStream.addStreamOutput(self, type: .screen, sampleHandlerQueue: captureQueue)
            try await captureStream.startCapture()
            
            self.stream = captureStream
            self.isActive = true
            self.lastError = nil
            
            return true
            
        } catch {
            lastError = "Error Code: MG-CAP-002 ScreenCaptureKit start error: \(error.localizedDescription)"
            return false
        }
    }
    
    func stopCapture() async {
        if let currentStream = stream {
            do {
                try await currentStream.stopCapture()
            } catch {
                let nsError = error as NSError
                if !(nsError.domain == SCStreamErrorDomain && nsError.code == -3808) {
                    lastError = "Error Code: MG-CAP-003 ScreenCaptureKit stop error: \(error.localizedDescription)"
                }
            }
        }
        stream = nil
        isActive = false
    }
    
    nonisolated func stream(_ stream: SCStream, didStopWithError error: Error) {
        Task { @MainActor in
            let nsError = error as NSError
            if nsError.domain == SCStreamErrorDomain && nsError.code == -3808 { return }
            self.lastError = "Error Code: MG-CAP-004 Stream stopped with error: \(error.localizedDescription)"
            self.isActive = false
            self.stream = nil
        }
    }
    
    nonisolated func stream(_ stream: SCStream, didOutputSampleBuffer sampleBuffer: CMSampleBuffer, of type: SCStreamOutputType) {
        guard type == .screen else { return }
        
        guard let attachmentsArray = CMSampleBufferGetSampleAttachmentsArray(sampleBuffer, createIfNecessary: false) as? [[SCStreamFrameInfo: Any]],
              let attachments = attachmentsArray.first else {
            return
        }
        
        guard let statusRawValue = attachments[SCStreamFrameInfo.status] as? Int,
              let status = SCFrameStatus(rawValue: statusRawValue),
              status == .complete else {
            return
        }
        
        guard let pixelBuffer = CMSampleBufferGetImageBuffer(sampleBuffer) else { return }
        guard let surface = CVPixelBufferGetIOSurface(pixelBuffer)?.takeUnretainedValue() else { return }
        
        let timestamp = CMTimeGetSeconds(CMSampleBufferGetPresentationTimeStamp(sampleBuffer))
        
        onFrameReceived?(surface, timestamp)
    }
}
