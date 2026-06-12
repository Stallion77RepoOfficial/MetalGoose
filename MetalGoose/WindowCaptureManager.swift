import Foundation
import AppKit
import ScreenCaptureKit
import CoreGraphics
import CoreMedia
import CoreVideo
import IOSurface
import os

@available(macOS 26.0, *)
final class WindowCaptureManager: NSObject, ObservableObject, SCStreamDelegate, SCStreamOutput, @unchecked Sendable {

    private let captureQueue = DispatchQueue(label: "com.metalgoose.capture", qos: .userInteractive)

    @Published private(set) var isActive: Bool = false
    @Published private(set) var lastError: String?
    @Published private(set) var currentResolution: CGSize = .zero

    private var stream: SCStream?

    // The pixel buffer must be retained until the GPU finishes reading the
    // IOSurface, otherwise ScreenCaptureKit recycles it mid-frame.
    var onFrameReceived: ((_ surface: IOSurfaceRef, _ pixelBuffer: CVPixelBuffer, _ timestamp: Double) -> Void)?

    // SCWindow frames are in global top-left coordinates while NSScreen
    // frames are bottom-left, so convert before matching the screen
    private static func backingScale(for windowFrame: CGRect) -> CGFloat {
        let primaryHeight = NSScreen.screens.first?.frame.height ?? 0
        let cocoaFrame = CGRect(
            x: windowFrame.origin.x,
            y: primaryHeight - windowFrame.maxY,
            width: windowFrame.width,
            height: windowFrame.height
        )
        let screen = NSScreen.screens.first { $0.frame.intersects(cocoaFrame) } ?? NSScreen.main
        return screen?.backingScaleFactor ?? 2.0
    }

    /// Captures the window at up to `maxFPS` frames per second (typically the
    /// display's refresh rate). The real source frame rate is recovered by
    /// dropping duplicate frames, so frame generation has headroom to work.
    func startCapture(windowID: CGWindowID, maxFPS: Int) async -> Bool {
        await stopCapture()

        do {
            let availableContent = try await SCShareableContent.excludingDesktopWindows(false, onScreenWindowsOnly: true)

            guard let targetWindow = availableContent.windows.first(where: { $0.windowID == windowID }) else {
                await MainActor.run { lastError = "Error Code: MG-CAP-001 Target window not found." }
                return false
            }

            let filter = SCContentFilter(desktopIndependentWindow: targetWindow)

            // Capture at pixel resolution, not point resolution, to avoid
            // halving the source quality on Retina displays
            let scale = Self.backingScale(for: targetWindow.frame)
            let pixelSize = CGSize(width: targetWindow.frame.width * scale,
                                   height: targetWindow.frame.height * scale)

            let config = SCStreamConfiguration()
            config.width = Int(pixelSize.width)
            config.height = Int(pixelSize.height)
            config.minimumFrameInterval = CMTime(value: 1, timescale: CMTimeScale(max(1, maxFPS)))
            config.pixelFormat = kCVPixelFormatType_32BGRA
            // The cursor is never baked into the captured frame; the real
            // system cursor floats over the overlay at native resolution and
            // zero latency, so there is exactly one cursor on screen.
            config.showsCursor = false

            // Critical settings for minimum latency and zero-copy mapping
            config.queueDepth = 3
            config.captureResolution = .best
            config.shouldBeOpaque = false
            config.backgroundColor = .clear

            let captureStream = SCStream(filter: filter, configuration: config, delegate: self)
            try captureStream.addStreamOutput(self, type: .screen, sampleHandlerQueue: captureQueue)
            try await captureStream.startCapture()

            self.stream = captureStream
            await MainActor.run {
                currentResolution = pixelSize
                isActive = true
                lastError = nil
            }

            return true

        } catch {
            await MainActor.run {
                lastError = "Error Code: MG-CAP-002 ScreenCaptureKit start error: \(error.localizedDescription)"
            }
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
                    await MainActor.run {
                        lastError = "Error Code: MG-CAP-003 ScreenCaptureKit stop error: \(error.localizedDescription)"
                    }
                }
            }
        }
        stream = nil
        await MainActor.run { isActive = false }
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
            // .idle / .blank / .suspended frames carry no new content
            return
        }

        // Drop duplicate frames: when ScreenCaptureKit reports no dirty rects,
        // the window content is identical to the previous frame. Skipping these
        // makes the reported capture rate reflect the real source frame rate and
        // gives frame generation genuinely different frames to interpolate.
        // If the attachment is absent we conservatively treat the frame as new.
        if let dirtyRects = attachments[SCStreamFrameInfo.dirtyRects] as? [[String: Any]],
           dirtyRects.isEmpty {
            return
        }

        guard let pixelBuffer = CMSampleBufferGetImageBuffer(sampleBuffer) else { return }
        guard let surface = CVPixelBufferGetIOSurface(pixelBuffer)?.takeUnretainedValue() else { return }

        let timestamp = CMTimeGetSeconds(CMSampleBufferGetPresentationTimeStamp(sampleBuffer))

        onFrameReceived?(surface, pixelBuffer, timestamp)
    }
}
