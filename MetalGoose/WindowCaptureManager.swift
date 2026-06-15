import Foundation
import AppKit
import ScreenCaptureKit
import CoreGraphics
import CoreMedia
import CoreVideo
import IOSurface
import os

final class WindowCaptureManager: NSObject, SCStreamDelegate, SCStreamOutput, @unchecked Sendable {

    private let captureQueue = DispatchQueue(label: "com.metalgoose.capture", qos: .userInteractive)

    private(set) var lastError: String?

    private var stream: SCStream?

    private var lastFrameSignature: UInt64 = 0
    private var hasLastSignature = false

    private var lastLumaGrid: [UInt8]?
    private var changedFractionEMA: Double?
    private var changedFractionVarEMA: Double = 0

    var onFrameReceived: ((_ surface: IOSurfaceRef, _ pixelBuffer: CVPixelBuffer, _ timestamp: Double, _ isSceneCut: Bool) -> Void)?

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

    func startCapture(windowID: CGWindowID, maxFPS: Int, showsCursor: Bool) async -> Bool {
        await stopCapture()

        do {
            let availableContent = try await SCShareableContent.excludingDesktopWindows(false, onScreenWindowsOnly: true)

            guard let targetWindow = availableContent.windows.first(where: { $0.windowID == windowID }) else {
                await MainActor.run { lastError = "Error Code: MG-CAP-001 Target window not found." }
                return false
            }

            let filter = SCContentFilter(desktopIndependentWindow: targetWindow)

            let scale = Self.backingScale(for: targetWindow.frame)
            let pixelSize = CGSize(width: targetWindow.frame.width * scale,
                                   height: targetWindow.frame.height * scale)

            let config = SCStreamConfiguration()
            config.width = Int(pixelSize.width)
            config.height = Int(pixelSize.height)
            config.minimumFrameInterval = CMTime(value: 1, timescale: CMTimeScale(max(1, maxFPS)))
            config.pixelFormat = kCVPixelFormatType_32BGRA
            config.showsCursor = showsCursor

            config.queueDepth = 3
            config.captureResolution = .best
            config.shouldBeOpaque = false
            config.backgroundColor = .clear

            let captureStream = SCStream(filter: filter, configuration: config, delegate: self)
            try captureStream.addStreamOutput(self, type: .screen, sampleHandlerQueue: captureQueue)
            try await captureStream.startCapture()

            self.stream = captureStream
            hasLastSignature = false
            lastLumaGrid = nil
            changedFractionEMA = nil
            changedFractionVarEMA = 0
            await MainActor.run {
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
    }

    nonisolated func stream(_ stream: SCStream, didStopWithError error: Error) {
        Task { @MainActor in
            let nsError = error as NSError
            if nsError.domain == SCStreamErrorDomain && nsError.code == -3808 { return }
            self.lastError = "Error Code: MG-CAP-004 Stream stopped with error: \(error.localizedDescription)"
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

        let (signature, lumaGrid) = Self.frameSignatureAndLuma(surface)
        if hasLastSignature && signature == lastFrameSignature {
            return
        }
        lastFrameSignature = signature
        hasLastSignature = true

        let isSceneCut = self.isSceneCut(previous: lastLumaGrid, current: lumaGrid)
        lastLumaGrid = lumaGrid

        let timestamp = CMTimeGetSeconds(CMSampleBufferGetPresentationTimeStamp(sampleBuffer))

        onFrameReceived?(surface, pixelBuffer, timestamp, isSceneCut)
    }

    private static func frameSignatureAndLuma(_ surface: IOSurfaceRef) -> (UInt64, [UInt8]) {
        IOSurfaceLock(surface, .readOnly, nil)
        defer { IOSurfaceUnlock(surface, .readOnly, nil) }

        let base = IOSurfaceGetBaseAddress(surface)
        let width = IOSurfaceGetWidth(surface)
        let height = IOSurfaceGetHeight(surface)
        let bytesPerRow = IOSurfaceGetBytesPerRow(surface)
        guard width > 0, height > 0, bytesPerRow > 0 else { return (0, []) }

        let ptr = base.assumingMemoryBound(to: UInt8.self)
        let gridX = 64
        let gridY = 64
        var hash: UInt64 = 0xcbf29ce484222325
        var luma = [UInt8](repeating: 0, count: gridX * gridY)

        for gy in 0..<gridY {
            let y = (height - 1) * gy / (gridY - 1)
            let row = ptr + y * bytesPerRow
            for gx in 0..<gridX {
                let x = (width - 1) * gx / (gridX - 1)
                let px = row + x * 4
                let value = UInt32(px[0]) | (UInt32(px[1]) << 8) |
                            (UInt32(px[2]) << 16) | (UInt32(px[3]) << 24)
                hash = (hash ^ UInt64(value)) &* 0x100000001b3
                let l = (UInt16(px[2]) * 54 + UInt16(px[1]) * 183 + UInt16(px[0]) * 19) >> 8
                luma[gy * gridX + gx] = UInt8(l)
            }
        }
        return (hash, luma)
    }

    private func isSceneCut(previous: [UInt8]?, current: [UInt8]) -> Bool {
        guard let previous, previous.count == current.count, !current.isEmpty else { return false }

        let perCellThreshold = 50
        var changedCells = 0
        for i in 0..<current.count {
            if abs(Int(current[i]) - Int(previous[i])) > perCellThreshold {
                changedCells += 1
            }
        }
        let changedFraction = Double(changedCells) / Double(current.count)

        let baseThreshold = 0.6

        guard let mean = changedFractionEMA else {
            changedFractionEMA = changedFraction
            return false
        }

        let stddev = sqrt(changedFractionVarEMA)
        let absoluteCap = 0.85
        let threshold = max(baseThreshold, min(mean + 4.0 * stddev, absoluteCap))
        let isCut = changedFraction > threshold

        if !isCut {
            let alpha = 0.1
            let deviation = changedFraction - mean
            changedFractionVarEMA = changedFractionVarEMA * (1 - alpha) + deviation * deviation * alpha
            changedFractionEMA = mean + alpha * deviation
        }

        return isCut
    }
}
