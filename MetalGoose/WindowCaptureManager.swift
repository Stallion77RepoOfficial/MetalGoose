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

    /// `startCapture` and `updateRenderScale` are nonisolated async, so they do
    /// not run on the main thread, while the HUD reads `capturePixelSize` from it.
    private let configLock = OSAllocatedUnfairLock()
    private var _basePixelSize: CGSize = .zero
    private var _currentRenderScale: Float = 1.0
    private var _capturePixelSize: CGSize = .zero
    private var _maxFPS: Int = 0
    private var _showsCursor: Bool = false
    private var _queueDepth: Int = 0

    /// Size ScreenCaptureKit is actually delivering. The compositor does the
    /// downscale, so nothing further down the pipeline pays for it.
    var capturePixelSize: CGSize {
        configLock.lock()
        defer { configLock.unlock() }
        return _capturePixelSize
    }

    /// Target window size in pixels before render scale. Frame generation runs
    /// here so that lowering render scale cannot degrade it.
    var nativePixelSize: CGSize {
        configLock.lock()
        defer { configLock.unlock() }
        return _basePixelSize
    }

    private var lastFrameSignature: UInt64 = 0
    private var hasLastSignature = false

    private var lastLumaGrid: [UInt8]?
    private var changedFractionEMA: Double?
    private var changedFractionVarEMA: Double = 0
    private var cutWarmupRemaining: Int = 0

    var onFrameReceived: ((_ surface: IOSurfaceRef, _ pixelBuffer: CVPixelBuffer, _ timestamp: Double, _ isSceneCut: Bool) -> Void)?

    /// The window's own display decides the pixel density. A missing screen means
    /// there is nothing to capture from, so 1.0 — unscaled — is the only neutral
    /// answer; assuming Retina would silently double the capture resolution.
    private static func backingScale(for windowFrame: CGRect) -> CGFloat {
        let primaryHeight = NSScreen.screens.first?.frame.height ?? 0
        let cocoaFrame = CGRect(
            x: windowFrame.origin.x,
            y: primaryHeight - windowFrame.maxY,
            width: windowFrame.width,
            height: windowFrame.height
        )
        let screen = NSScreen.screens.first { $0.frame.intersects(cocoaFrame) } ?? NSScreen.main
        return screen?.backingScaleFactor ?? 1.0
    }

    /// MetalFX will not build a scaler below this, so it is the floor for a
    /// render-scaled capture rather than a taste decision.
    private static let minimumCaptureDimension: CGFloat = 16

    private func makeConfiguration(renderScale: Float) -> SCStreamConfiguration {
        configLock.lock()
        let scaled = CGSize(
            width: max(Self.minimumCaptureDimension, (_basePixelSize.width * CGFloat(renderScale)).rounded()),
            height: max(Self.minimumCaptureDimension, (_basePixelSize.height * CGFloat(renderScale)).rounded()))
        _capturePixelSize = scaled
        _currentRenderScale = renderScale
        let maxFPS = _maxFPS
        let showsCursor = _showsCursor
        let queueDepth = _queueDepth
        configLock.unlock()

        let config = SCStreamConfiguration()
        config.width = Int(scaled.width)
        config.height = Int(scaled.height)
        if maxFPS > 0 {
            config.minimumFrameInterval = CMTime(value: 1, timescale: CMTimeScale(maxFPS))
        }
        config.pixelFormat = kCVPixelFormatType_32BGRA
        config.showsCursor = showsCursor

        // ScreenCaptureKit's queue and the render pipeline's buffer depth are the
        // same decision — a deeper capture queue than the pipeline will drain
        // only adds latency.
        if queueDepth > 0 {
            config.queueDepth = queueDepth
        }
        config.captureResolution = .best
        config.shouldBeOpaque = false
        config.backgroundColor = .clear
        return config
    }

    /// Applies render scale at the source. The frames arrive already reduced, so
    /// the GPU never encodes a downscale pass and every later stage — including
    /// the CPU surface scan — works on proportionally fewer pixels.
    func updateRenderScale(_ renderScale: Float) async {
        let (ready, changed) = configLock.withLock { () -> (Bool, Bool) in
            (_basePixelSize != .zero, abs(renderScale - _currentRenderScale) > 0.001)
        }
        guard let stream, ready, changed else { return }
        let config = makeConfiguration(renderScale: renderScale)
        do {
            try await stream.updateConfiguration(config)
        } catch {
            await MainActor.run {
                lastError = "Error Code: MG-CAP-007 Capture reconfiguration failed: \(error.localizedDescription)"
            }
        }
    }

    func startCapture(windowID: CGWindowID, maxFPS: Int, showsCursor: Bool,
                      renderScale: Float, queueDepth: Int) async -> Bool {
        await stopCapture()

        do {
            let availableContent = try await SCShareableContent.excludingDesktopWindows(false, onScreenWindowsOnly: true)

            guard let targetWindow = availableContent.windows.first(where: { $0.windowID == windowID }) else {
                await MainActor.run { lastError = "Error Code: MG-CAP-001 Target window not found." }
                return false
            }

            let filter = SCContentFilter(desktopIndependentWindow: targetWindow)

            let scale = Self.backingScale(for: targetWindow.frame)
            let base = CGSize(width: targetWindow.frame.width * scale,
                              height: targetWindow.frame.height * scale)
            configLock.withLock {
                _basePixelSize = base
                _maxFPS = maxFPS
                _showsCursor = showsCursor
                _queueDepth = queueDepth
            }
            let config = makeConfiguration(renderScale: renderScale)

            let captureStream = SCStream(filter: filter, configuration: config, delegate: self)
            try captureStream.addStreamOutput(self, type: .screen, sampleHandlerQueue: captureQueue)
            try await captureStream.startCapture()

            self.stream = captureStream
            hasLastSignature = false
            lastLumaGrid = nil
            changedFractionEMA = nil
            changedFractionVarEMA = 0
            cutWarmupRemaining = 0
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

        // A fixed sample budget, deliberately not derived from the frame size:
        // this runs on the capture thread for every frame, so its cost must stay
        // flat as resolution rises. A frame whose signature matches the previous
        // one is dropped, so the budget is spent on density — fine detail falling
        // between samples makes a genuinely new frame look like a duplicate.
        let ptr = base.assumingMemoryBound(to: UInt8.self)
        let gridX = 64
        let gridY = 64
        // The two outputs want opposite things from the same walk. The hash wants
        // the sharpest sample it can get, so it reads the grid point itself. The
        // luma statistic feeds a scene-cut test and wants the opposite: a single
        // pixel per grid point aliases badly once the frame is much larger than
        // the grid, and detail finer than the sample spacing — a pixel-scale
        // checkerboard, dense text — swings a sample the full range as it moves,
        // which reads as a cut in content that never cut. Averaging a small block
        // around each point resolves that structure into its mean, at a cost that
        // is still a constant multiple of the grid rather than of the frame.
        let blockSpan = 4
        let blockArea = blockSpan * blockSpan
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

                // Clamped so the block never leaves the surface at the right and
                // bottom edges, where the grid point is the last pixel.
                var sum = 0
                for by in 0..<blockSpan {
                    let sy = min(height - 1, y + by)
                    let srow = ptr + sy * bytesPerRow
                    for bx in 0..<blockSpan {
                        let sx = min(width - 1, x + bx)
                        let sp = srow + sx * 4
                        sum += Int((UInt16(sp[2]) * 54 + UInt16(sp[1]) * 183 + UInt16(sp[0]) * 19) >> 8)
                    }
                }
                luma[gy * gridX + gx] = UInt8(min(255, sum / blockArea))
            }
        }
        return (hash, luma)
    }

    /// A cut is an outlier in how much the frame changed, so it is detected as
    /// one: the mean absolute luma change over the sample grid is tracked with a
    /// running mean and variance, and a frame is a cut when it sits more than
    /// `cutSigma` standard deviations above that.
    ///
    /// This replaces a per-cell threshold plus a floor plus a cap, which were
    /// three fixed numbers that fought the adaptive part — content whose normal
    /// motion sat above the floor could never register a cut, and content quieter
    /// than the cap could never clear it.
    private static let cutSigma = 6.0

    /// A change smaller than this is not a cut whatever the running statistics
    /// say. A cut replaces the image, so it moves a large fraction of the grid a
    /// long way; ordinary camera motion does not reach a mean absolute luma
    /// change of an eighth of full range. Without a floor, an estimator whose
    /// variance has not grown yet puts the threshold at the mean and calls every
    /// above-average frame a cut.
    private static let minCutChange = 0.125

    /// Frames of history collected before any cut may be reported, so the spread
    /// is measured rather than assumed.
    private static let cutWarmupFrames = 30

    private func isSceneCut(previous: [UInt8]?, current: [UInt8]) -> Bool {
        guard let previous, previous.count == current.count, !current.isEmpty else { return false }

        var total = 0
        for i in 0..<current.count {
            total += abs(Int(current[i]) - Int(previous[i]))
        }
        // Normalised to 0...1 against the full luma range, so the statistic does
        // not depend on grid size or bit depth.
        let change = Double(total) / Double(current.count * Int(UInt8.max))

        guard let mean = changedFractionEMA else {
            changedFractionEMA = change
            changedFractionVarEMA = 0
            cutWarmupRemaining = Self.cutWarmupFrames
            return false
        }

        let threshold = max(Self.minCutChange,
                            mean + Self.cutSigma * sqrt(changedFractionVarEMA))
        let isCut = cutWarmupRemaining == 0 && change > threshold
        if cutWarmupRemaining > 0 { cutWarmupRemaining -= 1 }

        // Every frame updates the baseline, but a cut enters it held at the
        // threshold rather than at its own value: it must not pull the mean up
        // behind it, or the frames after a transition inherit its statistics and
        // the next cut is missed.
        //
        // Skipping the update entirely — which is what this did — censors the
        // sample down to the frames that passed the test. The spread then gets
        // measured from the quiet half alone, so it collapses toward zero, the
        // threshold collapses onto the mean, and from there every frame that
        // moves more than average is a cut. In gameplay that is about half of
        // them, which is what killed interpolation: the schedule reached the
        // generator and was turned away at the cut check.
        let observed = min(change, threshold)
        let alpha = emaAlpha
        let deviation = observed - mean
        changedFractionVarEMA += (deviation * deviation - changedFractionVarEMA) * alpha
        changedFractionEMA = mean + alpha * deviation

        return isCut
    }

    /// The baseline spans roughly one second of capture whatever the frame rate
    /// is, instead of a fixed number of frames that means half a second at 20 fps
    /// and a tenth of one at 120.
    private var emaAlpha: Double {
        let fps = configLock.withLock { _maxFPS }
        return 1.0 / Double(max(8, fps))
    }
}
