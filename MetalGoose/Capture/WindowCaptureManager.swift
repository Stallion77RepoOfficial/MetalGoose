import Foundation
import ScreenCaptureKit
import CoreGraphics
import CoreMedia
import CoreVideo
import IOSurface
import os

final class WindowCaptureManager: NSObject, SCStreamDelegate, SCStreamOutput, @unchecked Sendable {

    private let captureQueue = DispatchQueue(label: "com.metalgoose.capture", qos: .userInteractive)

    /// Everything written by the `async` start/stop/reconfigure methods, which run on the
    /// cooperative pool, and read from the main thread (the HUD) and the capture queue (every
    /// frame). One lock, one snapshot per use.
    private struct State {
        var stream: SCStream?
        var windowPixelSize: CGSize = .zero
        var capturePixelSize: CGSize = .zero
        var renderScale: Float = 1
        var maxFPS = 0
        var showsCursor = false
        var queueDepth = 0
        var lastError: MGError?
        var onFrame: (@Sendable (CapturedFrame) -> Void)?
        var onStop: (@Sendable (MGError?) -> Void)?
    }
    private let state = OSAllocatedUnfairLock(uncheckedState: State())

    /// Touched only from `captureQueue`, which is where every frame arrives.
    private var lastSignature: UInt64?
    private var lastLuma: [UInt8]?
    private var sceneCuts = SceneCutDetector()

    var lastError: MGError? { state.withLockUnchecked { $0.lastError } }

    /// Size ScreenCaptureKit is actually delivering. The compositor does the downscale, so
    /// nothing further down the pipeline pays for it.
    var capturePixelSize: CGSize { state.withLockUnchecked { $0.capturePixelSize } }

    /// Delivered for every frame that is not a repeat of the one before. Set before
    /// `startCapture`, because frames begin the moment the stream does.
    var onFrame: (@Sendable (CapturedFrame) -> Void)? {
        get { state.withLockUnchecked { $0.onFrame } }
        set { state.withLockUnchecked { $0.onFrame = newValue } }
    }

    /// Called when the stream ends on its own — the target window closed, or the system
    /// stopped it. `nil` means it ended without an error worth showing.
    var onStop: (@Sendable (MGError?) -> Void)? {
        get { state.withLockUnchecked { $0.onStop } }
        set { state.withLockUnchecked { $0.onStop = newValue } }
    }

    /// MetalFX will not build a scaler below this, so it is the floor for a render-scaled
    /// capture rather than a taste decision.
    private static let minimumCaptureDimension: CGFloat = 16

    private static func scaledSize(_ native: CGSize, by renderScale: Float) -> CGSize {
        CGSize(width: max(minimumCaptureDimension, (native.width * CGFloat(renderScale)).rounded()),
               height: max(minimumCaptureDimension, (native.height * CGFloat(renderScale)).rounded()))
    }

    private func makeConfiguration(_ snapshot: State) -> SCStreamConfiguration {
        let config = SCStreamConfiguration()
        config.width = Int(snapshot.capturePixelSize.width)
        config.height = Int(snapshot.capturePixelSize.height)
        if snapshot.maxFPS > 0 {
            config.minimumFrameInterval = CMTime(value: 1, timescale: CMTimeScale(snapshot.maxFPS))
        }
        config.pixelFormat = kCVPixelFormatType_32BGRA
        config.showsCursor = snapshot.showsCursor
        // The window's own shadow is not part of what is being scaled. Including it squeezes
        // the shadow and the window into a surface sized for the window alone, so the content
        // lands smaller than the window it is drawn over.
        config.ignoreShadowsSingleWindow = true

        // ScreenCaptureKit's queue and the render pipeline's buffer depth are the same
        // decision — a deeper capture queue than the pipeline will drain only adds latency.
        if snapshot.queueDepth > 0 {
            config.queueDepth = snapshot.queueDepth
        }
        config.captureResolution = .best
        config.shouldBeOpaque = false
        config.backgroundColor = .clear
        return config
    }

    func startCapture(target: CaptureTarget, maxFPS: Int, showsCursor: Bool,
                      renderScale: Float, queueDepth: Int) async -> Bool {
        await stopCapture()

        do {
            let content = try await SCShareableContent.excludingDesktopWindows(false, onScreenWindowsOnly: true)
            guard let window = content.windows.first(where: { $0.windowID == target.windowID }) else {
                state.withLockUnchecked { $0.lastError = .captureWindowNotFound }
                return false
            }

            let native = target.pixelSize
            let snapshot = state.withLockUnchecked { state -> State in
                state.windowPixelSize = native
                state.capturePixelSize = Self.scaledSize(native, by: renderScale)
                state.renderScale = renderScale
                state.maxFPS = maxFPS
                state.showsCursor = showsCursor
                state.queueDepth = queueDepth
                state.lastError = nil
                return state
            }

            // Reset ahead of the stream rather than after it starts: the queue is serial, so
            // this runs before the first frame can arrive, where a reset made once the stream
            // is running races the frames already being handled.
            captureQueue.async { [self] in
                lastSignature = nil
                lastLuma = nil
                sceneCuts.reset()
            }

            let stream = SCStream(filter: SCContentFilter(desktopIndependentWindow: window),
                                  configuration: makeConfiguration(snapshot), delegate: self)
            try stream.addStreamOutput(self, type: .screen, sampleHandlerQueue: captureQueue)
            try await stream.startCapture()
            state.withLockUnchecked { $0.stream = stream }
            return true
        } catch {
            state.withLockUnchecked { $0.lastError = .captureStartFailed(error) }
            return false
        }
    }

    func stopCapture() async {
        guard let stream = state.withLockUnchecked({ state -> SCStream? in
            defer { state.stream = nil }
            return state.stream
        }) else { return }

        do {
            try await stream.stopCapture()
        } catch {
            // -3808: the stream was already stopping, which is what was asked for.
            let nsError = error as NSError
            if !(nsError.domain == SCStreamErrorDomain && nsError.code == -3808) {
                state.withLockUnchecked { $0.lastError = .captureStopFailed(error) }
            }
        }
    }

    /// Applies render scale and the window's size at the source, without restarting the
    /// stream. The frames arrive already reduced, so the GPU never encodes a downscale pass
    /// and every later stage works on proportionally fewer pixels.
    func reconfigure(renderScale: Float? = nil, window: CaptureTarget? = nil) async {
        let update = state.withLockUnchecked { state -> (SCStream, State)? in
            guard let stream = state.stream else { return nil }
            let native = window?.pixelSize ?? state.windowPixelSize
            let scale = renderScale ?? state.renderScale
            let capture = Self.scaledSize(native, by: scale)
            guard capture != state.capturePixelSize else { return nil }

            state.windowPixelSize = native
            state.renderScale = scale
            state.capturePixelSize = capture
            return (stream, state)
        }
        guard let (stream, snapshot) = update else { return }

        do {
            try await stream.updateConfiguration(makeConfiguration(snapshot))
        } catch {
            state.withLockUnchecked { $0.lastError = .captureReconfigurationFailed(error) }
        }
    }

    // MARK: - SCStreamDelegate

    nonisolated func stream(_ stream: SCStream, didStopWithError error: Error) {
        let nsError = error as NSError
        let isSCKError = nsError.domain == SCStreamErrorDomain
        // -3808: stopped on request. -3817: the user stopped sharing from the system menu.
        // Neither is a failure.
        let benign = isSCKError && (nsError.code == -3808 || nsError.code == -3817)
        let failure: MGError? = benign ? nil : .captureStreamStopped(error)

        let callback = state.withLockUnchecked { state -> (@Sendable (MGError?) -> Void)? in
            state.stream = nil
            if let failure { state.lastError = failure }
            return state.onStop
        }
        callback?(failure)
    }

    // MARK: - SCStreamOutput

    nonisolated func stream(_ stream: SCStream, didOutputSampleBuffer sampleBuffer: CMSampleBuffer,
                            of type: SCStreamOutputType) {
        guard type == .screen,
              let attachments = (CMSampleBufferGetSampleAttachmentsArray(sampleBuffer, createIfNecessary: false)
                                    as? [[SCStreamFrameInfo: Any]])?.first,
              let rawStatus = attachments[SCStreamFrameInfo.status] as? Int,
              SCFrameStatus(rawValue: rawStatus) == .complete,
              let pixelBuffer = CMSampleBufferGetImageBuffer(sampleBuffer),
              let surface = CVPixelBufferGetIOSurface(pixelBuffer)?.takeUnretainedValue() else { return }

        let (callback, native, maxFPS) = state.withLockUnchecked { ($0.onFrame, $0.windowPixelSize, $0.maxFPS) }

        // A frame whose pixels match the previous one's carries nothing new. A surface that
        // cannot be sampled is passed through rather than guessed at.
        var isSceneCut = false
        if let sample = FrameSampler.sample(surface) {
            if sample.signature == lastSignature { return }
            lastSignature = sample.signature
            isSceneCut = sceneCuts.isCut(previous: lastLuma, current: sample.luma,
                                         alpha: SceneCutDetector.alpha(forFrameRate: maxFPS))
            lastLuma = sample.luma
        }

        callback?(CapturedFrame(pixelBuffer: pixelBuffer, surface: surface,
                                captureTime: CMTimeGetSeconds(CMSampleBufferGetPresentationTimeStamp(sampleBuffer)),
                                isSceneCut: isSceneCut, nativePixelSize: native))
    }
}
