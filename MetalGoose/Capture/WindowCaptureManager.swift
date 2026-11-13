import Foundation
import ScreenCaptureKit
import CoreGraphics
import CoreMedia
import CoreVideo
import IOSurface
import os

final class WindowCaptureManager: NSObject, SCStreamDelegate, SCStreamOutput, @unchecked Sendable {

    private let captureQueue = DispatchQueue(label: "com.metalgoose.capture", qos: .userInteractive)
    private let configurationGate = AsyncSerialGate()

    /// Everything written by the `async` start/stop/reconfigure methods, which run on the
    /// cooperative pool, and read from the main thread (the HUD) and the capture queue (every
    /// frame). One lock, one snapshot per use.
    private struct State {
        var stream: SCStream?
        /// The window alone, which the stream captures unless the app has windows over it.
        var windowFilter: SCContentFilter?
        var application: SCRunningApplication?
        var windowFrame: CGRect = .zero
        /// Pixels per point on the display the window is on.
        var backingScale: CGFloat = 1
        /// How the window's content was magnified to the window's pixels, where it is drawn at half of them
        /// (`ContentResolution`); nil where it is not, or not yet known to be.
        var magnification: ContentResolution.Magnification?
        /// Whether that is still to be found out.
        var testsContent = false
        var windowPixelSize: CGSize = .zero
        var capturePixelSize: CGSize = .zero
        var renderScale: Float = 1
        var maxFPS = 0
        var pipelineDepth = 0
        /// The display, in CoreGraphics coordinates, whose part under the window is captured with the app's other windows
        /// on it (`follow`); nil while the window alone is.
        var appWindowsDisplay: CGRect?
        /// The latest of what `follow` was asked for, which the serialised update applies whatever order it runs in.
        var followed: (target: CaptureTarget, includesAppWindows: Bool)?
        /// While the stream changes what it captures, the frames it delivers are of neither one nor the other.
        var isSwitching = false
        /// The first frame after a switch is not a neighbour of the one before it.
        var cutsNext = false
        var lastError: MGError?
        var onFrame: (@Sendable (CapturedFrame) -> Void)?
        var onStop: (@Sendable (MGError?) -> Void)?
    }
    private let state = OSAllocatedUnfairLock(uncheckedState: State())

    /// Touched only from `captureQueue`, which is where every frame arrives.
    private var lastSignature: UInt64?
    private var lastLuma: [UInt8]?
    private var lastPixelBuffer: CVPixelBuffer?
    private var sceneCuts = SceneCutDetector()
    private var contentFrames = 0

    var lastError: MGError? { state.withLockUnchecked { $0.lastError } }

    /// Size ScreenCaptureKit is actually delivering. The compositor does the downscale, so
    /// nothing further down the pipeline pays for it.
    var capturePixelSize: CGSize { state.withLockUnchecked { $0.capturePixelSize } }

    /// How the window's content was magnified to the frames being delivered, where they are its magnified picture.
    var contentMagnification: ContentResolution.Magnification? {
        state.withLockUnchecked { Self.magnification(of: $0, frameSize: $0.capturePixelSize) }
    }

    /// Size the pipeline works at: what is delivered, or half of it where that is content drawn at half the window's pixels
    /// and magnified, which the pipeline takes back to the content's own size first.
    var contentPixelSize: CGSize {
        state.withLockUnchecked { state in
            Self.magnification(of: state, frameSize: state.capturePixelSize) == nil
                ? state.capturePixelSize
                : CGSize(width: (state.capturePixelSize.width / 2).rounded(.down), height: (state.capturePixelSize.height / 2).rounded(.down))
        }
    }

    /// Delivered for every frame that is not a repeat of the one before. Set before
    /// `startCapture`, because frames begin the moment the stream does.
    var onFrame: (@Sendable (CapturedFrame) -> Void)? {
        get { state.withLockUnchecked { $0.onFrame } }
        set { state.withLockUnchecked { $0.onFrame = newValue } }
    }

    /// Called when the stream ends, or will not take even the configuration it is put back to after a change it would not
    /// take (`recover`). `nil` means it ended without an error worth showing.
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

    /// The magnification a frame of `frameSize` carries: only one of the window's full size on a display of two is the
    /// magnified picture; at a reduced render scale ScreenCaptureKit has made it small already, and on a display of one the
    /// content is the window's pixels.
    private static func magnification(of state: State, frameSize: CGSize) -> ContentResolution.Magnification? {
        guard state.backingScale == 2, Int(frameSize.width) == Int(state.windowPixelSize.width.rounded()),
              Int(frameSize.height) == Int(state.windowPixelSize.height.rounded()) else { return nil }
        return state.magnification
    }

    /// How often a frame is tested while the content's resolution is not known, in frames delivered: a test of a
    /// 3024x1964 frame takes up to 3 ms of the capture queue.
    private static let contentTestInterval = 30

    /// Surfaces ScreenCaptureKit keeps in its pool. The pipeline holds a frame for as long as the GPU works on it, up to its
    /// buffer depth of them at once; one more waits for a permit, one in the mailbox, and the last one delivered is kept to
    /// tell a repeat; and the compositor needs one free to draw the next into. A pool of the buffer depth alone ran dry
    /// whenever the GPU was busy: with each frame held for 30 ms, a pool of 3 delivered 83 frames a second and one of 2
    /// delivered 56, where 7 and 6 delivered 111. ScreenCaptureKit takes at most eight.
    private static func queueDepth(pipelineDepth: Int) -> Int {
        min(8, max(3, pipelineDepth + 4))
    }

    private func makeConfiguration(_ snapshot: State) -> SCStreamConfiguration {
        let config = SCStreamConfiguration()
        config.width = Int(snapshot.capturePixelSize.width)
        config.height = Int(snapshot.capturePixelSize.height)
        if snapshot.maxFPS > 0 {
            config.minimumFrameInterval = CMTime(value: 1, timescale: CMTimeScale(snapshot.maxFPS))
        }
        config.pixelFormat = kCVPixelFormatType_32BGRA
        config.showsCursor = false
        // The window's own shadow is not part of what is being scaled. Including it squeezes
        // the shadow and the window into a surface sized for the window alone, so the content
        // lands smaller than the window it is drawn over.
        config.ignoreShadowsSingleWindow = true
        // Where the app's windows on the display are captured, only the window's own rectangle is.
        if let display = snapshot.appWindowsDisplay {
            config.sourceRect = snapshot.windowFrame.offsetBy(dx: -display.minX, dy: -display.minY)
        }
        config.queueDepth = Self.queueDepth(pipelineDepth: snapshot.pipelineDepth)
        config.captureResolution = .best
        config.shouldBeOpaque = false
        config.backgroundColor = .clear
        return config
    }

    func startCapture(target: CaptureTarget, maxFPS: Int,
                      renderScale: Float, pipelineDepth: Int) async -> Bool {
        await configurationGate.perform { [self] in
            await startSerially(target: target, maxFPS: maxFPS, renderScale: renderScale,
                                pipelineDepth: pipelineDepth)
        }
    }

    private func startSerially(target: CaptureTarget, maxFPS: Int,
                               renderScale: Float, pipelineDepth: Int) async -> Bool {
        await stopSerially()

        do {
            let content = try await SCShareableContent.excludingDesktopWindows(false, onScreenWindowsOnly: true)
            guard let window = content.windows.first(where: {
                $0.windowID == target.windowID
                    && (target.ownerPID == nil || $0.owningApplication?.processID == target.ownerPID)
            }) else {
                state.withLockUnchecked { $0.lastError = .captureWindowNotFound }
                return false
            }

            let filter = SCContentFilter(desktopIndependentWindow: window)
            let snapshot = state.withLockUnchecked { state -> State in
                state.windowFilter = filter
                state.application = window.owningApplication
                state.windowFrame = target.frame
                state.backingScale = target.backingScale
                state.magnification = nil
                // Content drawn at one pixel a point is told by how a display of two magnifies it.
                state.testsContent = target.backingScale == 2
                state.windowPixelSize = target.pixelSize
                state.capturePixelSize = Self.scaledSize(target.pixelSize, by: renderScale)
                state.renderScale = renderScale
                state.maxFPS = maxFPS
                state.pipelineDepth = pipelineDepth
                state.appWindowsDisplay = nil
                state.followed = nil
                state.isSwitching = false
                state.cutsNext = false
                state.lastError = nil
                return state
            }

            // Reset ahead of the stream rather than after it starts: the queue is serial, so
            // this runs before the first frame can arrive, where a reset made once the stream
            // is running races the frames already being handled.
            captureQueue.async { [self] in
                lastSignature = nil
                lastLuma = nil
                lastPixelBuffer = nil
                sceneCuts.reset()
                contentFrames = 0
            }

            let stream = SCStream(filter: filter, configuration: makeConfiguration(snapshot), delegate: self)
            try stream.addStreamOutput(self, type: .screen, sampleHandlerQueue: captureQueue)
            state.withLockUnchecked { $0.stream = stream }
            try await stream.startCapture()
            return state.withLockUnchecked { $0.stream === stream }
        } catch {
            state.withLockUnchecked { $0.stream = nil; $0.lastError = .captureStartFailed(error) }
            return false
        }
    }

    func stopCapture() async {
        await configurationGate.perform { [self] in await stopSerially() }
    }

    private func stopSerially() async {
        captureQueue.async { [self] in lastPixelBuffer = nil }
        guard let stream = state.withLockUnchecked({ state -> SCStream? in
            defer {
                state.stream = nil
                state.followed = nil
                state.isSwitching = false
            }
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

    /// Applies render scale or a new frame-rate ceiling at the source, without restarting the
    /// stream. The frames arrive already reduced, so the GPU never encodes a downscale pass
    /// and every later stage works on proportionally fewer pixels.
    func reconfigure(renderScale: Float? = nil, maxFPS: Int? = nil) async {
        await configurationGate.perform { [self] in
            await reconfigureSerially(renderScale: renderScale, maxFPS: maxFPS)
        }
    }

    private func reconfigureSerially(renderScale: Float?, maxFPS: Int?) async {
        let update = state.withLockUnchecked { state -> (SCStream, State, State)? in
            guard let stream = state.stream else { return nil }
            var requested = state
            requested.renderScale = renderScale ?? state.renderScale
            requested.maxFPS = maxFPS.map { max(1, $0) } ?? state.maxFPS
            requested.capturePixelSize = Self.scaledSize(requested.windowPixelSize, by: requested.renderScale)
            guard requested.capturePixelSize != state.capturePixelSize || requested.maxFPS != state.maxFPS else {
                state.renderScale = requested.renderScale
                return nil
            }
            return (stream, state, requested)
        }
        guard let (stream, current, snapshot) = update else { return }

        do {
            try await stream.updateConfiguration(makeConfiguration(snapshot))
            state.withLockUnchecked {
                guard $0.stream === stream else { return }
                $0.renderScale = snapshot.renderScale
                $0.capturePixelSize = snapshot.capturePixelSize
                $0.maxFPS = snapshot.maxFPS
                $0.lastError = nil
            }
        } catch {
            await recover(from: error, on: stream, to: current, filter: nil)
        }
    }

    // MARK: - Following the window

    /// The window moved or changed size, or the app put windows of its own over it or took them away: menus, popups,
    /// tooltips, panels and dialogs, which are windows of their own and not part of the window's capture (its sheets and
    /// popovers are). While there are any, the stream takes in the app's windows on the display, cut to the window's
    /// rectangle, so that they are shown as they would be; otherwise the window alone, which follows the window by itself
    /// and delivered more frames for the same content (118 a second against 112 for a window drawn 60 times a second).
    /// Measured on one stream, switching took about 0.1 s either way, and the frames meanwhile are dropped.
    ///
    /// Called from the main thread; the latest call is what is applied, whatever order the updates run in.
    func follow(_ target: CaptureTarget, includesAppWindows: Bool) {
        state.withLockUnchecked { $0.followed = (target, includesAppWindows) }
        Task { await configurationGate.perform { [self] in await followSerially() } }
    }

    private func followSerially() async {
        guard let (stream, current, wanted) = state.withLockUnchecked({ state -> (SCStream, State, (target: CaptureTarget, includesAppWindows: Bool))? in
            guard let stream = state.stream, let followed = state.followed else { return nil }
            return (stream, state, followed)
        }) else { return }

        var next = current
        next.windowFrame = wanted.target.frame
        next.backingScale = wanted.target.backingScale
        next.windowPixelSize = wanted.target.pixelSize
        next.capturePixelSize = Self.scaledSize(wanted.target.pixelSize, by: current.renderScale)

        // A new filter where the app's windows come or go, or the window has left the display they are captured on.
        var filter: SCContentFilter?
        if wanted.includesAppWindows {
            let center = CGPoint(x: wanted.target.frame.midX, y: wanted.target.frame.midY)
            if current.appWindowsDisplay.map({ !$0.contains(center) }) ?? true {
                guard let application = current.application,
                      let content = try? await SCShareableContent.excludingDesktopWindows(false, onScreenWindowsOnly: true),
                      let display = Self.display(holding: wanted.target.frame, among: content.displays) else { return }
                let appWindows = SCContentFilter(display: display, including: [application], exceptingWindows: [])
                appWindows.includeMenuBar = false
                filter = appWindows
                next.appWindowsDisplay = display.frame
            }
        } else if current.appWindowsDisplay != nil {
            filter = current.windowFilter
            next.appWindowsDisplay = nil
        }

        // A move alone changes nothing for the window's own capture, which follows it.
        let movesRectangle = next.appWindowsDisplay != nil && next.windowFrame != current.windowFrame
        guard filter != nil || movesRectangle || next.capturePixelSize != current.capturePixelSize else {
            state.withLockUnchecked {
                guard $0.stream === stream else { return }
                $0.windowFrame = next.windowFrame
                $0.backingScale = next.backingScale
                $0.windowPixelSize = next.windowPixelSize
            }
            return
        }

        if filter != nil { state.withLockUnchecked { if $0.stream === stream { $0.isSwitching = true } } }
        do {
            if let filter { try await stream.updateContentFilter(filter) }
            try await stream.updateConfiguration(makeConfiguration(next))
            state.withLockUnchecked {
                guard $0.stream === stream else { return }
                $0.windowFrame = next.windowFrame
                $0.backingScale = next.backingScale
                $0.windowPixelSize = next.windowPixelSize
                $0.capturePixelSize = next.capturePixelSize
                $0.appWindowsDisplay = next.appWindowsDisplay
                if filter != nil { $0.cutsNext = true }
                $0.lastError = nil
            }
        } catch {
            // Back to the window alone, which is what the stream was started with, where a switch of what it captures is what
            // failed, rather than half of a switch; to the configuration it had where a change of size or rectangle did.
            if filter != nil, let windowFilter = current.windowFilter {
                var fallback = next
                fallback.appWindowsDisplay = nil
                await recover(from: error, on: stream, to: fallback, filter: windowFilter)
            } else {
                await recover(from: error, on: stream, to: current, filter: nil)
            }
        }
        if filter != nil { state.withLockUnchecked { if $0.stream === stream { $0.isSwitching = false } } }
    }

    /// After a change the stream would not take: back to `fallback`, which it has run with, so that the session goes on with
    /// what can be captured, and the error is kept. Only a stream that will not take even that ends the session
    /// (`configurationFailed`).
    private func recover(from error: Error, on stream: SCStream, to fallback: State, filter: SCContentFilter?) async {
        do {
            if let filter { try await stream.updateContentFilter(filter) }
            try await stream.updateConfiguration(makeConfiguration(fallback))
        } catch {
            configurationFailed(error, on: stream)
            return
        }
        state.withLockUnchecked {
            guard $0.stream === stream else { return }
            if filter != nil { $0.cutsNext = true }
            $0.windowFrame = fallback.windowFrame
            $0.backingScale = fallback.backingScale
            $0.windowPixelSize = fallback.windowPixelSize
            $0.capturePixelSize = fallback.capturePixelSize
            $0.renderScale = fallback.renderScale
            $0.maxFPS = fallback.maxFPS
            $0.appWindowsDisplay = fallback.appWindowsDisplay
            $0.lastError = .captureReconfigurationFailed(error)
        }
    }

    // MARK: - The content's resolution

    /// Now and then, until it is known: whether the window's content is drawn at half its pixels, while the window alone is
    /// captured at all of them (`ContentResolution`). Capture queue only.
    private func testContent(_ surface: IOSurfaceRef, of stream: SCStream) {
        contentFrames += 1
        guard contentFrames % Self.contentTestInterval == 1, let count = ContentResolution.count(surface) else { return }
        let verdict = ContentResolution.verdict(of: count)
        guard verdict != .undecided else { return }
        state.withLockUnchecked { state in
            guard state.stream === stream, state.testsContent else { return }
            state.testsContent = false
            if case .half(let magnification) = verdict { state.magnification = magnification }
        }
    }

    private func configurationFailed(_ error: Error, on stream: SCStream) {
        let failure = MGError.captureReconfigurationFailed(error)
        let callback = state.withLockUnchecked { state -> (@Sendable (MGError?) -> Void)? in
            guard state.stream === stream else { return nil }
            state.lastError = failure
            return state.onStop
        }
        callback?(failure)
    }

    /// The display holding most of `frame`, as AppKit decides a window's screen.
    private static func display(holding frame: CGRect, among displays: [SCDisplay]) -> SCDisplay? {
        func area(_ display: SCDisplay) -> CGFloat {
            let shared = display.frame.intersection(frame)
            return shared.isNull ? 0 : shared.width * shared.height
        }
        guard let best = displays.max(by: { area($0) < area($1) }), area(best) > 0 else { return nil }
        return best
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
            guard state.stream === stream else { return nil }
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

        let frameSize = CGSize(width: IOSurfaceGetWidth(surface), height: IOSurfaceGetHeight(surface))
        let snapshot = state.withLockUnchecked { state -> ((@Sendable (CapturedFrame) -> Void)?, CGSize, Int, Bool, Bool,
                                                            ContentResolution.Magnification?)? in
            guard state.stream === stream, !state.isSwitching else { return nil }
            defer { state.cutsNext = false }
            let magnification = Self.magnification(of: state, frameSize: frameSize)
            // At the window's full size only: a frame the render scale reduced is not the magnified picture the test reads.
            let testsContent = state.testsContent && state.appWindowsDisplay == nil && state.backingScale == 2
                && Int(frameSize.width) == Int(state.windowPixelSize.width.rounded())
                && Int(frameSize.height) == Int(state.windowPixelSize.height.rounded())
            // A magnified frame stands for content of half its size.
            let native = magnification == nil ? state.windowPixelSize
                : CGSize(width: (state.windowPixelSize.width / 2).rounded(.down), height: (state.windowPixelSize.height / 2).rounded(.down))
            return (state.onFrame, native, state.maxFPS, state.cutsNext, testsContent, magnification)
        }
        guard let (callback, native, maxFPS, cutsHere, testsContent, magnification) = snapshot else { return }

        // A frame whose pixels match the previous one's carries nothing new. A surface that
        // cannot be sampled is passed through rather than guessed at.
        var isSceneCut = cutsHere
        if let sample = FrameSampler.sample(surface) {
            if !cutsHere, sample.signature == lastSignature, let previous = lastPixelBuffer,
               let previousSurface = CVPixelBufferGetIOSurface(previous)?.takeUnretainedValue(),
               FrameSampler.isIdentical(surface, to: previousSurface) {
                lastPixelBuffer = pixelBuffer
                return
            }
            lastSignature = sample.signature
            let detected = sceneCuts.isCut(previous: lastLuma, current: sample.luma,
                                           alpha: SceneCutDetector.alpha(forFrameRate: maxFPS))
            isSceneCut = isSceneCut || detected
            lastLuma = sample.luma
        } else {
            lastSignature = nil
            lastLuma = nil
        }
        lastPixelBuffer = pixelBuffer
        if testsContent { testContent(surface, of: stream) }

        callback?(CapturedFrame(pixelBuffer: pixelBuffer, surface: surface,
                                captureTime: CMTimeGetSeconds(CMSampleBufferGetPresentationTimeStamp(sampleBuffer)),
                                isSceneCut: isSceneCut, nativePixelSize: native, magnification: magnification))
    }
}
