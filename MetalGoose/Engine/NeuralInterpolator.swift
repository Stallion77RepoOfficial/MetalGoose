import Foundation
@preconcurrency import Metal
@preconcurrency import VideoToolbox
@preconcurrency import CoreVideo
import CoreMedia
import QuartzCore
import os

/// A frame in the 4:2:0 form the Neural Engine takes, with the planes wrapped as textures so the GPU
/// can convert to and from it without a copy.
struct YUVFrame: @unchecked Sendable {
    let buffer: CVPixelBuffer
    let luma: MTLTexture
    let chroma: MTLTexture
    /// The session this buffer belongs to. A frame from a session that has been replaced is not the same size as the
    /// ones after it and must not be paired with them.
    let session: Int
    var lease: BufferLease? = nil
}

/// Frame interpolation on the Neural Engine, through VideoToolbox's low-latency frame interpolator.
///
/// The point is the GPU. MetalGoose shares it with the app being captured, and every millisecond it
/// spends interpolating is one the app does not get. This path costs the GPU one small conversion
/// pass per capture — about a tenth of a millisecond at a megapixel — and does the interpolation
/// itself on the Neural Engine, which nothing else is using. The way back is made as an image is shown, in the
/// blend that brings it toward the captures (`blendTowardCapturesFromYUV`), so an image that is not shown costs the GPU
/// nothing.
///
/// A pair is cut into two steps, which is one image, its midpoint, or into four, which is three: its
/// quarters. A quarter is made in two stages, the midpoint first and the quarter from it, so one alone takes
/// twice what the midpoint takes, and the three together about two and a half times: at 1280x1016, 11 ms
/// for the midpoint and 31 for all three back to back, and 17 and 38 at the pace of a capture, when the
/// Neural Engine has been idle in between. They are made in one call for that reason; three calls of one image
/// each took 54 ms. So four steps are a matter of whether three images fit the time between captures.
///
/// It is not a drop-in for MetalFX. VideoToolbox takes video-range 4:2:0 and nothing else, so each
/// frame is converted on the way in and the result on the way out; the processor is sized at creation
/// and builds its model for that size, which takes long enough that it happens off to the side while
/// captures go on being shown; and its size ceiling is a fraction of what MetalFX handles, so a larger frame is shrunk
/// to what it takes on the way in and the images are enlarged on the way out (`NeuralSizes`). Measured against an analytic
/// ground truth on a textured pan it lands at 39 to 41 dB at every phase, where the 4:2:0 conversion alone caps any
/// such path at 49 dB.
///
/// A session — a `Rig` here — is for one size. Moving to another size builds the new one beside the old, which goes on
/// serving until the new one has started, so that the images do not stop while a model is compiled.
///
/// Concurrency: the capture pipeline calls `prepare`, `isReady`, `encodeConversion` and the readers of the times on
/// its own queue, and none of them waits for this object's. Everything else — session set-up, submission, the
/// results — happens on this object's queue; the render thread reads the results from `GeneratedImages`.
final class NeuralInterpolator: @unchecked Sendable {

    /// The most images a pair has: the quarters.
    static let maximumImages = 3

    /// The phases of a pair cut into `steps`, in the order they are wanted.
    private static func phases(steps: Int) -> [Double] {
        (1..<max(2, steps)).map { Double($0) / Double(steps) }
    }

    private let gpu: GPUContext
    private let errors: ErrorLog
    private let latency: GenerationLatency
    private let images: GeneratedImages
    private let queue = DispatchQueue(label: "com.metalgoose.neural", qos: .userInitiated)

    // MARK: Capacity

    /// What VideoToolbox's frame interpolator takes; nil where it is not there at all.
    static let limits: NeuralSizes.Limits? = {
        guard VTLowLatencyFrameInterpolationConfiguration.isSupported else { return nil }
        return NeuralSizes.Limits(
            maximumDimension: Int(VTLowLatencyFrameInterpolationConfiguration.__maximumDimension(forSpatialScaleFactor: 1)),
            maximumPixels: Int(VTLowLatencyFrameInterpolationConfiguration.__maximumPixelCount(forSpatialScaleFactor: 1)))
    }()

    /// The sizes the Neural Engine works at for frames of this size, finest first; empty where it cannot take them at all.
    static func ladder(frameWidth: Int, frameHeight: Int) -> [NeuralSizes.Size] {
        limits.map { NeuralSizes.ladder(frameWidth: frameWidth, frameHeight: frameHeight, limits: $0) } ?? []
    }

    /// Whether frames of this size are taken as they are, which needs both dimensions even, because the chroma planes
    /// are half-size, and the frame inside what the processor supports.
    static func supports(width: Int, height: Int) -> Bool {
        ladder(frameWidth: width, frameHeight: height).first == NeuralSizes.Size(width: width, height: height)
    }

    /// Two captures that followed one another, and what the schedule behind them looks like.
    private struct Pair {
        let previous: (frame: YUVFrame, timestamp: CFTimeInterval)
        let current: (frame: YUVFrame, timestamp: CFTimeInterval)
        let steps: Int
        /// How far behind real time the frame clock runs, as the capture path expects it to.
        let delay: CFTimeInterval
    }

    // MARK: A session at one size

    /// Everything that belongs to a session at one size: the buffers it works in, the processor, and what has been seen of
    /// how long it takes.
    private final class Rig: @unchecked Sendable {
        let id: Int
        let size: NeuralSizes.Size

        /// Conversion buffers leased under a lock, independently of the Neural Engine's queue,
        /// which can be busy for seconds — ending a session waits for the engine's service to let go of its model, which
        /// took that long while another size was being compiled, and a capture queue that waited on it stopped taking
        /// frames.
        private let inputs: [YUVFrame]
        private let inputLeases: BufferLeasePool

        /// Output buffers remain leased through inference, storage, and GPU reads. Their
        /// planes are converted to colour by the presentation blend. Pool exhaustion skips
        /// new work instead of overwriting an image that still has a reader.
        let outputs: [YUVFrame]
        private let outputLeases: BufferLeasePool

        // Queue-confined, but for the times.
        var processor: VTFrameProcessor?
        var previous: (frame: YUVFrame, timestamp: CFTimeInterval)?
        var busy = false
        var isRetired = false

        /// The newest pair that arrived while a call was on the engine, made when the call is done if it can still be shown in time.
        var pending: Pair?

        /// How long a call takes, for the midpoint alone and for all three quarters.
        let midpointTimes = OSAllocatedUnfairLock(initialState: IntervalFilter())
        let quarterTimes = OSAllocatedUnfairLock(initialState: IntervalFilter())

        init(id: Int, size: NeuralSizes.Size, inputs: [YUVFrame], outputs: [YUVFrame]) {
            self.id = id
            self.size = size
            self.inputs = inputs
            self.outputs = outputs
            inputLeases = BufferLeasePool(capacity: inputs.count)
            outputLeases = BufferLeasePool(capacity: outputs.count)
        }

        /// Two captures' buffers and one for the image, for the call that tries the session.
        var probeFrames: (previous: YUVFrame, current: YUVFrame, destination: YUVFrame) { (inputs[0], inputs[1], outputs[0]) }

        func takeInput() -> YUVFrame? {
            guard let lease = inputLeases.acquire() else { return nil }
            var frame = inputs[lease.index]
            frame.lease = lease
            return frame
        }
        func takeOutput() -> YUVFrame? {
            guard let lease = outputLeases.acquire() else { return nil }
            var frame = outputs[lease.index]
            frame.lease = lease
            return frame
        }
    }

    // MARK: State

    private var active: Rig?
    private var building: Rig?
    private var nextRigID = 0

    /// What the capture queue reads: the session that is serving, and the size last asked for.
    private struct Published {
        var active: Rig?
        var wanted: NeuralSizes.Size?
    }
    private let published = OSAllocatedUnfairLock(uncheckedState: Published())

    private let failure = OSAllocatedUnfairLock(initialState: false)

    /// Sizes whose session started and then refused a call, until when. The processor reports the largest size it takes, but
    /// takes only the shapes its network was built for: 1874x1106 starts a session and fails every call with "Processor is not
    /// initialized", where 1920x1080 and 1708x1016 do not. A session is therefore tried with one call before it serves; a size
    /// it refuses is not asked for again, and one that failed for another reason is left alone for a while.
    private let rejected = OSAllocatedUnfairLock(initialState: [NeuralSizes.Size: CFTimeInterval]())
    private static let retryAfterFailedTrial: CFTimeInterval = 30

    /// Where sessions are ended, off to the side: ending one can take seconds.
    private static let retiring = DispatchQueue(label: "com.metalgoose.neural.retire", qos: .utility)

    /// Ends a session that no call is on, a little after its last call came back: VideoToolbox goes on with the call's own
    /// bookkeeping for a moment after it has run its completion, and a session ended under that is what brought the process
    /// down once (the bookkeeping read state that was gone). Nothing waits for it, and a model held for a third of a second
    /// longer costs nothing.
    private static func end(_ processor: VTFrameProcessor) {
        nonisolated(unsafe) let retired = processor
        retiring.asyncAfter(deadline: .now() + 0.3) { retired.endSession() }
    }

    init(gpu: GPUContext, errors: ErrorLog, latency: GenerationLatency, images: GeneratedImages) {
        self.gpu = gpu
        self.errors = errors
        self.latency = latency
        self.images = images
    }

    /// The processor failed at run time, and the caller should use another engine from now on.
    var hasFailed: Bool { failure.withLock { $0 } }

    /// Whether a session at this size was found to refuse calls.
    func isRejected(_ size: NeuralSizes.Size) -> Bool {
        rejected.withLock { state in state[size].map { CACurrentMediaTime() < $0 } ?? false }
    }

    /// The size of the session that is serving, if one is.
    var activeSize: NeuralSizes.Size? { published.withLockUnchecked { $0.active?.size } }

    /// How long a call takes at the size that is serving, for the midpoint alone and for all three quarters; 0 until they
    /// have been timed.
    var midpointTime: CFTimeInterval { published.withLockUnchecked { $0.active }?.midpointTimes.withLock { $0.value } ?? 0 }
    var quartersTime: CFTimeInterval { published.withLockUnchecked { $0.active }?.quarterTimes.withLock { $0.value } ?? 0 }

    func reset() {
        queue.async { [self] in
            retire(active)
            retire(building)
            active = nil
            building = nil
            published.withLockUnchecked { $0 = Published() }
            failure.withLock { $0 = false }
        }
    }

    /// Lets go of a session. Its processor is ended off to the side once no call is on it (`endIfIdle`).
    private func retire(_ rig: Rig?) {
        guard let rig else { return }
        rig.isRetired = true
        rig.previous = nil
        rig.pending = nil
        endIfIdle(rig)
    }

    /// Ends the processor of a session that has been let go of, if no call is on it; the call's completion does it otherwise.
    /// A call is made on this object's queue, as this is, so a processor is never called after it has been ended: calling
    /// one that was ended reads state that is gone and brings the process down (a `process` that was still to start on a
    /// task when its session was replaced did exactly that).
    private func endIfIdle(_ rig: Rig) {
        guard rig.isRetired, !rig.busy, let processor = rig.processor else { return }
        rig.processor = nil
        Self.end(processor)
    }

    // MARK: - Capture side

    /// Asks for a session at this size, which is built beside the one that is serving, if there is one, and takes over when
    /// it has started. Never waits. Called from the capture pipeline's queue.
    func prepare(_ size: NeuralSizes.Size) {
        let build = published.withLockUnchecked { state -> Bool in
            if state.wanted == size { return false }
            state.wanted = size
            return true
        }
        guard build else { return }
        queue.async { [self] in
            guard let wanted = published.withLockUnchecked({ $0.wanted }) else { return }
            startSession(for: wanted)
        }
    }

    /// Whether the session that is serving is at this size.
    func isReady(_ size: NeuralSizes.Size) -> Bool {
        !hasFailed && activeSize == size
    }

    /// Encodes the conversion of one captured frame to 4:2:0 into the next buffer of the serving session, shrinking it on
    /// the way where it is larger than the session's size, and returns that buffer; nil where no session is serving. Called
    /// on the capture pipeline's queue, and never waits for this object's.
    func encodeConversion(of frame: MTLTexture, commandBuffer: MTLCommandBuffer) -> YUVFrame? {
        guard let rig = published.withLockUnchecked({ $0.active }),
              let target = rig.takeInput(),
              let encoder = commandBuffer.makeComputeCommandEncoder() else { return nil }

        let resamples = frame.width != rig.size.width || frame.height != rig.size.height
        let pipeline = resamples ? gpu.pipelines.convertTo420Resampled : gpu.pipelines.convertTo420
        encoder.setComputePipelineState(pipeline)
        encoder.setTexture(frame, index: 0)
        encoder.setTexture(target.luma, index: 1)
        encoder.setTexture(target.chroma, index: 2)
        // One thread per 2x2 block.
        gpu.dispatch(pipeline, on: encoder, width: rig.size.width / 2, height: rig.size.height / 2)
        encoder.endEncoding()
        commandBuffer.addCompletedHandler { _ in withExtendedLifetime(target) {} }
        return target
    }

    /// The conversion of `frame` has finished, so VideoToolbox may read it. Pairs it with `previous`, the capture before
    /// it, and interpolates between them, into `steps` steps: 2 makes the midpoint, 4 its quarters. A pair is two captures that
    /// follow one another: where the session was not given the one before — it was not the engine then, or it was still
    /// being converted — this capture only becomes the one the next is paired with.
    ///
    /// Only one pair is worked on at a time. One that arrives while the Neural Engine is on the last waits, the newest
    /// alone, for the call to finish, and is made then if its first image can still be there when the render clock wants it
    /// (`delay`, how far behind real time that clock runs): captures do not arrive evenly, and a pair that follows its
    /// predecessor closely is wanted later after its own arrival than one a whole interval long. Otherwise it has missed its
    /// moment, and the screen stays on a capture.
    func frameConverted(_ frame: YUVFrame, timestamp: CFTimeInterval, previous partner: CFTimeInterval?, steps: Int,
                        delay: CFTimeInterval) {
        queue.async { [self] in
            guard let rig = active, rig.id == frame.session else { return }
            defer { rig.previous = (frame, timestamp) }
            guard rig.processor != nil, let previous = rig.previous, let partner, previous.timestamp == partner else { return }
            let pair = Pair(previous: previous, current: (frame, timestamp), steps: steps, delay: delay)
            if rig.busy {
                rig.pending = pair
            } else {
                start(pair, on: rig, waited: false)
            }
        }
    }

    /// Whether the first image of a pair that has been waiting for the engine can still be made before it is wanted.
    private func isStillWanted(_ pair: Pair, on rig: Rig) -> Bool {
        let call = pair.steps >= InterpolationSteps.quarters ? rig.quarterTimes : rig.midpointTimes
        let wanted = FramePlanner.firstImageWanted(previous: pair.previous.timestamp, next: pair.current.timestamp,
                                                   delay: pair.delay, steps: pair.steps)
        return CACurrentMediaTime() + call.withLock { $0.value } <= wanted
    }

    private func start(_ pair: Pair, on rig: Rig, waited: Bool) {
        guard let processor = rig.processor else { return }
        rig.busy = true
        submit(rig: rig, processor: processor, previous: pair.previous, current: pair.current,
               phases: Self.phases(steps: pair.steps), waited: waited)
    }

    // MARK: - Interpolating

    /// One image of a pair: where it sits, and the buffer it is made into.
    private struct Image {
        let phase: Double
        let destination: YUVFrame
    }

    private func submit(rig: Rig, processor: VTFrameProcessor,
                        previous: (frame: YUVFrame, timestamp: CFTimeInterval),
                        current: (frame: YUVFrame, timestamp: CFTimeInterval),
                        phases: [Double], waited: Bool) {
        func frame(_ yuv: YUVFrame, _ seconds: CFTimeInterval) -> VTFrameProcessorFrame? {
            VTFrameProcessorFrame(buffer: yuv.buffer, presentationTimeStamp: CMTime(seconds: seconds, preferredTimescale: 1_000_000))
        }
        let duration = current.timestamp - previous.timestamp
        var made: [Image] = []
        var outputFrames: [VTFrameProcessorFrame] = []
        for phase in phases {
            guard let destination = rig.takeOutput(),
                  let output = frame(destination, previous.timestamp + phase * duration) else {
                rig.busy = false
                return
            }
            made.append(Image(phase: phase, destination: destination))
            outputFrames.append(output)
        }
        guard let source = frame(current.frame, current.timestamp),
              let reference = frame(previous.frame, previous.timestamp),
              let parameters = VTLowLatencyFrameInterpolationParameters(
                sourceFrame: source, previousFrame: reference,
                interpolationPhase: phases.map { Float($0) }, destinationFrames: outputFrames) else {
            rig.busy = false
            return
        }

        nonisolated(unsafe) let request = parameters
        let pair = (previous: previous.timestamp, next: current.timestamp)
        let images = made
        let started = CACurrentMediaTime()
        processor.process(parameters: request) { [self] _, error in
            // The lease prevents our capture writer from reusing the ANE's inputs.
            withExtendedLifetime((previous.frame, current.frame)) {}
            let spent = CACurrentMediaTime() - started
            queue.async { [self] in
                rig.busy = false
                // A session that was replaced while it was on a call is not a failure of the engine, and is ended now.
                guard !rig.isRetired else {
                    endIfIdle(rig)
                    return
                }
                if error != nil {
                    // Not worth an alert: the capture path switches to MetalFX, which is slower on the GPU
                    // but always there. What would be worth knowing is that this happened.
                    failure.withLock { $0 = true }
                    return
                }
                (images.count > 1 ? rig.quarterTimes : rig.midpointTimes).withLock {
                    $0.add(spent, window: EngineShared.measurementWindow, elapsed: duration)
                }
                publish(images, previous: pair.previous, next: pair.next, measuresLatency: !waited)
                if let next = rig.pending {
                    rig.pending = nil
                    if isStillWanted(next, on: rig) { start(next, on: rig, waited: true) }
                }
            }
        }
    }

    /// Makes the images of a pair available to the render thread: they are complete, and are shown from the planes the
    /// processor wrote, so the GPU is not asked for anything until one of them is shown. How long they took to arrive sets
    /// how far behind the schedule runs, except for a pair that waited for the engine: that one was only taken because it was
    /// still in time, and must not move the schedule for the pairs that were not kept waiting.
    private func publish(_ made: [Image], previous: CFTimeInterval, next: CFTimeInterval, measuresLatency: Bool) {
        images.publish(made.map {
            GeneratedImages.Image(previous: previous, next: next, phase: $0.phase,
                                  source: .planes($0.destination), engine: .neuralEngine)
        })
        if measuresLatency { latency.record(previous: previous, next: next) }
    }

    // MARK: - Set-up

    /// Builds the buffers for this size at once, and the processor in the background: starting a session builds the model for
    /// the size, which takes long enough that nothing should wait for it. The session that is serving goes on until the new
    /// one has started.
    private func startSession(for size: NeuralSizes.Size) {
        if active?.size == size {
            // Already serving at it: whatever was being built for another size is not wanted.
            retire(building)
            building = nil
            return
        }
        if building?.size == size { return }
        retire(building)
        building = nil

        // Configured for the quarters, which a call for the midpoint alone takes as well and at the same cost,
        // so that going from two steps to four does not mean a new session.
        guard let configuration = VTLowLatencyFrameInterpolationConfiguration(frameWidth: size.width, frameHeight: size.height,
                                                                               numberOfInterpolatedFrames: 2) else {
            failure.withLock { $0 = true }
            return
        }

        nextRigID += 1
        let id = nextRigID
        // An image's 4:2:0 buffer is needed until the GPU has converted it, while the next of its pair is being made.
        guard let inputs = makeBuffers(count: FrameRing.capacity + GooseEngine.maxInFlight, size: size, session: id,
                                       attributes: configuration.sourcePixelBufferAttributes),
              let outputs = makeBuffers(count: GeneratedImages.capacity(of: .neuralEngine) + Self.maximumImages, size: size,
                                        session: id, attributes: configuration.destinationPixelBufferAttributes) else {
            failure.withLock { $0 = true }
            return
        }

        let rig = Rig(id: id, size: size, inputs: inputs, outputs: outputs)
        building = rig

        DispatchQueue.global(qos: .userInitiated).async { [self] in
            let session = VTFrameProcessor()
            do { try session.startSession(configuration: configuration) } catch {
                queue.async { [self] in
                    if building === rig { failure.withLock { $0 = true } }
                }
                return
            }
            nonisolated(unsafe) let started = session
            let result = Self.tryCall(on: session, with: rig.probeFrames)
            queue.async { [self] in
                guard result.trial == .accepted else {
                    // Not a failure of the engine: this is a size it does not take, and the others may be. Asked for again
                    // if the size is wanted after all (a session that failed for another reason may well work later).
                    rejected.withLock {
                        $0[size] = result.trial == .refused ? .infinity : CACurrentMediaTime() + Self.retryAfterFailedTrial
                    }
                    published.withLockUnchecked { if $0.wanted == size { $0.wanted = nil } }
                    if building === rig { building = nil }
                    retire(rig)
                    if result.returned { Self.end(started) }
                    return
                }
                // Replaced, or the size is not the one wanted any more, while it was starting.
                guard building === rig, published.withLockUnchecked({ $0.wanted }) == size else {
                    Self.end(started)
                    return
                }
                rig.processor = started
                building = nil
                let before = active
                active = rig
                published.withLockUnchecked { $0.active = rig }
                retire(before)
            }
        }
    }

    private enum Trial { case accepted, refused, failed }

    /// How a trial call went, and whether it came back. One that did not is ended with its session when it does.
    private struct TrialResult {
        let trial: Trial
        let returned: Bool
    }

    private struct TrialState {
        var trial = Trial.accepted
        var returned = false
        var abandoned = false
    }

    /// One call for the midpoint of two blank frames, to see whether the session takes calls at all. A call that has not come
    /// back in five seconds is given up on, and its session is not ended until it has: ending one under a call that is still
    /// to finish is what a session may not have done to it.
    private static func tryCall(on session: VTFrameProcessor,
                                with frames: (previous: YUVFrame, current: YUVFrame, destination: YUVFrame)) -> TrialResult {
        func frame(_ yuv: YUVFrame, _ seconds: Double) -> VTFrameProcessorFrame? {
            VTFrameProcessorFrame(buffer: yuv.buffer, presentationTimeStamp: CMTime(seconds: seconds, preferredTimescale: 1_000_000))
        }
        guard let reference = frame(frames.previous, 0), let source = frame(frames.current, 1), let destination = frame(frames.destination, 0.5),
              let parameters = VTLowLatencyFrameInterpolationParameters(sourceFrame: source, previousFrame: reference,
                                                                       interpolationPhase: [0.5], destinationFrames: [destination]) else {
            return TrialResult(trial: .failed, returned: true)
        }
        let finished = DispatchSemaphore(value: 0)
        let state = OSAllocatedUnfairLock(initialState: TrialState())
        nonisolated(unsafe) let request = parameters
        nonisolated(unsafe) let processor = session
        processor.process(parameters: request) { _, error in
            let endsSession = state.withLock { value -> Bool in
                value.returned = true
                if let error {
                    // -19730, "Processor is not initialized": what a size it does not take answers, at once, to every call.
                    let code = (error as NSError)
                    value.trial = code.domain == VTFrameProcessorErrorDomain && code.code == -19730 ? .refused : .failed
                }
                return value.abandoned
            }
            if endsSession { end(processor) }
            finished.signal()
        }
        if finished.wait(timeout: .now() + 5) == .success { return TrialResult(trial: state.withLock { $0.trial }, returned: true) }
        return state.withLock { value in
            if !value.returned { value.abandoned = true }
            return value.returned ? TrialResult(trial: value.trial, returned: true) : TrialResult(trial: .failed, returned: false)
        }
    }

    private func makeBuffers(count: Int, size: NeuralSizes.Size, session: Int, attributes: [String: Any]) -> [YUVFrame]? {
        var attributes = attributes
        attributes[kCVPixelBufferIOSurfacePropertiesKey as String] = [:] as CFDictionary
        attributes[kCVPixelBufferMetalCompatibilityKey as String] = true

        var frames: [YUVFrame] = []
        for _ in 0..<count {
            var created: CVPixelBuffer?
            guard CVPixelBufferCreate(kCFAllocatorDefault, size.width, size.height, kCVPixelFormatType_420YpCbCr8BiPlanarVideoRange,
                                      attributes as CFDictionary, &created) == kCVReturnSuccess,
                  let buffer = created,
                  let surface = CVPixelBufferGetIOSurface(buffer)?.takeUnretainedValue() else { return nil }

            // The planes are BT.709, which is what the conversion kernels write and read.
            CVBufferSetAttachment(buffer, kCVImageBufferYCbCrMatrixKey, kCVImageBufferYCbCrMatrix_ITU_R_709_2, .shouldPropagate)
            CVBufferSetAttachment(buffer, kCVImageBufferColorPrimariesKey, kCVImageBufferColorPrimaries_ITU_R_709_2, .shouldPropagate)
            CVBufferSetAttachment(buffer, kCVImageBufferTransferFunctionKey, kCVImageBufferTransferFunction_ITU_R_709_2, .shouldPropagate)

            func plane(_ format: MTLPixelFormat, _ index: Int, _ width: Int, _ height: Int) -> MTLTexture? {
                let descriptor = MTLTextureDescriptor.texture2DDescriptor(pixelFormat: format, width: width, height: height,
                                                                          mipmapped: false)
                descriptor.usage = [.shaderRead, .shaderWrite]
                return gpu.device.makeTexture(descriptor: descriptor, iosurface: surface, plane: index)
            }
            guard let luma = plane(.r8Unorm, 0, size.width, size.height),
                  let chroma = plane(.rg8Unorm, 1, size.width / 2, size.height / 2) else { return nil }
            frames.append(YUVFrame(buffer: buffer, luma: luma, chroma: chroma, session: session))
        }
        return frames
    }
}
