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

    // MARK: A session at one size

    /// Everything that belongs to a session at one size: the buffers it works in, the processor, and what has been seen of
    /// how long it takes.
    private final class Rig: @unchecked Sendable {
        let id: Int
        let size: NeuralSizes.Size

        /// What the capture queue converts into, in turn. Taken under a lock and not through the Neural Engine's queue,
        /// which can be busy for seconds — ending a session waits for the engine's service to let go of its model, which
        /// took that long while another size was being compiled, and a capture queue that waited on it stopped taking
        /// frames.
        private let inputs: [YUVFrame]
        private let nextInput = OSAllocatedUnfairLock(initialState: 0)

        /// What the processor writes into, in turn, which the images are shown from as they are: the planes are turned into
        /// colour by the blend that shows them. The pool outnumbers the images the store keeps by the images a pair has, so
        /// that a buffer is never written while it can still be read.
        let outputs: [YUVFrame]

        // Queue-confined, but for the times.
        var processor: VTFrameProcessor?
        var outputIndex = 0
        var previous: (frame: YUVFrame, timestamp: CFTimeInterval)?
        var busy = false
        var isRetired = false

        /// How long a call takes, for the midpoint alone and for all three quarters.
        let midpointTimes = OSAllocatedUnfairLock(initialState: IntervalFilter())
        let quarterTimes = OSAllocatedUnfairLock(initialState: IntervalFilter())

        init(id: Int, size: NeuralSizes.Size, inputs: [YUVFrame], outputs: [YUVFrame]) {
            self.id = id
            self.size = size
            self.inputs = inputs
            self.outputs = outputs
        }

        func takeInput() -> YUVFrame {
            let index = nextInput.withLock { value -> Int in
                defer { value += 1 }
                return value
            }
            return inputs[index % inputs.count]
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

    /// Where sessions are ended, off to the side: ending one can take seconds.
    private static let retiring = DispatchQueue(label: "com.metalgoose.neural.retire", qos: .utility)

    init(gpu: GPUContext, errors: ErrorLog, latency: GenerationLatency, images: GeneratedImages) {
        self.gpu = gpu
        self.errors = errors
        self.latency = latency
        self.images = images
    }

    /// The processor failed at run time, and the caller should use another engine from now on.
    var hasFailed: Bool { failure.withLock { $0 } }

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

    /// Lets go of a session, and ends its processor off to the side.
    private func retire(_ rig: Rig?) {
        guard let rig else { return }
        rig.isRetired = true
        guard let processor = rig.processor else { return }
        rig.processor = nil
        nonisolated(unsafe) let retired = processor
        Self.retiring.async { retired.endSession() }
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
              let encoder = commandBuffer.makeComputeCommandEncoder() else { return nil }
        let target = rig.takeInput()

        let resamples = frame.width != rig.size.width || frame.height != rig.size.height
        let pipeline = resamples ? gpu.pipelines.convertTo420Resampled : gpu.pipelines.convertTo420
        encoder.setComputePipelineState(pipeline)
        encoder.setTexture(frame, index: 0)
        encoder.setTexture(target.luma, index: 1)
        encoder.setTexture(target.chroma, index: 2)
        // One thread per 2x2 block.
        gpu.dispatch(pipeline, on: encoder, width: rig.size.width / 2, height: rig.size.height / 2)
        encoder.endEncoding()
        return target
    }

    /// The conversion of `frame` has finished, so VideoToolbox may read it. Pairs it with `previous`, the capture before
    /// it, and interpolates between them, into `steps` steps: 2 makes the midpoint, 4 its quarters. A pair is two captures that
    /// follow one another: where the session was not given the one before — it was not the engine then, or it was still
    /// being converted — this capture only becomes the one the next is paired with.
    func frameConverted(_ frame: YUVFrame, timestamp: CFTimeInterval, previous partner: CFTimeInterval?, steps: Int) {
        queue.async { [self] in
            guard let rig = active, rig.id == frame.session else { return }
            defer { rig.previous = (frame, timestamp) }
            // Only one pair is worked on at a time. A pair that arrives while the Neural Engine is still on the last has
            // missed its moment: the render clock will have moved on before it finished.
            guard let processor = rig.processor, let previous = rig.previous, !rig.busy,
                  let partner, previous.timestamp == partner else { return }
            rig.busy = true
            submit(rig: rig, processor: processor, previous: previous, current: (frame, timestamp),
                   phases: Self.phases(steps: steps))
        }
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
                        phases: [Double]) {
        func frame(_ yuv: YUVFrame, _ seconds: CFTimeInterval) -> VTFrameProcessorFrame? {
            VTFrameProcessorFrame(buffer: yuv.buffer, presentationTimeStamp: CMTime(seconds: seconds, preferredTimescale: 1_000_000))
        }
        let duration = current.timestamp - previous.timestamp
        var made: [Image] = []
        var outputFrames: [VTFrameProcessorFrame] = []
        for phase in phases {
            let destination = rig.outputs[rig.outputIndex % rig.outputs.count]
            rig.outputIndex += 1
            guard let output = frame(destination, previous.timestamp + phase * duration) else {
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

        nonisolated(unsafe) let processor = processor
        nonisolated(unsafe) let request = parameters
        let pair = (previous: previous.timestamp, next: current.timestamp)
        let images = made
        Task { [self] in
            let started = CACurrentMediaTime()
            let failed: Bool
            do { try await processor.process(parameters: request); failed = false } catch { failed = true }
            let spent = CACurrentMediaTime() - started
            queue.async { [self] in
                rig.busy = false
                // A session that was replaced while it was on a call is not a failure of the engine.
                guard !rig.isRetired else { return }
                if failed {
                    // Not worth an alert: the capture path switches to MetalFX, which is slower on the GPU
                    // but always there. What would be worth knowing is that this happened.
                    failure.withLock { $0 = true }
                    return
                }
                (images.count > 1 ? rig.quarterTimes : rig.midpointTimes).withLock {
                    $0.add(spent, window: EngineShared.measurementWindow, elapsed: duration)
                }
                publish(images, previous: pair.previous, next: pair.next)
            }
        }
    }

    /// Makes the images of a pair available to the render thread: they are complete, and are shown from the planes the
    /// processor wrote, so the GPU is not asked for anything until one of them is shown.
    private func publish(_ made: [Image], previous: CFTimeInterval, next: CFTimeInterval) {
        images.publish(made.map {
            GeneratedImages.Image(previous: previous, next: next, phase: $0.phase,
                                  source: .planes(luma: $0.destination.luma, chroma: $0.destination.chroma), engine: .neuralEngine)
        })
        latency.record(previous: previous, next: next)
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
            queue.async { [self] in
                // Replaced, or the size is not the one wanted any more, while it was starting.
                guard building === rig, published.withLockUnchecked({ $0.wanted }) == size else {
                    Self.retiring.async { started.endSession() }
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
