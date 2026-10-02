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
    /// The interpolator's generation when this buffer was built. A frame from before a teardown is not
    /// the same size as the ones after it and must not be paired with them.
    let epoch: Int
}

/// Frame interpolation on the Neural Engine, through VideoToolbox's low-latency frame interpolator.
///
/// The point is the GPU. MetalGoose shares it with the app being captured, and every millisecond it
/// spends interpolating is one the app does not get. This path costs the GPU two small conversion
/// passes per image — about a tenth of a millisecond each at a megapixel — and does the interpolation
/// itself on the Neural Engine, which nothing else is using.
///
/// A pair is cut into two steps, which is one image, its midpoint, or into four, which is three: its
/// quarters. A quarter is made in two stages, the midpoint first and the quarter from it, so one alone takes
/// twice what the midpoint takes, and the three together about two and a half times: at 1280x1016, 11 ms
/// for the midpoint and 31 for all three back to back, and 17 and 38 at the pace of a capture, when the
/// Neural Engine has been idle in between. They are made in one call for that reason; three calls of one image
/// each took 54 ms. So four steps are a matter of whether three images fit the time between captures, which
/// they do at 1280x720 up to about 30 fps, at 1280x1016 below about 20 fps, and at 1920x1080 below about 15.
///
/// It is not a drop-in for MetalFX. VideoToolbox takes video-range 4:2:0 and nothing else, so each
/// frame is converted on the way in and the result on the way out; the processor is sized at creation
/// and builds its model for that size, which takes long enough that it happens off to the side while
/// captures go on being shown; and its size ceiling is a fraction of what MetalFX handles, so a large
/// window is declined and the caller uses MetalFX instead. Measured against an analytic ground truth on
/// a textured pan it lands at 39 to 41 dB at every phase, where the 4:2:0 conversion alone caps any such
/// path at 49 dB.
///
/// Concurrency: the capture pipeline calls `encodeConversion` on its own queue. Everything else —
/// session set-up, submission, the results — happens on this object's queue, and the render thread
/// reads results through a lock.
final class NeuralInterpolator: @unchecked Sendable {

    struct Output {
        let previous: CFTimeInterval
        let next: CFTimeInterval
        /// How far from `previous` to `next` the image sits.
        let phase: Double
        let texture: MTLTexture
    }

    /// The most images a pair has: the quarters.
    private static let maximumImages = 3

    /// The phases of a pair cut into `steps`, in the order they are wanted.
    private static func phases(steps: Int) -> [Double] {
        (1..<max(2, steps)).map { Double($0) / Double(steps) }
    }

    private let gpu: GPUContext
    private let errors: ErrorLog
    private let latency: GenerationLatency
    private let queue = DispatchQueue(label: "com.metalgoose.neural", qos: .userInitiated)

    // MARK: Capacity

    /// Whether this size can be interpolated here. Both dimensions have to be even, because the chroma
    /// planes are half-size, and the frame has to fit inside what the processor supports.
    static func supports(width: Int, height: Int) -> Bool {
        guard VTLowLatencyFrameInterpolationConfiguration.isSupported,
              width >= 64, height >= 64, width % 2 == 0, height % 2 == 0 else { return false }

        let maximumDimension = Int(VTLowLatencyFrameInterpolationConfiguration.__maximumDimension(forSpatialScaleFactor: 1))
        let maximumPixels = Int(VTLowLatencyFrameInterpolationConfiguration.__maximumPixelCount(forSpatialScaleFactor: 1))
        return max(width, height) <= maximumDimension && width * height <= maximumPixels
    }

    // MARK: State (queue-confined unless noted)

    private var size = (width: 0, height: 0)
    private var processor: VTFrameProcessor?
    private var inputs: [YUVFrame] = []
    private var outputs: [YUVFrame] = []
    private var outputTextures: [MTLTexture] = []

    private var inputIndex = 0
    private var outputIndex = 0
    private var previous: (frame: YUVFrame, timestamp: CFTimeInterval)?
    private var busy = false

    /// How long a call takes, for the midpoint alone and for all three quarters. The caller weighs them against
    /// the capture interval to decide how many images a pair can have. Written on this object's queue, read from
    /// the capture path.
    private let midpointTimes = OSAllocatedUnfairLock(initialState: IntervalFilter())
    private let quarterTimes = OSAllocatedUnfairLock(initialState: IntervalFilter())
    var midpointTime: CFTimeInterval { midpointTimes.withLock { $0.value } }
    var quartersTime: CFTimeInterval { quarterTimes.withLock { $0.value } }

    /// Bumped on every teardown, so a callback that outlives its session cannot write into the next one.
    private var generation = 0

    private struct Shared {
        var failed = false
        var results: [Output] = []
    }
    private let shared = OSAllocatedUnfairLock(uncheckedState: Shared())

    /// Two pairs' worth of images: the render clock reads the pair it is in the middle of and the one
    /// before it, and a pair further back has been replaced by a newer one.
    private static let resultCapacity = 2 * maximumImages

    init(gpu: GPUContext, errors: ErrorLog, latency: GenerationLatency) {
        self.gpu = gpu
        self.errors = errors
        self.latency = latency
    }

    /// The processor failed at run time, and the caller should use another backend from now on.
    var hasFailed: Bool { shared.withLockUnchecked { $0.failed } }

    func reset() {
        queue.async { [self] in tearDown() }
    }

    private func tearDown() {
        generation &+= 1
        processor?.endSession()
        processor = nil
        inputs.removeAll()
        outputs.removeAll()
        outputTextures.removeAll()
        previous = nil
        busy = false
        size = (0, 0)
        // A processor at another size takes another time.
        midpointTimes.withLock { $0.reset() }
        quarterTimes.withLock { $0.reset() }
        shared.withLockUnchecked { $0.failed = false; $0.results.removeAll() }
    }

    // MARK: - Capture side

    /// Encodes the conversion of one captured frame to 4:2:0 into the next buffer of the ring and
    /// returns that buffer, building the processor first when the size has changed. Called on the
    /// capture pipeline's queue.
    func encodeConversion(of frame: MTLTexture, commandBuffer: MTLCommandBuffer) -> YUVFrame? {
        let width = frame.width
        let height = frame.height
        // The ring and its index belong to this object's queue, which is also where a teardown runs.
        let target: YUVFrame? = queue.sync {
            configureIfNeeded(width: width, height: height)
            guard !inputs.isEmpty else { return nil }
            defer { inputIndex += 1 }
            return inputs[inputIndex % inputs.count]
        }
        guard let target, let encoder = commandBuffer.makeComputeCommandEncoder() else { return nil }

        encoder.setComputePipelineState(gpu.pipelines.convertTo420)
        encoder.setTexture(frame, index: 0)
        encoder.setTexture(target.luma, index: 1)
        encoder.setTexture(target.chroma, index: 2)
        // One thread per 2x2 block.
        gpu.dispatch(gpu.pipelines.convertTo420, on: encoder, width: width / 2, height: height / 2)
        encoder.endEncoding()
        return target
    }

    /// The conversion of `frame` has finished, so VideoToolbox may read it. Pairs it with the capture
    /// before it and interpolates between them, into `steps` steps: 2 makes the midpoint, 4 its quarters.
    func frameConverted(_ frame: YUVFrame, timestamp: CFTimeInterval, steps: Int) {
        queue.async { [self] in
            guard frame.epoch == generation else { return }
            defer { previous = (frame, timestamp) }
            guard let processor, let previous, !busy else { return }
            // Only one pair is worked on at a time. A pair that arrives while the Neural Engine is still on
            // the last has missed its moment: the render clock will have moved on before it finished.
            busy = true
            submit(processor: processor, previous: previous, current: (frame, timestamp),
                   phases: Self.phases(steps: steps), generation: generation)
        }
    }

    // MARK: - Interpolating

    /// One image of a pair: where it sits, and the buffers it is made into and read from.
    private struct Image {
        let phase: Double
        let destination: YUVFrame
        let texture: MTLTexture
    }

    private func submit(processor: VTFrameProcessor,
                        previous: (frame: YUVFrame, timestamp: CFTimeInterval),
                        current: (frame: YUVFrame, timestamp: CFTimeInterval),
                        phases: [Double], generation expected: Int) {
        func frame(_ yuv: YUVFrame, _ seconds: CFTimeInterval) -> VTFrameProcessorFrame? {
            VTFrameProcessorFrame(buffer: yuv.buffer, presentationTimeStamp: CMTime(seconds: seconds, preferredTimescale: 1_000_000))
        }
        let duration = current.timestamp - previous.timestamp
        var images: [Image] = []
        var outputFrames: [VTFrameProcessorFrame] = []
        for phase in phases {
            let destination = outputs[outputIndex % outputs.count]
            let texture = outputTextures[outputIndex % outputTextures.count]
            outputIndex += 1
            guard let output = frame(destination, previous.timestamp + phase * duration) else {
                busy = false
                return
            }
            images.append(Image(phase: phase, destination: destination, texture: texture))
            outputFrames.append(output)
        }
        guard let source = frame(current.frame, current.timestamp),
              let reference = frame(previous.frame, previous.timestamp),
              let parameters = VTLowLatencyFrameInterpolationParameters(
                sourceFrame: source, previousFrame: reference,
                interpolationPhase: phases.map { Float($0) }, destinationFrames: outputFrames) else {
            busy = false
            return
        }

        nonisolated(unsafe) let processor = processor
        nonisolated(unsafe) let request = parameters
        let made = images
        let (earlier, later) = (previous.timestamp, current.timestamp)
        Task { [self] in
            let started = CACurrentMediaTime()
            let failed: Bool
            do { try await processor.process(parameters: request); failed = false } catch { failed = true }
            let spent = CACurrentMediaTime() - started
            queue.async { [self] in
                busy = false
                guard generation == expected else { return }
                if failed {
                    // Not worth an alert: the capture path switches to MetalFX, which is slower on the GPU
                    // but always there. What would be worth knowing is that this happened.
                    shared.withLockUnchecked { $0.failed = true }
                    return
                }
                (made.count > 1 ? quarterTimes : midpointTimes).withLock {
                    $0.add(spent, window: EngineShared.measurementWindow, elapsed: duration)
                }
                publish(made, previous: earlier, next: later, generation: expected)
            }
        }
    }

    /// Converts the processor's 4:2:0 output back to BGRA and makes it available to the render thread
    /// once the GPU has written it.
    private func publish(_ images: [Image], previous: CFTimeInterval, next: CFTimeInterval, generation expected: Int) {
        guard let commandBuffer = gpu.makeCommandBuffer("MetalGoose neural interpolation"),
              let encoder = commandBuffer.makeComputeCommandEncoder() else { return }

        for image in images {
            encoder.setComputePipelineState(gpu.pipelines.convertFrom420)
            encoder.setTexture(image.destination.luma, index: 0)
            encoder.setTexture(image.destination.chroma, index: 1)
            encoder.setTexture(image.texture, index: 2)
            gpu.dispatch(gpu.pipelines.convertFrom420, on: encoder, width: image.texture.width, height: image.texture.height)
        }
        encoder.endEncoding()

        // The images of a pair come out of one call, so they are ready together, and the one that is wanted
        // first is ready when the last is: there is nothing to discount.
        nonisolated(unsafe) let textures = images.map { ($0.phase, $0.texture) }
        commandBuffer.addCompletedHandler { [self] _ in
            queue.async { [self] in
                guard generation == expected else { return }
                shared.withLockUnchecked { shared in
                    for (phase, texture) in textures {
                        shared.results.append(Output(previous: previous, next: next, phase: phase, texture: texture))
                    }
                    if shared.results.count > Self.resultCapacity {
                        shared.results.removeFirst(shared.results.count - Self.resultCapacity)
                    }
                }
                latency.record(previous: previous, next: next)
            }
        }
        commandBuffer.commit()
    }

    /// The image `phase` of the way from `previous` to `next`, once it has been produced. Called from the
    /// render thread.
    func texture(previous: CFTimeInterval, next: CFTimeInterval, phase: Double) -> MTLTexture? {
        shared.withLockUnchecked { shared in
            shared.results.last { $0.previous == previous && $0.next == next && $0.phase == phase }?.texture
        }
    }

    // MARK: - Set-up

    /// Builds the buffers for this size at once, and the processor in the background: starting a
    /// session builds the model for the size, which takes long enough that nothing should wait for it.
    private func configureIfNeeded(width: Int, height: Int) {
        guard size != (width, height) else { return }
        tearDown()
        size = (width, height)
        let expected = generation

        // Configured for the quarters, which a call for the midpoint alone takes as well and at the same cost,
        // so that going from two steps to four does not mean a new session.
        guard let configuration = VTLowLatencyFrameInterpolationConfiguration(frameWidth: width, frameHeight: height,
                                                                               numberOfInterpolatedFrames: 2) else {
            shared.withLockUnchecked { $0.failed = true }
            return
        }

        let count = FrameRing.capacity + GooseEngine.maxInFlight
        // An image's 4:2:0 buffer is needed until the GPU has converted it, while the next of its pair is being made.
        guard let inputs = makeBuffers(count: count, width: width, height: height,
                                       attributes: configuration.sourcePixelBufferAttributes),
              let outputs = makeBuffers(count: Self.maximumImages + 1, width: width, height: height,
                                        attributes: configuration.destinationPixelBufferAttributes) else {
            shared.withLockUnchecked { $0.failed = true }
            return
        }
        self.inputs = inputs
        self.outputs = outputs

        let descriptor = MTLTextureDescriptor.texture2DDescriptor(pixelFormat: .bgra8Unorm, width: width, height: height,
                                                                  mipmapped: false)
        descriptor.usage = [.shaderRead, .shaderWrite]
        descriptor.storageMode = .private
        outputTextures = (0..<Self.resultCapacity).compactMap { _ in gpu.device.makeTexture(descriptor: descriptor) }
        guard outputTextures.count == Self.resultCapacity else {
            shared.withLockUnchecked { $0.failed = true }
            return
        }

        DispatchQueue.global(qos: .userInitiated).async { [self] in
            let session = VTFrameProcessor()
            do { try session.startSession(configuration: configuration) } catch {
                queue.async { [self] in
                    if generation == expected { shared.withLockUnchecked { $0.failed = true } }
                }
                return
            }
            nonisolated(unsafe) let started = session
            queue.async { [self] in
                guard generation == expected else {
                    started.endSession()
                    return
                }
                processor = started
            }
        }
    }

    private func makeBuffers(count: Int, width: Int, height: Int, attributes: [String: Any]) -> [YUVFrame]? {
        var attributes = attributes
        attributes[kCVPixelBufferIOSurfacePropertiesKey as String] = [:] as CFDictionary
        attributes[kCVPixelBufferMetalCompatibilityKey as String] = true

        var frames: [YUVFrame] = []
        for _ in 0..<count {
            var created: CVPixelBuffer?
            guard CVPixelBufferCreate(kCFAllocatorDefault, width, height, kCVPixelFormatType_420YpCbCr8BiPlanarVideoRange,
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
            guard let luma = plane(.r8Unorm, 0, width, height),
                  let chroma = plane(.rg8Unorm, 1, width / 2, height / 2) else { return nil }
            frames.append(YUVFrame(buffer: buffer, luma: luma, chroma: chroma, epoch: generation))
        }
        return frames
    }
}
