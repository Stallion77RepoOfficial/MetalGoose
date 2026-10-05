import Foundation
@preconcurrency import Metal
@preconcurrency import CoreVideo
@preconcurrency import VideoToolbox
@preconcurrency import IOSurface
import QuartzCore
import os

/// The motion between two consecutive captures, one vector per block of the frame.
struct MotionField {
    /// Backward motion against the previous capture, in pixels of the frame MetalFX works on.
    let vectors: any MTLTexture
    /// Timestamp of the newer frame of the pair this was measured from.
    let timestamp: CFTimeInterval
    /// The slot the field is in, held by every command buffer that reads it until that has run.
    let lease: BufferLease
    /// Where on the capture lane the field is written (`GPULane.submitted`).
    let written: UInt64
    let validity: WorkValidity
}

// MARK: - Media engine

/// Wraps VTMotionEstimationSession, which runs on the media engine rather than the GPU. It takes
/// single-component luma, so each frame is converted first, and it returns backward vectors in
/// pixels at one vector per block.
///
/// How the frame is searched trades accuracy against time. Measured on real video, against the
/// capture the field has to explain:
///
/// - One search pass, the default, finds vectors that explain a large frame *worse* than assuming nothing
///   moved (31 dB against 35 on 1440p video), and an image made along the wrong ones swims like dough. Several
///   passes ("true motion") cost a fraction of a millisecond more and gave 38 dB.
/// - 4x4 blocks are more accurate again (43 dB) but much slower, and the media engine will not build them at
///   4K at all. `MotionAnalysis` decides how finely a frame can afford to be searched, and in what blocks;
///   this follows it.
private final class MediaEngineEstimator {

    private let device: MTLDevice
    private var session: __VTMotionEstimationSession?
    private var size: (width: Int, height: Int) = (0, 0)
    private var analysis = MotionAnalysis(rung: MotionAnalysis.coarsest, frameWidth: 0, frameHeight: 0)
    private var rung: Int?

    /// Rungs the media engine would not build for the frame size it was last asked for — it refuses 4x4
    /// blocks at 3840x2160 and above. A rung that gave way to the next is not tried again every frame.
    private var refused: Set<Int> = []
    private var refusedSize: (width: Int, height: Int) = (0, 0)

    /// How much longer than the measurements say a pair's vectors take to come back, as the median of what
    /// they have taken — a slower chip, a GPU shared with a game, a video decoder in the captured window
    /// sharing the media engine. Written on VideoToolbox's callback thread, read on the processing queue.
    private let slowdown = OSAllocatedUnfairLock(initialState: IntervalFilter())
    private var lumaBuffers: [CVPixelBuffer] = []
    private var lumaTextures: [MTLTexture] = []
    private var slot = 0
    private var hasReference = false

    /// Bumped on every teardown, so an estimate scheduled against the old session cannot be
    /// submitted against the new one's buffers.
    private var generation = 0

    /// Four. The media engine reads a pair outside Metal's ordering, so a buffer is not written while a
    /// pair that reads it is out: a ring that wrote it anyway would have VideoToolbox measure against a
    /// half-overwritten reference. A pair is out from the moment its frame arrives to the moment the vectors
    /// come back, and a GPU shared with a game stretches that past a capture interval — a ring of three had
    /// no free buffer for the frames that arrived meanwhile, and a frame that is not converted is a pair that
    /// is not measured. Four hold three pairs at once.
    private static let lumaSlotCount = 4

    /// A pair the media engine has been given and has not answered, so that nothing writes one of its two
    /// buffers meanwhile.
    private struct Measurement {
        let id: Int
        let reference: Int
        let current: Int
        let since: CFTimeInterval
    }
    private let measuring = OSAllocatedUnfairLock<[Measurement]>(initialState: [])
    private var measurementCount = 0

    /// A measurement VideoToolbox never answers is given up on after this long.
    private static let measurementTimeout: CFTimeInterval = 0.5

    /// VideoToolbox recycles its output buffers from a pool, so the same few IOSurfaces come
    /// back frame after frame. Building a descriptor and a driver texture for each one at up
    /// to 240 fps allocated for surfaces that were already wrapped.
    private var vectorTextures: [IOSurfaceID: MTLTexture] = [:]

    /// The pair `estimate` should hand to the media engine, resolved when the luma destination
    /// is handed out rather than when the GPU finishes.
    struct PendingPair {
        let id: Int
        let current: CVPixelBuffer
        let reference: CVPixelBuffer
        let generation: Int
        /// How the pair was analysed, which says what its field means.
        let analysis: MotionAnalysis
        /// When it was started, and how long it should take by the measurements: what is learnt from the
        /// difference. The capture interval is what the sample stands for.
        let since: CFTimeInterval
        let expected: CFTimeInterval
        let interval: CFTimeInterval
    }

    init(device: MTLDevice) { self.device = device }

    /// The next frame starts a new pair instead of completing one with the last frame this was given, which after a
    /// stretch in which none was given is old.
    func breakSequence() {
        hasReference = false
    }

    func reset() {
        if let session { __VTMotionEstimationSessionInvalidate(session) }
        session = nil
        size = (0, 0)
        rung = nil
        lumaBuffers.removeAll()
        lumaTextures.removeAll()
        vectorTextures.removeAll()
        slot = 0
        hasReference = false
        measuring.withLock { $0.removeAll() }
        generation &+= 1
    }

    private func ensureSession(width: Int, height: Int, interval: CFTimeInterval) -> Bool {
        if refusedSize != (width, height) { refused = []; refusedSize = (width, height) }
        let wanted = MotionAnalysis.choose(current: rung, frameWidth: width, frameHeight: height, interval: interval,
                                           slowdown: slowdown.withLock { $0.value }, refused: refused)
        if session != nil, size == (width, height), rung == wanted { return true }
        reset()

        for index in wanted...MotionAnalysis.coarsest {
            if build(rung: index, width: width, height: height) {
                rung = index
                return true
            }
            refused.insert(index)
        }
        return false
    }

    private func build(rung index: Int, width: Int, height: Int) -> Bool {
        let plan = MotionAnalysis(rung: index, frameWidth: width, frameHeight: height)
        let options: [String: Any] = [
            kVTMotionEstimationSessionCreationOption_UseMultiPassSearch as String: true as CFBoolean,
            kVTMotionEstimationSessionCreationOption_MotionVectorSize as String: plan.blockSize as CFNumber
        ]
        var created: __VTMotionEstimationSession?
        guard __VTMotionEstimationSessionCreate(kCFAllocatorDefault, options as CFDictionary,
                                                UInt32(plan.width), UInt32(plan.height), &created) == noErr,
              let created else { return false }

        var attributes: CFDictionary?
        __VTMotionEstimationSessionCopySourcePixelBufferAttributes(created, &attributes)
        var descriptor = (attributes as? [String: Any]) ?? [:]
        descriptor[kCVPixelBufferIOSurfacePropertiesKey as String] = [:] as CFDictionary

        var buffers: [CVPixelBuffer] = []
        var textures: [MTLTexture] = []
        for _ in 0..<Self.lumaSlotCount {
            var buffer: CVPixelBuffer?
            let textureDescriptor = MTLTextureDescriptor.texture2DDescriptor(
                pixelFormat: .r8Unorm, width: plan.width, height: plan.height, mipmapped: false)
            textureDescriptor.usage = [.shaderRead, .shaderWrite]
            guard CVPixelBufferCreate(kCFAllocatorDefault, plan.width, plan.height,
                                      kCVPixelFormatType_OneComponent8,
                                      descriptor as CFDictionary, &buffer) == kCVReturnSuccess,
                  let buffer,
                  let surface = CVPixelBufferGetIOSurface(buffer)?.takeUnretainedValue(),
                  let texture = device.makeTexture(descriptor: textureDescriptor, iosurface: surface, plane: 0) else {
                __VTMotionEstimationSessionInvalidate(created)
                return false
            }
            buffers.append(buffer)
            textures.append(texture)
        }

        session = created
        size = (width, height)
        analysis = plan
        lumaBuffers = buffers
        lumaTextures = textures
        return true
    }

    /// Destination for this frame's luma conversion, how the frame is analysed, and the pair that
    /// destination completes — none when there is nothing to measure yet.
    ///
    /// Nothing is measured until the capture interval is known, because the analysis follows it.
    ///
    /// The slot advances here, where the destination is handed out. Advancing it inside
    /// `estimate` tied it to a GPU completion handler instead: under load that completion
    /// landed after the next capture had already asked for a destination, both frames were
    /// converted into the same buffer, and the estimate then compared a frame against a stale
    /// partner. The resulting vectors describe a pair that never existed — which is exactly the
    /// smearing the images showed when the GPU was busiest.
    ///
    /// Requests queue inside the media engine, so one that takes longer than a capture interval would fall
    /// further behind with every frame — measured, 200 ms late at 1440p. What bounds the queue is the
    /// buffers: a frame that finds its buffer still being read is not converted, and the next one starts a
    /// new pair, so every pair is still two consecutive captures and its field still spans one interval. The
    /// analysis is chosen so that the engine keeps up, and this is what holds when it does not.
    func prepare(width: Int, height: Int, interval: CFTimeInterval)
        -> (texture: MTLTexture, analysis: MotionAnalysis, pending: PendingPair?)? {
        let count = Self.lumaSlotCount
        guard interval > 0 else {
            hasReference = false
            return nil
        }
        guard ensureSession(width: width, height: height, interval: interval), let rung,
              lumaBuffers.count == count else { return nil }

        let now = CACurrentMediaTime()
        let destination = slot
        let reference = (destination + count - 1) % count
        // The destination is one of the buffers being measured: writing it would corrupt the measurement,
        // so this frame is skipped and the next one starts a new pair.
        let isFree = measuring.withLock { pairs -> Bool in
            pairs.removeAll { now - $0.since > Self.measurementTimeout }
            return !pairs.contains { $0.reference == destination || $0.current == destination }
        }
        guard isFree else {
            hasReference = false
            return nil
        }
        slot = (slot + 1) % count

        let texture = lumaTextures[destination]
        guard hasReference else {
            hasReference = true
            return (texture, analysis, nil)
        }

        measurementCount += 1
        let id = measurementCount
        measuring.withLock { $0.append(Measurement(id: id, reference: reference, current: destination, since: now)) }
        let expected = MotionAnalysis.cost(rung: rung, frameWidth: width, frameHeight: height)
        return (texture, analysis, PendingPair(id: id, current: lumaBuffers[destination],
                                               reference: lumaBuffers[reference], generation: generation,
                                               analysis: analysis, since: now, expected: expected, interval: interval))
    }

    /// Submits the pair and returns. The vectors arrive on VideoToolbox's own callback thread and
    /// are picked up by a later frame, so a slow or silent media engine costs a missing update
    /// rather than a stalled capture queue.
    ///
    /// Must be called only once the luma write has actually landed.
    func estimate(_ pending: PendingPair, completion: @escaping @Sendable (CVPixelBuffer) -> Void) {
        let (id, since, expected, interval) = (pending.id, pending.since, pending.expected, pending.interval)
        guard let session, pending.generation == generation else {
            measuring.withLock { $0.removeAll { $0.id == id } }
            return
        }
        let status = __VTMotionEstimationSessionEstimateMotionVectors(
            session, pending.reference, pending.current, [], nil
        ) { [measuring, slowdown] status, _, _, vectors in
            measuring.withLock { $0.removeAll { $0.id == id } }
            guard status == noErr, let vectors else { return }
            slowdown.withLock {
                $0.add((CACurrentMediaTime() - since) / expected, window: EngineShared.measurementWindow, elapsed: interval)
            }
            completion(vectors)
        }
        if status != noErr { measuring.withLock { $0.removeAll { $0.id == id } } }
    }

    func texture(for vectors: CVPixelBuffer) -> MTLTexture? {
        guard let surface = CVPixelBufferGetIOSurface(vectors)?.takeUnretainedValue() else { return nil }
        let id = IOSurfaceGetID(surface)
        if let cached = vectorTextures[id] { return cached }

        let descriptor = MTLTextureDescriptor.texture2DDescriptor(
            pixelFormat: .rg16Float,
            width: CVPixelBufferGetWidth(vectors),
            height: CVPixelBufferGetHeight(vectors),
            mipmapped: false)
        descriptor.usage = [.shaderRead]
        guard let texture = device.makeTexture(descriptor: descriptor, iosurface: surface, plane: 0) else {
            return nil
        }
        // The pool is a handful of surfaces. More than that means it is not being recycled
        // after all, and the cache is only holding memory.
        if vectorTextures.count >= 8 { vectorTextures.removeAll() }
        vectorTextures[id] = texture
        return texture
    }
}

// MARK: - Pipeline

/// Produces the motion field MetalFX interpolates along: measures it with the media engine, then removes the vectors
/// the block matcher got wildly wrong. Every method runs on the processing queue, which owns the estimator's state.
final class MotionPipeline: @unchecked Sendable {

    private struct Slot {
        /// The field as measured, scaled into the units of the frame MetalFX works on. Everything downstream reads
        /// `vectors`, which is this with its outliers removed.
        var raw: (any MTLTexture)?
        var vectors: (any MTLTexture)?
        /// What the frame as a whole is doing, which the wild vectors are judged against.
        var global: (any MTLBuffer)?
    }

    /// Fields are replaced as new ones arrive, so only the latest and the few still referenced by
    /// in-flight command buffers need to stay distinct. A slot is leased (`MotionField.lease`): MetalFX reads a field on a
    /// lane of its own, and the slot is not written again until it has. A field that finds no free slot is not stored.
    private static let slotCount = 4

    private let gpu: GPUContext
    private let queue: DispatchQueue
    private let mediaEngine: MediaEngineEstimator
    private var slots = [Slot](repeating: Slot(), count: MotionPipeline.slotCount)
    private var slotLeases = BufferLeasePool(capacity: MotionPipeline.slotCount)

    /// Called on the processing queue each time a field has been stored, so that the pair it belongs to is fed to
    /// MetalFX the moment its motion exists.
    var onField: ((MotionField) -> Void)?

    init(gpu: GPUContext, queue: DispatchQueue) {
        self.gpu = gpu
        self.queue = queue
        self.mediaEngine = MediaEngineEstimator(device: gpu.device)
    }

    /// For when frames stop being submitted and start again, as when another engine made the images for a while: the
    /// first one then has nothing before it to be measured against, rather than a frame from long ago.
    func breakSequence() {
        mediaEngine.breakSequence()
    }

    func reset() {
        slots = [Slot](repeating: Slot(), count: Self.slotCount)
        slotLeases = BufferLeasePool(capacity: Self.slotCount)
        mediaEngine.reset()
    }

    /// Starts measuring the motion that brought the capture to where it is. The field lands later, from the media
    /// engine's own thread, and is handed to `onField`.
    ///
    /// Converts the frame to luma, hands it to the media engine, and copies the resulting vectors
    /// into a slot we own — VideoToolbox recycles its own buffers, and a field is read long after
    /// its frame left the estimator.
    ///
    /// - Parameter interval: the time between captures, which decides how finely the media engine can
    ///   afford to search the frame. 0 while it is not known yet.
    func submit(frame: FrameHistory, interval: CFTimeInterval) {
        let texture = frame.texture
        guard let prepared = mediaEngine.prepare(width: texture.width, height: texture.height, interval: interval),
              let command = gpu.capture.makeCommand("MetalGoose motion luma"),
              let encoder = command.makeComputePass() else { return }

        var divisor = UInt32(prepared.analysis.divisor)
        encoder.setComputePipelineState(gpu.pipelines.luma)
        encoder.setTexture(texture, index: 0)
        encoder.setTexture(prepared.texture, index: 1)
        encoder.setBytes(&divisor, length: MemoryLayout<UInt32>.size, index: 0)
        gpu.dispatch(gpu.pipelines.luma, on: encoder, width: prepared.texture.width, height: prepared.texture.height)
        encoder.endEncoding()
        command.retain(frame.lease)

        // VideoToolbox reads the IOSurface outside Metal's ordering, so the estimate has to follow
        // the write — but nothing waits for it. It is submitted from the completion handler and the
        // vectors are picked up by a later frame, so a slow or silent media engine costs a missing
        // update rather than a stalled capture queue.
        if let pending = prepared.pending {
            nonisolated(unsafe) let pending = pending
            let (width, timestamp, validity) = (texture.width, frame.timestamp, frame.validity)
            command.onCompleted { [self] completion in
                guard completion.succeeded, validity.isValid else { return }
                estimate(pending, frameWidth: width, timestamp: timestamp)
            }
        }
        command.commit()
    }

    /// Estimator state belongs to the processing queue; the completion handler that gets here runs
    /// on a Metal thread, and the vectors arrive on VideoToolbox's.
    private func estimate(_ pending: MediaEngineEstimator.PendingPair, frameWidth: Int, timestamp: CFTimeInterval) {
        nonisolated(unsafe) let pending = pending
        let analysis = pending.analysis
        queue.async { [self] in
            mediaEngine.estimate(pending) { vectors in
                nonisolated(unsafe) let vectors = vectors
                self.queue.async {
                    guard let field = self.mediaEngine.texture(for: vectors) else { return }
                    self.store(field, retaining: vectors, analysis: analysis, frameWidth: frameWidth, timestamp: timestamp)
                }
            }
        }
    }

    // MARK: Field post-processing

    private func ensureGlobalBuffer(_ buffer: inout (any MTLBuffer)?) -> (any MTLBuffer)? {
        if buffer == nil {
            buffer = gpu.device.makeBuffer(length: MemoryLayout<SIMD2<Float>>.stride, options: .storageModePrivate)
        }
        return buffer
    }

    /// Copies the estimator's output into a slot we own — VideoToolbox recycles its buffers while the field is still
    /// being read — and removes the vectors the block matcher got wildly wrong.
    ///
    /// The vectors are measured on the averaged-down image the media engine searched, so they are carried
    /// into the units of the frame MetalFX works on; and the field is cut to the vectors that cover the
    /// frame, because what the media engine pads it with is not the image.
    private func store(_ field: any MTLTexture, retaining buffer: CVPixelBuffer, analysis: MotionAnalysis, frameWidth: Int,
                       timestamp: CFTimeInterval) {
        let width = min(field.width, analysis.vectorWidth)
        let height = min(field.height, analysis.vectorHeight)
        guard let lease = slotLeases.acquire() else { return }
        let index = lease.index
        guard let raw = gpu.ensureTexture(&slots[index].raw, width: width, height: height, pixelFormat: .rg16Float),
              let vectors = gpu.ensureTexture(&slots[index].vectors, width: width, height: height,
                                              pixelFormat: .rg16Float),
              let global = ensureGlobalBuffer(&slots[index].global),
              let command = gpu.capture.makeCommand("MetalGoose motion field"),
              let encoder = command.makeComputePass() else { return }

        // A compute copy rather than a blit, because the field has to be scaled on the way in and a
        // blit cannot touch the values it moves.
        var scale = Float(analysis.divisor)
        var coverage = SIMD2<Float>(Float(width) / Float(field.width), Float(height) / Float(field.height))
        encoder.setComputePipelineState(gpu.pipelines.copyMotion)
        encoder.setTexture(field, index: 0)
        encoder.setTexture(raw, index: 1)
        encoder.setBytes(&scale, length: MemoryLayout<Float>.size, index: 0)
        encoder.setBytes(&coverage, length: MemoryLayout<SIMD2<Float>>.size, index: 1)
        gpu.dispatch(gpu.pipelines.copyMotion, on: encoder, width: width, height: height)

        // What the frame as a whole is doing is measured on the raw field — it is a trimmed mean, so a
        // few wild blocks do not move it — and is what the wild blocks are judged against.
        encoder.setComputePipelineState(gpu.pipelines.globalMotion)
        encoder.setTexture(raw, index: 0)
        encoder.setBuffer(global, index: 0)
        encoder.dispatchThreadgroups(MTLSize(width: 1, height: 1, depth: 1),
                                     threadsPerThreadgroup: MTLSize(width: Int(MG_MOTION_GRID * MG_MOTION_GRID),
                                                                    height: 1, depth: 1))

        var frame = Float(frameWidth)
        encoder.setComputePipelineState(gpu.pipelines.despeckle)
        encoder.setTexture(raw, index: 0)
        encoder.setTexture(vectors, index: 1)
        encoder.setBuffer(global, index: 0)
        encoder.setBytes(&frame, length: MemoryLayout<Float>.size, index: 1)
        gpu.dispatch(gpu.pipelines.despeckle, on: encoder, width: width, height: height)
        encoder.endEncoding()
        command.retain(buffer)
        command.retain(lease)
        let validity = WorkValidity()
        command.onCompleted { completion in
            if !completion.succeeded { validity.invalidate() }
        }
        let written = command.commit()

        onField?(MotionField(vectors: vectors, timestamp: timestamp, lease: lease, written: written, validity: validity))
    }
}
