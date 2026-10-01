import Foundation
@preconcurrency import Metal
@preconcurrency import CoreVideo
@preconcurrency import VideoToolbox
@preconcurrency import Vision
@preconcurrency import IOSurface
import QuartzCore
import os

/// A motion field and the maps derived from it.
///
/// Two of them are properties of the field alone — one value per block, not one per pixel —
/// so they are built once per capture at the field's own resolution.
struct MotionField {
    /// Backward motion against the previous capture, in pixels.
    let vectors: MTLTexture
    /// Where neighbouring vectors disagree, the field is straddling a motion boundary it
    /// cannot represent, and the warp fades there.
    let disagreement: MTLTexture
    /// One vector: what the frame as a whole is doing. Pixels whose own vector is not trusted
    /// fall back to this instead of freezing, which keeps the image internally coherent — a
    /// slightly misplaced frame reads as motion, a half-frozen one reads as a fault.
    let global: MTLBuffer
    /// Timestamp of the newer frame of the pair this was measured from. The warp extrapolates
    /// forward assuming the field describes the interval that just ended; a field measured
    /// earlier describes a velocity the scene has already left.
    let timestamp: CFTimeInterval
}

// MARK: - Media engine

/// Wraps VTMotionEstimationSession, which runs on the media engine rather than the GPU. It takes
/// single-component luma, so each frame is converted first, and it returns backward vectors in
/// pixels at one vector per block.
private final class MediaEngineEstimator {
    private let device: MTLDevice
    private var session: __VTMotionEstimationSession?
    private var size: (width: Int, height: Int) = (0, 0)
    private var lumaBuffers: [CVPixelBuffer] = []
    private var lumaTextures: [MTLTexture] = []
    private var slot = 0
    private var hasReference = false

    /// Bumped on every teardown, so an estimate scheduled against the old session cannot be
    /// submitted against the new one's buffers.
    private var generation = 0

    /// Three, not two. The media engine reads the pair outside Metal's ordering, so a two-slot
    /// ring hands the next capture the very buffer the pending estimate is still using as its
    /// reference: the luma pass for frame N+1 can be committed before frame N's estimate has
    /// been submitted, and VideoToolbox then measures against a half-overwritten reference. A
    /// third slot puts a whole capture between the write and the buffer's reuse, which is more
    /// than the pipeline ever has in flight, for one extra single-component frame of memory.
    private static let lumaSlotCount = 3

    /// VideoToolbox recycles its output buffers from a pool, so the same few IOSurfaces come
    /// back frame after frame. Building a descriptor and a driver texture for each one at up
    /// to 240 fps allocated for surfaces that were already wrapped.
    private var vectorTextures: [IOSurfaceID: MTLTexture] = [:]

    /// The pair `estimate` should hand to the media engine, resolved when the luma destination
    /// is handed out rather than when the GPU finishes.
    struct PendingPair {
        let current: CVPixelBuffer
        let reference: CVPixelBuffer
        let generation: Int
    }

    init(device: MTLDevice) { self.device = device }

    func reset() {
        if let session { __VTMotionEstimationSessionInvalidate(session) }
        session = nil
        size = (0, 0)
        lumaBuffers.removeAll()
        lumaTextures.removeAll()
        vectorTextures.removeAll()
        slot = 0
        hasReference = false
        generation &+= 1
    }

    /// Default block size and a single search pass: a 4x4 grid measures 12x slower, which no
    /// real-time budget can absorb.
    private func ensureSession(width: Int, height: Int) -> Bool {
        if session != nil, size == (width, height) { return true }
        reset()

        let options: [String: Any] = [
            kVTMotionEstimationSessionCreationOption_UseMultiPassSearch as String: false as CFBoolean
        ]
        var created: __VTMotionEstimationSession?
        guard __VTMotionEstimationSessionCreate(kCFAllocatorDefault, options as CFDictionary,
                                                UInt32(width), UInt32(height), &created) == noErr,
              let created else { return false }

        var attributes: CFDictionary?
        __VTMotionEstimationSessionCopySourcePixelBufferAttributes(created, &attributes)
        var descriptor = (attributes as? [String: Any]) ?? [:]
        descriptor[kCVPixelBufferIOSurfacePropertiesKey as String] = [:] as CFDictionary

        for _ in 0..<Self.lumaSlotCount {
            var buffer: CVPixelBuffer?
            guard CVPixelBufferCreate(kCFAllocatorDefault, width, height,
                                      kCVPixelFormatType_OneComponent8,
                                      descriptor as CFDictionary, &buffer) == kCVReturnSuccess,
                  let buffer,
                  let surface = CVPixelBufferGetIOSurface(buffer)?.takeUnretainedValue() else { return false }

            let textureDescriptor = MTLTextureDescriptor.texture2DDescriptor(
                pixelFormat: .r8Unorm, width: width, height: height, mipmapped: false)
            textureDescriptor.usage = [.shaderRead, .shaderWrite]
            guard let texture = device.makeTexture(descriptor: textureDescriptor, iosurface: surface, plane: 0) else {
                return false
            }
            lumaBuffers.append(buffer)
            lumaTextures.append(texture)
        }

        session = created
        size = (width, height)
        return true
    }

    /// Destination for this frame's luma conversion, together with the pair that destination
    /// completes.
    ///
    /// The slot advances here, where the destination is handed out. Advancing it inside
    /// `estimate` tied it to a GPU completion handler instead: under load that completion
    /// landed after the next capture had already asked for a destination, both frames were
    /// converted into the same buffer, and the estimate then compared a frame against a stale
    /// partner. The resulting vectors describe a pair that never existed — which is exactly the
    /// smearing the warp showed when the GPU was busiest.
    func prepare(width: Int, height: Int) -> (texture: MTLTexture, pending: PendingPair?)? {
        let count = Self.lumaSlotCount
        guard ensureSession(width: width, height: height), lumaBuffers.count == count else { return nil }
        let texture = lumaTextures[slot]
        let current = lumaBuffers[slot]
        let reference = lumaBuffers[(slot + count - 1) % count]
        slot = (slot + 1) % count

        guard hasReference else {
            hasReference = true
            return (texture, nil)
        }
        return (texture, PendingPair(current: current, reference: reference, generation: generation))
    }

    /// Submits the pair and returns. The vectors arrive on VideoToolbox's own callback thread and
    /// are picked up by a later frame, so a slow or silent media engine costs a missing update
    /// rather than a stalled capture queue.
    ///
    /// Must be called only once the luma write has actually landed.
    func estimate(_ pending: PendingPair, completion: @escaping @Sendable (CVPixelBuffer) -> Void) {
        guard let session, pending.generation == generation else { return }
        _ = __VTMotionEstimationSessionEstimateMotionVectors(
            session, pending.reference, pending.current, [], nil
        ) { status, _, _, vectors in
            guard status == noErr, let vectors else { return }
            completion(vectors)
        }
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

// MARK: - Optical flow

/// Optical flow from Vision, as an alternative to the media engine's block matcher.
///
/// The block matcher is built for video compression, where any match that shrinks the residual is
/// a good match. On a repeating texture — a tiled floor, a brick wall — it locks onto the wrong
/// repeat and every block agrees on the same wrong answer, which no neighbour-agreement test can
/// detect and which the warp turns into liquid. This returns the exact displacement on textured
/// content and falls to near zero where the motion is genuinely ambiguous, and a zero vector
/// simply holds the source pixel.
///
/// It costs about six times the block matcher, so it only fits while captures arrive slowly.
private final class OpticalFlowEstimator: @unchecked Sendable {
    private let device: MTLDevice
    /// Its own queue: `perform` blocks for the whole computation, and the processing queue is
    /// where every captured frame is handled.
    private let queue = DispatchQueue(label: "com.metalgoose.opticalflow", qos: .userInitiated)
    private struct State {
        var previous: CVPixelBuffer?
        var inFlight = false
        var pool: CVPixelBufferPool?
        var poolSize: (width: Int, height: Int) = (0, 0)
    }
    private let state = OSAllocatedUnfairLock(uncheckedState: State())

    init(device: MTLDevice) { self.device = device }

    func reset() {
        state.withLockUnchecked { $0 = State() }
    }

    /// Frames are copied out of ScreenCaptureKit's buffers rather than retained. Its pool is
    /// `queueDepth` deep — two or three — and the pipeline already holds one per in-flight command
    /// buffer; holding the previous frame and the one being measured on top of that starves the
    /// pool and the capture rate falls. A pair of buffers we own costs one copy per frame and takes
    /// the pool out of the question entirely.
    private static func copy(_ source: CVPixelBuffer, using state: inout State) -> CVPixelBuffer? {
        let width = CVPixelBufferGetWidth(source)
        let height = CVPixelBufferGetHeight(source)
        if state.pool == nil || state.poolSize != (width, height) {
            var created: CVPixelBufferPool?
            let attributes: [String: Any] = [
                kCVPixelBufferWidthKey as String: width,
                kCVPixelBufferHeightKey as String: height,
                kCVPixelBufferPixelFormatTypeKey as String: CVPixelBufferGetPixelFormatType(source),
                kCVPixelBufferIOSurfacePropertiesKey as String: [:] as CFDictionary
            ]
            // Three: the reference, the frame being measured against it, and the one arriving
            // while that runs.
            guard CVPixelBufferPoolCreate(kCFAllocatorDefault,
                                          [kCVPixelBufferPoolMinimumBufferCountKey as String: 3] as CFDictionary,
                                          attributes as CFDictionary, &created) == kCVReturnSuccess,
                  let created else { return nil }
            state.pool = created
            state.poolSize = (width, height)
        }
        guard let pool = state.pool else { return nil }
        var destination: CVPixelBuffer?
        guard CVPixelBufferPoolCreatePixelBuffer(kCFAllocatorDefault, pool, &destination) == kCVReturnSuccess,
              let destination else { return nil }

        CVPixelBufferLockBaseAddress(source, .readOnly)
        CVPixelBufferLockBaseAddress(destination, [])
        defer {
            CVPixelBufferUnlockBaseAddress(destination, [])
            CVPixelBufferUnlockBaseAddress(source, .readOnly)
        }
        guard let src = CVPixelBufferGetBaseAddress(source),
              let dst = CVPixelBufferGetBaseAddress(destination) else { return nil }
        let srcStride = CVPixelBufferGetBytesPerRow(source)
        let dstStride = CVPixelBufferGetBytesPerRow(destination)
        if srcStride == dstStride {
            memcpy(dst, src, srcStride * height)
        } else {
            let row = min(srcStride, dstStride)
            for y in 0..<height {
                memcpy(dst + y * dstStride, src + y * srcStride, row)
            }
        }
        return destination
    }

    /// Hands in the newest captured buffer and, if a request is not already running, starts one
    /// against the buffer before it.
    ///
    /// Only one request is ever in flight. Vision has no way to cancel, and a queue of them would
    /// each be measuring a pair the pipeline had already moved past — so a capture that arrives
    /// mid-flight simply becomes the next reference.
    func submit(_ pixelBuffer: CVPixelBuffer, completion: @escaping @Sendable (MTLTexture) -> Void) {
        let job = state.withLockUnchecked { state -> (from: CVPixelBuffer, to: CVPixelBuffer)? in
            let reference = state.previous
            // Copied even when a request is already running: this frame is the next reference
            // either way, and the source buffer goes back to ScreenCaptureKit the moment we return.
            state.previous = Self.copy(pixelBuffer, using: &state) ?? state.previous
            guard !state.inFlight, let reference, let current = state.previous else { return nil }
            state.inFlight = true
            return (reference, current)
        }
        guard let job else { return }

        nonisolated(unsafe) let from = job.from
        nonisolated(unsafe) let to = job.to
        queue.async { [self] in
            let texture = flow(from: from, to: to)
            state.withLockUnchecked { $0.inFlight = false }
            if let texture { completion(texture) }
        }
    }

    /// Revision 1 rather than 2: the ML path measured both slower and badly short on displacement.
    /// The accuracy level looked irrelevant against a single rigid shift — every level returned the
    /// same exact displacement and the higher ones only cost time — but that measurement had no
    /// thin structures and no motion boundaries in it, which is the only place the level can
    /// matter: a coarser pyramid smooths further, and smoothing is what drags a static overlay
    /// along with the scene moving behind it.
    private func flow(from: CVPixelBuffer, to: CVPixelBuffer) -> MTLTexture? {
        let request = VNGenerateOpticalFlowRequest(targetedCVPixelBuffer: to, options: [:])
        request.computationAccuracy = .high
        request.outputPixelFormat = kCVPixelFormatType_TwoComponent16Half
        if VNGenerateOpticalFlowRequest.supportedRevisions.contains(1) {
            request.revision = 1
        }
        do { try VNImageRequestHandler(cvPixelBuffer: from, options: [:]).perform([request]) } catch { return nil }
        guard let buffer = request.results?.first?.pixelBuffer else { return nil }
        return texture(for: buffer)
    }

    /// Vision's buffer is not guaranteed to be IOSurface-backed, so there are two ways in: wrap it
    /// where it is, and copy it where that is not possible. The wrap is the common path.
    private func texture(for buffer: CVPixelBuffer) -> MTLTexture? {
        let width = CVPixelBufferGetWidth(buffer)
        let height = CVPixelBufferGetHeight(buffer)
        let descriptor = MTLTextureDescriptor.texture2DDescriptor(
            pixelFormat: .rg16Float, width: width, height: height, mipmapped: false)
        descriptor.usage = [.shaderRead]

        if let surface = CVPixelBufferGetIOSurface(buffer)?.takeUnretainedValue() {
            return device.makeTexture(descriptor: descriptor, iosurface: surface, plane: 0)
        }

        descriptor.storageMode = .shared
        guard let texture = device.makeTexture(descriptor: descriptor) else { return nil }
        CVPixelBufferLockBaseAddress(buffer, .readOnly)
        defer { CVPixelBufferUnlockBaseAddress(buffer, .readOnly) }
        guard let base = CVPixelBufferGetBaseAddress(buffer) else { return nil }
        texture.replace(region: MTLRegionMake2D(0, 0, width, height), mipmapLevel: 0,
                        withBytes: base, bytesPerRow: CVPixelBufferGetBytesPerRow(buffer))
        return texture
    }
}

// MARK: - Pipeline

/// Produces the motion field extrapolation warps along: measures it with the chosen estimator,
/// then derives the maps the warp reads from it. Every method runs on the processing queue,
/// which owns the estimators' state; the latest field is the one thing the render thread reads.
final class MotionPipeline: @unchecked Sendable {

    private struct Slot {
        /// The field as measured, scaled into the warp's units. Everything downstream reads `vectors`,
        /// which is this with its outliers removed.
        var raw: MTLTexture?
        var vectors: MTLTexture?
        var disagreement: MTLTexture?
        var global: MTLBuffer?
    }

    /// Fields are replaced as new ones arrive, so only the latest and the few still referenced by
    /// in-flight render command buffers need to stay distinct.
    private static let slotCount = 4

    /// VideoToolbox's default block. A dense field has one vector per pixel, so neighbours are
    /// compared this far apart in frame pixels whatever produced it.
    private static let blockSpan = 16

    private let gpu: GPUContext
    private let queue: DispatchQueue
    private let mediaEngine: MediaEngineEstimator
    private let opticalFlow: OpticalFlowEstimator
    private var slots = [Slot](repeating: Slot(), count: MotionPipeline.slotCount)
    private var slotIndex = 0
    private let latestField = OSAllocatedUnfairLock<MotionField?>(uncheckedState: nil)

    /// Called on the processing queue each time a field has been stored, for whoever needs the
    /// measurement the moment it exists rather than when the next display callback looks for it.
    var onField: ((MotionField) -> Void)?

    init(gpu: GPUContext, queue: DispatchQueue) {
        self.gpu = gpu
        self.queue = queue
        self.mediaEngine = MediaEngineEstimator(device: gpu.device)
        self.opticalFlow = OpticalFlowEstimator(device: gpu.device)
    }

    /// The most recent field, from whichever thread asks.
    var latest: MotionField? { latestField.withLockUnchecked { $0 } }

    func reset() {
        slots = [Slot](repeating: Slot(), count: Self.slotCount)
        slotIndex = 0
        latestField.withLockUnchecked { $0 = nil }
        mediaEngine.reset()
        opticalFlow.reset()
    }

    /// Starts measuring the motion that brought the capture to where it is. The field lands later,
    /// from the estimator's own thread, and is picked up through `latest`.
    func submit(frame: MTLTexture, capture: CVPixelBuffer, source: MotionSource, timestamp: CFTimeInterval) {
        switch source {
        case .mediaEngine:
            submitToMediaEngine(frame, timestamp: timestamp)
        case .opticalFlow:
            submitToOpticalFlow(capture, frameWidth: frame.width, timestamp: timestamp)
        }
    }

    // MARK: Media engine

    /// Converts the frame to luma, hands it to the media engine, and copies the resulting vectors
    /// into a slot we own — VideoToolbox recycles its own buffers, and a field is read long after
    /// its frame left the estimator.
    private func submitToMediaEngine(_ frame: MTLTexture, timestamp: CFTimeInterval) {
        guard let prepared = mediaEngine.prepare(width: frame.width, height: frame.height),
              let commandBuffer = gpu.makeCommandBuffer("MetalGoose motion luma"),
              let encoder = commandBuffer.makeComputeCommandEncoder() else { return }

        encoder.setComputePipelineState(gpu.pipelines.luma)
        encoder.setTexture(frame, index: 0)
        encoder.setTexture(prepared.texture, index: 1)
        gpu.dispatch(gpu.pipelines.luma, on: encoder, width: frame.width, height: frame.height)
        encoder.endEncoding()

        // VideoToolbox reads the IOSurface outside Metal's ordering, so the estimate has to follow
        // the write — but nothing waits for it. It is submitted from the completion handler and the
        // vectors are picked up by a later frame, so a slow or silent media engine costs a missing
        // update rather than a stalled capture queue.
        if let pending = prepared.pending {
            nonisolated(unsafe) let pending = pending
            let width = frame.width
            commandBuffer.addCompletedHandler { [self] _ in
                estimate(pending, frameWidth: width, timestamp: timestamp)
            }
        }
        commandBuffer.commit()
    }

    /// Estimator state belongs to the processing queue; the completion handler that gets here runs
    /// on a Metal thread, and the vectors arrive on VideoToolbox's.
    private func estimate(_ pending: MediaEngineEstimator.PendingPair, frameWidth: Int, timestamp: CFTimeInterval) {
        nonisolated(unsafe) let pending = pending
        queue.async { [self] in
            mediaEngine.estimate(pending) { vectors in
                nonisolated(unsafe) let vectors = vectors
                self.queue.async {
                    guard let field = self.mediaEngine.texture(for: vectors) else { return }
                    self.store(field, scale: 1, frameWidth: frameWidth, timestamp: timestamp)
                }
            }
        }
    }

    // MARK: Optical flow

    /// Vision reads the captured image directly, so no luma pass and no GPU round trip — but the
    /// field it returns is in captured pixels, while the warp runs on the history frame, which
    /// render scale may have restored to a larger size. The ratio rides in with the sign, which is
    /// the other correction: Vision reports where content went, the warp reads where it came from.
    private func submitToOpticalFlow(_ capture: CVPixelBuffer, frameWidth: Int, timestamp: CFTimeInterval) {
        let ratio = Float(frameWidth) / Float(max(1, CVPixelBufferGetWidth(capture)))
        nonisolated(unsafe) let capture = capture
        opticalFlow.submit(capture) { [self] field in
            nonisolated(unsafe) let field = field
            queue.async {
                self.store(field, scale: -ratio, frameWidth: frameWidth, timestamp: timestamp)
            }
        }
    }

    // MARK: Field post-processing

    private func ensureGlobalBuffer(_ buffer: inout MTLBuffer?) -> MTLBuffer? {
        if buffer == nil {
            buffer = gpu.device.makeBuffer(length: MemoryLayout<SIMD2<Float>>.stride, options: .storageModePrivate)
        }
        return buffer
    }

    /// Copies the estimator's output into a slot we own — VideoToolbox recycles its buffers while
    /// the warp still reads the field — and derives the maps the warp needs alongside it.
    ///
    /// `scale` carries both corrections the incoming field needs: the sign that reconciles the
    /// source's convention with the warp's, and the ratio that carries vectors measured on the
    /// captured image into the units of the frame the warp runs on.
    private func store(_ field: MTLTexture, scale: Float, frameWidth: Int, timestamp: CFTimeInterval) {
        let index = slotIndex % slots.count
        slotIndex += 1
        guard let raw = gpu.ensureTexture(&slots[index].raw, width: field.width, height: field.height,
                                          pixelFormat: .rg16Float),
              let vectors = gpu.ensureTexture(&slots[index].vectors, width: field.width, height: field.height,
                                              pixelFormat: .rg16Float),
              let disagreement = gpu.ensureTexture(&slots[index].disagreement, width: field.width, height: field.height,
                                                   pixelFormat: .r16Float),
              let global = ensureGlobalBuffer(&slots[index].global),
              let commandBuffer = gpu.makeCommandBuffer("MetalGoose motion field"),
              let encoder = commandBuffer.makeComputeCommandEncoder() else { return }

        // A compute copy rather than a blit, because the field has to be scaled on the way in and a
        // blit cannot touch the values it moves.
        var scaleValue = scale
        encoder.setComputePipelineState(gpu.pipelines.copyMotion)
        encoder.setTexture(field, index: 0)
        encoder.setTexture(raw, index: 1)
        encoder.setBytes(&scaleValue, length: MemoryLayout<Float>.size, index: 0)
        gpu.dispatch(gpu.pipelines.copyMotion, on: encoder, width: field.width, height: field.height)

        // What the frame as a whole is doing is measured on the raw field — it is a trimmed mean, so a
        // few wild blocks do not move it — and is what the wild blocks are judged against.
        encoder.setComputePipelineState(gpu.pipelines.globalMotion)
        encoder.setTexture(raw, index: 0)
        encoder.setBuffer(global, offset: 0, index: 0)
        encoder.dispatchThreadgroups(MTLSize(width: 1, height: 1, depth: 1),
                                     threadsPerThreadgroup: MTLSize(width: Int(MG_MOTION_GRID * MG_MOTION_GRID),
                                                                    height: 1, depth: 1))

        var width = Float(frameWidth)
        encoder.setComputePipelineState(gpu.pipelines.despeckle)
        encoder.setTexture(raw, index: 0)
        encoder.setTexture(vectors, index: 1)
        encoder.setBuffer(global, offset: 0, index: 0)
        encoder.setBytes(&width, length: MemoryLayout<Float>.size, index: 1)
        gpu.dispatch(gpu.pipelines.despeckle, on: encoder, width: field.width, height: field.height)

        // The media engine's field is one vector per 16x16 block, so one texel of it already spans a
        // block. A dense field needs to step that same distance in frame pixels to be measuring the
        // same quantity.
        var stride = Int32(max(1, (field.width * Self.blockSpan) / max(1, frameWidth)))
        encoder.setComputePipelineState(gpu.pipelines.disagreement)
        encoder.setTexture(vectors, index: 0)
        encoder.setTexture(disagreement, index: 1)
        encoder.setBytes(&stride, length: MemoryLayout<Int32>.size, index: 0)
        gpu.dispatch(gpu.pipelines.disagreement, on: encoder, width: field.width, height: field.height)
        encoder.endEncoding()
        commandBuffer.commit()

        let stored = MotionField(vectors: vectors, disagreement: disagreement, global: global, timestamp: timestamp)
        latestField.withLockUnchecked { $0 = stored }
        onField?(stored)
    }
}
