import Foundation
@preconcurrency import Metal
@preconcurrency import MetalFX
import os

/// Frame interpolation on the GPU, through MetalFX.
///
/// Fed from the capture side, once per pair, as soon as the motion between the pair has been measured —
/// not from the display callback that wants the result. MetalFX builds its answer from a short history
/// of the calls it has had:
///
/// - a pair that is never fed makes the next call come back unchanged, and the one after is whole again;
/// - a history reset in the middle of a stream costs three calls, one more than at the start of one, so
///   it is never used there: a stream is begun by a new interpolator.
///
/// Feeding every pair the capture pipeline completes, once and in order, keeps the history whole
/// whatever the display is doing; the render thread only picks up what exists.
///
/// The motion is not optional. MetalFX's validation rejects a nil motion texture, and without it the
/// interpolator guesses the motion itself (an image 30 dB from the real one against 50 dB with it). A
/// pair whose motion has not arrived is not interpolated; the capture is shown instead.
///
/// Concurrency: everything but `texture(previous:next:)` runs on the capture pipeline's queue.
final class MetalFXInterpolator: @unchecked Sendable {

    private struct Output {
        let previous: CFTimeInterval
        let next: CFTimeInterval
        let texture: MTLTexture
    }

    private struct Results {
        /// Bumped on every reset, so the completion of a command buffer from before it cannot publish into
        /// what follows.
        var epoch = 0
        var outputs: [Output] = []
    }

    /// One result per pair the ring can hold: the render clock never reads further back than that.
    private static let resultCapacity = FrameRing.capacity - 1

    /// Calls that return the current frame unchanged before MetalFX interpolates: after a new
    /// interpolator's first call, and after a pair that was never fed.
    private static let warmUpAfterStart = 2
    private static let warmUpAfterGap = 1

    /// Interpolations that may be on the GPU at once. A GPU that cannot keep up — sharing it with the
    /// captured app, it sometimes cannot — would otherwise be handed a pair per capture for ever, and
    /// every image would arrive later than the last. One running and one queued behind it is already a
    /// capture interval behind; past that the pair is not fed, and the screen shows the capture.
    private static let maximumPending = 2

    private let gpu: GPUContext
    private let errors: ErrorLog
    private let latency: GenerationLatency

    // Processing queue only.
    private var interpolator: MTLFXFrameInterpolator?
    private var warmUpCalls = 0
    private var lastFedNext: CFTimeInterval?
    /// The motion field at the interpolator's resolution: one vector per pixel, expanded from the block
    /// field the media engine measures.
    private var denseMotion: MTLTexture?
    /// MetalFX rejects a nil depth texture — its validation layer asserts with "Input content width
    /// exceed input texture dimension". Captured content is a flat 2D plane, so a constant far-plane
    /// depth is the honest answer rather than a workaround. Cleared once per size change.
    private var flatDepth: MTLTexture?
    private var flatDepthIsCleared = false
    /// The images are written round-robin. One more than the results held, so the slot being written is
    /// never one the render thread can still ask for.
    private var outputs: [MTLTexture] = []
    private var outputIndex = 0

    private let results = OSAllocatedUnfairLock<Results>(uncheckedState: Results())
    private let pending = OSAllocatedUnfairLock(initialState: 0)

    init(gpu: GPUContext, errors: ErrorLog, latency: GenerationLatency) {
        self.gpu = gpu
        self.errors = errors
        self.latency = latency
    }

    /// Drops everything. The captures the interpolator held references to are gone, and a stream begun
    /// again is begun by a new interpolator.
    func reset() {
        interpolator = nil
        warmUpCalls = 0
        lastFedNext = nil
        denseMotion = nil
        flatDepth = nil
        flatDepthIsCleared = false
        outputs.removeAll()
        outputIndex = 0
        results.withLockUnchecked {
            $0.epoch &+= 1
            $0.outputs.removeAll()
        }
    }

    /// The midpoint of a pair, once it has been produced. Called from the render thread.
    func texture(previous: CFTimeInterval, next: CFTimeInterval) -> MTLTexture? {
        results.withLockUnchecked { state in
            state.outputs.last { $0.previous == previous && $0.next == next }?.texture
        }
    }

    // MARK: - Interpolating

    /// Encodes the interpolation of one pair, with the motion measured between them. The result is
    /// published when the GPU has finished it.
    func feed(previous: FrameHistory, next: FrameHistory, field: MotionField) {
        let width = next.texture.width
        let height = next.texture.height
        guard pending.withLock({ $0 < Self.maximumPending }),
              previous.texture.width == width, previous.texture.height == height,
              let interpolator = ensureInterpolator(width: width, height: height),
              let depth = ensureFlatDepth(width: width, height: height),
              let motion = gpu.ensureTexture(&denseMotion, width: width, height: height, pixelFormat: .rg16Float),
              let output = nextOutput(width: width, height: height),
              let commandBuffer = gpu.makeCommandBuffer("MetalGoose interpolation") else { return }

        if !flatDepthIsCleared {
            encodeClear(of: depth, commandBuffer: commandBuffer)
            flatDepthIsCleared = true
        }
        guard encodeExpansion(of: field, into: motion, commandBuffer: commandBuffer) else {
            commandBuffer.commit()
            return
        }

        // Continuity: this call's previous frame is meant to be the last call's current frame. When it
        // is not, a pair went by unfed, and the call after the gap is the one that comes back unchanged.
        let isFirstCall = lastFedNext == nil
        if isFirstCall {
            warmUpCalls = Self.warmUpAfterStart
        } else if lastFedNext != previous.timestamp {
            warmUpCalls = max(warmUpCalls, Self.warmUpAfterGap)
        }

        interpolator.colorTexture = next.texture
        interpolator.prevColorTexture = previous.texture
        interpolator.outputTexture = output
        interpolator.depthTexture = depth
        interpolator.motionTexture = motion
        interpolator.motionVectorScaleX = 1
        interpolator.motionVectorScaleY = 1
        interpolator.isDepthReversed = false
        interpolator.nearPlane = 0.1
        interpolator.farPlane = 1000.0
        interpolator.fieldOfView = 1.0
        interpolator.aspectRatio = Float(width) / Float(max(1, height))
        interpolator.deltaTime = Float(max(0.0001, next.timestamp - previous.timestamp))
        interpolator.shouldResetHistory = isFirstCall
        interpolator.encode(commandBuffer: commandBuffer)
        lastFedNext = next.timestamp

        // The call was made — the history cannot form without it — but what it produced is not an
        // interpolation while the history is still forming, so it is not offered as one.
        let producesImage = warmUpCalls == 0
        if !producesImage { warmUpCalls -= 1 }

        let pair = (previous: previous.timestamp, next: next.timestamp)
        let epoch = results.withLockUnchecked { $0.epoch }
        nonisolated(unsafe) let image = output
        pending.withLock { $0 += 1 }
        commandBuffer.addCompletedHandler { [results, latency, pending] _ in
            pending.withLock { $0 -= 1 }
            guard producesImage else { return }
            let published = results.withLockUnchecked { state -> Bool in
                guard state.epoch == epoch else { return false }
                state.outputs.append(Output(previous: pair.previous, next: pair.next, texture: image))
                if state.outputs.count > Self.resultCapacity { state.outputs.removeFirst() }
                return true
            }
            if published { latency.record(previous: pair.previous, next: pair.next) }
        }
        commandBuffer.commit()
    }

    /// MetalFX wants one vector per pixel, pointing to where that pixel was in the previous frame —
    /// exactly the media engine's convention — but the field is one vector per block. The same
    /// kernel that copies a field into the pipeline's own slot resamples it bilinearly to any size.
    private func encodeExpansion(of field: MotionField, into dense: MTLTexture, commandBuffer: MTLCommandBuffer) -> Bool {
        guard let encoder = commandBuffer.makeComputeCommandEncoder() else { return false }
        var scale: Float = 1
        var coverage = SIMD2<Float>(1, 1)
        encoder.setComputePipelineState(gpu.pipelines.copyMotion)
        encoder.setTexture(field.vectors, index: 0)
        encoder.setTexture(dense, index: 1)
        encoder.setBytes(&scale, length: MemoryLayout<Float>.size, index: 0)
        encoder.setBytes(&coverage, length: MemoryLayout<SIMD2<Float>>.size, index: 1)
        gpu.dispatch(gpu.pipelines.copyMotion, on: encoder, width: dense.width, height: dense.height)
        encoder.endEncoding()
        return true
    }

    // MARK: - Resources

    private func ensureInterpolator(width: Int, height: Int) -> MTLFXFrameInterpolator? {
        if let interpolator, interpolator.inputWidth == width, interpolator.inputHeight == height,
           interpolator.outputWidth == width, interpolator.outputHeight == height {
            return interpolator
        }

        let descriptor = MTLFXFrameInterpolatorDescriptor()
        descriptor.colorTextureFormat = .bgra8Unorm
        descriptor.outputTextureFormat = .bgra8Unorm
        descriptor.depthTextureFormat = .r32Float
        descriptor.motionTextureFormat = .rg16Float
        descriptor.inputWidth = width
        descriptor.inputHeight = height
        descriptor.outputWidth = width
        descriptor.outputHeight = height

        guard let created = descriptor.makeFrameInterpolator(device: gpu.device) else {
            errors.report(.interpolatorFailed)
            return nil
        }
        // A new interpolator has no history: it begins a stream, with the warm-up that comes with one.
        // Nothing the old one was fed carries over, and nothing it produced can be published.
        lastFedNext = nil
        results.withLockUnchecked {
            $0.epoch &+= 1
            $0.outputs.removeAll()
        }
        interpolator = created
        return created
    }

    private func ensureFlatDepth(width: Int, height: Int) -> MTLTexture? {
        if let flatDepth, flatDepth.width == width, flatDepth.height == height { return flatDepth }

        let descriptor = MTLTextureDescriptor.texture2DDescriptor(pixelFormat: .r32Float,
                                                                  width: width, height: height, mipmapped: false)
        descriptor.usage = [.shaderRead, .renderTarget]
        descriptor.storageMode = .private
        flatDepth = gpu.device.makeTexture(descriptor: descriptor)
        flatDepthIsCleared = false
        return flatDepth
    }

    /// A clear-only render pass, encoded ahead of the interpolator in the same command buffer.
    private func encodeClear(of texture: MTLTexture, commandBuffer: MTLCommandBuffer) {
        let pass = MTLRenderPassDescriptor()
        pass.colorAttachments[0].texture = texture
        pass.colorAttachments[0].loadAction = .clear
        pass.colorAttachments[0].storeAction = .store
        pass.colorAttachments[0].clearColor = MTLClearColor(red: 1, green: 0, blue: 0, alpha: 0)
        commandBuffer.makeRenderCommandEncoder(descriptor: pass)?.endEncoding()
    }

    private func nextOutput(width: Int, height: Int) -> MTLTexture? {
        if outputs.first.map({ $0.width != width || $0.height != height }) ?? true {
            let descriptor = MTLTextureDescriptor.texture2DDescriptor(pixelFormat: .bgra8Unorm,
                                                                      width: width, height: height, mipmapped: false)
            // MetalFX writes it, and the presentation step reads it.
            descriptor.usage = [.shaderRead, .shaderWrite, .renderTarget]
            descriptor.storageMode = .private
            let made = (0...Self.resultCapacity).compactMap { _ in gpu.device.makeTexture(descriptor: descriptor) }
            guard made.count == Self.resultCapacity + 1 else { return nil }
            outputs = made
            outputIndex = 0
        }
        defer { outputIndex += 1 }
        return outputs[outputIndex % outputs.count]
    }
}
