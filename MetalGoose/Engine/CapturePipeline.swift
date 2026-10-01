import Foundation
@preconcurrency import Metal
@preconcurrency import MetalFX
@preconcurrency import IOSurface
@preconcurrency import CoreVideo
import QuartzCore
import os

/// Everything a captured frame goes through before it can be shown: restoring render scale,
/// sharpening, anti-aliasing, and — for extrapolation — the motion it will be warped along.
///
/// Runs on its own serial queue, which owns every texture and counter in here. The only things
/// shared with the render thread are the ring the finished frames go into and the handful of
/// locked values in `EngineShared`.
final class CapturePipeline: @unchecked Sendable {

    let queue = DispatchQueue(label: "com.metalgoose.processing", qos: .userInteractive)
    let motion: MotionPipeline

    private let shared: EngineShared
    private let gpu: GPUContext

    // MARK: Back-pressure

    /// The newest frame waiting for the queue. A frame that arrives while another is still waiting
    /// replaces it, so the pipeline always works on the newest capture; the replaced frame is counted
    /// as dropped instead of being shown late.
    private let mailbox = OSAllocatedUnfairLock<CapturedFrame?>(initialState: nil)

    /// Never replaced. A completion handler resolves this when the GPU finishes, so swapping the
    /// object mid-flight would signal a semaphore the frame never waited on. Shallower pipelines
    /// park permits instead.
    private let inFlight = DispatchSemaphore(value: GooseEngine.maxInFlight)
    private var parkedPermits = 0

    // MARK: Textures (processing queue only)

    /// A slot must not be rewritten while the ring still hands it out, so the pool holds everything
    /// the ring can reference plus everything the pipeline can have in flight behind it.
    private static let poolDepth = FrameRing.capacity + GooseEngine.maxInFlight
    private var historyTextures = [MTLTexture?](repeating: nil, count: CapturePipeline.poolDepth)
    private var maskTextures = [MTLTexture?](repeating: nil, count: CapturePipeline.poolDepth)
    private var historyIndex = 0
    private var restoreTexture: MTLTexture?
    private var restoreScaler: MTLFXSpatialScaler?
    private var sharpenTexture: MTLTexture?
    private var smaaEdges: MTLTexture?
    private var smaaWeights: MTLTexture?
    private var processedSize: CGSize = .zero

    /// ScreenCaptureKit recycles a small pool of surfaces, so the same few come back frame after
    /// frame. Wrapping each in a texture again for every capture allocated for surfaces that were
    /// already wrapped.
    private var surfaceTextures: [IOSurfaceID: MTLTexture] = [:]
    private static let surfaceCacheLimit = 16

    // MARK: Timing (processing queue only)

    private var lastArrival: CFTimeInterval = 0
    private var fpsWindowStart: CFTimeInterval = 0
    private var fpsWindowFrames = 0
    private var lastResourceSample: CFTimeInterval = 0
    private var lastCPUNanos: UInt64 = 0

    init(shared: EngineShared) {
        self.shared = shared
        self.gpu = shared.gpu
        self.motion = MotionPipeline(gpu: shared.gpu, queue: queue)
        motion.onField = { [weak self] field in self?.fieldMeasured(field) }
    }

    // MARK: - Entry points

    /// Hands a captured frame to the pipeline, from whichever thread it arrived on.
    func receive(_ frame: CapturedFrame) {
        let presenting = shared.isPresenting.withLock { $0 }
        let superseded = mailbox.withLock { slot -> Bool in
            defer { slot = frame }
            return slot != nil
        }
        if superseded {
            // Frames replaced while the overlay is hidden were never going to be shown.
            if presenting { shared.stats.withLock { $0.droppedFrames += 1 } }
        } else if presenting {
            queue.async { [self] in drain() }
        }
    }

    private func drain() {
        while shared.isPresenting.withLock({ $0 }),
              let frame = mailbox.withLock({ slot -> CapturedFrame? in
                  defer { slot = nil }
                  return slot
              }) {
            process(frame)
        }
    }

    /// The overlay was hidden or shown. While it is hidden nothing can be seen, so the newest frame is
    /// held and not processed.
    ///
    /// ScreenCaptureKit delivers a frame only when the window changes, so the held frame is processed on
    /// return. Frames captured before the pause are dropped: they are not neighbours of the ones after it.
    func setPresenting(_ presenting: Bool) {
        shared.isPresenting.withLock { $0 = presenting }
        queue.async { [self] in
            if presenting {
                drain()
            } else {
                shared.ring.clear()
                motion.reset()
                lastArrival = 0
            }
        }
    }

    /// Emulates a shallower pipeline by parking permits on the queue rather than replacing the
    /// semaphore out from under in-flight frames.
    func applyBufferDepth(_ depth: Int) {
        let wanted = GooseEngine.maxInFlight - max(2, min(GooseEngine.maxInFlight, depth))
        queue.async { [self] in
            while parkedPermits < wanted {
                inFlight.wait()
                parkedPermits += 1
            }
            while parkedPermits > wanted {
                inFlight.signal()
                parkedPermits -= 1
            }
        }
    }

    /// Drops every texture and estimator the pipeline holds, and optionally the frames and
    /// counters that describe what was captured with them.
    func reset(clearFrames: Bool) {
        queue.async { [self] in resetState(clearFrames: clearFrames) }
    }

    /// The timing scalars belong to the queue, so their reset is handed over like everything else.
    func resetTiming() {
        queue.async { [self] in
            lastArrival = 0
            fpsWindowStart = CACurrentMediaTime()
            fpsWindowFrames = 0
            lastResourceSample = 0
            lastCPUNanos = 0
        }
    }

    private func resetState(clearFrames: Bool) {
        restoreTexture = nil
        restoreScaler = nil
        sharpenTexture = nil
        smaaEdges = nil
        smaaWeights = nil
        historyTextures = [MTLTexture?](repeating: nil, count: Self.poolDepth)
        maskTextures = [MTLTexture?](repeating: nil, count: Self.poolDepth)
        historyIndex = 0
        surfaceTextures.removeAll()
        processedSize = .zero
        motion.reset()
        shared.neural.reset()
        shared.metalFX.reset()
        shared.generationLatency.reset()
        shared.renderResetRequested.withLock { $0 = true }

        if clearFrames {
            shared.ring.clear()
            shared.stats.withLock { $0.resetCumulativeCounters() }
        }
    }

    // MARK: - One frame

    private func process(_ frame: CapturedFrame) {
        inFlight.wait()
        let now = CACurrentMediaTime()
        recordArrival(now: now, captureTime: frame.captureTime)

        guard let commandBuffer = gpu.makeCommandBuffer("MetalGoose capture") else {
            inFlight.signal()
            return
        }

        // Keeps ScreenCaptureKit's buffer out of its pool until the GPU has finished reading it,
        // and returns the in-flight permit whichever way this frame ends.
        nonisolated(unsafe) let retained = frame.pixelBuffer
        commandBuffer.addCompletedHandler { [inFlight] _ in
            _ = retained
            inFlight.signal()
        }

        guard let input = surfaceTexture(for: frame.surface) else {
            shared.errors.report(.surfaceTextureFailed)
            drop(commandBuffer)
            return
        }

        let config = shared.config.withLock { $0 }

        // ScreenCaptureKit delivers the render resolution directly and the spatial upscale runs once
        // per presented frame, so the capture path only sharpens and anti-aliases before the frame
        // enters the ring. Render scale reduces the capture, but frame generation must not inherit
        // that reduction: interpolation and the motion field would then work on a fraction of the
        // pixels and smear. Bring the frame back to the window's native size first.
        //
        // Frame generation is the only reason this pass exists, so it is also its only condition.
        // Without it the restore would be a second full MetalFX pass per captured frame, with the
        // sharpening and anti-aliasing running at native size instead of the reduced one — the
        // opposite of what Render Scale is for. The presentation step covers the whole gap from the
        // capture to the drawable on its own.
        let native = frame.nativePixelSize
        let restores = config.upscaling && config.generatesFrames
            && native.width >= CGFloat(input.width) + 1 && native.height >= CGFloat(input.height) + 1
        let width = restores ? Int(native.width) : input.width
        let height = restores ? Int(native.height) : input.height

        let size = CGSize(width: width, height: height)
        if size != processedSize {
            resetState(clearFrames: true)
            processedSize = size
        }

        // Every scratch texture is reused next frame, so the LAST active stage writes straight into
        // the ring slot; a passthrough configuration needs a copy.
        let slot = historyIndex % Self.poolDepth
        historyIndex += 1
        let targetUsage: MTLTextureUsage = [.shaderRead, .shaderWrite, .renderTarget]
        guard let history = gpu.ensureTexture(&historyTextures[slot], width: width, height: height, usage: targetUsage) else {
            drop(commandBuffer)
            return
        }

        let sharpens = config.sharpness > 0.01
        let antiAliases = config.antiAliasing != .off
        var working = input

        if restores {
            let destination: MTLTexture?
            if sharpens || antiAliases {
                destination = gpu.ensureTexture(&restoreTexture, width: width, height: height, usage: targetUsage)
            } else {
                destination = history
            }
            guard let destination,
                  let scaler = gpu.ensureSpatialScaler(&restoreScaler, inputWidth: input.width, inputHeight: input.height,
                                                       outputWidth: width, outputHeight: height) else {
                shared.errors.report(.spatialScalerFailed)
                drop(commandBuffer)
                return
            }
            scaler.colorTexture = input
            scaler.outputTexture = destination
            scaler.encode(commandBuffer: commandBuffer)
            working = destination
        }

        if sharpens {
            let destination: MTLTexture?
            if antiAliases {
                destination = gpu.ensureTexture(&sharpenTexture, width: width, height: height)
            } else {
                destination = history
            }
            guard let destination, encodeSharpen(working, to: destination, strength: config.sharpness,
                                                 commandBuffer: commandBuffer) else {
                shared.errors.report(.sharpeningUnavailable)
                drop(commandBuffer)
                return
            }
            working = destination
        }

        if antiAliases {
            guard encodeAntiAliasing(working, to: history, config: config, commandBuffer: commandBuffer) else {
                drop(commandBuffer)
                return
            }
        } else if working !== history {
            guard let blit = commandBuffer.makeBlitCommandEncoder() else {
                drop(commandBuffer)
                return
            }
            blit.copy(from: working, to: history)
            blit.endEncoding()
        }

        // The mask compares this capture with the one before it, so the first capture of a
        // session has none and cannot be warped from.
        var mask: MTLTexture?
        if config.frameGeneration == .extrapolation,
           let previous = shared.ring.snapshot().last,
           previous.texture.width == width, previous.texture.height == height {
            mask = encodeStaticMask(history, previous: previous.texture, slot: slot, commandBuffer: commandBuffer)
        }

        // The Neural Engine takes the capture in 4:2:0, converted here and handed over once the GPU has
        // written it. A size it cannot take, or a processor that failed, falls back to MetalFX.
        let backend = interpolationBackend(for: config, width: width, height: height)
        let backendChanged = shared.interpolationBackend.withLock { current -> Bool in
            defer { current = backend }
            return current != backend
        }
        // The two engines take different times to make an image, so one's measurement says nothing about
        // the other's.
        if backendChanged { shared.generationLatency.reset() }
        let yuv = backend == .neuralEngine
            ? shared.neural.encodeConversion(of: history, commandBuffer: commandBuffer) : nil

        commandBuffer.addCompletedHandler { [shared] buffer in
            let gpuTime = Float((buffer.gpuEndTime - buffer.gpuStartTime) * 1000)
            shared.stats.withLock { $0.captureGPUTime = gpuTime }
            if let yuv { shared.neural.frameConverted(yuv, timestamp: now) }
        }
        commandBuffer.commit()

        // MetalFX takes the motion between the pair as an input, and extrapolation warps along it; the
        // Neural Engine needs none. Only extrapolation lets the user pick how it is measured.
        if config.frameGeneration == .extrapolation
            || (config.frameGeneration == .interpolation && backend == .metalFX) {
            let source = config.frameGeneration == .extrapolation ? config.motionSource : .mediaEngine
            motion.submit(frame: history, capture: frame.pixelBuffer, source: source, timestamp: now)
        }

        shared.ring.push(FrameHistory(texture: history, timestamp: now, isSceneCut: frame.isSceneCut, staticMask: mask))
    }

    /// A motion field has just been stored. When MetalFX is interpolating, it is the last thing the pair
    /// it belongs to was waiting for, so the pair is fed now rather than when a display callback asks.
    private func fieldMeasured(_ field: MotionField) {
        guard shared.config.withLock({ $0.frameGeneration }) == .interpolation,
              shared.interpolationBackend.withLock({ $0 }) == .metalFX else { return }

        let frames = shared.ring.snapshot()
        guard let index = frames.lastIndex(where: { $0.timestamp == field.timestamp }), index > 0 else { return }

        // Nothing is generated across a cut. The pair is simply not fed, which the interpolator takes in
        // its stride as any other gap.
        let next = frames[index]
        guard !next.isSceneCut else { return }
        shared.metalFX.feed(previous: frames[index - 1], next: next, field: field)
    }

    /// Which engine interpolates this frame's pair. The setting is a preference: the Neural Engine is
    /// used only while it can take frames of this size and has not failed, and MetalFX is what is left.
    private func interpolationBackend(for config: EngineConfig, width: Int, height: Int) -> InterpolationEngine {
        guard config.frameGeneration == .interpolation, config.interpolationEngine == .neuralEngine,
              NeuralInterpolator.supports(width: width, height: height), !shared.neural.hasFailed else {
            return .metalFX
        }
        return .neuralEngine
    }

    /// The single way a captured frame is abandoned. Going through one function makes the accounting
    /// uniform by construction: the HUD's Dropped row must read non-zero through exactly the failures
    /// someone would be looking at it to diagnose — a scaler that will not rebuild drops every frame.
    private func drop(_ commandBuffer: MTLCommandBuffer) {
        shared.stats.withLock { $0.droppedFrames += 1 }
        commandBuffer.commit()
    }

    private func surfaceTexture(for surface: IOSurfaceRef) -> MTLTexture? {
        let id = IOSurfaceGetID(surface)
        if let cached = surfaceTextures[id] { return cached }

        let descriptor = MTLTextureDescriptor.texture2DDescriptor(
            pixelFormat: .bgra8Unorm, width: IOSurfaceGetWidth(surface), height: IOSurfaceGetHeight(surface),
            mipmapped: false)
        descriptor.usage = [.shaderRead]
        guard let texture = gpu.device.makeTexture(descriptor: descriptor, iosurface: surface, plane: 0) else {
            return nil
        }
        // A pool this size is not being recycled after all, and the cache is only holding surfaces.
        if surfaceTextures.count >= Self.surfaceCacheLimit { surfaceTextures.removeAll() }
        surfaceTextures[id] = texture
        return texture
    }

    // MARK: - Stages

    private func encodeSharpen(_ input: MTLTexture, to output: MTLTexture, strength: Float,
                               commandBuffer: MTLCommandBuffer) -> Bool {
        guard let encoder = commandBuffer.makeComputeCommandEncoder() else { return false }
        var params = SharpenParams(sharpness: strength)
        encoder.setComputePipelineState(gpu.pipelines.sharpen)
        encoder.setTexture(input, index: 0)
        encoder.setTexture(output, index: 1)
        encoder.setBytes(&params, length: MemoryLayout<SharpenParams>.size, index: 0)
        gpu.dispatch(gpu.pipelines.sharpen, on: encoder, width: output.width, height: output.height)
        encoder.endEncoding()
        return true
    }

    private func encodeAntiAliasing(_ input: MTLTexture, to output: MTLTexture, config: EngineConfig,
                                    commandBuffer: MTLCommandBuffer) -> Bool {
        let width = output.width
        let height = output.height

        switch config.antiAliasing {
        case .off:
            return true

        case .fxaa:
            guard let encoder = commandBuffer.makeComputeCommandEncoder() else {
                shared.errors.report(.antiAliasingUnavailable("FXAA"))
                return false
            }
            var threshold = config.profile.aaThreshold
            encoder.setComputePipelineState(gpu.pipelines.fxaa)
            encoder.setTexture(input, index: 0)
            encoder.setTexture(output, index: 1)
            encoder.setBytes(&threshold, length: MemoryLayout<Float>.size, index: 0)
            gpu.dispatch(gpu.pipelines.fxaa, on: encoder, width: width, height: height)
            encoder.endEncoding()
            return true

        case .smaa:
            // Edges carry two channels and need no more; weights carry four.
            guard let edges = gpu.ensureTexture(&smaaEdges, width: width, height: height, pixelFormat: .rg8Unorm),
                  let weights = gpu.ensureTexture(&smaaWeights, width: width, height: height),
                  let encoder = commandBuffer.makeComputeCommandEncoder() else {
                shared.errors.report(.antiAliasingUnavailable("SMAA"))
                return false
            }
            var params = AntiAliasParams(threshold: config.profile.aaThreshold,
                                         maxSearchSteps: Int32(config.profile.smaaSearchSteps))
            encoder.setComputePipelineState(gpu.pipelines.smaaEdges)
            encoder.setTexture(input, index: 0)
            encoder.setTexture(edges, index: 1)
            encoder.setBytes(&params, length: MemoryLayout<AntiAliasParams>.size, index: 0)
            gpu.dispatch(gpu.pipelines.smaaEdges, on: encoder, width: width, height: height)

            encoder.setComputePipelineState(gpu.pipelines.smaaWeights)
            encoder.setTexture(edges, index: 0)
            encoder.setTexture(weights, index: 1)
            encoder.setBytes(&params, length: MemoryLayout<AntiAliasParams>.size, index: 0)
            gpu.dispatch(gpu.pipelines.smaaWeights, on: encoder, width: width, height: height)

            encoder.setComputePipelineState(gpu.pipelines.smaaBlend)
            encoder.setTexture(input, index: 0)
            encoder.setTexture(weights, index: 1)
            encoder.setTexture(output, index: 2)
            gpu.dispatch(gpu.pipelines.smaaBlend, on: encoder, width: width, height: height)
            encoder.endEncoding()
            return true
        }
    }

    /// Where this capture and the one before it are pixel-identical. `staticMask` loads its halo
    /// cooperatively, so every threadgroup of the grid has to be full-size; the grid is rounded up
    /// and the kernel discards what falls outside the image.
    private func encodeStaticMask(_ frame: MTLTexture, previous: MTLTexture, slot: Int,
                                  commandBuffer: MTLCommandBuffer) -> MTLTexture? {
        guard let mask = gpu.ensureTexture(&maskTextures[slot], width: frame.width, height: frame.height,
                                           pixelFormat: .r8Unorm),
              let encoder = commandBuffer.makeComputeCommandEncoder() else { return nil }

        let tileWidth = Int(MG_MASK_TILE_WIDTH)
        let tileHeight = Int(MG_MASK_TILE_HEIGHT)
        encoder.setComputePipelineState(gpu.pipelines.staticMask)
        encoder.setTexture(frame, index: 0)
        encoder.setTexture(previous, index: 1)
        encoder.setTexture(mask, index: 2)
        encoder.dispatchThreadgroups(
            MTLSize(width: (frame.width + tileWidth - 1) / tileWidth,
                    height: (frame.height + tileHeight - 1) / tileHeight, depth: 1),
            threadsPerThreadgroup: MTLSize(width: tileWidth, height: tileHeight, depth: 1))
        encoder.endEncoding()
        return mask
    }

    // MARK: - Statistics

    private func recordArrival(now: CFTimeInterval, captureTime: CFTimeInterval) {
        fpsWindowFrames += 1

        if lastArrival > 0 {
            let interval = now - lastArrival
            shared.captureInterval.withLock { $0.add(interval, window: EngineShared.measurementWindow) }
        }
        let delta = lastArrival > 0 ? (now - lastArrival) * 1000 : 0
        lastArrival = now

        let elapsed = now - fpsWindowStart
        let publishesRate = elapsed >= 1.0
        let captureFPS = publishesRate ? Float(fpsWindowFrames) / Float(elapsed) : nil
        if publishesRate {
            fpsWindowFrames = 0
            fpsWindowStart = now
        }

        let resources = now - lastResourceSample >= EngineShared.measurementWindow ? sampleResources(now: now) : nil

        shared.stats.withLock { stats in
            stats.frameCount += 1
            if delta > 0 { stats.frameTime = Float(delta) }
            // `captureTime` is ScreenCaptureKit's stamp for when the compositor drew the frame; the
            // distance to now is how long the frame sat before the pipeline reached it.
            stats.captureLatency = captureTime > 0 ? Float((now - captureTime) * 1000) : Float(delta)
            if let captureFPS { stats.captureFPS = captureFPS }
            if let resources {
                stats.gpuMemoryUsed = resources.gpuUsed
                stats.gpuMemoryTotal = resources.gpuTotal
                stats.processMemoryUsed = resources.processMemory
                if let cpu = resources.cpuUsage { stats.cpuUsage = cpu }
            }
        }
    }

    private struct ResourceSample {
        let gpuUsed: UInt64
        let gpuTotal: UInt64
        let processMemory: UInt64
        let cpuUsage: Float?
    }

    /// Memory and CPU change slowly, and reading them costs a system call each — nothing to do on
    /// every one of up to 240 frames a second, so they are sampled once per measurement window.
    private func sampleResources(now: CFTimeInterval) -> ResourceSample {
        var cpu: Float?
        var usage = rusage_info_current()
        let result = withUnsafeMutablePointer(to: &usage) { pointer -> Int32 in
            pointer.withMemoryRebound(to: rusage_info_t?.self, capacity: 1) {
                proc_pid_rusage(getpid(), RUSAGE_INFO_CURRENT, $0)
            }
        }
        if result == 0 {
            let nanos = usage.ri_user_time + usage.ri_system_time
            let wall = now - lastResourceSample
            if lastResourceSample > 0, wall > 0, nanos >= lastCPUNanos {
                cpu = Float(Double(nanos - lastCPUNanos) / 1_000_000_000.0 / wall * 100.0)
            }
            lastCPUNanos = nanos
        }
        lastResourceSample = now

        return ResourceSample(gpuUsed: UInt64(gpu.device.currentAllocatedSize),
                              gpuTotal: UInt64(gpu.device.recommendedMaxWorkingSetSize),
                              processMemory: Self.processMemoryFootprint(),
                              cpuUsage: cpu)
    }

    private static func processMemoryFootprint() -> UInt64 {
        var info = task_vm_info_data_t()
        var count = mach_msg_type_number_t(MemoryLayout<task_vm_info_data_t>.size / MemoryLayout<integer_t>.size)
        let result = withUnsafeMutablePointer(to: &info) {
            $0.withMemoryRebound(to: integer_t.self, capacity: Int(count)) {
                task_info(mach_task_self_, task_flavor_t(TASK_VM_INFO), $0, &count)
            }
        }
        return result == KERN_SUCCESS ? info.phys_footprint : 0
    }
}
