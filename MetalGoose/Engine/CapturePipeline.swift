import Foundation
@preconcurrency import Metal
@preconcurrency import MetalFX
@preconcurrency import IOSurface
@preconcurrency import CoreVideo
import QuartzCore
import os

/// Everything a captured frame goes through before it can be shown: restoring render scale,
/// sharpening, anti-aliasing, and what the engine making the images between captures needs of it — its conversion to
/// 4:2:0 for the Neural Engine, or the motion between it and the capture before for MetalFX.
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

    /// Where the processing queue is, and since when. When frames pile up behind it for seconds, `receive` says where it
    /// is stuck: a pipeline that stops taking frames looks the same from the HUD whatever it is waiting for.
    private struct Progress {
        var stage = "idle"
        var since = CACurrentMediaTime()
        var reported = false
    }
    private let progress = OSAllocatedUnfairLock(initialState: Progress())
    private static let stallReport: CFTimeInterval = 3

    private func enter(_ stage: String) {
        progress.withLock { $0 = Progress(stage: stage, since: CACurrentMediaTime(), reported: false) }
    }

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
    private var historyIndex = 0
    private var restoreTexture: MTLTexture?
    private var restoreScaler: MTLFXSpatialScaler?
    private var cropTexture: MTLTexture?
    private var sharpenTexture: MTLTexture?
    private var smaaEdges: MTLTexture?
    private var smaaWeights: MTLTexture?
    private var processedSize: CGSize = .zero
    private var generating = false

    /// Which engine makes the images between captures, and how many. The choice has memory: a rate near the limit of
    /// what an engine needs does not flip between two of them.
    private var selector = GenerationSelector()

    /// What the previous capture was given to, so that the engine the work goes to next can be handed over to without a
    /// pair going by: the one that made the last pair finishes it, and the one that takes over takes this capture in.
    private var lastChoice = GenerationChoice.nothing

    /// The capture the media engine was last given, so that the motion of a pair is only measured between two captures that
    /// followed one another: after a stretch in which none was given — the Neural Engine making the images — the reference
    /// is old.
    private var lastMotionTimestamp: CFTimeInterval?

    /// The sizes the Neural Engine can work at for the frames this pipeline holds, finest first.
    private var neuralSizes: [NeuralSizes.Size] = []
    private var neuralSizesFor = CGSize.zero

    /// ScreenCaptureKit recycles a small pool of surfaces, so the same few come back frame after
    /// frame. Wrapping each in a texture again for every capture allocated for surfaces that were
    /// already wrapped.
    private var surfaceTextures: [IOSurfaceID: MTLTexture] = [:]
    private static let surfaceCacheLimit = 16

    // MARK: Timing (processing queue only)

    private var lastArrival: CFTimeInterval = 0
    /// The last capture's presentation time, and how long after it the capture arrived (`presentationTime(of:arrival:)`).
    private var lastPresentation: CFTimeInterval = 0
    private var lastDelivery: CFTimeInterval = 0
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
            let stuck = progress.withLock { state -> (String, CFTimeInterval)? in
                let waited = CACurrentMediaTime() - state.since
                guard presenting, state.stage != "idle", !state.reported, waited > Self.stallReport else { return nil }
                state.reported = true
                return (state.stage, waited)
            }
            if let (stage, waited) = stuck {
                NSLog("MetalGoose: the capture queue has been %@ for %.1f s and frames are being dropped", stage, waited)
            }
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
            enter("idle")
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

    /// Drops every texture and estimator the pipeline holds, and the frames and counters that
    /// describe what was captured with them.
    func reset() {
        queue.async { [self] in resetState() }
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

    private func resetState() {
        restoreTexture = nil
        restoreScaler = nil
        cropTexture = nil
        sharpenTexture = nil
        smaaEdges = nil
        smaaWeights = nil
        historyTextures = [MTLTexture?](repeating: nil, count: Self.poolDepth)
        historyIndex = 0
        surfaceTextures.removeAll()
        processedSize = .zero
        motion.reset()
        shared.neural.reset()
        shared.metalFX.reset()
        shared.images.reset()
        shared.neuralLatency.reset()
        shared.metalFXLatency.reset()
        shared.renderResetRequested.withLock { $0 = true }
        shared.ring.clear()
        shared.stats.withLock { $0.resetCumulativeCounters() }
        selector.reset()
        lastChoice = .nothing
        lastMotionTimestamp = nil
        neuralSizes = []
        neuralSizesFor = .zero
        shared.generation.withLock { $0 = .nothing }
    }

    // MARK: - One frame

    private func process(_ frame: CapturedFrame) {
        enter("waiting for the GPU to finish earlier frames")
        inFlight.wait()
        enter("encoding a capture")
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
        let generates = config.generatesFrames
        if generates != generating {
            generating = generates
            if !generates { stopGeneration() }
        }

        // ScreenCaptureKit delivers the render resolution directly and the spatial upscale runs once
        // per presented frame, so the capture path only sharpens and anti-aliases before the frame
        // enters the ring, at the reduced size. Generation works on the capture as it is: bringing it
        // back to the window's size first was measured to add nothing for MetalFX, and costs a MetalFX
        // pass per capture and every later stage at four times the pixels. Only the Neural Engine takes
        // the window's own size, when that fits it (`CaptureRestore`).
        //
        // Sharpening the capture before MetalFX, rather than MetalFX's output, was measured (luma PSNR
        // against the original, 24 photos and 4 drawn interfaces reduced by area and by point and brought
        // back by MetalFX, at each Sharpening strength): 0.07 to 0.21 dB closer on interface and text, and
        // up to 0.21 dB further on photos. It is also one pass at the capture's size a capture, where after
        // MetalFX it would be one at the screen's size for every image presented.
        let native = frame.nativePixelSize
        let isReduced = native.width >= CGFloat(input.width) + 1 && native.height >= CGFloat(input.height) + 1
        // What the Neural Engine is given whole is even in both dimensions, because its chroma planes are half-size. A
        // window or a capture that is odd in one is taken one pixel short: a restored frame is made that much smaller, and
        // a capture is cropped by it where what is left fits. A frame that is larger than the Neural Engine takes is not
        // cropped: it is shrunk to what it takes, whatever its parity.
        let windowWidth = (isReduced ? Int(native.width) : input.width) & ~1
        let windowHeight = (isReduced ? Int(native.height) : input.height) & ~1
        let neuralEngineFitsWindow = generates && NeuralInterpolator.supports(width: windowWidth, height: windowHeight)
        let restores = CaptureRestore.isRestored(upscaling: config.upscaling, generating: generates, isReduced: isReduced,
                                                 neuralEngineFitsWindow: neuralEngineFitsWindow)
        let evenWidth = input.width & ~1
        let evenHeight = input.height & ~1
        let crops = !restores && generates && (evenWidth != input.width || evenHeight != input.height)
            && NeuralInterpolator.supports(width: evenWidth, height: evenHeight)
        let width = restores ? windowWidth : (crops ? evenWidth : input.width)
        let height = restores ? windowHeight : (crops ? evenHeight : input.height)

        let size = CGSize(width: width, height: height)
        if size != processedSize {
            resetState()
            processedSize = size
        }
        if neuralSizesFor != size {
            neuralSizes = NeuralInterpolator.ladder(frameWidth: width, frameHeight: height)
            neuralSizesFor = size
        }

        // Which engine makes this frame's pair, from what the settings, the size and the engines' own times say, and what
        // each engine is to be given of this capture: the engine that makes the pair, and the one that made the pair
        // before it, which finishes it as the new one takes this capture in.
        let choice = generates ? chooseGeneration(config: config, width: width, height: height, now: now) : GenerationChoice.nothing
        let outgoing = lastChoice.engine
        let handsOver = outgoing != nil && outgoing != choice.engine
        let neuralSteps = choice.engine == .neuralEngine ? choice.multiplier : lastChoice.multiplier
        lastChoice = choice
        let runsNeuralEngine = choice.engine == .neuralEngine || (handsOver && outgoing == .neuralEngine)
        let runsMetalFX = choice.engine == .metalFX || (handsOver && outgoing == .metalFX)

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

        if crops {
            guard let cropped = gpu.ensureTexture(&cropTexture, width: width, height: height, usage: targetUsage),
                  let blit = commandBuffer.makeBlitCommandEncoder() else {
                drop(commandBuffer)
                return
            }
            blit.copy(from: input, sourceSlice: 0, sourceLevel: 0, sourceOrigin: MTLOrigin(x: 0, y: 0, z: 0),
                      sourceSize: MTLSize(width: width, height: height, depth: 1),
                      to: cropped, destinationSlice: 0, destinationLevel: 0, destinationOrigin: MTLOrigin(x: 0, y: 0, z: 0))
            blit.endEncoding()
            working = cropped
        }

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

        // The Neural Engine takes the capture in 4:2:0, converted here and handed over once the GPU has written it. It is
        // paired with the capture before it, which is the one the render clock will bracket it with.
        let previousCapture = shared.ring.snapshot().last
        enter("handing the capture to the Neural Engine")
        let yuv = runsNeuralEngine ? shared.neural.encodeConversion(of: history, commandBuffer: commandBuffer) : nil
        let partner = frame.isSceneCut ? nil : previousCapture?.timestamp

        // How far behind real time the render clock runs for this pair, which tells the Neural Engine how long a pair may wait
        // for it and still be shown.
        let scheduleDelay = FramePlanner.interpolationDelay(captureInterval: shared.captureInterval.withLock { $0.value },
                                                            generationLatency: choice.latency, steps: neuralSteps)

        commandBuffer.addCompletedHandler { [shared] buffer in
            let gpuTime = Float((buffer.gpuEndTime - buffer.gpuStartTime) * 1000)
            shared.stats.withLock { $0.captureGPUTime = gpuTime }
            if let yuv { shared.neural.frameConverted(yuv, timestamp: now, previous: partner, steps: neuralSteps, delay: scheduleDelay) }
        }
        commandBuffer.commit()

        // MetalFX takes the motion between the pair as an input; the Neural Engine needs none. The motion of a pair is
        // measured between two captures that followed one another, so after a stretch in which the media engine was given
        // none the next capture starts a pair, rather than being compared with one from long ago.
        if runsMetalFX {
            if lastMotionTimestamp != previousCapture?.timestamp { motion.breakSequence() }
            lastMotionTimestamp = now
            motion.submit(frame: history, timestamp: now, interval: shared.captureInterval.withLock { $0.value })
        }

        shared.ring.push(FrameHistory(texture: history, timestamp: now, presentationTime: presentationTime(of: frame, arrival: now),
                                      isSceneCut: frame.isSceneCut))
    }

    /// When the compositor showed `frame`: ScreenCaptureKit's time for it, where that is one — not in the future, not from
    /// long ago, after the last; otherwise its arrival less how late the last frame that had one arrived. A frame whose time
    /// is off keeps the timeline going rather than moving it to a clock of its own.
    private func presentationTime(of frame: CapturedFrame, arrival: CFTimeInterval) -> CFTimeInterval {
        let stamped = frame.captureTime
        let time: CFTimeInterval
        if stamped > lastPresentation, stamped <= arrival, arrival - stamped < Self.longestDelivery {
            lastDelivery = arrival - stamped
            time = stamped
        } else {
            time = max(arrival - lastDelivery, lastPresentation + Self.leastPresentationStep)
        }
        lastPresentation = time
        return time
    }

    /// A presentation time further back than this is not the frame's.
    private static let longestDelivery: CFTimeInterval = 0.25
    private static let leastPresentationStep: CFTimeInterval = 0.0001

    /// Frame generation was switched off: what the engines hold is released, and nothing is made until it is on again.
    private func stopGeneration() {
        motion.reset()
        shared.neural.reset()
        shared.metalFX.reset()
        shared.images.reset()
        shared.neuralLatency.reset()
        shared.metalFXLatency.reset()
        selector.reset()
        lastChoice = .nothing
        lastMotionTimestamp = nil
        shared.generation.withLock { $0 = .nothing }
    }

    /// A motion field has just been stored. It is the last thing the pair it belongs to was waiting for, so the pair is fed
    /// to MetalFX now rather than when a display callback asks. Also when MetalFX has just been left for the Neural Engine:
    /// the pair it was still on is finished.
    private func fieldMeasured(_ field: MotionField) {
        guard shared.config.withLock({ $0.generatesFrames }) else { return }

        let frames = shared.ring.snapshot()
        guard let index = frames.lastIndex(where: { $0.timestamp == field.timestamp }), index > 0 else { return }

        // Nothing is generated across a cut. The pair is simply not fed, which the interpolator takes in
        // its stride as any other gap.
        let next = frames[index]
        guard !next.isSceneCut else { return }
        shared.metalFX.feed(previous: frames[index - 1], next: next, field: field)
    }

    /// Which engine makes this frame's pair and how many images it carries, published for the render thread, which
    /// plans with it. The Neural Engine is asked for the size the choice wants beside the one that is serving, which goes
    /// on until the new one has started.
    private func chooseGeneration(config: EngineConfig, width: Int, height: Int, now: CFTimeInterval) -> GenerationChoice {
        let usable = !neuralSizes.isEmpty && !shared.neural.hasFailed
        let inputs = GenerationSelector.Inputs(
            requested: config.multiplier,
            captureInterval: shared.captureInterval.withLock { $0.value },
            shortestInterval: shared.captureSpread.withLock { $0.shortest },
            refreshRate: shared.stats.withLock { $0.screenRefreshRate },
            framePixels: width * height,
            // A size the processor refused is as good as one with no time to spare.
            neuralRungs: usable ? neuralSizes.map { shared.neural.isRejected($0) ? Int.max / 8 : $0.pixels } : [],
            neuralActive: usable ? shared.neural.activeSize.flatMap { neuralSizes.firstIndex(of: $0) } : nil,
            neuralMidpointTime: shared.neural.midpointTime, neuralQuartersTime: shared.neural.quartersTime,
            neuralLatency: shared.neuralLatency.value, metalFXLatency: shared.metalFXLatency.value)

        var choice = selector.choose(inputs, now: now)
        if let rung = choice.neuralRung, rung < neuralSizes.count { shared.neural.prepare(neuralSizes[rung]) }
        // Where the Neural Engine works at a size other than the frames' own, the size is reported.
        if choice.engine == .neuralEngine, let size = shared.neural.activeSize, size.width != width || size.height != height {
            choice.neuralSize = size
        }
        let published = choice
        shared.generation.withLock { $0 = published }
        return choice
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

    // MARK: - Statistics

    private func recordArrival(now: CFTimeInterval, captureTime: CFTimeInterval) {
        fpsWindowFrames += 1

        if lastArrival > 0 {
            let interval = now - lastArrival
            shared.captureInterval.withLock { $0.add(interval, window: EngineShared.measurementWindow) }
        }
        shared.captureSpread.withLock { $0.add(arrival: now) }
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
