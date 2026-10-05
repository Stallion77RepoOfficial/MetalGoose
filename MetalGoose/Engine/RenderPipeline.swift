import Foundation
@preconcurrency import Metal
@preconcurrency import MetalFX
import QuartzCore
import os

/// Puts the right image on screen for each display callback: decides what the callback should
/// show, blends it with the captures it sits between if it was generated, scales it up to the drawable, and presents.
///
/// Runs on the render thread alone, and owns everything it keeps — the scaler, the blender and
/// their textures. It shares only what `EngineShared` locks. Interpolated
/// images are not made here: the engines make each as its pair completes, and this only picks
/// up the ones that exist.
final class RenderPipeline: @unchecked Sendable {

    private let shared: EngineShared
    private let gpu: GPUContext
    private let stability: StabilityBlender

    /// How far behind the newest capture the schedule runs, as it is allowed to change.
    private var delay = DelayGovernor()

    /// Whether the captures keep a beat of the display's refreshes, where on the refresh grid they and the display callbacks
    /// fall, and the delay that was chosen from them last (`ScheduleAlignment`).
    private var cadence = CaptureCadence()
    private var callbackPhase = GridPhase()
    private var capturePhase = GridPhase()
    private var lastNoted: CFTimeInterval = 0
    private var alignedDelay: CFTimeInterval?
    /// The time between refreshes, from the link's own target times: the panel's nominal rate is a whole number, and 59.94 Hz
    /// read as 60 would slide the grid a millisecond a second.
    private var refreshInterval = IntervalFilter()
    private var lastTarget: CFTimeInterval = 0

    private var spatialScaler: MTLFXSpatialScaler?
    private var upscaled: MTLTexture?

    /// What the screen is showing now. A callback whose image is the one already there does not
    /// draw at all: the layer keeps its contents, so presenting the same pixels again would cost a
    /// full pass over the drawable and a recomposite by the window server for nothing.
    private var lastPresented: PresentedImage?

    /// Set from any thread when the overlay changes shape or reappears: the drawable then has to
    /// be filled again even though no new image has arrived.
    private let redrawRequested = OSAllocatedUnfairLock(initialState: true)

    private let pacing = OSAllocatedUnfairLock(initialState: PacingTracker())

    private var windowStart: CFTimeInterval = 0
    private var windowPresents = 0
    private var windowGenerated = 0
    private var windowBusyStart = 0.0

    init(shared: EngineShared) {
        self.shared = shared
        self.gpu = shared.gpu
        self.stability = StabilityBlender(gpu: shared.gpu)
    }

    func requestRedraw() {
        redrawRequested.withLock { $0 = true }
    }

    /// Drops what the render side holds. The next callback starts from nothing.
    func reset() {
        spatialScaler = nil
        upscaled = nil
        stability.reset()
        delay.reset()
        cadence.reset()
        callbackPhase.reset()
        capturePhase.reset()
        lastNoted = 0
        alignedDelay = nil
        refreshInterval.reset()
        lastTarget = 0
        lastPresented = nil
        requestRedraw()
        pacing.withLock { $0.reset() }
    }

    // MARK: - One display callback

    /// What a callback ends up putting on screen.
    private struct Content {
        /// What is actually being shown, which is not always what was planned: a midpoint that does not
        /// exist yet falls back to a capture, and that capture is what the screen then holds.
        let image: PresentedImage
        /// A capture, as it is, or an image an engine made, which is shown blended with the captures it sits between.
        let picture: Picture
        /// Age is reported from this: the newest real information on screen. A generated image could only be
        /// made once the capture it ends at had arrived, so that is its age.
        let sourceTimestamp: CFTimeInterval
        let isGenerated: Bool
    }

    private enum Picture {
        case capture(MTLTexture)
        case generated(Blend)
    }

    /// An image an engine made, the two captures it sits between, where, and how far the image is trusted.
    private struct Blend {
        let image: GeneratedImages.Source
        let previous: MTLTexture
        let next: MTLTexture
        let phase: Double
        let motion: Float
    }

    /// A capture as the schedule sees it: at the compositor's time for it.
    private struct ScheduledCapture: TimedFrame {
        let timestamp: CFTimeInterval
        let isSceneCut: Bool
    }

    /// - Parameter targetTime: when the image this callback presents is to reach the screen, which is on the display's
    ///   refresh grid.
    func render(into drawable: CAMetalDrawable, displayRate: DisplayRate, targetTime: CFTimeInterval) {
        if shared.renderResetRequested.withLock({ requested -> Bool in
            defer { requested = false }
            return requested
        }) {
            reset()
        }

        let config = shared.config.withLock { $0 }
        let frames = shared.ring.snapshot()

        // The schedule is read at the callback, not at the time the link says its image will reach the
        // screen: that latency is the same for every image and drops out, where a clock that included it
        // would start every capture interval most of the way through.
        let now = CACurrentMediaTime()
        if lastTarget > 0 { refreshInterval.add(targetTime - lastTarget, window: EngineShared.measurementWindow) }
        lastTarget = targetTime
        let refreshPeriod = refreshInterval.value > 0 ? refreshInterval.value
            : displayRate.maximum > 0 ? 1 / Double(displayRate.maximum) : 0
        note(frames, now: now, targetTime: targetTime, refreshPeriod: refreshPeriod)

        // The schedule runs on the compositor's times for the captures, which are the content's own and on the display's
        // grid; their arrivals wander by how long each took to be delivered and reached.
        let timeline = frames.map { ScheduledCapture(timestamp: $0.presentationTime, isSceneCut: $0.isSceneCut) }
        let generation = config.generatesFrames ? shared.generation.withLock { $0 } : .nothing
        let plan = FramePlanner.plan(timeline, PlanningInput(
            multiplier: generation.multiplier, sampleTime: now,
            delay: scheduleDelay(for: generation, frames: frames, now: now, refreshPeriod: refreshPeriod)))

        publishRates(now: now, displayRate: displayRate)

        guard let planned = FramePlanner.image(of: plan, in: timeline) else { return }
        let redraw = redrawRequested.withLock { $0 }
        guard planned != lastPresented || redraw else { return }

        let content = realize(plan, frames: frames)

        // The planned image is not always the one that came out: a midpoint that has not been made yet —
        // the motion for the pair is still being measured — falls back to a capture. Remembering the
        // *plan* as shown would never try again once the image arrives, and presenting the fallback
        // over and over would redraw what is already on screen, so only the realised image counts. The
        // command buffer is made only now: a callback that waits for an image used to commit an empty one.
        guard content.image != lastPresented || redraw,
              let commandBuffer = gpu.makeCommandBuffer("MetalGoose present") else { return }

        // Only now, with the image known to be shown: a pass for one that is not would be spent for nothing.
        let shown: MTLTexture
        switch content.picture {
        case .capture(let texture):
            shown = texture
        case .generated(let blend):
            guard let blended = stability.blend(blend.image, previous: blend.previous, next: blend.next, phase: blend.phase,
                                                motion: blend.motion, commandBuffer: commandBuffer) else {
                commandBuffer.commit()
                return
            }
            shown = blended
        }
        encodePresent(shown, to: drawable.texture, upscaling: config.upscaling, commandBuffer: commandBuffer)

        // A redraw of the image already on screen — the overlay changed shape — is not a new generated image.
        let isNewImage = content.image != lastPresented
        lastPresented = content.image
        redrawRequested.withLock { $0 = false }
        recordPresent(content, isNewImage: isNewImage,
                      drawableSize: CGSize(width: drawable.texture.width, height: drawable.texture.height))

        let source = content.sourceTimestamp
        let capacity = max(8, Int((Double(displayRate.maximum) * EngineShared.measurementWindow).rounded()))
        drawable.addPresentedHandler { [shared, pacing] presented in
            // Zero means the system could not say when it reached the screen.
            let time = presented.presentedTime
            guard time > 0 else { return }
            pacing.withLock { $0.record(time, capacity: capacity) }
            let latency = Float((time - source) * 1000)
            shared.stats.withLock {
                $0.presentLatency = latency
                $0.endToEndLatency = $0.captureLatency + latency
            }
        }
        commandBuffer.addCompletedHandler { [shared] buffer in
            // Generation and the upscale live here, so the capture buffer alone does not represent
            // the pipeline's GPU cost.
            let renderTime = Float((buffer.gpuEndTime - buffer.gpuStartTime) * 1000)
            shared.stats.withLock { $0.gpuTime = $0.captureGPUTime + renderTime }
        }
        commandBuffer.present(drawable)
        commandBuffer.commit()
    }

    // MARK: - Realising a plan

    /// How far behind the newest capture the schedule runs: most of a capture interval, plus how long the engine takes to
    /// make the first image of a pair, eased where it falls (`DelayGovernor`). Where the captures keep a beat of the refreshes
    /// it is chosen from where the callbacks' samples land (`ScheduleAlignment`), and otherwise for the worst place they
    /// could. Nothing is held back where nothing is made.
    private func scheduleDelay(for generation: GenerationChoice, frames: [FrameHistory], now: CFTimeInterval,
                               refreshPeriod: CFTimeInterval) -> CFTimeInterval {
        guard generation.engine != nil else {
            delay.reset()
            alignedDelay = nil
            return 0
        }
        let steps = InterpolationSteps.steps(for: generation.multiplier)
        // The engines measure their latency from a capture's arrival; the schedule runs on the compositor's times, which
        // are earlier by how long the capture took to be delivered.
        let latency = generation.latency + Self.delivery(of: frames)
        var needed = FramePlanner.interpolationDelay(captureInterval: shared.captureInterval.withLock { $0.value },
                                                     generationLatency: latency, steps: steps)
        if let beat = cadence.beat(refreshPeriod: refreshPeriod),
           let callbacks = callbackPhase.offset(period: refreshPeriod),
           let captures = capturePhase.offset(period: refreshPeriod),
           let aligned = ScheduleAlignment.delay(captureInterval: beat, generationLatency: latency, steps: steps,
                                                 phase: callbacks - captures, refreshPeriod: refreshPeriod,
                                                 current: alignedDelay) {
            alignedDelay = aligned
            needed = aligned
        } else {
            alignedDelay = nil
        }
        return delay.apply(needed, now: now)
    }

    /// Takes in what this callback adds to what the schedule knows: where on the refresh grid the callback fell, and the
    /// beat and the place on the grid of the captures that have come in since the last. The grid is the target time's.
    private func note(_ frames: [FrameHistory], now: CFTimeInterval, targetTime: CFTimeInterval, refreshPeriod: CFTimeInterval) {
        guard refreshPeriod > 0 else { return }
        callbackPhase.add(now - targetTime, period: refreshPeriod)
        for frame in frames where frame.timestamp > lastNoted {
            cadence.add(frame.presentationTime)
            capturePhase.add(frame.presentationTime - targetTime, period: refreshPeriod)
            lastNoted = frame.timestamp
        }
    }

    /// How long after the compositor showed them the captures reached the pipeline, the median of those held.
    private static func delivery(of frames: [FrameHistory]) -> CFTimeInterval {
        let delays = frames.map { $0.timestamp - $0.presentationTime }.sorted()
        return delays.isEmpty ? 0 : delays[delays.count / 2]
    }

    /// Turns the plan into a texture. A generated image that does not exist yet falls back to a capture
    /// rather than leaving the screen without an image.
    private func realize(_ plan: PresentationPlan, frames: [FrameHistory]) -> Content {
        // What is shown is named on the schedule's timeline, as the plan names it; the images are looked up by the captures'
        // arrivals, under which the engines publish them.
        func captured(_ index: Int) -> Content {
            let frame = frames[index]
            return Content(image: .captured(frame.presentationTime), picture: .capture(frame.texture),
                           sourceTimestamp: frame.timestamp, isGenerated: false)
        }

        switch plan {
        case .nothing:
            preconditionFailure("an empty plan has no image to realise")

        case .captured(let index):
            return captured(index)

        case .interpolated(let previous, let next, let step, let steps):
            // Not made yet — the motion for the pair is still being measured, or the engine is still on
            // it — so the screen stays where it is: on the last image of the pair that exists, or on the
            // capture before it. Showing a later one early would put time forward, and the image, when it
            // did arrive, would take it back.
            let (earlier, later) = (frames[previous].timestamp, frames[next].timestamp)
            for candidate in stride(from: step, through: 1, by: -1) {
                let phase = Double(candidate) / Double(steps)
                guard let made = shared.images.image(previous: earlier, next: later, phase: phase) else { continue }
                // The image stands for a time between the pair, but it could only be made once `next` had
                // arrived, so that is the honest age to report — and it makes the two engines directly comparable.
                // It is shown blended with the captures it sits between, as the engine that made it is trusted.
                let blend = Blend(image: made.source, previous: frames[previous].texture, next: frames[next].texture, phase: phase,
                                  motion: StabilityBlend.motion(for: made.engine))
                return Content(image: .interpolated(previous: frames[previous].presentationTime,
                                                    next: frames[next].presentationTime, phase: phase),
                               picture: .generated(blend), sourceTimestamp: later, isGenerated: true)
            }
            return captured(previous)
        }
    }

    // MARK: - Presentation

    /// Fills the drawable from `content`, upscaling on the way when scaling is on.
    ///
    /// MetalFX only writes private textures, and a drawable is not one, so the scaled image is
    /// built in a texture of our own and then copied across. A blit does that at memory speed;
    /// the render pass is kept for the cases that need resampling — a drawable smaller than the
    /// capture, or scaling off with an overlay that is not the capture's size.
    private func encodePresent(_ content: MTLTexture, to target: MTLTexture, upscaling: Bool,
                               commandBuffer: MTLCommandBuffer) {
        var source = content
        let growsInBothDimensions = target.width >= content.width && target.height >= content.height
        let sameSize = target.width == content.width && target.height == content.height

        // Scale Factor is already expressed in the overlay's size, so the target is simply the
        // drawable: MetalFX covers the whole gap and the copy stays 1:1 instead of bilinearly
        // stretching on top of the upscale. The overlay frame is capped at the screen, so the
        // drawable is bounded by the panel and can never approach Metal's texture limit.
        if upscaling, growsInBothDimensions, !sameSize {
            if let output = gpu.ensureTexture(&upscaled, width: target.width, height: target.height,
                                              usage: [.shaderRead, .shaderWrite, .renderTarget]),
               let scaler = gpu.ensureSpatialScaler(&spatialScaler, inputWidth: content.width, inputHeight: content.height,
                                                    outputWidth: target.width, outputHeight: target.height) {
                scaler.colorTexture = content
                scaler.outputTexture = output
                scaler.encode(commandBuffer: commandBuffer)
                source = output
            } else {
                shared.errors.report(.spatialScalerFailed)
            }
        }

        if source.width == target.width, source.height == target.height {
            guard let blit = commandBuffer.makeBlitCommandEncoder() else { return }
            blit.copy(from: source, to: target)
            blit.endEncoding()
        } else {
            let pass = MTLRenderPassDescriptor()
            pass.colorAttachments[0].texture = target
            // Every pixel is drawn, so what was there is not read back.
            pass.colorAttachments[0].loadAction = .dontCare
            pass.colorAttachments[0].storeAction = .store
            guard let encoder = commandBuffer.makeRenderCommandEncoder(descriptor: pass) else { return }
            encoder.setRenderPipelineState(gpu.pipelines.present)
            encoder.setFragmentTexture(source, index: 0)
            encoder.drawPrimitives(type: .triangleStrip, vertexStart: 0, vertexCount: 4)
            encoder.endEncoding()
        }
    }

    // MARK: - Statistics

    private func recordPresent(_ content: Content, isNewImage: Bool, drawableSize: CGSize) {
        windowPresents += 1
        if content.isGenerated && isNewImage { windowGenerated += 1 }
        shared.stats.withLock {
            $0.outputResolution = drawableSize
            $0.outputFrameCount += 1
            if content.isGenerated {
                $0.generatedFrameCount += 1
            } else {
                $0.passthroughFrameCount += 1
            }
        }
    }

    /// Once a second: the rates over that window, and the pacing of what actually reached the
    /// screen. Runs whether or not this callback presents, so a pipeline that has gone quiet shows
    /// zero rather than its last good figure.
    private func publishRates(now: CFTimeInterval, displayRate: DisplayRate) {
        if windowStart == 0 { windowStart = now }
        let elapsed = now - windowStart
        guard elapsed >= 1.0 else { return }

        let summary = pacing.withLock { $0.summary }
        let presents = Float(windowPresents) / Float(elapsed)
        let generated = Float(windowGenerated) / Float(elapsed)
        let busy = gpu.busyTime
        let load = Float((busy - windowBusyStart) / elapsed * 100)
        windowBusyStart = busy
        let config = shared.config.withLock { $0 }
        let imagesPerCapture = config.generatesFrames ? max(1, shared.generation.withLock { $0.multiplier }) : 1
        shared.stats.withLock {
            $0.outputFPS = presents
            $0.generatedFPS = generated
            // What the screen should be given: every capture, and the images generated between them, up to one a
            // refresh — four steps are made with fewer than four refreshes a capture, and not all of them are shown.
            $0.targetOutputFPS = min(Int(($0.captureFPS * Float(imagesPerCapture)).rounded()), displayRate.maximum)
            $0.gpuLoad = load
            if let summary {
                $0.avgFrameTime = Float(summary.averageInterval * 1000)
                $0.framePacingScore = Float(summary.score)
            }
            $0.screenRefreshRate = displayRate.maximum
            $0.isProMotion = displayRate.isVariable
        }
        windowPresents = 0
        windowGenerated = 0
        windowStart = now
    }
}
