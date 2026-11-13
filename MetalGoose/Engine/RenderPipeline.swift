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

    private var spatialScaler: (any MTL4FXSpatialScaler)?
    private var upscaled: (any MTLTexture)?

    /// What the screen is showing now. A callback whose image is the one already there does not
    /// draw at all: the layer keeps its contents, so presenting the same pixels again would cost a
    /// full pass over the drawable and a recomposite by the window server for nothing.
    private var lastPresented: PresentedImage?

    /// Set from any thread when the overlay changes shape or reappears: the drawable then has to
    /// be filled again even though no new image has arrived.
    private let redrawRequested = OSAllocatedUnfairLock(initialState: true)

    private let pacing = OSAllocatedUnfairLock(initialState: PacingTracker())

    private var windowStart: CFTimeInterval = 0
    private let presentations = PresentationCounter()
    private var windowBusyStart = 0.0

    /// What was drawn over the image on screen (`OverlayGraphics`), and when the last image, and the last new one, were
    /// committed.
    private var drawnHUD: Int?
    private var drawnPointer: CGPoint?
    private var lastCommit: CFTimeInterval = 0
    private var lastNewImageCommit: CFTimeInterval = 0

    init(shared: EngineShared) {
        self.shared = shared
        self.gpu = shared.gpu
        self.stability = StabilityBlender(gpu: shared.gpu)
        windowBusyStart = shared.gpu.busyTime
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
        presentations.reset()
        windowStart = 0
        windowBusyStart = gpu.busyTime
        drawnHUD = nil
        drawnPointer = nil
        lastCommit = 0
        lastNewImageCommit = 0
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
        let compositorTimestamp: CFTimeInterval
        let validity: WorkValidity
        let referenceValidity: WorkValidity?
        let isGenerated: Bool
        /// The captures the image is made of, which the presentation waits for and holds.
        let captures: [FrameHistory]
        var isValid: Bool { validity.isValid && referenceValidity?.isValid != false }
    }

    private enum Picture {
        case capture(any MTLTexture)
        case generated(Blend)
    }

    /// An image an engine made, the two captures it sits between, where, and how far the image is trusted.
    private struct Blend {
        let image: GeneratedImages.Source
        let previous: any MTLTexture
        let next: any MTLTexture
        let phase: Double
        let motion: Float
    }

    /// A capture as the schedule sees it: at the compositor's time for it.
    private struct ScheduledCapture: TimedFrame {
        let timestamp: CFTimeInterval
        let isSceneCut: Bool
    }

    /// - Parameters:
    ///   - acquire: the drawable to present into, taken only once there is something new to show.
    ///   - targetTime: when an image presented now would reach the screen: the display link's target time, which is on the
    ///     display's refresh grid, or now plus the frame latency where each image is presented at its own time.
    ///   - atImageTime: whether each new image is presented at the moment it is due (`present(at:)`) rather than on the link's
    ///     refresh, which a panel that can vary its refresh (VRR) then shows it at. There is no refresh grid then to align
    ///     the schedule to.
    func render(drawable acquire: () -> CAMetalDrawable?, displayRate: DisplayRate, targetTime: CFTimeInterval,
                atImageTime: Bool) {
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
        var refreshPeriod: CFTimeInterval = 0
        if !atImageTime {
            if lastTarget > 0 { refreshInterval.add(targetTime - lastTarget, window: EngineShared.measurementWindow) }
            lastTarget = targetTime
            refreshPeriod = refreshInterval.value > 0 ? refreshInterval.value
                : displayRate.maximum > 0 ? 1 / Double(displayRate.maximum) : 0
            note(frames, now: now, targetTime: targetTime, refreshPeriod: refreshPeriod)
        }

        // The schedule runs on the compositor's times for the captures, which are the content's own and on the display's
        // grid; their arrivals wander by how long each took to be delivered and reached.
        let timeline = frames.map { ScheduledCapture(timestamp: $0.presentationTime, isSceneCut: $0.isSceneCut) }
        let generation = config.generatesFrames ? shared.generation.withLock { $0 } : .nothing
        let delay = scheduleDelay(for: generation, frames: frames, now: now, refreshPeriod: refreshPeriod)
        let plan = FramePlanner.plan(timeline, PlanningInput(multiplier: generation.multiplier, sampleTime: now, delay: delay))

        publishRates(now: now, displayRate: displayRate)

        guard let planned = FramePlanner.image(of: plan, in: timeline) else { return }
        let graphics = shared.graphics.snapshot(includesPointer: config.variableRefresh)
        let panelPeriod = displayRate.maximum > 0 ? 1 / Double(displayRate.maximum) : 1.0 / 60
        let redraw = redrawRequested.withLock { $0 } || redrawsGraphics(graphics, now: now, period: panelPeriod)
        guard planned != lastPresented || redraw else { return }

        let content = realize(plan, frames: frames)
        guard content.isValid else { return }

        // The planned image is not always the one that came out: a midpoint that has not been made yet —
        // the motion for the pair is still being measured — falls back to a capture. Remembering the
        // *plan* as shown would never try again once the image arrives, and presenting the fallback
        // over and over would redraw what is already on screen, so only the realised image counts. The
        // command buffer is made only now: a callback that waits for an image used to commit an empty one.
        guard content.image != lastPresented || redraw, let drawable = acquire(),
              let command = gpu.render.makeCommand("MetalGoose present") else { return }

        // The captures are written on the capture lane, and their slots are not written again until this has run.
        for capture in content.captures {
            command.wait(for: gpu.capture, value: capture.written)
            command.retain(capture.lease)
        }

        // Only now, with the image known to be shown: a pass for one that is not would be spent for nothing.
        let shown: any MTLTexture
        switch content.picture {
        case .capture(let texture):
            shown = texture
        case .generated(let blend):
            guard let blended = stability.blend(blend.image, previous: blend.previous, next: blend.next, phase: blend.phase,
                                                motion: blend.motion, command: command) else {
                command.commit()
                return
            }
            shown = blended
        }
        guard encodePresent(shown, graphics: graphics, to: drawable.texture, upscaling: config.upscaling,
                            magnification: content.captures.first?.magnification, command: command) else {
            command.commit()
            return
        }

        // A redraw of the image already on screen — the overlay changed shape, or what is drawn over it — is not a new
        // generated image.
        let isNewImage = content.image != lastPresented
        lastPresented = content.image
        redrawRequested.withLock { $0 = false }
        drawnHUD = graphics.hudGeneration
        drawnPointer = graphics.pointer?.fraction
        lastCommit = now
        if isNewImage { lastNewImageCommit = now }

        let source = content.sourceTimestamp
        let compositor = content.compositorTimestamp
        let validity = content.validity
        let referenceValidity = content.referenceValidity
        let generated = content.isGenerated
        let drawableSize = CGSize(width: drawable.texture.width, height: drawable.texture.height)
        let epoch = presentations.epoch
        let counterEpoch = shared.stats.withLock { $0.counterEpoch }
        let capacity = max(8, Int((Double(displayRate.maximum) * EngineShared.measurementWindow).rounded()))
        drawable.addPresentedHandler { [shared, pacing, presentations] presented in
            // Zero means the system could not say when it reached the screen.
            let time = presented.presentedTime
            guard time > 0, validity.isValid, referenceValidity?.isValid != false else { return }
            let latency = Float((time - source) * 1000)
            shared.stats.withLock {
                guard $0.counterEpoch == counterEpoch,
                      presentations.record(epoch: epoch, isNewImage: isNewImage, isGenerated: generated) else { return }
                if isNewImage {
                    pacing.withLock { $0.record(time, capacity: capacity) }
                    $0.outputFrameCount += 1
                    if generated { $0.generatedFrameCount += 1 }
                    else { $0.passthroughFrameCount += 1 }
                }
                $0.outputResolution = drawableSize
                $0.presentLatency = latency
                $0.endToEndLatency = Float((time - compositor) * 1000)
            }
        }
        command.onCompleted { [shared] succeeded in
            if !succeeded { shared.renderResetRequested.withLock { $0 = true } }
        }
        // Presented at its own time, a new image reaches the screen when its moment on the timeline comes round on the
        // schedule — as long after now as the driver presents ahead, give or take where in its turn it was picked — rather
        // than on the next refresh. Never before now, and never so far ahead that it holds a drawable for long; a redraw
        // goes at once.
        let due: CFTimeInterval? = atImageTime && isNewImage
            ? min(max(content.image.time + delay + (targetTime - now), now), now + Self.longestHold) : nil
        command.commit(presenting: drawable, at: due)
    }

    /// The furthest ahead an image presented at its own time is given its moment.
    private static let longestHold: CFTimeInterval = 0.1

    /// Whether what is drawn over the picture is drawn again on its own, having changed since the image on screen was drawn.
    /// Where new images keep coming, each carries it as it is when it is drawn, and a pass of its own would only crowd the
    /// panel between them; where none has come for `stillAfter`, it is, once a refresh at the most.
    private func redrawsGraphics(_ graphics: OverlayGraphics.Snapshot, now: CFTimeInterval, period: CFTimeInterval) -> Bool {
        guard graphics.hudGeneration != drawnHUD || graphics.pointer?.fraction != drawnPointer else { return false }
        return now >= max(lastNewImageCommit + Self.stillAfter, lastCommit + period)
    }

    /// How long without a new image before the HUD or the pointer is drawn again on its own: below 20 images a second.
    private static let stillAfter: CFTimeInterval = 0.05

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
                           sourceTimestamp: frame.timestamp, compositorTimestamp: frame.presentationTime,
                           validity: frame.validity, referenceValidity: nil, isGenerated: false, captures: [frame])
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
                               picture: .generated(blend), sourceTimestamp: later, compositorTimestamp: frames[next].presentationTime,
                               validity: frames[next].validity, referenceValidity: frames[previous].validity, isGenerated: true,
                               captures: [frames[previous], frames[next]])
            }
            return captured(previous)
        }
    }

    // MARK: - Presentation

    /// Fills the drawable from `content`, upscaling on the way when scaling is on, with `graphics` over it.
    ///
    /// MetalFX only writes private textures, and a drawable is not one, so the scaled image is
    /// built in a texture of our own and then copied across. A blit does that at memory speed;
    /// the render pass is kept for the cases that need resampling — a drawable smaller than the
    /// capture, or scaling off with an overlay that is not the capture's size — and for an image with the HUD or the pointer
    /// over it, which the pass draws after the picture without reading the drawable back.
    /// Content an app drew at half the window's pixels and showed as blocks (`magnification`) is shown so again where the
    /// drawable is twice its size and nothing scales it up, as the app's own window showed it; the bilinear present made it
    /// softer.
    private func encodePresent(_ content: any MTLTexture, graphics: OverlayGraphics.Snapshot, to target: any MTLTexture,
                                upscaling: Bool, magnification: ContentResolution.Magnification?, command: GPUCommand) -> Bool {
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
                gpu.encodeUpscale(scaler, from: content, to: output, on: command)
                source = output
            } else {
                shared.errors.report(.spatialScalerFailed)
            }
        }

        // The drawable is made resident by its layer's own residency set (`RenderDriver`), not by the command buffer.
        if source.width == target.width, source.height == target.height, graphics.hud == nil, graphics.pointer == nil {
            guard let pass = command.makeComputePass() else { return false }
            pass.copy(from: source, toDrawable: target)
            pass.endEncoding()
        } else {
            // Every pixel is drawn, so what was there is not read back.
            guard let pass = command.makeRenderPass(target: target) else { return false }
            let asBlocks = magnification == .nearest && source.width * 2 == target.width && source.height * 2 == target.height
            pass.setRenderPipelineState(asBlocks ? gpu.pipelines.presentNearest : gpu.pipelines.present)
            pass.setFragmentTexture(source, index: 0)
            pass.drawFullTarget()
            drawGraphics(graphics, in: pass, target: target)
            pass.endEncoding()
        }
        return true
    }

    /// Draws the HUD and the pointer over the picture `pass` has drawn, a pixel of theirs to a pixel of the drawable's.
    private func drawGraphics(_ graphics: OverlayGraphics.Snapshot, in pass: RenderPass, target: any MTLTexture) {
        var sprites: [(sprite: OverlayGraphics.Sprite, at: CGPoint)] = []
        if let hud = graphics.hud { sprites.append((hud.sprite, hud.origin)) }
        if let pointer = graphics.pointer {
            sprites.append((pointer.sprite, CGPoint(x: pointer.fraction.x * CGFloat(target.width),
                                                    y: pointer.fraction.y * CGFloat(target.height))))
        }
        guard !sprites.isEmpty else { return }
        pass.setRenderPipelineState(gpu.pipelines.sprite)
        let width = Float(target.width), height = Float(target.height)
        for (sprite, point) in sprites {
            // On whole pixels, so that the image is sampled at its texels' centres and stays sharp.
            let left = Float((point.x - sprite.anchor.x).rounded())
            let top = Float((point.y - sprite.anchor.y).rounded())
            var rect = SIMD4<Float>(left / width, top / height,
                                    (left + Float(sprite.texture.width)) / width, (top + Float(sprite.texture.height)) / height)
            pass.setFragmentTexture(sprite.texture, index: 0)
            pass.setVertexBytes(&rect, length: MemoryLayout<SIMD4<Float>>.stride, index: 0)
            pass.drawFullTarget()
        }
    }

    // MARK: - Statistics

    /// Once a second: the rates over that window, and the pacing of what actually reached the
    /// screen. Runs whether or not this callback presents, so a pipeline that has gone quiet shows
    /// zero rather than its last good figure.
    private func publishRates(now: CFTimeInterval, displayRate: DisplayRate) {
        if windowStart == 0 { windowStart = now }
        let elapsed = now - windowStart
        guard elapsed >= 1.0 else { return }

        let summary = pacing.withLock { $0.summary }
        let window = presentations.takeWindow()
        let presents = Float(window.images) / Float(elapsed)
        let generated = Float(window.generated) / Float(elapsed)
        let busy = gpu.busyTime
        let duration = max(0, busy - windowBusyStart)
        let load = Float(duration / elapsed * 100)
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
            // Average cost of all GPU stages per new image actually presented.
            $0.gpuTime = window.images > 0 ? Float(duration * 1000 / Double(window.images)) : 0
            if let summary {
                $0.avgFrameTime = Float(summary.averageInterval * 1000)
                $0.framePacingScore = Float(summary.score)
            }
            $0.screenRefreshRate = displayRate.maximum
            $0.isProMotion = displayRate.isVariable
        }
        windowStart = now
    }
}
