import Foundation
@preconcurrency import Metal
@preconcurrency import MetalFX
import QuartzCore
import os

/// Puts the right image on screen for each display callback: decides what the callback should
/// show, extrapolates it if that is what it is, scales it up to the drawable, and presents.
///
/// Runs on the render thread alone, and owns everything it keeps — the scaler, the extrapolator and
/// their textures. It shares only what `EngineShared` and the motion pipeline lock. Interpolated
/// images are not made here: the capture side makes each as its pair completes, and this only picks
/// up the ones that exist.
final class RenderPipeline: @unchecked Sendable {

    private let shared: EngineShared
    private let gpu: GPUContext
    private let motion: MotionPipeline
    private let extrapolator: FrameExtrapolator

    private var spatialScaler: MTLFXSpatialScaler?
    private var upscaled: MTLTexture?

    /// What the screen is showing now. A callback whose image is the one already there does not
    /// draw at all: the layer keeps its contents, so presenting the same pixels again would cost a
    /// full pass over the drawable and a recomposite by the window server for nothing.
    private var lastPresented: PresentedImage?
    private var newestPresentedTimestamp: CFTimeInterval = -1

    /// Set from any thread when the overlay changes shape or reappears: the drawable then has to
    /// be filled again even though no new image has arrived.
    private let redrawRequested = OSAllocatedUnfairLock(initialState: true)

    private let pacing = OSAllocatedUnfairLock(initialState: PacingTracker())

    private var windowStart: CFTimeInterval = 0
    private var windowPresents = 0
    private var windowGenerated = 0
    private var windowBusyStart = 0.0

    init(shared: EngineShared, motion: MotionPipeline) {
        self.shared = shared
        self.gpu = shared.gpu
        self.motion = motion
        self.extrapolator = FrameExtrapolator(gpu: shared.gpu)
    }

    func requestRedraw() {
        redrawRequested.withLock { $0 = true }
    }

    /// Drops what the render side holds. The next callback starts from nothing.
    func reset() {
        spatialScaler = nil
        upscaled = nil
        extrapolator.reset()
        lastPresented = nil
        newestPresentedTimestamp = -1
        requestRedraw()
        pacing.withLock { $0.reset() }
    }

    // MARK: - One display callback

    /// What a callback ends up putting on screen.
    private struct Content {
        /// What is actually being shown, which is not always what was planned: a midpoint that does not
        /// exist yet falls back to a capture, and that capture is what the screen then holds.
        let image: PresentedImage
        let texture: MTLTexture
        /// Age is reported from this: the newest real information on screen. Extrapolated pixels
        /// are a guess, and crediting them made the figure read as zero.
        let sourceTimestamp: CFTimeInterval
        let isGenerated: Bool
    }

    func render(into drawable: CAMetalDrawable, displayRate: DisplayRate) {
        if shared.renderResetRequested.withLock({ requested -> Bool in
            defer { requested = false }
            return requested
        }) {
            reset()
        }

        let config = shared.config.withLock { $0 }
        let frames = shared.ring.snapshot()
        let field = config.generatesFrames ? motion.latest : nil

        // The schedule is read at the callback, not at the time the link says its image will reach the
        // screen: that latency is the same for every image and drops out, where a clock that included it
        // would start every capture interval most of the way through.
        let now = CACurrentMediaTime()
        let plan = FramePlanner.plan(frames, PlanningInput(
            mode: config.frameGeneration,
            multiplier: multiplier(for: config),
            sampleTime: now,
            captureInterval: shared.captureInterval.withLock { $0.value },
            generationLatency: shared.generationLatency.value,
            newestWasPresented: frames.last?.timestamp == newestPresentedTimestamp,
            motionTimestamp: field?.timestamp))

        publishRates(now: now, displayRate: displayRate)

        guard let planned = FramePlanner.image(of: plan, in: frames) else { return }
        let redraw = redrawRequested.withLock { $0 }
        guard planned != lastPresented || redraw,
              let commandBuffer = gpu.makeCommandBuffer("MetalGoose present") else { return }

        let content = realize(plan, frames: frames, field: field, commandBuffer: commandBuffer)

        // The planned image is not always the one that came out: a midpoint that has not been made yet —
        // the motion for the pair is still being measured — falls back to a capture. Remembering the
        // *plan* as shown would never try again once the image arrives, and presenting the fallback
        // over and over would redraw what is already on screen, so only the realised image counts.
        guard content.image != lastPresented || redraw else {
            commandBuffer.commit()
            return
        }

        encodePresent(content.texture, to: drawable.texture, upscaling: config.upscaling, commandBuffer: commandBuffer)

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

    /// How many images each capture interval is planned to carry: what extrapolation was asked for, and
    /// what interpolation is delivering, which is not always what it was asked for.
    private func multiplier(for config: EngineConfig) -> Int {
        config.frameGeneration == .interpolation ? shared.interpolationSteps.withLock { $0 } : config.multiplier
    }

    // MARK: - Realising a plan

    /// Turns the plan into a texture. A generated image that does not exist yet falls back to a capture
    /// rather than leaving the screen without an image.
    private func realize(_ plan: PresentationPlan, frames: [FrameHistory], field: MotionField?,
                         commandBuffer: MTLCommandBuffer) -> Content {
        func captured(_ index: Int) -> Content {
            let frame = frames[index]
            if index == frames.count - 1 { newestPresentedTimestamp = frame.timestamp }
            return Content(image: .captured(frame.timestamp), texture: frame.texture,
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
                guard let texture = shared.interpolated(previous: earlier, next: later, phase: phase) else { continue }
                // The image stands for a time between the pair, but it could only be made once `next` had
                // arrived, so that is the honest age to report — and it makes the two generation modes
                // directly comparable.
                return Content(image: .interpolated(previous: earlier, next: later, phase: phase),
                               texture: texture, sourceTimestamp: later, isGenerated: true)
            }
            return captured(previous)

        case .extrapolated(let source, let step, let steps):
            guard let field,
                  let texture = extrapolator.extrapolate(source: frames[source], field: field, step: step, steps: steps,
                                                         commandBuffer: commandBuffer) else {
                return captured(source)
            }
            return Content(image: .extrapolated(source: frames[source].timestamp, step: step),
                           texture: texture, sourceTimestamp: frames[source].timestamp, isGenerated: true)
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
        let imagesPerCapture = config.generatesFrames ? max(1, multiplier(for: config)) : 1
        shared.stats.withLock {
            $0.outputFPS = presents
            $0.generatedFPS = generated
            // What the screen should be given: every capture, and the images generated between them. The
            // panel's refresh rate is not it — it repeats whatever it last showed.
            $0.targetOutputFPS = Int(($0.captureFPS * Float(imagesPerCapture)).rounded())
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
