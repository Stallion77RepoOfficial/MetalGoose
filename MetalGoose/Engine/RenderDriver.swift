import Foundation
@preconcurrency import Metal
import QuartzCore
import os

/// Calls the render pipeline, from a thread of its own: once per display refresh, or, with variable refresh (VRR) on, often
/// enough to present each image at its own moment.
///
/// A display link on a dedicated run loop is not behind the main thread (SwiftUI layout, the HUD, the
/// mouse event tap), and it hands over a drawable that is already acquired, so the render thread never
/// blocks inside `nextDrawable`. Callbacks arrive once per refresh, so the schedule is read at each one.
///
/// The link always runs at the panel's own rate, whatever is being captured: its frame latency is
/// counted in link frames, so a link slowed to a 30 fps capture queues each image for two 30 Hz frames,
/// against two 8 ms frames at 120 Hz. A callback with nothing new to show returns before it allocates
/// anything, so the faster rate costs next to nothing.
///
/// A link's drawables are presented on its refreshes; one presented at a time of its own raises an exception. So variable
/// refresh has a presenter of its own (`PresentationThread`): it reads the schedule every quarter refresh, takes a
/// drawable only for a new image, and presents that at the time it is due (`RenderPipeline`). A panel that can vary its refresh shows
/// it then — measured on the built-in ProMotion panel, an opaque window covering the display, with nothing moving above it,
/// on a 4.17 ms grid — and any other shows it on its next refresh, as the link would have.
final class RenderDriver: NSObject, CAMetalDisplayLinkDelegate, @unchecked Sendable {

    private let shared: EngineShared
    private let pipeline: RenderPipeline
    private let displayRate: OSAllocatedUnfairLock<DisplayRate>
    private let thread = RenderThread()

    // Render thread only.
    private var link: CAMetalDisplayLink?

    // Main thread only.
    /// The layer's drawables, made resident for the render lane while the driver runs.
    private var drawables: (any MTLResidencySet)?
    private var layer: CAMetalLayer?
    /// Whether images are presented at their own times (`presenter`) rather than on the link's refreshes.
    private var variableRefresh = false
    private var presenter: PresentationThread?
    private var isPaused = false

    init(shared: EngineShared, pipeline: RenderPipeline, displayRate: DisplayRate) {
        self.shared = shared
        self.pipeline = pipeline
        self.displayRate = OSAllocatedUnfairLock(initialState: displayRate)
        super.init()
    }

    // MARK: - Lifecycle (main thread)

    /// Configures the layer for presentation and starts presenting into it.
    @MainActor
    func start(layer: CAMetalLayer) {
        let config = shared.config.withLock { $0 }
        configure(layer: layer, config: config)
        let drawables = layer.residencySet
        shared.gpu.render.queue.addResidencySet(drawables)
        self.drawables = drawables
        self.layer = layer

        thread.start()
        thread.waitUntilRunning()
        variableRefresh = config.variableRefresh
        if variableRefresh { startPresenter(layer: layer) } else { startLink(layer: layer) }
    }

    /// Stops presenting and waits until the threads are done, so nothing is encoding when the caller goes on to release
    /// what the pipeline holds.
    @MainActor
    func stop() {
        defer {
            if let drawables { shared.gpu.render.queue.removeResidencySet(drawables) }
            drawables = nil
            layer = nil
        }
        stopPresenter()
        stopLink()
        thread.stop()
    }

    /// Presents on the link's refreshes or at each image's own time from now on. The one stops before the other starts, so
    /// the pipeline is driven from one thread at a time.
    @MainActor
    func setVariableRefresh(_ on: Bool) {
        guard on != variableRefresh, let layer else { return }
        variableRefresh = on
        if on {
            stopLink()
            startPresenter(layer: layer)
        } else {
            stopPresenter()
            startLink(layer: layer)
        }
        shared.renderResetRequested.withLock { $0 = true }
    }

    @MainActor
    func setPaused(_ paused: Bool) {
        isPaused = paused
        presenter?.setPaused(paused)
        guard let runLoop = thread.runLoop else { return }
        runLoop.perform { [self] in link?.isPaused = paused }
        CFRunLoopWakeUp(runLoop.getCFRunLoop())
    }

    /// Re-applies what depends on the settings, from the main thread.
    @MainActor
    func configure(layer: CAMetalLayer, config: EngineConfig) {
        layer.device = shared.gpu.device
        layer.pixelFormat = GPUContext.drawablePixelFormat
        // The pipeline copies into the drawable, which a framebuffer-only texture does not allow.
        layer.framebufferOnly = false
        layer.presentsWithTransaction = false
        // The layer is fully covered by what is drawn into it, so the compositor can skip blending
        // it with whatever is behind.
        layer.isOpaque = true
        layer.displaySyncEnabled = config.vsync
        // CAMetalDisplayLink manages the drawable queue. Setting maximumDrawableCount
        // after a link is attached raises CAMetalLayerInvalidOperation on macOS 27.

        // One frame of latency for double buffering, two for triple: the toggle is a latency
        // choice, and this is where it takes effect.
        let latency = Self.frameLatency(for: config)
        if let runLoop = thread.runLoop {
            runLoop.perform { [self] in link?.preferredFrameLatency = latency }
            CFRunLoopWakeUp(runLoop.getCFRunLoop())
        }
    }

    // MARK: - Driving (main thread)

    @MainActor
    private func startLink(layer: CAMetalLayer) {
        nonisolated(unsafe) let layer = layer
        guard let runLoop = thread.runLoop else { return }
        let paused = isPaused
        runLoop.perform { [self] in
            let link = CAMetalDisplayLink(metalLayer: layer)
            link.delegate = self
            self.link = link
            link.preferredFrameLatency = Self.frameLatency(for: shared.config.withLock { $0 })
            let rate = Float(displayRate.withLock { $0.maximum })
            link.preferredFrameRateRange = CAFrameRateRange(minimum: rate, maximum: rate, preferred: rate)
            link.isPaused = paused
            link.add(to: .current, forMode: .default)
        }
        // Scheduling a block does not wake a run loop that is asleep waiting for a source.
        CFRunLoopWakeUp(runLoop.getCFRunLoop())
    }

    @MainActor
    private func stopLink() {
        guard let runLoop = thread.runLoop else { return }
        let done = DispatchSemaphore(value: 0)
        runLoop.perform { [self] in
            link?.invalidate()
            link = nil
            done.signal()
        }
        CFRunLoopWakeUp(runLoop.getCFRunLoop())
        done.wait()
    }

    @MainActor
    private func startPresenter(layer: CAMetalLayer) {
        nonisolated(unsafe) let layer = layer
        let presenter = PresentationThread { [pipeline, displayRate, shared] in
            let rate = displayRate.withLock { $0 }
            let period = rate.maximum > 0 ? 1 / Double(rate.maximum) : 1.0 / 60
            // An image presented at a time of its own reaches the screen about a refresh after that time (8 to 12 ms at
            // 120 Hz, measured), which the link's frame latency already counts: one refresh less ahead than the link keeps
            // about its latency. At least half a refresh, for the image to be encoded in.
            let latency = max(Double(Self.frameLatency(for: shared.config.withLock { $0 })) - 1, 0.5) * period
            pipeline.render(drawable: { layer.nextDrawable() }, displayRate: rate,
                            targetTime: CACurrentMediaTime() + latency, atImageTime: true)
            // An image is taken up at most a quarter refresh after its moment, which leaves most of the lead to encode it
            // in; taken up every half refresh, some at 90 a second missed their time. Sleeping until the plan's next image
            // instead, and waking for each capture, took 0.4 to 0.5% of a core less but left the timing off by more in
            // 14 of 26 runs against 7, measured.
            return period / 4
        }
        presenter.setPaused(isPaused)
        presenter.start()
        self.presenter = presenter
    }

    @MainActor
    private func stopPresenter() {
        presenter?.stopAndWait()
        presenter = nil
    }

    // MARK: - Callback (render thread)

    private static func frameLatency(for config: EngineConfig) -> Float {
        Float(min(max(2, config.bufferDepth), GooseEngine.maxInFlight) - 1)
    }

    func updateDisplayRate(_ rate: DisplayRate) {
        let changed = displayRate.withLock { current -> Bool in
            guard current != rate else { return false }
            current = rate
            return true
        }
        guard changed else { return }
        // The presenter reads the rate every time it wakes; the pipeline starts over on its next pass either way.
        shared.renderResetRequested.withLock { $0 = true }
        guard let runLoop = thread.runLoop else { return }
        runLoop.perform { [self] in
            let maximum = Float(rate.maximum)
            link?.preferredFrameRateRange = CAFrameRateRange(minimum: maximum, maximum: maximum, preferred: maximum)
        }
        CFRunLoopWakeUp(runLoop.getCFRunLoop())
    }

    func metalDisplayLink(_ link: CAMetalDisplayLink, needsUpdate update: CAMetalDisplayLink.Update) {
        let drawable = update.drawable
        pipeline.render(drawable: { drawable }, displayRate: displayRate.withLock { $0 },
                        targetTime: update.targetPresentationTimestamp, atImageTime: false)
    }
}

/// A thread that exists to host a run loop.
private final class RenderThread: Thread, @unchecked Sendable {
    private let running = DispatchSemaphore(value: 0)
    private(set) var runLoop: RunLoop?

    override init() {
        super.init()
        name = "com.metalgoose.render"
        qualityOfService = .userInteractive
    }

    override func main() {
        autoreleasepool {
            let loop = RunLoop.current
            runLoop = loop
            // A run loop with nothing scheduled returns at once. The port keeps it waiting until the
            // display link is added.
            loop.add(Port(), forMode: .default)
            running.signal()
            while !isCancelled {
                loop.run(mode: .default, before: .distantFuture)
            }
        }
    }

    func waitUntilRunning() {
        running.wait()
    }

    func stop() {
        cancel()
        if let loop = runLoop { CFRunLoopStop(loop.getCFRunLoop()) }
    }
}

/// Runs `pass` again and again until stopped, sleeping between passes for as long as each returns; while paused it only
/// sleeps. Presents for variable refresh (`RenderDriver`).
private final class PresentationThread: Thread, @unchecked Sendable {
    private let pass: @Sendable () -> CFTimeInterval
    private let paused = OSAllocatedUnfairLock(initialState: false)
    private let stopped = DispatchSemaphore(value: 0)

    /// How long a paused presenter sleeps between looks at whether it still is.
    private static let pausedNap: CFTimeInterval = 0.05

    init(pass: @escaping @Sendable () -> CFTimeInterval) {
        self.pass = pass
        super.init()
        name = "com.metalgoose.present"
        qualityOfService = .userInteractive
    }

    func setPaused(_ value: Bool) {
        paused.withLock { $0 = value }
    }

    override func main() {
        while !isCancelled {
            let nap = paused.withLock { $0 } ? Self.pausedNap : autoreleasepool { pass() }
            Thread.sleep(forTimeInterval: nap)
        }
        stopped.signal()
    }

    /// Stops after the pass under way, and returns once it has.
    func stopAndWait() {
        cancel()
        stopped.wait()
    }
}
