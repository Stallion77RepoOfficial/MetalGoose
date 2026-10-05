import Foundation
@preconcurrency import Metal
import QuartzCore
import os

/// Calls the render pipeline once per display refresh, from a thread of its own.
///
/// A display link on a dedicated run loop is not behind the main thread (SwiftUI layout, the HUD, the
/// mouse event tap), and it hands over a drawable that is already acquired, so the render thread never
/// blocks inside `nextDrawable`. Callbacks arrive once per refresh, so the schedule is read at each one.
///
/// The link always runs at the panel's own rate, whatever is being captured: its frame latency is
/// counted in link frames, so a link slowed to a 30 fps capture queues each image for two 30 Hz frames,
/// against two 8 ms frames at 120 Hz. A callback with nothing new to show returns before it allocates
/// anything, so the faster rate costs next to nothing.
final class RenderDriver: NSObject, CAMetalDisplayLinkDelegate, @unchecked Sendable {

    private let shared: EngineShared
    private let pipeline: RenderPipeline
    private let displayRate: OSAllocatedUnfairLock<DisplayRate>
    private let thread = RenderThread()

    // Render thread only.
    private var link: CAMetalDisplayLink?

    /// The layer's drawables, made resident for the render lane while the link runs. Main thread only.
    private var drawables: (any MTLResidencySet)?

    init(shared: EngineShared, pipeline: RenderPipeline, displayRate: DisplayRate) {
        self.shared = shared
        self.pipeline = pipeline
        self.displayRate = OSAllocatedUnfairLock(initialState: displayRate)
        super.init()
    }

    // MARK: - Lifecycle (main thread)

    /// Configures the layer for presentation and starts the callbacks.
    @MainActor
    func start(layer: CAMetalLayer) {
        configure(layer: layer, config: shared.config.withLock { $0 })
        let drawables = layer.residencySet
        shared.gpu.render.queue.addResidencySet(drawables)
        self.drawables = drawables

        thread.start()
        thread.waitUntilRunning()
        nonisolated(unsafe) let layer = layer
        guard let runLoop = thread.runLoop else { return }
        runLoop.perform { [self] in
            let link = CAMetalDisplayLink(metalLayer: layer)
            link.delegate = self
            self.link = link
            link.preferredFrameLatency = Self.frameLatency(for: shared.config.withLock { $0 })
            let rate = Float(displayRate.withLock { $0.maximum })
            link.preferredFrameRateRange = CAFrameRateRange(minimum: rate, maximum: rate, preferred: rate)
            link.add(to: .current, forMode: .default)
        }
        // Scheduling a block does not wake a run loop that is asleep waiting for a source.
        CFRunLoopWakeUp(runLoop.getCFRunLoop())
    }

    /// Stops the callbacks and waits until the render thread is gone, so nothing is encoding when
    /// the caller goes on to release what the pipeline holds.
    @MainActor
    func stop() {
        defer {
            if let drawables { shared.gpu.render.queue.removeResidencySet(drawables) }
            drawables = nil
        }
        guard let runLoop = thread.runLoop else { return }
        let done = DispatchSemaphore(value: 0)
        runLoop.perform { [self] in
            link?.invalidate()
            link = nil
            done.signal()
        }
        CFRunLoopWakeUp(runLoop.getCFRunLoop())
        done.wait()
        thread.stop()
    }

    func setPaused(_ paused: Bool) {
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
        guard changed, let runLoop = thread.runLoop else { return }
        runLoop.perform { [self] in
            let maximum = Float(rate.maximum)
            link?.preferredFrameRateRange = CAFrameRateRange(minimum: maximum, maximum: maximum, preferred: maximum)
            pipeline.reset()
        }
        CFRunLoopWakeUp(runLoop.getCFRunLoop())
    }

    func metalDisplayLink(_ link: CAMetalDisplayLink, needsUpdate update: CAMetalDisplayLink.Update) {
        pipeline.render(into: update.drawable, displayRate: displayRate.withLock { $0 }, targetTime: update.targetPresentationTimestamp)
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
