import AppKit
import CoreGraphics
import os

/// Keeps the system cursor inside the captured window while the overlay magnifies it, and has the overlay draw the
/// pointer where the cursor appears in the magnified image (`PointerMapping`).
///
/// The cursor moves as the system moves it. An event tap only holds it in: an event that would take it out of the window
/// is given the nearest point inside instead, and the cursor is put there. Nothing else about an event is changed, so its
/// location, its deltas and the cursor an app polls all say the same thing, and a click lands where the pointer is drawn
/// whichever of them the app reads. The system pointer is hidden — it is under the overlay, where the window really is —
/// and the overlay draws one where the user sees the window.
///
/// The tap runs on a thread of its own. An active tap holds every mouse event of the session until it has answered, and on
/// the main thread each one the game was sent waited for whatever MetalGoose's interface was doing at the time: the HUD
/// being laid out, the window list being read.
///
/// Needs Accessibility: an active event tap is how the events are changed.
final class MouseConstraintManager: @unchecked Sendable {
    static let shared = MouseConstraintManager()

    private init() {}

    private struct State {
        var eventTap: CFMachPort?
        var runLoopSource: CFRunLoopSource?
        var mapping = PointerMapping(window: .zero, displays: [])
        /// Where the cursor is, as the last event held it.
        var pointer: CGPoint = .zero
        var isConstraining = false
        var pointerVisible = true

        /// Set while some app other than the capture target is frontmost. The constraint exists to
        /// hold the pointer inside the captured window while the user is playing, and both halves of
        /// it are hostile once they have switched away: the hold drags the pointer back into the
        /// game on the first mouse movement, and the hide leaves the whole session without a visible
        /// cursor. Between them, switching away with Cmd+Tab looked like the game was pulling focus
        /// back, and nothing else on screen could be used or even seen.
        var isSuspended = false

        /// The pointer has moved since the overlay last drew it, and an update is on its way to the main thread. Events come
        /// at up to a thousand a second; the update draws wherever the pointer is when it runs, so one in flight is enough.
        var drawPending = false
    }
    private let state = OSAllocatedUnfairLock(uncheckedState: State())

    private let tapThread = EventTapThread()

    private var cursorHideTimer: Timer?

    var isConstraining: Bool { state.withLockUnchecked { $0.isConstraining } }

    /// Called on the main thread whenever the pointer sprite should move: with where it is as a fraction of the captured
    /// window, measured from its top-left corner, or `nil` when it should not be drawn.
    @MainActor var onPointerChange: ((CGPoint?) -> Void)?

    /// The hide is reference counted per connection, and each tick is two synchronous window-server
    /// round trips on the main run loop. Nothing between ticks changes the cursor state, so a faster
    /// rate buys nothing except how quickly a cursor another process revealed goes away again.
    private static let cursorReassertInterval: TimeInterval = 0.5

    // MARK: - Starting and stopping

    /// Starts holding the cursor in `window`, which can be on any of `displays` (both in CoreGraphics coordinates).
    @MainActor
    func startConstraining(window: CGRect, displays: [CGRect]) {
        let alreadyOn = state.withLockUnchecked { state -> Bool in
            state.mapping = PointerMapping(window: window, displays: displays)
            return state.isConstraining
        }
        if alreadyOn { return }

        let eventMask: CGEventMask = [
            CGEventType.mouseMoved, .leftMouseDragged, .rightMouseDragged, .otherMouseDragged,
            .leftMouseDown, .leftMouseUp, .rightMouseDown, .rightMouseUp, .otherMouseDown, .otherMouseUp
        ].reduce(0) { $0 | (1 << CGEventMask($1.rawValue)) }

        let callback: CGEventTapCallBack = { _, type, event, refcon in
            guard let refcon else { return Unmanaged.passUnretained(event) }
            let manager = Unmanaged<MouseConstraintManager>.fromOpaque(refcon).takeUnretainedValue()
            manager.handle(type: type, event: event)
            return Unmanaged.passUnretained(event)
        }

        guard let runLoop = tapThread.runLoop(),
              let tap = CGEvent.tapCreate(tap: .cgSessionEventTap, place: .headInsertEventTap, options: .defaultTap,
                                          eventsOfInterest: eventMask, callback: callback,
                                          userInfo: Unmanaged.passUnretained(self).toOpaque()) else { return }

        let source = CFMachPortCreateRunLoopSource(kCFAllocatorDefault, tap, 0)
        state.withLockUnchecked { state in
            state.eventTap = tap
            state.runLoopSource = source
            state.isConstraining = true
        }
        CFRunLoopAddSource(runLoop, source, .commonModes)
        CFRunLoopWakeUp(runLoop)
        CGEvent.tapEnable(tap: tap, enable: true)

        holdCursor()
        Self.enableBackgroundCursorControl()
        CGDisplayHideCursor(CGMainDisplayID())
        startCursorReassertTimer()
        notifyPointer()
    }

    /// Takes the tap down and gives the system pointer back.
    @MainActor
    func stopConstraining() {
        let previous = state.withLockUnchecked { state -> State in
            let previous = state
            state.eventTap = nil
            state.runLoopSource = nil
            state.isConstraining = false
            state.isSuspended = false
            state.pointerVisible = true
            return previous
        }

        cursorHideTimer?.invalidate()
        cursorHideTimer = nil

        guard previous.isConstraining else { return }

        if let tap = previous.eventTap {
            CGEvent.tapEnable(tap: tap, enable: false)
            CFMachPortInvalidate(tap)
        }
        if let source = previous.runLoopSource, let runLoop = tapThread.runLoop() {
            CFRunLoopRemoveSource(runLoop, source, .commonModes)
        }

        // The re-assert timer balances itself, so exactly one hide level is ours. A suspended
        // constraint has already given it back; releasing again here would take the count below zero
        // and hand a permanent extra show to whoever hid the cursor next.
        if !previous.isSuspended {
            CGDisplayShowCursor(CGMainDisplayID())
        }

        if let location = CGEvent(source: nil)?.location {
            CGWarpMouseCursorPosition(location)
        }
        onPointerChange?(nil)
    }

    /// Suspending releases exactly the one hide level `startConstraining` took, so the cursor comes
    /// back for the rest of the system; resuming takes it again. Anything that leaves the count
    /// unbalanced either strands the user without a pointer or needs thousands of releases to recover.
    @MainActor
    func setSuspended(_ suspended: Bool) {
        let changed = state.withLockUnchecked { state -> Bool in
            guard state.isConstraining, state.isSuspended != suspended else { return false }
            state.isSuspended = suspended
            return true
        }
        guard changed else { return }

        if suspended {
            cursorHideTimer?.invalidate()
            cursorHideTimer = nil
            CGDisplayShowCursor(CGMainDisplayID())
        } else {
            // The cursor went wherever the user took it meanwhile.
            holdCursor()
            CGDisplayHideCursor(CGMainDisplayID())
            startCursorReassertTimer()
        }
        notifyPointer()
    }

    @MainActor
    func toggleCursorSpriteVisible() {
        state.withLockUnchecked { $0.pointerVisible.toggle() }
        notifyPointer()
    }

    /// Follows the captured window when it moves or resizes. The cursor is left where it is: the next event holds it in
    /// the window, and putting it there now would pull at a window being dragged by its title bar.
    @MainActor
    func update(window: CGRect, displays: [CGRect]) {
        state.withLockUnchecked { $0.mapping = PointerMapping(window: window, displays: displays) }
        notifyPointer()
    }

    // MARK: - Pointer

    /// Puts the cursor in the window if it is not, and takes where it is as the pointer.
    @MainActor
    private func holdCursor() {
        guard let location = CGEvent(source: nil)?.location else { return }
        let held = state.withLockUnchecked { state -> CGPoint? in
            guard let held = state.mapping.constrain(location) else { return nil }
            state.pointer = held
            return held
        }
        if let held, held != location { CGWarpMouseCursorPosition(held) }
    }

    /// Where the pointer is, as a fraction of the captured window, or `nil` when it is not drawn.
    private func pointerFraction() -> CGPoint? {
        state.withLockUnchecked { state in
            guard state.isConstraining, !state.isSuspended, state.pointerVisible else { return nil }
            return state.mapping.fraction(of: state.pointer)
        }
    }

    @MainActor
    private func notifyPointer() {
        state.withLockUnchecked { $0.drawPending = false }
        onPointerChange?(pointerFraction())
    }

    // MARK: - Event tap (tap thread)

    /// Runs for every mouse event of the session, which waits for it, so it stays short.
    private func handle(type: CGEventType, event: CGEvent) {
        // The system takes a tap away that is too slow to answer, or when the user's input is
        // being redirected. Left alone it stays off for the rest of the session and the pointer
        // silently stops being held.
        if type == .tapDisabledByTimeout || type == .tapDisabledByUserInput {
            if let tap = state.withLockUnchecked({ $0.eventTap }) { CGEvent.tapEnable(tap: tap, enable: true) }
            return
        }

        let incoming = event.location
        let outcome = state.withLockUnchecked { state -> (held: CGPoint, draws: Bool)? in
            guard state.isConstraining, !state.isSuspended, let held = state.mapping.constrain(incoming) else { return nil }
            state.pointer = held
            let draws = !state.drawPending
            state.drawPending = true
            return (held, draws)
        }
        guard let outcome else { return }

        if outcome.held != incoming {
            event.location = outcome.held
            CGWarpMouseCursorPosition(outcome.held)
        }
        if outcome.draws {
            DispatchQueue.main.async { [self] in
                MainActor.assumeIsolated { self.notifyPointer() }
            }
        }
    }

    // MARK: - System cursor

    @MainActor
    private func startCursorReassertTimer() {
        cursorHideTimer?.invalidate()
        let timer = Timer(timeInterval: Self.cursorReassertInterval, repeats: true) { _ in
            CGDisplayShowCursor(CGMainDisplayID())
            CGDisplayHideCursor(CGMainDisplayID())
        }
        RunLoop.current.add(timer, forMode: .common)
        cursorHideTimer = timer
    }

    /// Lets this process hide the cursor while another app is frontmost. There is no public API for
    /// it; the connection property is the one the system's own tools use. The functions are looked up
    /// by their SkyLight names too, which are the ones the window server's own framework exports.
    private static func enableBackgroundCursorControl() {
        typealias MainConnectionFunction = @convention(c) () -> Int32
        typealias SetPropertyFunction = @convention(c) (Int32, Int32, CFString, CFTypeRef) -> Int32
        func symbol(_ names: String...) -> UnsafeMutableRawPointer? {
            names.lazy.compactMap { dlsym(UnsafeMutableRawPointer(bitPattern: -2), $0) }.first
        }
        guard let main = symbol("CGSMainConnectionID", "SLSMainConnectionID"),
              let set = symbol("CGSSetConnectionProperty", "SLSSetConnectionProperty") else { return }
        let connection = unsafeBitCast(main, to: MainConnectionFunction.self)()
        _ = unsafeBitCast(set, to: SetPropertyFunction.self)(connection, connection, "SetsCursorInBackground" as CFString, kCFBooleanTrue)
    }
}

/// The thread the event tap runs on: a run loop and nothing else, started the first time it is wanted and kept for the life
/// of the process.
private final class EventTapThread: Thread, @unchecked Sendable {
    private let running = DispatchSemaphore(value: 0)
    private var loop: CFRunLoop?
    private var started = false

    override init() {
        super.init()
        name = "com.metalgoose.eventtap"
        qualityOfService = .userInteractive
    }

    override func main() {
        loop = CFRunLoopGetCurrent()
        // A run loop with nothing scheduled returns at once. The port keeps it waiting for the tap.
        RunLoop.current.add(Port(), forMode: .default)
        running.signal()
        while true {
            autoreleasepool {
                _ = RunLoop.current.run(mode: .default, before: .distantFuture)
            }
        }
    }

    /// The thread's run loop, starting the thread the first time. Main thread only.
    @MainActor
    func runLoop() -> CFRunLoop? {
        if !started {
            started = true
            start()
            running.wait()
        }
        return loop
    }
}
