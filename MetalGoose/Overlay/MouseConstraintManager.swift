import AppKit
import CoreGraphics
import os

/// Keeps the physical mouse inside the captured window while the overlay magnifies it.
///
/// The overlay is larger than the window beneath it, so the pointer the user sees and the pointer
/// the game receives are at different places. An event tap rewrites each mouse event so the game
/// sees the position that corresponds to where the user is pointing in the overlay, the system
/// pointer is hidden, and the overlay draws its own where the user expects it.
///
/// Needs Accessibility: an active event tap is how the events are rewritten.
final class MouseConstraintManager: @unchecked Sendable {
    static let shared = MouseConstraintManager()

    private init() {}

    private struct State {
        var eventTap: CFMachPort?
        var runLoopSource: CFRunLoopSource?
        var sourceRect: CGRect = .zero
        var displayBounds: CGRect = .zero
        var virtualPosition: CGPoint = .zero
        var lastMappedPoint: CGPoint = .zero
        var isConstraining = false
        var pointerVisible = true

        /// Set while some app other than the capture target is frontmost. The constraint exists to
        /// hold the pointer inside the captured window while the user is playing, and both halves of
        /// it are hostile once they have switched away: the warp drags the pointer back into the
        /// game on the first mouse movement, and the hide leaves the whole session without a visible
        /// cursor. Between them, switching away with Cmd+Tab looked like the game was pulling focus
        /// back, and nothing else on screen could be used or even seen.
        var isSuspended = false
    }
    private let state = OSAllocatedUnfairLock(uncheckedState: State())

    private var cursorHideTimer: Timer?

    var isConstraining: Bool { state.withLockUnchecked { $0.isConstraining } }

    /// Called on the main thread, where the tap runs, whenever the pointer sprite should move: with
    /// where it is as a fraction of the captured window, measured from its top-left corner, or `nil`
    /// when it should not be drawn.
    @MainActor var onPointerChange: ((CGPoint?) -> Void)?

    /// The hide is reference counted per connection, and each tick is two synchronous window-server
    /// round trips on the main run loop — the same loop that services the mouse events themselves.
    /// Nothing between ticks changes the cursor state, so a faster rate buys nothing except how
    /// quickly a cursor another process revealed goes away again.
    private static let cursorReassertInterval: TimeInterval = 0.5

    // MARK: - Starting and stopping

    @MainActor
    func startConstraining(sourceRect: CGRect, displayBounds: CGRect) {
        let alreadyOn = state.withLockUnchecked { state -> Bool in
            state.sourceRect = sourceRect
            state.displayBounds = displayBounds
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

        guard let tap = CGEvent.tapCreate(tap: .cgSessionEventTap, place: .headInsertEventTap, options: .defaultTap,
                                          eventsOfInterest: eventMask, callback: callback,
                                          userInfo: Unmanaged.passUnretained(self).toOpaque()) else { return }

        let source = CFMachPortCreateRunLoopSource(kCFAllocatorDefault, tap, 0)
        CFRunLoopAddSource(CFRunLoopGetCurrent(), source, .commonModes)
        CGEvent.tapEnable(tap: tap, enable: true)

        state.withLockUnchecked { state in
            state.eventTap = tap
            state.runLoopSource = source
            state.virtualPosition = CGPoint(x: displayBounds.midX, y: displayBounds.midY)
            state.lastMappedPoint = CGPoint(x: sourceRect.midX, y: sourceRect.midY)
            state.isConstraining = true
        }

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
        if let source = previous.runLoopSource {
            CFRunLoopRemoveSource(CFRunLoopGetCurrent(), source, .commonModes)
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

    /// Follows the captured window when it moves or resizes.
    @MainActor
    func update(sourceRect: CGRect, displayBounds: CGRect) {
        state.withLockUnchecked {
            $0.sourceRect = sourceRect
            $0.displayBounds = displayBounds
        }
        notifyPointer()
    }

    // MARK: - Pointer

    /// Where the pointer is, as a fraction of the captured window, or `nil` when it is not drawn.
    private func pointerFraction() -> CGPoint? {
        state.withLockUnchecked { state in
            guard state.isConstraining, !state.isSuspended, state.pointerVisible,
                  state.sourceRect.width > 0, state.sourceRect.height > 0 else { return nil }
            return CGPoint(x: (state.lastMappedPoint.x - state.sourceRect.minX) / state.sourceRect.width,
                           y: (state.lastMappedPoint.y - state.sourceRect.minY) / state.sourceRect.height)
        }
    }

    @MainActor
    private func notifyPointer() {
        onPointerChange?(pointerFraction())
    }

    // MARK: - Event tap

    /// Runs on the main run loop for every mouse event, so it stays short.
    private func handle(type: CGEventType, event: CGEvent) {
        // The system takes a tap away that is too slow to answer, or when the user's input is
        // being redirected. Left alone it stays off for the rest of the session and the pointer
        // silently stops being remapped.
        if type == .tapDisabledByTimeout || type == .tapDisabledByUserInput {
            if let tap = state.withLockUnchecked({ $0.eventTap }) { CGEvent.tapEnable(tap: tap, enable: true) }
            return
        }

        let mapped: CGPoint? = state.withLockUnchecked { state in
            guard !state.isSuspended, state.sourceRect.width > 0, state.sourceRect.height > 0,
                  state.displayBounds.width > 0, state.displayBounds.height > 0 else { return nil }
            let source = state.sourceRect
            let display = state.displayBounds

            switch type {
            case .mouseMoved, .leftMouseDragged, .rightMouseDragged, .otherMouseDragged:
                let dx = CGFloat(event.getIntegerValueField(.mouseEventDeltaX))
                let dy = CGFloat(event.getIntegerValueField(.mouseEventDeltaY))
                state.virtualPosition.x = min(max(display.minX, state.virtualPosition.x + dx), display.maxX)
                state.virtualPosition.y = min(max(display.minY, state.virtualPosition.y + dy), display.maxY)
                state.lastMappedPoint = CGPoint(
                    x: source.minX + (state.virtualPosition.x - display.minX) / display.width * source.width,
                    y: source.minY + (state.virtualPosition.y - display.minY) / display.height * source.height)
            default:
                break
            }
            return state.lastMappedPoint
        }

        guard let mapped else { return }
        event.location = mapped
        switch type {
        case .mouseMoved, .leftMouseDragged, .rightMouseDragged, .otherMouseDragged:
            CGWarpMouseCursorPosition(mapped)
            // The tap runs on the main run loop, so the sprite moves in the same turn of it.
            MainActor.assumeIsolated { notifyPointer() }
        default:
            break
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
    /// it; the connection property is the one the system's own tools use.
    private static func enableBackgroundCursorControl() {
        typealias MainConnectionFunction = @convention(c) () -> Int32
        typealias SetPropertyFunction = @convention(c) (Int32, Int32, CFString, CFTypeRef) -> Int32
        guard let main = dlsym(UnsafeMutableRawPointer(bitPattern: -2), "CGSMainConnectionID"),
              let set = dlsym(UnsafeMutableRawPointer(bitPattern: -2), "CGSSetConnectionProperty") else { return }
        let connection = unsafeBitCast(main, to: MainConnectionFunction.self)()
        _ = unsafeBitCast(set, to: SetPropertyFunction.self)(connection, connection, "SetsCursorInBackground" as CFString, kCFBooleanTrue)
    }
}
