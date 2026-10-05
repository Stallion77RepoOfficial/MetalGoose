import os

/// Counts actual presentations, excluding redraws and callbacks from a reset schedule.
final class PresentationCounter: Sendable {
    struct Window: Sendable, Equatable { var images = 0; var generated = 0 }
    private struct State { var epoch = 0; var window = Window() }
    private let state = OSAllocatedUnfairLock(initialState: State())
    var epoch: Int { state.withLock { $0.epoch } }
    @discardableResult
    func record(epoch: Int, isNewImage: Bool, isGenerated: Bool) -> Bool {
        state.withLock {
            guard $0.epoch == epoch else { return false }
            if isNewImage {
                $0.window.images += 1
                if isGenerated { $0.window.generated += 1 }
            }
            return true
        }
    }
    func takeWindow() -> Window {
        state.withLock { defer { $0.window = Window() }; return $0.window }
    }
    func reset() { state.withLock { $0.epoch &+= 1; $0.window = Window() } }
}
