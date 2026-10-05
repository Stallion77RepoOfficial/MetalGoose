import os

/// Copies of a frame share one lease. Its last owner returns the slot, including GPU
/// and ANE readers; retaining a texture alone does not reserve its contents.
final class BufferLease: Sendable {
    let index: Int
    private let release: @Sendable () -> Void
    fileprivate init(index: Int, release: @escaping @Sendable () -> Void) {
        self.index = index
        self.release = release
    }
    deinit { release() }
}

final class BufferLeasePool: Sendable {
    private struct State { var occupied: [Bool]; var next = 0 }
    private let state: OSAllocatedUnfairLock<State>
    init(capacity: Int) {
        precondition(capacity > 0)
        state = OSAllocatedUnfairLock(initialState: State(occupied: Array(repeating: false, count: capacity)))
    }
    /// Never waits for a reader. No free slot means this work should be skipped.
    func acquire() -> BufferLease? {
        let index = state.withLock { state -> Int? in
            for offset in state.occupied.indices {
                let index = (state.next + offset) % state.occupied.count
                guard !state.occupied[index] else { continue }
                state.occupied[index] = true
                state.next = (index + 1) % state.occupied.count
                return index
            }
            return nil
        }
        guard let index else { return nil }
        return BufferLease(index: index) { [state] in state.withLock { $0.occupied[index] = false } }
    }
}
