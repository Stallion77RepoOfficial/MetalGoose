import os

/// Same-queue GPU readers can be encoded before the producer completes. If it fails,
/// its history and downstream results must no longer be published.
final class WorkValidity: Sendable {
    private let valid = OSAllocatedUnfairLock(initialState: true)
    var isValid: Bool { valid.withLock { $0 } }
    func invalidate() { valid.withLock { $0 = false } }
}
