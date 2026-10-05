/// Serialises whole async transactions; actor isolation alone allows another update
/// to start while the first is awaiting ScreenCaptureKit.
actor AsyncSerialGate {
    private var busy = false
    private var waiting: [CheckedContinuation<Void, Never>] = []
    func perform<T: Sendable>(_ operation: @Sendable () async throws -> T) async rethrows -> T {
        if busy { await withCheckedContinuation { waiting.append($0) } }
        else { busy = true }
        defer {
            if waiting.isEmpty { busy = false }
            else { waiting.removeFirst().resume() }
        }
        return try await operation()
    }
}
