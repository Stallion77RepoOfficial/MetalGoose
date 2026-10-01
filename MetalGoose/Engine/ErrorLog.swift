import Foundation
import os

/// Errors raised off the main thread, held until the UI picks them up. Each distinct error
/// is reported once per capture session: a failure that recurs every frame is one alert, not
/// one hundred and twenty a second.
final class ErrorLog: @unchecked Sendable {
    private struct State {
        var reported: Set<String> = []
        var pending: [MGError] = []
    }
    private let state = OSAllocatedUnfairLock(initialState: State())

    func report(_ error: MGError) {
        state.withLock { state in
            guard state.reported.insert(error.message).inserted else { return }
            state.pending.append(error)
        }
    }

    func take() -> MGError? {
        state.withLock { $0.pending.isEmpty ? nil : $0.pending.removeFirst() }
    }

    func reset() {
        state.withLock { $0 = State() }
    }
}
