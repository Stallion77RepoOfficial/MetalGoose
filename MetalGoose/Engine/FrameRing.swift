import Foundation
@preconcurrency import Metal
import os

/// A capture that has been through the capture path and is waiting to be shown or used to
/// generate something.
struct FrameHistory: TimedFrame {
    let texture: MTLTexture
    let timestamp: CFTimeInterval
    let isSceneCut: Bool
}

/// The most recent captures, oldest first.
///
/// The render clock samples at most one capture interval behind the newest frame, so two
/// entries always bracket it; the rest is headroom for captures still in flight behind the
/// one being read.
final class FrameRing: @unchecked Sendable {
    static let capacity = GooseEngine.maxInFlight + 1

    private let frames = OSAllocatedUnfairLock<[FrameHistory]>(uncheckedState: [])

    func push(_ frame: FrameHistory) {
        frames.withLockUnchecked {
            $0.append(frame)
            if $0.count > Self.capacity { $0.removeFirst() }
        }
    }

    func snapshot() -> [FrameHistory] {
        frames.withLockUnchecked { $0 }
    }

    func clear() {
        frames.withLockUnchecked { $0.removeAll() }
    }
}
