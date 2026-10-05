import Foundation
@preconcurrency import Metal
import os

/// A capture that has been through the capture path and is waiting to be shown or used to
/// generate something.
struct FrameHistory: TimedFrame {
    let texture: MTLTexture
    /// When the capture reached the pipeline. It names the capture: the engines publish its pair's images under it, and
    /// measure their latency from it.
    let timestamp: CFTimeInterval
    /// When the compositor showed it — ScreenCaptureKit's presentation time — on the same clock: the moment its content
    /// stands for, on the display's refresh grid, without the wander of its delivery. The render clock plans on this.
    let presentationTime: CFTimeInterval
    let isSceneCut: Bool
    let validity = WorkValidity()
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
        frames.withLockUnchecked { $0.filter { $0.validity.isValid } }
    }

    func clear() {
        frames.withLockUnchecked { $0.removeAll() }
    }
}
