import Foundation
import IOSurface
import Darwin

/// A cheap fingerprint of a captured surface: a hash that selects candidates for exact
/// duplicate verification, and a coarse luma grid that tells how
/// far the picture moved.
///
/// Generic by construction: ScreenCaptureKit's own dirty rectangles are not used, because
/// many apps never report empty ones, so a frame that did not change cannot be told apart
/// by them. Whether pixels differ is the only signal every app supports.
struct FrameSample {
    let signature: UInt64
    let luma: [UInt8]
}

enum FrameSampler {
    private static let gridSize = 64

    /// A fixed sample budget, deliberately not derived from the frame size: this runs on the
    /// capture thread for every frame, so its cost must stay flat as resolution rises. A
    /// frame whose signature matches is verified against the retained preceding surface;
    /// detail between grid points must not be discarded as a duplicate.
    ///
    /// The two outputs want opposite things from the same walk. The hash wants the sharpest
    /// sample it can get, so it reads the grid point itself. The luma statistic feeds a
    /// scene-cut test and wants the opposite: a single pixel per grid point aliases badly
    /// once the frame is much larger than the grid, and detail finer than the sample spacing
    /// — a pixel-scale checkerboard, dense text — swings a sample the full range as it
    /// moves, which reads as a cut in content that never cut. Averaging a small block around
    /// each point resolves that structure into its mean at a cost that is still a constant
    /// multiple of the grid rather than of the frame.
    ///
    /// The block is that multiplier, and this walk runs on the capture thread inside an
    /// IOSurface lock for every frame the stream delivers — up to 240 a second — and before
    /// the duplicate test that discards most of them. A 4x4 block made it 65k strided reads
    /// per frame; 2x2 resolves the same pixel-scale structure for a quarter of that, since
    /// anything finer than two pixels is what the aliasing was.
    private static let blockSpan = 2

    /// Verify equal fingerprints against retained BGRA pixels without an image copy.
    /// Padding is excluded and different strides are supported.
    static func isIdentical(_ surface: IOSurfaceRef, to previous: IOSurfaceRef) -> Bool {
        let width = IOSurfaceGetWidth(surface), height = IOSurfaceGetHeight(surface)
        guard width == IOSurfaceGetWidth(previous), height == IOSurfaceGetHeight(previous),
              width > 0, height > 0 else { return false }
        if IOSurfaceGetID(surface) == IOSurfaceGetID(previous) { return true }
        guard IOSurfaceLock(surface, .readOnly, nil) == 0 else { return false }
        defer { IOSurfaceUnlock(surface, .readOnly, nil) }
        guard IOSurfaceLock(previous, .readOnly, nil) == 0 else { return false }
        defer { IOSurfaceUnlock(previous, .readOnly, nil) }
        let stride = IOSurfaceGetBytesPerRow(surface), oldStride = IOSurfaceGetBytesPerRow(previous)
        let bytes = width * 4
        guard stride >= bytes, oldStride >= bytes else { return false }
        let current = IOSurfaceGetBaseAddress(surface), old = IOSurfaceGetBaseAddress(previous)
        if stride == bytes, oldStride == bytes { return memcmp(current, old, bytes * height) == 0 }
        for y in 0..<height where memcmp(current + y * stride, old + y * oldStride, bytes) != 0 { return false }
        return true
    }

    static func sample(_ surface: IOSurfaceRef) -> FrameSample? {
        guard IOSurfaceLock(surface, .readOnly, nil) == kIOReturnSuccess else { return nil }
        defer { IOSurfaceUnlock(surface, .readOnly, nil) }

        let width = IOSurfaceGetWidth(surface)
        let height = IOSurfaceGetHeight(surface)
        let bytesPerRow = IOSurfaceGetBytesPerRow(surface)
        guard width > 0, height > 0, bytesPerRow > 0 else { return nil }

        let base = IOSurfaceGetBaseAddress(surface).assumingMemoryBound(to: UInt8.self)
        let blockArea = blockSpan * blockSpan
        var hash: UInt64 = 0xcbf29ce484222325
        var luma = [UInt8](repeating: 0, count: gridSize * gridSize)

        for gy in 0..<gridSize {
            let y = (height - 1) * gy / (gridSize - 1)
            let row = base + y * bytesPerRow
            for gx in 0..<gridSize {
                let x = (width - 1) * gx / (gridSize - 1)
                let pixel = row + x * 4
                let value = UInt32(pixel[0]) | (UInt32(pixel[1]) << 8) |
                            (UInt32(pixel[2]) << 16) | (UInt32(pixel[3]) << 24)
                hash = (hash ^ UInt64(value)) &* 0x100000001b3

                // Clamped so the block never leaves the surface at the right and bottom
                // edges, where the grid point is the last pixel.
                var sum = 0
                for by in 0..<blockSpan {
                    let blockRow = base + min(height - 1, y + by) * bytesPerRow
                    for bx in 0..<blockSpan {
                        let p = blockRow + min(width - 1, x + bx) * 4
                        sum += Int((UInt16(p[2]) * 54 + UInt16(p[1]) * 183 + UInt16(p[0]) * 19) >> 8)
                    }
                }
                luma[gy * gridSize + gx] = UInt8(min(255, sum / blockArea))
            }
        }
        return FrameSample(signature: hash, luma: luma)
    }
}
