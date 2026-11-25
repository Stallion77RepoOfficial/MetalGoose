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
    static let lumaCount = gridSize * gridSize

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
        var luma = [UInt8](repeating: 0, count: lumaCount)
        guard let result = sample(surface, luma: &luma) else { return nil }
        return FrameSample(signature: result.signature, luma: luma)
    }

    /// Samples `surface` into a caller-owned `luma` buffer (`lumaCount` bytes) under a single IOSurface lock,
    /// optionally running `ContentResolution.count` on the same locked base pointer.
    static func sample(_ surface: IOSurfaceRef, luma: inout [UInt8],
                       testContent: Bool = false) -> (signature: UInt64, contentCount: ContentResolution.Count?)? {
        guard IOSurfaceLock(surface, .readOnly, nil) == kIOReturnSuccess else { return nil }
        defer { IOSurfaceUnlock(surface, .readOnly, nil) }

        let width = IOSurfaceGetWidth(surface)
        let height = IOSurfaceGetHeight(surface)
        let bytesPerRow = IOSurfaceGetBytesPerRow(surface)
        guard width > 0, height > 0, bytesPerRow >= width * 4 else { return nil }

        let base = IOSurfaceGetBaseAddress(surface).assumingMemoryBound(to: UInt8.self)
        if luma.count != lumaCount { luma = [UInt8](repeating: 0, count: lumaCount) }

        let blockArea = blockSpan * blockSpan
        var hash: UInt64 = 0xcbf29ce484222325

        // Precompute the 64 column coordinates once rather than dividing 4,096 times inside the loop.
        var xCoords = [Int](repeating: 0, count: gridSize)
        for gx in 0..<gridSize { xCoords[gx] = (width - 1) * gx / (gridSize - 1) }

        luma.withUnsafeMutableBufferPointer { lumaBuffer in
            for gy in 0..<gridSize {
                let y = (height - 1) * gy / (gridSize - 1)
                let y1 = min(height - 1, y + 1)
                let row0 = base + y * bytesPerRow
                let row1 = base + y1 * bytesPerRow
                let rowBase = gy * gridSize

                for gx in 0..<gridSize {
                    let x = xCoords[gx]
                    let p00 = row0 + x * 4
                    let value = UInt32(p00[0]) | (UInt32(p00[1]) << 8)
                              | (UInt32(p00[2]) << 16) | (UInt32(p00[3]) << 24)
                    hash = (hash ^ UInt64(value)) &* 0x100000001b3

                    let x1Offset = min(width - 1, x + 1) * 4
                    let p01 = row0 + x1Offset
                    let p10 = row1 + x * 4
                    let p11 = row1 + x1Offset
                    let sum = Int((UInt16(p00[2]) * 54 + UInt16(p00[1]) * 183 + UInt16(p00[0]) * 19) >> 8)
                            + Int((UInt16(p01[2]) * 54 + UInt16(p01[1]) * 183 + UInt16(p01[0]) * 19) >> 8)
                            + Int((UInt16(p10[2]) * 54 + UInt16(p10[1]) * 183 + UInt16(p10[0]) * 19) >> 8)
                            + Int((UInt16(p11[2]) * 54 + UInt16(p11[1]) * 183 + UInt16(p11[0]) * 19) >> 8)
                    lumaBuffer[rowBase + gx] = UInt8(min(255, sum / blockArea))
                }
            }
        }

        let contentCount = testContent && width >= 8 && height >= 8
            ? ContentResolution.count(base, width: width, height: height, bytesPerRow: bytesPerRow)
            : nil
        return (hash, contentCount)
    }
}
