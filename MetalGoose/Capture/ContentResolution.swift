import Foundation
import IOSurface

/// Whether a window's content is drawn at half the window's pixels — one pixel a point on a display of two, as an app that
/// is not drawn for Retina displays is — and magnified to them.
///
/// The magnification is one of two kinds, and each leaves a relation between neighbouring pixels that detail drawn at the
/// display's own pixels does not keep to:
/// - Bilinear, samples a quarter and three quarters of the way between the content's, as the window server magnifies a
///   layer: measured, a one-pixel checkerboard drawn at one pixel a point came out at 159 and 96 at two, 0.625 and 0.375 of
///   255. Every content pixel `q[i]` is then given exactly by either neighbouring pair: `3·P[2i+1] − P[2i+2] =
///   3·P[2i] − P[2i−1] = 2·q[i]`.
/// - Nearest, each content pixel a block of two by two: `P[2i] = P[2i+1]`. A game under Wine without its Retina mode was
///   this, measured: every 2x2 block of its captured text was one colour.
///
/// Both across and down. Where a picture's edges all keep to one of them, the content is had back from it exactly
/// (`demagnifyNearest`, `demagnifyBilinear`), at a quarter of the pixels. ScreenCaptureKit asked for the content's size
/// does not give it: at the best resolution it magnifies and makes small again (a one-pixel checkerboard at a fifteenth of
/// its contrast), and at the nominal one it gave the content exactly where a layer was drawn at one pixel a point, but the
/// Wine game's softened, its edges' peaks a quarter down (186 to 140), measured.
enum ContentResolution {

    /// How content drawn at half a window's pixels was magnified to them, which says how it is had back exactly.
    enum Magnification: Sendable, Equatable {
        case nearest
        case bilinear
    }

    enum Verdict: Equatable {
        /// Drawn at half the window's pixels, and magnified so.
        case half(Magnification)
        /// Detail at the window's own pixels.
        case full
        /// Too few edges to tell, or too many that do not keep to either relation to be sure either way.
        case undecided
    }

    /// The places that can tell, and those of them that do not keep to a bilinear magnification's relation, and to a nearest
    /// one's.
    struct Count: Equatable {
        var edges = 0
        var breaks = 0
        var nearestBreaks = 0
    }

    /// Rows and columns looked at: one in `stride`, so that the test costs a fixed share of the frame (2.6 to 2.9 ms for one
    /// of 3024x1964 that is all edges, measured).
    private static let stride = 8
    /// How far apart neighbours are, in 8-bit steps of luma, for the place to be an edge that can tell.
    private static let edge = 24
    /// How far apart the two estimates of a content pixel may be, in 8-bit steps of twice its luma: rounding each channel of
    /// the magnified picture to eight bits moves the difference by up to 4.
    private static let tolerance = 6
    private static let fewestEdges = 500

    static func verdict(of count: Count) -> Verdict {
        guard count.edges >= fewestEdges else { return .undecided }
        let magnification: Magnification = count.nearestBreaks <= count.breaks ? .nearest : .bilinear
        let share = Double(min(count.breaks, count.nearestBreaks)) / Double(count.edges)
        if share <= 0.01 { return .half(magnification) }
        if share >= 0.05 { return .full }
        return .undecided
    }

    /// Counts a BGRA surface's edges and breaks, across rows and down columns.
    static func count(_ surface: IOSurfaceRef) -> Count? {
        guard IOSurfaceLock(surface, .readOnly, nil) == kIOReturnSuccess else { return nil }
        defer { IOSurfaceUnlock(surface, .readOnly, nil) }
        let width = IOSurfaceGetWidth(surface), height = IOSurfaceGetHeight(surface)
        let bytesPerRow = IOSurfaceGetBytesPerRow(surface)
        guard width >= 8, height >= 8, bytesPerRow >= width * 4 else { return nil }
        return count(IOSurfaceGetBaseAddress(surface).assumingMemoryBound(to: UInt8.self),
                     width: width, height: height, bytesPerRow: bytesPerRow)
    }

    static func count(_ base: UnsafePointer<UInt8>, width: Int, height: Int, bytesPerRow: Int) -> Count {
        let edge = Self.edge * 256, tolerance = Self.tolerance * 256
        var count = Count()
        // Content pixel i covers the picture's 2i and 2i+1; the first and the last are left out, where the magnification
        // clamps at the edge of the content.
        func test(_ line: UnsafeBufferPointer<Int>) {
            for i in 1..<(line.count / 2 - 1) {
                let l0 = line[2 * i - 1], l1 = line[2 * i], l2 = line[2 * i + 1], l3 = line[2 * i + 2]
                guard max(abs(l1 - l0), abs(l2 - l1), abs(l3 - l2)) >= edge else { continue }
                count.edges += 1
                if abs((3 * l2 - l3) - (3 * l1 - l0)) > tolerance { count.breaks += 1 }
                if abs(l2 - l1) > tolerance { count.nearestBreaks += 1 }
            }
        }
        // Luma at 256 times the scale of a channel, unrounded, so that it is as linear in the channels as they are.
        func luma(_ p: UnsafePointer<UInt8>) -> Int { Int(p[2]) * 54 + Int(p[1]) * 183 + Int(p[0]) * 19 }
        var line = [Int](repeating: 0, count: max(width, height))
        line.withUnsafeMutableBufferPointer { buffer in
            for y in Swift.stride(from: 0, to: height, by: stride) {
                let row = base + y * bytesPerRow
                for x in 0..<width { buffer[x] = luma(row + x * 4) }
                test(UnsafeBufferPointer(rebasing: buffer[0..<width]))
            }
            for x in Swift.stride(from: 0, to: width, by: stride) {
                let column = base + x * 4
                for y in 0..<height { buffer[y] = luma(column + y * bytesPerRow) }
                test(UnsafeBufferPointer(rebasing: buffer[0..<height]))
            }
        }
        return count
    }
}
