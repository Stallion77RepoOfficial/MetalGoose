import Foundation
import VideoToolbox
import CoreVideo

/// Handles high-quality buffer scaling via VideoToolbox to keep lossless fidelity before handing frames to Metal.
final class FrameProcessor {
    private var transferSession: VTPixelTransferSession?
    private let bufferAttributes: CFDictionary = [
        kCVPixelBufferIOSurfacePropertiesKey: [:] as CFDictionary
    ] as CFDictionary
    private let maxDimension: Int = 12288
    private let maxPixelBudget: Int = 120_000_000
    
    init() {
        VTPixelTransferSessionCreate(allocator: kCFAllocatorDefault, pixelTransferSessionOut: &transferSession)
    }
    
    func prepare(buffer: CVPixelBuffer, scaleFactor: Float, scalingMode: CFString) -> CVPixelBuffer {
        guard scaleFactor > 1.0,
              let session = transferSession else {
            return buffer
        }
        let baseWidth = CVPixelBufferGetWidth(buffer)
        let baseHeight = CVPixelBufferGetHeight(buffer)
        let pixelFormat = CVPixelBufferGetPixelFormatType(buffer)
        let safeScale = resolveSafeScaleFactor(requested: scaleFactor, width: baseWidth, height: baseHeight)
        if safeScale <= 1.001 {
            return buffer
        }
        let scaledWidth = max(1, Int(Float(baseWidth) * safeScale))
        let scaledHeight = max(1, Int(Float(baseHeight) * safeScale))
        
        var dstBuffer: CVPixelBuffer?
        CVPixelBufferCreate(kCFAllocatorDefault,
                            scaledWidth,
                            scaledHeight,
                            pixelFormat,
                            bufferAttributes,
                            &dstBuffer)
        guard let dst = dstBuffer else { return buffer }
        VTSessionSetProperty(session,
                     key: kVTPixelTransferPropertyKey_ScalingMode,
                     value: scalingMode)
        VTPixelTransferSessionTransferImage(session,
                            from: buffer,
                            to: dst)
        return dst
    }

    private func resolveSafeScaleFactor(requested: Float, width: Int, height: Int) -> Float {
        guard width > 0, height > 0 else { return 1.0 }
        let dimCapWidth = Float(maxDimension) / Float(width)
        let dimCapHeight = Float(maxDimension) / Float(height)
        let dimensionLimited = min(requested, dimCapWidth, dimCapHeight)
        let basePixels = width * height
        let pixelLimit = sqrt(Double(maxPixelBudget) / Double(max(1, basePixels)))
        let pixelLimited = min(dimensionLimited, Float(pixelLimit))
        return max(1.0, pixelLimited)
    }
}
