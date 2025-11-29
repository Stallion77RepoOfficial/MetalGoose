import Foundation
import VideoToolbox
import CoreVideo

/// Prepares frames for Metal with VideoToolbox (scaling, formatting, color space)
final class FrameProcessor {
    private var transferSession: VTPixelTransferSession?
    
    // Metal-compatible output attributes
    private let destinationAttributes: CFDictionary = [
        kCVPixelBufferMetalCompatibilityKey: true,
        kCVPixelBufferIOSurfacePropertiesKey: [:] as CFDictionary
    ] as CFDictionary
    
    init() {
        // Create the transfer session (hardware accelerated)
        let status = VTPixelTransferSessionCreate(allocator: kCFAllocatorDefault, pixelTransferSessionOut: &transferSession)
        if status != kCVReturnSuccess || transferSession == nil {
            fatalError("Failed to create VTPixelTransferSession. Error code: \(status)")
        }
    }
    
    /// Prepare the buffer, optionally scaling and fixing its format.
    func prepare(buffer: CVPixelBuffer, scalingMode: CFString) -> CVPixelBuffer {
        guard let session = transferSession else {
            fatalError("VTPixelTransferSession is invalid.")
        }
        
        let width = CVPixelBufferGetWidth(buffer)
        let height = CVPixelBufferGetHeight(buffer)
        
        // Explicitly create a destination buffer.
        // We do not fallback to the original buffer to ensure the format is strictly BGRA for MetalFX.
        var dstBuffer: CVPixelBuffer?
        let status = CVPixelBufferCreate(
            kCFAllocatorDefault,
            width,
            height,
            kCVPixelFormatType_32BGRA, // Ideal format for MetalFX
            destinationAttributes,
            &dstBuffer
        )
        
        guard status == kCVReturnSuccess, let destination = dstBuffer else {
            fatalError("Failed to create CVPixelBuffer for frame processing. Error code: \(status)")
        }
        
        // Configure VideoToolbox transfer and scaling settings
        VTSessionSetProperty(session, key: kVTPixelTransferPropertyKey_ScalingMode, value: scalingMode)
        
        // Execute the transfer using the GPU/Media Engine path
        let transferStatus = VTPixelTransferSessionTransferImage(session, from: buffer, to: destination)
        if transferStatus != kCVReturnSuccess {
            fatalError("VTPixelTransferSession failed to transfer image. Error code: \(transferStatus)")
        }
        
        return destination
    }
}
