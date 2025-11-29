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
        VTPixelTransferSessionCreate(allocator: kCFAllocatorDefault, pixelTransferSessionOut: &transferSession)
    }
    
    /// Prepare the buffer, optionally scaling and fixing its format.
    func prepare(buffer: CVPixelBuffer, scalingMode: CFString) -> CVPixelBuffer {
        guard let session = transferSession else { return buffer }
        
        let width = CVPixelBufferGetWidth(buffer)
        let height = CVPixelBufferGetHeight(buffer)
        
        // If the buffer is already Metal-friendly (BGRA) and the size won't change, skip extra work.
        // Many window captures differ, so we create a safe buffer every time.
        
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
            return buffer
        }
        
        // Configure VideoToolbox transfer and scaling settings
        VTSessionSetProperty(session, key: kVTPixelTransferPropertyKey_ScalingMode, value: scalingMode)
        
        // Execute the transfer using the GPU/Media Engine path
        VTPixelTransferSessionTransferImage(session, from: buffer, to: destination)
        
        return destination
    }
}

