import Foundation
import VideoToolbox
import CoreVideo

/// VideoToolbox kullanarak frame'i Metal için hazırlar (Ölçekleme/Formatlama/Renk Uzayı)
final class FrameProcessor {
    private var transferSession: VTPixelTransferSession?
    
    // Metal uyumlu çıktı özellikleri
    private let destinationAttributes: CFDictionary = [
        kCVPixelBufferMetalCompatibilityKey: true,
        kCVPixelBufferIOSurfacePropertiesKey: [:] as CFDictionary
    ] as CFDictionary
    
    init() {
        // Transfer oturumunu başlat (Donanım hızlandırmalı)
        VTPixelTransferSessionCreate(allocator: kCFAllocatorDefault, pixelTransferSessionOut: &transferSession)
    }
    
    /// Buffer'ı hazırlar. Gerekirse ön ölçekleme yapar veya formatı düzeltir.
    func prepare(buffer: CVPixelBuffer, scalingMode: CFString) -> CVPixelBuffer {
        guard let session = transferSession else { return buffer }
        
        let width = CVPixelBufferGetWidth(buffer)
        let height = CVPixelBufferGetHeight(buffer)
        
        // Eğer format zaten Metal dostuysa (BGRA) ve boyut değişmeyecekse işlem yapma (Zero-copy optimization)
        // Ancak çoğu oyun penceresi farklı formatta olabilir, o yüzden garantiye alıyoruz.
        
        var dstBuffer: CVPixelBuffer?
        let status = CVPixelBufferCreate(
            kCFAllocatorDefault,
            width,
            height,
            kCVPixelFormatType_32BGRA, // MetalFX için en iyi format
            destinationAttributes,
            &dstBuffer
        )
        
        guard status == kCVReturnSuccess, let destination = dstBuffer else {
            return buffer
        }
        
        // VideoToolbox ile transfer ve ölçekleme ayarları
        VTSessionSetProperty(session, key: kVTPixelTransferPropertyKey_ScalingMode, value: scalingMode)
        
        // Transfer işlemini gerçekleştir (GPU/Media Engine kullanır)
        VTPixelTransferSessionTransferImage(session, from: buffer, to: destination)
        
        return destination
    }
}

