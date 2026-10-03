import Foundation

/// What Render Scale leaves of a window's size when frames are generated from it, and which engine takes it.
///
/// Generation works on the capture as it is, and the presentation step scales every image up to the overlay alike.
/// Bringing a reduced capture back to the window's own size first was measured to add nothing for MetalFX
/// interpolation (a 2560x1440 window at 50%, real 4K clips: within 0.4 dB of the real in-between frame either way), and
/// it costs a MetalFX pass per capture and every later stage at four times the pixels.
///
/// The Neural Engine is the exception. On the restored frame it came out 2 to 3 dB closer to the real image than on the
/// capture, at no more GPU; and on the capture it came out 3 to 5 dB further from it than MetalFX did on the same
/// capture. So it takes the window's own size, restored, when that fits it. A window it does not fit is not restored: the
/// Neural Engine works on the capture, shrunk to what it takes where that is more (`NeuralSizes`).
///
/// Decided from the sizes and the settings alone, never from whether the processor has failed: a restore that came and
/// went with a failure would change the size of every frame each time, and every change of size starts the pipeline
/// over.
enum CaptureRestore {

    /// Whether a reduced capture is brought back to the window's own size: when frames are being generated, which the
    /// Neural Engine is to make where it can, and the window's own size fits it.
    static func isRestored(upscaling: Bool, generating: Bool, isReduced: Bool, neuralEngineFitsWindow: Bool) -> Bool {
        upscaling && generating && isReduced && neuralEngineFitsWindow
    }
}
