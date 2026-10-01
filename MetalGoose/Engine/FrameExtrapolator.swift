import Foundation
@preconcurrency import Metal

/// Produces the images between captures when nothing is waited for: the newest frame warped forward
/// along its motion.
///
/// Owned by the render thread alone. Interpolation is not here: it needs the next capture, so it runs
/// from the capture side as the pair completes (`MetalFXInterpolator`, `NeuralInterpolator`).
final class FrameExtrapolator {

    private let gpu: GPUContext

    private var extrapolated: MTLTexture?
    private var extrapolatedImage: (source: CFTimeInterval, step: Int)?

    init(gpu: GPUContext) {
        self.gpu = gpu
    }

    /// Drops everything. The captures the extrapolator held references to are gone.
    func reset() {
        extrapolated = nil
        extrapolatedImage = nil
    }

    /// Warps a capture forward along the motion field. `step` of `steps` is how far into the next capture
    /// interval the image sits, so step 0 would reproduce the source.
    ///
    /// The warp for a given capture and step is the same image every time it is asked for, so it is
    /// encoded once and held: re-warping for every present paid full cost for an image already
    /// produced.
    func extrapolate(source: FrameHistory, field: MotionField, step: Int, steps: Int,
                     commandBuffer: MTLCommandBuffer) -> MTLTexture? {
        guard let mask = source.staticMask,
              let output = gpu.ensureTexture(&extrapolated, width: source.texture.width, height: source.texture.height,
                                             usage: [.shaderRead, .shaderWrite, .renderTarget]) else { return nil }

        if let image = extrapolatedImage, image.source == source.timestamp, image.step == step {
            return output
        }

        guard let encoder = commandBuffer.makeComputeCommandEncoder() else { return nil }

        var phase = Float(step) / Float(steps)
        encoder.setComputePipelineState(gpu.pipelines.extrapolate)
        encoder.setTexture(source.texture, index: 0)
        encoder.setTexture(field.vectors, index: 1)
        encoder.setTexture(output, index: 2)
        encoder.setTexture(field.disagreement, index: 3)
        encoder.setTexture(mask, index: 4)
        encoder.setBytes(&phase, length: MemoryLayout<Float>.size, index: 0)
        encoder.setBuffer(field.global, offset: 0, index: 1)
        gpu.dispatch(gpu.pipelines.extrapolate, on: encoder, width: output.width, height: output.height)
        encoder.endEncoding()

        extrapolatedImage = (source.timestamp, step)
        return output
    }
}
