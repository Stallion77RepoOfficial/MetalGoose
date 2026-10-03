import Foundation
@preconcurrency import Metal

/// Brings a generated image back toward the captures it was made from, where they barely differ
/// (`StabilityBlend`, `blendTowardCaptures`). Owned by the render thread, which is the only one to call it.
final class StabilityBlender {

    private let gpu: GPUContext

    /// The blended images. One is read by the presentation pass of the command buffer that follows its blend, and
    /// the next callback's blend must not overwrite it before that has run, which three slots in turn guarantee.
    private var textures: [MTLTexture?] = Array(repeating: nil, count: 3)
    private var turn = 0

    init(gpu: GPUContext) {
        self.gpu = gpu
    }

    func reset() {
        textures = Array(repeating: nil, count: textures.count)
        turn = 0
    }

    /// `generated`, blended with the captures `previous` and `next` it sits between at `phase` (0 at `previous`, 1 at
    /// `next`), encoded on `commandBuffer`, at the captures' size: an image the Neural Engine made at another size is
    /// enlarged as it is blended, and one it made in planes is turned into colour. Nil when the two captures do not have one
    /// size, which no pair of one session does; the caller shows the capture.
    func blend(_ generated: GeneratedImages.Source, previous: MTLTexture, next: MTLTexture, phase: Double, tolerance: Float,
               commandBuffer: MTLCommandBuffer) -> MTLTexture? {
        let (width, height) = (previous.width, previous.height)
        guard next.width == width, next.height == height,
              let output = gpu.ensureTexture(&textures[turn], width: width, height: height),
              let encoder = commandBuffer.makeComputeCommandEncoder() else { return nil }
        turn = (turn + 1) % textures.count

        var parameters = StabilityBlendParams(phase: Float(phase), tolerance: tolerance)
        let pipeline: MTLComputePipelineState
        let captures: Int
        switch generated {
        case .colour(let image):
            pipeline = gpu.pipelines.stabilityBlend
            encoder.setTexture(image, index: 0)
            captures = 1
        case .planes(let luma, let chroma):
            pipeline = luma.width == width && luma.height == height
                ? gpu.pipelines.stabilityBlendFromYUV : gpu.pipelines.stabilityBlendFromYUVResampled
            encoder.setTexture(luma, index: 0)
            encoder.setTexture(chroma, index: 1)
            captures = 2
        }
        encoder.setComputePipelineState(pipeline)
        encoder.setTexture(previous, index: captures)
        encoder.setTexture(next, index: captures + 1)
        encoder.setTexture(output, index: captures + 2)
        encoder.setBytes(&parameters, length: MemoryLayout<StabilityBlendParams>.stride, index: 0)
        let tileWidth = Int(MG_BLEND_TILE_WIDTH)
        let tileHeight = Int(MG_BLEND_TILE_HEIGHT)
        encoder.dispatchThreadgroups(
            MTLSize(width: (width + tileWidth - 1) / tileWidth, height: (height + tileHeight - 1) / tileHeight, depth: 1),
            threadsPerThreadgroup: MTLSize(width: tileWidth, height: tileHeight, depth: 1))
        encoder.endEncoding()
        return output
    }
}
