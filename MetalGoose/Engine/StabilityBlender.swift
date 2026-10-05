import Foundation
@preconcurrency import Metal

/// Brings a generated image back toward the captures it was made from, where they barely differ
/// (`StabilityBlend`, `blendTowardCaptures`). Owned by the render thread, which is the only one to call it.
final class StabilityBlender {

    private let gpu: GPUContext

    /// The blended images. One is read by the presentation pass of the command buffer that follows its blend, and
    /// the next callback's blend must not overwrite it before that has run, which three slots in turn guarantee.
    private var textures: [(any MTLTexture)?] = Array(repeating: nil, count: 3)
    private var turn = 0

    init(gpu: GPUContext) {
        self.gpu = gpu
    }

    func reset() {
        textures = Array(repeating: nil, count: textures.count)
        turn = 0
    }

    /// `generated`, blended with the captures `previous` and `next` it sits between at `phase` (0 at `previous`, 1 at
    /// `next`), encoded on `command`, at the captures' size: an image the Neural Engine made at another size is
    /// enlarged as it is blended, and one it made in planes is turned into colour. Nil when the two captures do not have one
    /// size, which no pair of one session does; the caller shows the capture.
    func blend(_ generated: GeneratedImages.Source, previous: any MTLTexture, next: any MTLTexture, phase: Double,
               motion: Float, command: GPUCommand) -> (any MTLTexture)? {
        let (width, height) = (previous.width, previous.height)
        guard next.width == width, next.height == height,
              let output = gpu.ensureTexture(&textures[turn], width: width, height: height),
              let encoder = command.makeComputePass() else { return nil }
        turn = (turn + 1) % textures.count

        var parameters = StabilityBlendParams(phase: Float(phase), motion: motion, noiseFloor: StabilityBlend.noiseFloor)
        let pipeline: any MTLComputePipelineState
        let captures: Int
        switch generated {
        case .colour(let image, let lease, let lane, let written):
            pipeline = gpu.pipelines.stabilityBlend
            encoder.setTexture(image, index: 0)
            captures = 1
            command.wait(for: lane, value: written)
            command.retain(lease)
        case .planes(let frame):
            let (luma, chroma) = (frame.luma, frame.chroma)
            pipeline = luma.width == width && luma.height == height
                ? gpu.pipelines.stabilityBlendFromYUV : gpu.pipelines.stabilityBlendFromYUVResampled
            encoder.setTexture(luma, index: 0)
            encoder.setTexture(chroma, index: 1)
            captures = 2
            if let lease = frame.lease { command.retain(lease) }
            command.retain(frame.buffer)
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
