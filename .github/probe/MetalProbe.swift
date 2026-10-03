// What the machine a workflow runs on can tell about the experiments that need a GPU: whether MetalFX can write the
// upscale straight into a drawable, and which VideoToolbox processors exist. Run with Metal's validation on, so a texture
// MetalFX does not take shows as an error rather than as a wrong image.
import AppKit
import Metal
import MetalFX
import QuartzCore
import VideoToolbox

guard let device = MTLCreateSystemDefaultDevice(), let queue = device.makeCommandQueue() else {
    print("PROBE no Metal device"); exit(0)
}
print("PROBE device:", device.name, "apple9:", device.supportsFamily(.apple9), "apple7:", device.supportsFamily(.apple7))
print("PROBE MetalFX spatial supported:", MTLFXSpatialScalerDescriptor.supportsDevice(device))
print("PROBE MetalFX frame interpolator supported:", MTLFXFrameInterpolatorDescriptor.supportsDevice(device))
print("PROBE VT low-latency interpolation supported:", VTLowLatencyFrameInterpolationConfiguration.isSupported)
print("PROBE VT low-latency super resolution supported:", VTLowLatencySuperResolutionScalerConfiguration.isSupported)

let layer = CAMetalLayer()
layer.device = device
layer.pixelFormat = .bgra8Unorm
layer.framebufferOnly = false
layer.drawableSize = CGSize(width: 1920, height: 1080)
guard let drawable = layer.nextDrawable() else { print("PROBE no drawable"); exit(0) }
let target = drawable.texture
print("PROBE drawable storageMode:", target.storageMode.rawValue, "(0 shared, 1 managed, 2 private, 3 memoryless) usage:",
      target.usage.rawValue)

guard MTLFXSpatialScalerDescriptor.supportsDevice(device) else { exit(0) }
let descriptor = MTLFXSpatialScalerDescriptor()
descriptor.inputWidth = 960
descriptor.inputHeight = 540
descriptor.outputWidth = 1920
descriptor.outputHeight = 1080
descriptor.colorTextureFormat = .bgra8Unorm
descriptor.outputTextureFormat = .bgra8Unorm
descriptor.colorProcessingMode = .perceptual
guard let scaler = descriptor.makeSpatialScaler(device: device) else { print("PROBE no scaler"); exit(0) }
print("PROBE scaler outputTextureUsage:", scaler.outputTextureUsage.rawValue, "colorTextureUsage:", scaler.colorTextureUsage.rawValue)

let inputDescriptor = MTLTextureDescriptor.texture2DDescriptor(pixelFormat: .bgra8Unorm, width: 960, height: 540, mipmapped: false)
inputDescriptor.usage = [.shaderRead, .shaderWrite, .renderTarget]
inputDescriptor.storageMode = .private
let input = device.makeTexture(descriptor: inputDescriptor)!

func run(into output: MTLTexture, label: String) {
    let buffer = queue.makeCommandBuffer()!
    scaler.colorTexture = input
    scaler.outputTexture = output
    scaler.encode(commandBuffer: buffer)
    buffer.commit()
    buffer.waitUntilCompleted()
    print("PROBE \(label): status \(buffer.status.rawValue) error \(String(describing: buffer.error)) gpu \(String(format: "%.3f", (buffer.gpuEndTime - buffer.gpuStartTime) * 1000)) ms")
}

let privateDescriptor = MTLTextureDescriptor.texture2DDescriptor(pixelFormat: .bgra8Unorm, width: 1920, height: 1080, mipmapped: false)
privateDescriptor.usage = [.shaderRead, .shaderWrite, .renderTarget]
privateDescriptor.storageMode = .private
run(into: device.makeTexture(descriptor: privateDescriptor)!, label: "MetalFX into a private texture")
run(into: target, label: "MetalFX into the drawable")
print("PROBE done")
