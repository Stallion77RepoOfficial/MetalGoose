// Upscaling quality, measured on the runner's GPU with the real MetalFX and the app's own sharpening kernel: each test image
// is reduced (by area, as a render that is anti-aliased, and by point, as one that is not), brought back to its size by each
// method, and scored against the original — PSNR and SSIM of luma. Timing on a virtual GPU means nothing; quality does.
//
// Methods: bilinear; MetalFX spatial alone, perceptual (as the app) and linear; VideoToolbox's low-latency super resolution
// (with -D VT_SR); the app's order, contrast-adaptive sharpening on the capture and MetalFX after
// it; and the other order, MetalFX and then the sharpening on its output — at the three strengths of the Sharpening picker.
import AppKit
import CoreGraphics
import ImageIO
import Metal
import MetalFX
#if VT_SR
import CoreMedia
import CoreVideo
import VideoToolbox
#endif

// MARK: - Images

struct Image {
    var width: Int
    var height: Int
    var rgba: [UInt8]
}

func load(_ url: URL) -> Image? {
    guard let source = CGImageSourceCreateWithURL(url as CFURL, nil),
          let image = CGImageSourceCreateImageAtIndex(source, 0, nil) else { return nil }
    return render(width: image.width, height: image.height) { $0.draw(image, in: CGRect(x: 0, y: 0, width: image.width, height: image.height)) }
}

func render(width: Int, height: Int, _ draw: (CGContext) -> Void) -> Image {
    var pixels = [UInt8](repeating: 0, count: width * height * 4)
    pixels.withUnsafeMutableBytes { bytes in
        let context = CGContext(data: bytes.baseAddress, width: width, height: height, bitsPerComponent: 8, bytesPerRow: width * 4,
                                space: CGColorSpace(name: CGColorSpace.sRGB)!,
                                bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue)!
        draw(context)
    }
    return Image(width: width, height: height, rgba: pixels)
}

/// What a game's interface is made of: text at several sizes, thin lines, a grid, small shapes, over a gradient.
func interfaceImage(width: Int, height: Int, seed: Int) -> Image {
    render(width: width, height: height) { context in
        let colours = [CGColor(red: 0.08, green: 0.1, blue: 0.16, alpha: 1), CGColor(red: 0.3, green: 0.22, blue: 0.12, alpha: 1)]
        let gradient = CGGradient(colorsSpace: CGColorSpace(name: CGColorSpace.sRGB), colors: colours as CFArray, locations: [0, 1])!
        context.drawLinearGradient(gradient, start: .zero, end: CGPoint(x: width, y: height), options: [])
        NSGraphicsContext.current = NSGraphicsContext(cgContext: context, flipped: false)
        var y = CGFloat(height) - 60
        var random = UInt64(seed * 7919 + 1)
        func next() -> CGFloat {
            random = random &* 6364136223846793005 &+ 1442695040888963407
            return CGFloat(random >> 40) / CGFloat(1 << 24)
        }
        for size in [11.0, 13.0, 16.0, 20.0, 28.0, 40.0] as [CGFloat] {
            let attributes: [NSAttributedString.Key: Any] = [.font: NSFont.systemFont(ofSize: size),
                                                             .foregroundColor: NSColor(white: 0.92, alpha: 1)]
            let text = "Quest log 12/40 — Inventory: 3x Potion, Iron Sword (+7), Gold 1,284 · HP 87% MP 42%"
            NSAttributedString(string: text, attributes: attributes).draw(at: CGPoint(x: 40, y: y))
            y -= size * 1.9
        }
        context.setLineWidth(1)
        for i in 0..<60 {
            context.setStrokeColor(CGColor(red: next(), green: next(), blue: next(), alpha: 1))
            let x = CGFloat(i) * CGFloat(width) / 60
            context.move(to: CGPoint(x: x, y: 0))
            context.addLine(to: CGPoint(x: x + CGFloat(height) * 0.4, y: CGFloat(height) * 0.45))
            context.strokePath()
        }
        for row in 0..<8 {
            for column in 0..<14 {
                let w: Int = 26 + Int(next() * 20), h: Int = 14 + Int(next() * 10)
                let rect = CGRect(x: 60 + column * 60, y: 40 + row * 34, width: w, height: h)
                context.setFillColor(CGColor(red: next(), green: next(), blue: next(), alpha: 1))
                if (row + column).isMultiple(of: 2) { context.fillEllipse(in: rect) } else { context.fill(rect) }
            }
        }
        context.setStrokeColor(CGColor(gray: 1, alpha: 0.6))
        for i in stride(from: CGFloat(width) * 0.55, to: CGFloat(width), by: 6) {
            context.move(to: CGPoint(x: i, y: 40))
            context.addLine(to: CGPoint(x: i, y: CGFloat(height) * 0.4))
            context.strokePath()
        }
        NSGraphicsContext.current = nil
    }
}

/// Area average to the smaller size, or the nearest pixel.
func reduce(_ image: Image, to width: Int, _ height: Int, byArea: Bool) -> Image {
    var out = [UInt8](repeating: 255, count: width * height * 4)
    let sx = Double(image.width) / Double(width), sy = Double(image.height) / Double(height)
    for y in 0..<height {
        for x in 0..<width {
            for c in 0..<3 {
                if byArea {
                    let x0 = Double(x) * sx, x1 = x0 + sx, y0 = Double(y) * sy, y1 = y0 + sy
                    var sum = 0.0, weight = 0.0
                    for yy in Int(y0)..<min(image.height, Int(y1.rounded(.up))) {
                        let wy = min(y1, Double(yy + 1)) - max(y0, Double(yy))
                        for xx in Int(x0)..<min(image.width, Int(x1.rounded(.up))) {
                            let w = wy * (min(x1, Double(xx + 1)) - max(x0, Double(xx)))
                            sum += w * Double(image.rgba[(yy * image.width + xx) * 4 + c])
                            weight += w
                        }
                    }
                    out[(y * width + x) * 4 + c] = UInt8(min(255, max(0, (sum / weight).rounded())))
                } else {
                    let xx = min(image.width - 1, Int((Double(x) + 0.5) * sx)), yy = min(image.height - 1, Int((Double(y) + 0.5) * sy))
                    out[(y * width + x) * 4 + c] = image.rgba[(yy * image.width + xx) * 4 + c]
                }
            }
        }
    }
    return Image(width: width, height: height, rgba: out)
}

// MARK: - Scores

func luma(_ image: Image) -> [Double] {
    var out = [Double](repeating: 0, count: image.width * image.height)
    for i in out.indices {
        let r: Double = 0.2126 * Double(image.rgba[i * 4])
        let g: Double = 0.7152 * Double(image.rgba[i * 4 + 1])
        let b: Double = 0.0722 * Double(image.rgba[i * 4 + 2])
        out[i] = r + g + b
    }
    return out
}

func psnr(_ a: [Double], _ b: [Double]) -> Double {
    let mse = zip(a, b).map { ($0 - $1) * ($0 - $1) }.reduce(0, +) / Double(a.count)
    return 10 * log10(255 * 255 / max(mse, 1e-9))
}

/// SSIM over 8x8 windows, stepped by 4.
func ssim(_ a: [Double], _ b: [Double], width: Int, height: Int) -> Double {
    let k1: Double = 0.01 * 255, k2: Double = 0.03 * 255
    let c1 = k1 * k1, c2 = k2 * k2
    var total = 0.0, count = 0.0
    for y in stride(from: 0, to: height - 8, by: 4) {
        for x in stride(from: 0, to: width - 8, by: 4) {
            var ma = 0.0, mb = 0.0
            for j in 0..<8 { for i in 0..<8 { ma += a[(y + j) * width + x + i]; mb += b[(y + j) * width + x + i] } }
            ma /= 64; mb /= 64
            var va = 0.0, vb = 0.0, cov = 0.0
            for j in 0..<8 {
                for i in 0..<8 {
                    let da = a[(y + j) * width + x + i] - ma, db = b[(y + j) * width + x + i] - mb
                    va += da * da; vb += db * db; cov += da * db
                }
            }
            va /= 63; vb /= 63; cov /= 63
            let numerator: Double = (2 * ma * mb + c1) * (2 * cov + c2)
            let denominator: Double = (ma * ma + mb * mb + c1) * (va + vb + c2)
            total += numerator / denominator
            count += 1
        }
    }
    return total / count
}

// MARK: - GPU

let device = MTLCreateSystemDefaultDevice()!
let queue = device.makeCommandQueue()!

func shaderSource() -> String {
    func read(_ path: String) -> String {
        (try? String(contentsOfFile: path, encoding: .utf8))?.split(separator: "\n", omittingEmptySubsequences: false)
            .filter { !$0.hasPrefix("#include \"") }.joined(separator: "\n") ?? ""
    }
    return read("MetalGoose/Shaders/ShaderTypes.h") + "\n" + read("MetalGoose/Shaders/ShaderCommon.h") + "\n"
        + read("MetalGoose/Shaders/PostProcess.metal")
}

let library = try! device.makeLibrary(source: shaderSource(), options: nil)
let sharpen = try! device.makeComputePipelineState(function: library.makeFunction(name: "contrastAdaptiveSharpening")!)

func texture(_ width: Int, _ height: Int, _ storage: MTLStorageMode) -> MTLTexture {
    let descriptor = MTLTextureDescriptor.texture2DDescriptor(pixelFormat: .rgba8Unorm, width: width, height: height, mipmapped: false)
    descriptor.usage = [.shaderRead, .shaderWrite, .renderTarget]
    descriptor.storageMode = storage
    return device.makeTexture(descriptor: descriptor)!
}

func upload(_ image: Image) -> MTLTexture {
    let shared = texture(image.width, image.height, .shared)
    image.rgba.withUnsafeBytes {
        shared.replace(region: MTLRegionMake2D(0, 0, image.width, image.height), mipmapLevel: 0, withBytes: $0.baseAddress!,
                       bytesPerRow: image.width * 4)
    }
    let gpu = texture(image.width, image.height, .private)
    let buffer = queue.makeCommandBuffer()!
    let blit = buffer.makeBlitCommandEncoder()!
    blit.copy(from: shared, to: gpu)
    blit.endEncoding()
    buffer.commit()
    buffer.waitUntilCompleted()
    return gpu
}

func download(_ texture: MTLTexture) -> Image {
    let shared = texture.storageMode == .shared ? texture : {
        let copy = device.makeTexture(descriptor: {
            let d = MTLTextureDescriptor.texture2DDescriptor(pixelFormat: .rgba8Unorm, width: texture.width, height: texture.height, mipmapped: false)
            d.storageMode = .shared
            return d
        }())!
        let buffer = queue.makeCommandBuffer()!
        let blit = buffer.makeBlitCommandEncoder()!
        blit.copy(from: texture, to: copy)
        blit.endEncoding()
        buffer.commit()
        buffer.waitUntilCompleted()
        return copy
    }()
    var bytes = [UInt8](repeating: 0, count: texture.width * texture.height * 4)
    bytes.withUnsafeMutableBytes {
        shared.getBytes($0.baseAddress!, bytesPerRow: texture.width * 4, from: MTLRegionMake2D(0, 0, texture.width, texture.height), mipmapLevel: 0)
    }
    return Image(width: texture.width, height: texture.height, rgba: bytes)
}

func run(_ encode: (MTLCommandBuffer) -> Void) {
    let buffer = queue.makeCommandBuffer()!
    encode(buffer)
    buffer.commit()
    buffer.waitUntilCompleted()
}

func encodeSharpen(_ input: MTLTexture, _ output: MTLTexture, strength: Float, _ buffer: MTLCommandBuffer) {
    let encoder = buffer.makeComputeCommandEncoder()!
    var strength = strength
    encoder.setComputePipelineState(sharpen)
    encoder.setTexture(input, index: 0)
    encoder.setTexture(output, index: 1)
    encoder.setBytes(&strength, length: 4, index: 0)
    let w = sharpen.threadExecutionWidth, h = sharpen.maxTotalThreadsPerThreadgroup / w
    encoder.dispatchThreadgroups(MTLSize(width: (output.width + w - 1) / w, height: (output.height + h - 1) / h, depth: 1),
                                 threadsPerThreadgroup: MTLSize(width: w, height: h, depth: 1))
    encoder.endEncoding()
}

func scaler(_ inW: Int, _ inH: Int, _ outW: Int, _ outH: Int, mode: MTLFXSpatialScalerColorProcessingMode = .perceptual) -> MTLFXSpatialScaler {
    let descriptor = MTLFXSpatialScalerDescriptor()
    descriptor.inputWidth = inW
    descriptor.inputHeight = inH
    descriptor.outputWidth = outW
    descriptor.outputHeight = outH
    descriptor.colorTextureFormat = .rgba8Unorm
    descriptor.outputTextureFormat = .rgba8Unorm
    descriptor.colorProcessingMode = mode
    return descriptor.makeSpatialScaler(device: device)!
}

func bilinear(_ image: Image, to width: Int, _ height: Int) -> Image {
    var out = [UInt8](repeating: 255, count: width * height * 4)
    for y in 0..<height {
        for x in 0..<width {
            let fx = (Double(x) + 0.5) * Double(image.width) / Double(width) - 0.5
            let fy = (Double(y) + 0.5) * Double(image.height) / Double(height) - 0.5
            let x0 = max(0, min(image.width - 1, Int(fx.rounded(.down)))), y0 = max(0, min(image.height - 1, Int(fy.rounded(.down))))
            let x1 = min(image.width - 1, x0 + 1), y1 = min(image.height - 1, y0 + 1)
            let ax = min(1, max(0, fx - Double(x0))), ay = min(1, max(0, fy - Double(y0)))
            for c in 0..<3 {
                func p(_ xx: Int, _ yy: Int) -> Double { Double(image.rgba[(yy * image.width + xx) * 4 + c]) }
                let top: Double = p(x0, y0) * (1 - ax) + p(x1, y0) * ax
                let bottom: Double = p(x0, y1) * (1 - ax) + p(x1, y1) * ax
                let v: Double = top * (1 - ay) + bottom * ay
                out[(y * width + x) * 4 + c] = UInt8(min(255, max(0, v.rounded())))
            }
        }
    }
    return Image(width: width, height: height, rgba: out)
}

#if VT_SR
// MARK: - VideoToolbox low-latency super resolution

func pixelBuffer(_ width: Int, _ height: Int, format: OSType, attributes: [String: Any] = [:]) -> CVPixelBuffer? {
    var attributes = attributes
    attributes[kCVPixelBufferIOSurfacePropertiesKey as String] = [:] as CFDictionary
    attributes[kCVPixelBufferMetalCompatibilityKey as String] = true
    attributes.removeValue(forKey: kCVPixelBufferPixelFormatTypeKey as String)
    attributes.removeValue(forKey: kCVPixelBufferWidthKey as String)
    attributes.removeValue(forKey: kCVPixelBufferHeightKey as String)
    var created: CVPixelBuffer?
    guard CVPixelBufferCreate(kCFAllocatorDefault, width, height, format, attributes as CFDictionary, &created) == kCVReturnSuccess,
          let buffer = created else { return nil }
    CVBufferSetAttachment(buffer, kCVImageBufferYCbCrMatrixKey, kCVImageBufferYCbCrMatrix_ITU_R_709_2, .shouldPropagate)
    CVBufferSetAttachment(buffer, kCVImageBufferColorPrimariesKey, kCVImageBufferColorPrimaries_ITU_R_709_2, .shouldPropagate)
    CVBufferSetAttachment(buffer, kCVImageBufferTransferFunctionKey, kCVImageBufferTransferFunction_ITU_R_709_2, .shouldPropagate)
    return buffer
}

func bgra(_ image: Image) -> CVPixelBuffer {
    let buffer = pixelBuffer(image.width, image.height, format: kCVPixelFormatType_32BGRA)!
    CVPixelBufferLockBaseAddress(buffer, [])
    let base = CVPixelBufferGetBaseAddress(buffer)!.assumingMemoryBound(to: UInt8.self)
    let row = CVPixelBufferGetBytesPerRow(buffer)
    for y in 0..<image.height {
        for x in 0..<image.width {
            let s = (y * image.width + x) * 4, d = y * row + x * 4
            base[d] = image.rgba[s + 2]; base[d + 1] = image.rgba[s + 1]; base[d + 2] = image.rgba[s]; base[d + 3] = 255
        }
    }
    CVPixelBufferUnlockBaseAddress(buffer, [])
    return buffer
}

func image(of buffer: CVPixelBuffer) -> Image {
    let width = CVPixelBufferGetWidth(buffer), height = CVPixelBufferGetHeight(buffer)
    CVPixelBufferLockBaseAddress(buffer, .readOnly)
    let base = CVPixelBufferGetBaseAddress(buffer)!.assumingMemoryBound(to: UInt8.self)
    let row = CVPixelBufferGetBytesPerRow(buffer)
    var rgba = [UInt8](repeating: 255, count: width * height * 4)
    for y in 0..<height {
        for x in 0..<width {
            let s = y * row + x * 4, d = (y * width + x) * 4
            rgba[d] = base[s + 2]; rgba[d + 1] = base[s + 1]; rgba[d + 2] = base[s]
        }
    }
    CVPixelBufferUnlockBaseAddress(buffer, .readOnly)
    return Image(width: width, height: height, rgba: rgba)
}

let transfer: VTPixelTransferSession = {
    var session: VTPixelTransferSession?
    VTPixelTransferSessionCreate(allocator: nil, pixelTransferSessionOut: &session)
    return session!
}()

func formats(_ attributes: [String: Any]) -> [OSType] {
    let value = attributes[kCVPixelBufferPixelFormatTypeKey as String]
    if let number = value as? NSNumber { return [number.uint32Value] }
    if let numbers = value as? [NSNumber] { return numbers.map(\.uint32Value) }
    return []
}

func fourCC(_ code: OSType) -> String {
    String(bytes: [24, 16, 8, 0].map { UInt8((code >> $0) & 0xFF) }, encoding: .ascii) ?? "\(code)"
}

var sessions: [String: (VTFrameProcessor, VTLowLatencySuperResolutionScalerConfiguration)?] = [:]
var reported: Set<String> = []

func superResolution(_ small: Image, to width: Int, _ height: Int) -> Image? {
    let factor = Float(width) / Float(small.width)
    let key = "\(small.width)x\(small.height)@\(factor)"
    if sessions[key] == nil {
        let made: VTLowLatencySuperResolutionScalerConfiguration? =
            VTLowLatencySuperResolutionScalerConfiguration(frameWidth: small.width, frameHeight: small.height, scaleFactor: factor)
        guard let configuration = made else {
            sessions[key] = .some(nil)
            print("UPSCALE VT: no configuration for \(key)")
            return nil
        }
        let session = VTFrameProcessor()
        do { try session.startSession(configuration: configuration) } catch {
            sessions[key] = .some(nil)
            print("UPSCALE VT: session for \(key) refused: \(error)")
            return nil
        }
        print("UPSCALE VT: session for \(key), source \(formats(configuration.sourcePixelBufferAttributes).map(fourCC)), "
              + "destination \(formats(configuration.destinationPixelBufferAttributes).map(fourCC))")
        sessions[key] = (session, configuration)
    }
    guard let entry = sessions[key], let pair = entry else { return nil }
    let (session, configuration) = pair
    guard let sourceFormat = formats(configuration.sourcePixelBufferAttributes).first,
          let destinationFormat = formats(configuration.destinationPixelBufferAttributes).first,
          let source = pixelBuffer(small.width, small.height, format: sourceFormat, attributes: configuration.sourcePixelBufferAttributes),
          let destination = pixelBuffer(width, height, format: destinationFormat, attributes: configuration.destinationPixelBufferAttributes)
    else { return nil }
    guard VTPixelTransferSessionTransferImage(transfer, from: bgra(small), to: source) == noErr else { return nil }
    let sourceFrame: VTFrameProcessorFrame? = VTFrameProcessorFrame(buffer: source, presentationTimeStamp: .zero)
    let destinationFrame: VTFrameProcessorFrame? = VTFrameProcessorFrame(buffer: destination, presentationTimeStamp: .zero)
    guard let sourceFrame, let destinationFrame else { return nil }
    let made: VTLowLatencySuperResolutionScalerParameters? =
        VTLowLatencySuperResolutionScalerParameters(sourceFrame: sourceFrame, destinationFrame: destinationFrame)
    guard let parameters = made else { return nil }
    final class Outcome: @unchecked Sendable { var error: Error? }
    let done = DispatchSemaphore(value: 0)
    let outcome = Outcome()
    session.process(parameters: parameters) { _, error in
        outcome.error = error
        done.signal()
    }
    guard done.wait(timeout: .now() + 10) == .success else {
        if reported.insert(key).inserted { print("UPSCALE VT: \(key) did not come back") }
        return nil
    }
    if let failure = outcome.error {
        if reported.insert(key).inserted { print("UPSCALE VT: \(key) failed: \(failure)") }
        return nil
    }
    let out = pixelBuffer(width, height, format: kCVPixelFormatType_32BGRA)!
    guard VTPixelTransferSessionTransferImage(transfer, from: destination, to: out) == noErr else { return nil }
    return image(of: out)
}
#endif

// MARK: - The experiment

var images: [(String, Image)] = []
let kodak = URL(fileURLWithPath: "kodak")
if let files = try? FileManager.default.contentsOfDirectory(at: kodak, includingPropertiesForKeys: nil) {
    for file in files.sorted(by: { $0.lastPathComponent < $1.lastPathComponent }) where file.pathExtension == "png" {
        if let image = load(file) { images.append(("photo", image)) }
    }
}
for seed in 0..<4 { images.append(("interface", interfaceImage(width: 1280, height: 720, seed: seed))) }
#if VT_SR
let least = VTLowLatencySuperResolutionScalerConfiguration.minimumDimensions
let most = VTLowLatencySuperResolutionScalerConfiguration.maximumDimensions
print("UPSCALE VT: supported \(VTLowLatencySuperResolutionScalerConfiguration.isSupported), "
      + "dimensions \(least.width)x\(least.height) to \(most.width)x\(most.height)")
#endif
print("UPSCALE images: \(images.filter { $0.0 == "photo" }.count) photos, \(images.filter { $0.0 == "interface" }.count) interfaces")

struct Score { var psnr = 0.0; var ssim = 0.0; var count = 0.0 }
var scores: [String: Score] = [:]
func record(_ key: String, _ result: Image, _ truth: [Double]) {
    let l = luma(result)
    var s = scores[key] ?? Score()
    s.psnr += psnr(l, truth)
    s.ssim += ssim(l, truth, width: result.width, height: result.height)
    s.count += 1
    scores[key] = s
}

let strengths: [(String, Float)] = [("light 0.8", 0.8), ("balanced 1.0", 1.0), ("strong 1.2", 1.2)]
for (kind, original) in images {
    // Sizes that both factors divide, so that the reductions are exact.
    let width = original.width - original.width % 12, height = original.height - original.height % 12
    var cropped: [UInt8] = []
    cropped.reserveCapacity(width * height * 4)
    for y in 0..<height {
        let start = y * original.width * 4
        cropped += original.rgba[start..<(start + width * 4)]
    }
    let truthImage = Image(width: width, height: height, rgba: cropped)
    let truth = luma(truthImage)
    for (scaleName, factor) in [("2x", 2.0), ("1.5x", 1.5)] {
        let smallW = Int((Double(width) / factor).rounded()), smallH = Int((Double(height) / factor).rounded())
        for byArea in [true, false] {
            let tag = "\(kind) \(scaleName) \(byArea ? "area" : "point")"
            let small = reduce(truthImage, to: smallW, smallH, byArea: byArea)
            record("\(tag) | bilinear", bilinear(small, to: width, height), truth)

            let input = upload(small)
            let up = scaler(smallW, smallH, width, height)
            let output = texture(width, height, .private)
            run { up.colorTexture = input; up.outputTexture = output; up.encode(commandBuffer: $0) }
            let metalFX = download(output)
            record("\(tag) | MetalFX", metalFX, truth)

            let linear = scaler(smallW, smallH, width, height, mode: .linear)
            let linearOutput = texture(width, height, .private)
            run { linear.colorTexture = input; linear.outputTexture = linearOutput; linear.encode(commandBuffer: $0) }
            record("\(tag) | MetalFX linear", download(linearOutput), truth)

            #if VT_SR
            if let result = superResolution(small, to: width, height) {
                record("\(tag) | VideoToolbox super resolution", result, truth)
                record("\(tag) | VideoToolbox's images: MetalFX", metalFX, truth)
            }
            #endif

            for (name, strength) in strengths {
                // The app's order: the capture is sharpened, then MetalFX brings it up.
                let sharpened = texture(smallW, smallH, .private)
                let before = texture(width, height, .private)
                run {
                    encodeSharpen(input, sharpened, strength: strength, $0)
                    up.colorTexture = sharpened; up.outputTexture = before; up.encode(commandBuffer: $0)
                }
                record("\(tag) | sharpen \(name), then MetalFX", download(before), truth)

                // The other order: MetalFX, then the sharpening on what it made.
                let after = texture(width, height, .private)
                run { encodeSharpen(output, after, strength: strength, $0) }
                record("\(tag) | MetalFX, then sharpen \(name)", download(after), truth)
            }
        }
    }
}

for key in scores.keys.sorted() {
    let s = scores[key]!
    let name = key.padding(toLength: 62, withPad: " ", startingAt: 0)
    print("UPSCALE \(name)" + String(format: " PSNR %6.2f dB  SSIM %.4f", s.psnr / s.count, s.ssim / s.count))
}
