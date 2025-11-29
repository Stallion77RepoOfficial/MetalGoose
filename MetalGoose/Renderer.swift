import MetalKit
import MetalFX
import Vision
import CoreVideo
import QuartzCore
import AppKit
import ApplicationServices
import simd

@available(macOS 26.0, *)
final class Renderer: NSObject, MTKViewDelegate {
    let device: MTLDevice
    let commandQueue: MTLCommandQueue
    var textureCache: CVMetalTextureCache?
    
    // Pipelines
    var flowPipeline: MTLComputePipelineState?
    var integerPipeline: MTLComputePipelineState?
    var bicubicPipeline: MTLComputePipelineState?
    var renderPipeline: MTLRenderPipelineState?
    
    // Processor & Scaler
    var spatialScaler: MTLFXSpatialScaler?
    var spatialScalerDesc: MTLFXSpatialScalerDescriptor?
    var temporalScaler: MTLFXTemporalScaler?
    var temporalScalerDesc: MTLFXTemporalScalerDescriptor?
    var temporalDenoisedScaler: MTLFXTemporalDenoisedScaler?
    var temporalDenoisedScalerDesc: MTLFXTemporalDenoisedScalerDescriptor?
    var frameInterpolator: MTLFXFrameInterpolator?
    var frameInterpolatorDesc: MTLFXFrameInterpolatorDescriptor?
    var zeroMotionTexture: MTLTexture?
    var flatDepthTexture: MTLTexture?
    var scalerHistoryNeedsReset: Bool = true
    var interpolatorHistoryNeedsReset: Bool = true
    var lastMetalFXMode: CaptureSettings.MetalFXMode
    
    // Vision
    let sequenceHandler = VNSequenceRequestHandler()
    
    // State
    var settings: CaptureSettings
    var prevTexture: MTLTexture?
    var prevBuffer: CVPixelBuffer?
    var privateOutputTexture: MTLTexture?
    weak var hostView: MTKView?
    private var fpsLayer: CATextLayer?
    private var lastFPSTimestamp: CFTimeInterval = CACurrentMediaTime()
    private var smoothedFPS: Double = 0.0
    
    private var trackedAXWindow: AXUIElement?
    private var trackedPID: pid_t = 0
    
    // Window Tracking
    var overlayWindow: NSWindow?
    var trackedWindowID: CGWindowID = 0
    var trackingTimer: Timer?
    
    private func markAllTemporalHistoriesDirty() {
        scalerHistoryNeedsReset = true
        interpolatorHistoryNeedsReset = true
    }
    
    init?(metalKitView: MTKView, settings: CaptureSettings) {
        guard let dev = MTLCreateSystemDefaultDevice(),
              let queue = dev.makeCommandQueue() else { return nil }
        
        self.device = dev
        self.commandQueue = queue
        self.settings = settings
        self.lastMetalFXMode = settings.metalFXMode
        
        super.init()
        
        self.hostView = metalKitView
        
        metalKitView.device = dev
        metalKitView.isPaused = false
        metalKitView.enableSetNeedsDisplay = true
        metalKitView.preferredFramesPerSecond = settings.vsync ? 60 : 0
        metalKitView.framebufferOnly = false
        metalKitView.colorPixelFormat = .bgra8Unorm
        metalKitView.clearColor = MTLClearColor(red: 0, green: 0, blue: 0, alpha: 0)
        metalKitView.delegate = self
        metalKitView.layer?.isOpaque = false
        setupHUDLayerIfNeeded(on: metalKitView)
        
        CVMetalTextureCacheCreate(kCFAllocatorDefault, nil, device, nil, &textureCache)
        setupPipelines()
    }
    
    func setupPipelines() {
        guard let lib = try? device.makeDefaultLibrary(bundle: .main) else { return }
        
        if let fn = lib.makeFunction(name: "flow_warp") { flowPipeline = try? device.makeComputePipelineState(function: fn) }
        if let fn = lib.makeFunction(name: "integer_scale") { integerPipeline = try? device.makeComputePipelineState(function: fn) }
        if let fn = lib.makeFunction(name: "bicubic_scale") { bicubicPipeline = try? device.makeComputePipelineState(function: fn) }
        
        let pipeDesc = MTLRenderPipelineDescriptor()
        pipeDesc.vertexFunction = lib.makeFunction(name: "texture_vertex")
        pipeDesc.fragmentFunction = lib.makeFunction(name: "texture_fragment")
        pipeDesc.colorAttachments[0].pixelFormat = .bgra8Unorm
        pipeDesc.colorAttachments[0].isBlendingEnabled = true
        pipeDesc.colorAttachments[0].rgbBlendOperation = .add
        pipeDesc.colorAttachments[0].sourceRGBBlendFactor = .sourceAlpha
        pipeDesc.colorAttachments[0].destinationRGBBlendFactor = .oneMinusSourceAlpha
        renderPipeline = try? device.makeRenderPipelineState(descriptor: pipeDesc)
    }
    
    func processAndRender(buffer: CVPixelBuffer, view: MTKView) {
        setupHUDLayerIfNeeded(on: view)
        // 1. FRAME PROCESSOR STEP (VideoToolbox Scaling/Converting)
        let profile = settings.qualityMode.profile
        let processedBuffer = buffer
        
          guard let drawable = view.currentDrawable,
              let cmdBuffer = commandQueue.makeCommandBuffer(),
              let srcTex = createTexture(from: processedBuffer) else { return }
          let previousTexture = prevTexture
          let previousBuffer = prevBuffer
        
        if settings.metalFXMode != lastMetalFXMode {
            markAllTemporalHistoriesDirty()
            lastMetalFXMode = settings.metalFXMode
        }
        if settings.scalingType != .metalFX {
            markAllTemporalHistoriesDirty()
        }
        if settings.frameGenBackend != .metalFX {
            frameInterpolator = nil
            frameInterpolatorDesc = nil
            markAllTemporalHistoriesDirty()
        }
        
        let needsMotionVectors = (settings.frameGenMode != .off && settings.frameGenBackend == .vision) ||
        (settings.scalingType == .metalFX && settings.metalFXMode.requiresMotionVectors)
        var texToScale = srcTex
        var motionTexture: MTLTexture?
        
        if settings.frameGenMode != .off && settings.frameGenBackend == .metalFX {
            motionTexture = ensureZeroMotionTexture(width: srcTex.width, height: srcTex.height)
        } else if needsMotionVectors, let pTex = previousTexture, let pBuf = previousBuffer {
            let w0 = CVPixelBufferGetWidth(processedBuffer)
            let h0 = CVPixelBufferGetHeight(processedBuffer)
            let w1 = CVPixelBufferGetWidth(pBuf)
            let h1 = CVPixelBufferGetHeight(pBuf)
            let f0 = CVPixelBufferGetPixelFormatType(processedBuffer)
            let f1 = CVPixelBufferGetPixelFormatType(pBuf)
            if w0 == w1, h0 == h1, f0 == f1 {
                let request = VNGenerateOpticalFlowRequest(targetedCVPixelBuffer: processedBuffer, options: [:])
                request.computationAccuracy = profile.flowAccuracy
                request.outputPixelFormat = kCVPixelFormatType_TwoComponent32Float
                do {
                    try sequenceHandler.perform([request], on: pBuf, orientation: .up)
                          if let res = request.results?.first,
                              let flowTex = createTexture(from: res.pixelBuffer) {
                                motionTexture = flowTex
                                if settings.frameGenMode != .off && settings.frameGenBackend == .vision,
                           let midTex = makeIntermediateTextureLike(srcTex),
                           let enc = cmdBuffer.makeComputeCommandEncoder(),
                           let pipe = flowPipeline {
                            enc.setComputePipelineState(pipe)
                            enc.setTexture(pTex, index: 0)
                            enc.setTexture(srcTex, index: 1)
                            enc.setTexture(flowTex, index: 2)
                            enc.setTexture(midTex, index: 3)
                            var t: Float = 0.5
                            enc.setBytes(&t, length: 4, index: 0)
                            let w = pipe.threadExecutionWidth
                            let h = pipe.maxTotalThreadsPerThreadgroup / w
                            let grids = MTLSize(width: (srcTex.width + w - 1)/w, height: (srcTex.height + h - 1)/h, depth: 1)
                            enc.dispatchThreadgroups(grids, threadsPerThreadgroup: MTLSize(width: w, height: h, depth: 1))
                            enc.endEncoding()
                            texToScale = midTex
                        }
                    }
                } catch {
                    markAllTemporalHistoriesDirty()
                }
            } else {
                markAllTemporalHistoriesDirty()
            }
        }
        
        if motionTexture == nil, settings.scalingType == .metalFX, settings.metalFXMode.requiresMotionVectors {
            motionTexture = ensureZeroMotionTexture(width: srcTex.width, height: srcTex.height)
            markAllTemporalHistoriesDirty()
        }
        
        // 3. Frame Interpolation (optional)
        if settings.frameGenMode != .off && settings.frameGenBackend == .metalFX,
              let prevColor = previousTexture,
           let motionTex = motionTexture {
            texToScale = encodeFrameInterpolation(current: texToScale,
                              previous: prevColor,
                              motion: motionTex,
                              commandBuffer: cmdBuffer)
        }
        
        // 4. UPSCALING (MetalFX or Shader)
        let dstTex = drawable.texture
        
        if settings.scalingType == .metalFX {
            switch settings.metalFXMode {
            case .spatial:
                let scaler = setupSpatialMetalFX(src: texToScale, dst: dstTex, profile: profile)
                let outTex = ensurePrivateOutputTexture(matching: dstTex)
                scaler.colorTexture = texToScale
                scaler.outputTexture = outTex
                scaler.encode(commandBuffer: cmdBuffer)
                copy(texture: outTex, to: dstTex, using: cmdBuffer)
            case .temporal:
                guard let motion = motionTexture else {
                    fatalError("Temporal MetalFX mode requires motion data.")
                }
                encodeTemporalScaler(src: texToScale, dst: dstTex, motion: motion, commandBuffer: cmdBuffer)
            case .temporalDenoised:
                guard let motion = motionTexture else {
                    fatalError("Temporal Denoised MetalFX mode requires motion data.")
                }
                encodeTemporalDenoisedScaler(src: texToScale, dst: dstTex, motion: motion, commandBuffer: cmdBuffer)
            }
        } else {
            // Integer / Bicubic via compute into intermediate texture, then blit to drawable
            let outTex = ensurePrivateOutputTexture(matching: dstTex)

            guard let enc = cmdBuffer.makeComputeCommandEncoder() else { return }
            let pipe = (settings.scalingType == .integer) ? integerPipeline : bicubicPipeline

            if let p = pipe {
                enc.setComputePipelineState(p)
                enc.setTexture(texToScale, index: 0)
                enc.setTexture(outTex, index: 1)

                if settings.scalingType == .integer {
                    var factor = settings.scaleFactor
                    enc.setBytes(&factor, length: 4, index: 0)
                }

                let w = p.threadExecutionWidth
                let h = p.maxTotalThreadsPerThreadgroup / w
                let grids = MTLSize(width: (outTex.width + w - 1)/w, height: (outTex.height + h - 1)/h, depth: 1)
                enc.dispatchThreadgroups(grids, threadsPerThreadgroup: MTLSize(width: w, height: h, depth: 1))
            }
            enc.endEncoding()

            if let blit = cmdBuffer.makeBlitCommandEncoder() {
                blit.copy(from: outTex, to: dstTex)
                blit.endEncoding()
            }
        }
        
        cmdBuffer.present(drawable)
        cmdBuffer.addCompletedHandler { [weak self] _ in
            guard let self = self else { return }
            let now = CACurrentMediaTime()
            let dt = now - self.lastFPSTimestamp
            self.lastFPSTimestamp = now
            let inst = dt > 0 ? (1.0 / dt) : 0.0
            self.smoothedFPS = self.smoothedFPS == 0 ? inst : (self.smoothedFPS * 0.85 + inst * 0.15)
            if self.settings.showFPS, let layer = self.fpsLayer {
                DispatchQueue.main.async {
                    layer.string = String(format: "FPS: %.0f", self.smoothedFPS)
                }
            }
        }
        cmdBuffer.commit()
        prevTexture = srcTex
        prevBuffer = buffer
    }
    
    func setupSpatialMetalFX(src: MTLTexture, dst: MTLTexture, profile: QualityProfile) -> MTLFXSpatialScaler {
        guard MTLFXSpatialScalerDescriptor.supportsDevice(device) else {
            fatalError("MetalFX Spatial Scaler is not supported on this device.")
        }
        let needsNewScaler = spatialScaler == nil ||
        spatialScalerDesc?.inputWidth != src.width ||
        spatialScalerDesc?.inputHeight != src.height ||
        spatialScalerDesc?.outputWidth != dst.width ||
        spatialScalerDesc?.outputHeight != dst.height ||
        spatialScalerDesc?.colorTextureFormat != src.pixelFormat ||
        spatialScalerDesc?.outputTextureFormat != dst.pixelFormat ||
        spatialScalerDesc?.colorProcessingMode != profile.scalerMode
        
        if needsNewScaler {
            let desc = MTLFXSpatialScalerDescriptor()
            desc.inputWidth = src.width
            desc.inputHeight = src.height
            desc.outputWidth = dst.width
            desc.outputHeight = dst.height
            desc.colorTextureFormat = src.pixelFormat
            desc.outputTextureFormat = dst.pixelFormat
            desc.colorProcessingMode = profile.scalerMode
            spatialScalerDesc = desc
            guard let created = desc.makeSpatialScaler(device: device) else {
                fatalError("Failed to create MetalFX Spatial Scaler.")
            }
            spatialScaler = created
        }
        guard let scaler = spatialScaler else {
            fatalError("MetalFX Spatial Scaler is not available.")
        }
        return scaler
    }
    
    func encodeTemporalScaler(src: MTLTexture, dst: MTLTexture, motion: MTLTexture, commandBuffer: MTLCommandBuffer) {
        let scaler = setupTemporalScaler(src: src, dst: dst, motion: motion)
        let depth = ensureFlatDepthTexture(width: src.width, height: src.height)
        let outTex = ensurePrivateOutputTexture(matching: dst)
        scaler.colorTexture = src
        scaler.outputTexture = outTex
        scaler.motionTexture = motion
        scaler.depthTexture = depth
        scaler.preExposure = 1.0
        scaler.motionVectorScaleX = 1.0
        scaler.motionVectorScaleY = 1.0
        scaler.inputContentWidth = src.width
        scaler.inputContentHeight = src.height
        scaler.reset = scalerHistoryNeedsReset
        scaler.encode(commandBuffer: commandBuffer)
        scalerHistoryNeedsReset = false
        copy(texture: outTex, to: dst, using: commandBuffer)
    }
    
    func encodeTemporalDenoisedScaler(src: MTLTexture, dst: MTLTexture, motion: MTLTexture, commandBuffer: MTLCommandBuffer) {
        let scaler = setupTemporalDenoisedScaler(src: src, dst: dst, motion: motion)
        let depth = ensureFlatDepthTexture(width: src.width, height: src.height)
        let outTex = ensurePrivateOutputTexture(matching: dst)
        scaler.colorTexture = src
        scaler.outputTexture = outTex
        scaler.motionTexture = motion
        scaler.depthTexture = depth
        scaler.preExposure = 1.0
        scaler.motionVectorScaleX = 1.0
        scaler.motionVectorScaleY = 1.0
        scaler.shouldResetHistory = scalerHistoryNeedsReset
        scaler.worldToViewMatrix = matrix_identity_float4x4
        scaler.viewToClipMatrix = matrix_identity_float4x4
        scaler.isDepthReversed = false
        scaler.encode(commandBuffer: commandBuffer)
        scalerHistoryNeedsReset = false
        copy(texture: outTex, to: dst, using: commandBuffer)
    }
    
    func setupTemporalScaler(src: MTLTexture, dst: MTLTexture, motion: MTLTexture) -> MTLFXTemporalScaler {
        guard MTLFXTemporalScalerDescriptor.supportsDevice(device) else {
            fatalError("MetalFX Temporal Scaler is not supported on this device.")
        }
        let scaleX = Float(dst.width) / Float(src.width)
        let scaleY = Float(dst.height) / Float(src.height)
        let minScale = min(scaleX, scaleY)
        let maxScale = max(scaleX, scaleY)
        let needsNew = temporalScaler == nil ||
        temporalScalerDesc?.inputWidth != src.width ||
        temporalScalerDesc?.inputHeight != src.height ||
        temporalScalerDesc?.outputWidth != dst.width ||
        temporalScalerDesc?.outputHeight != dst.height ||
        temporalScalerDesc?.colorTextureFormat != src.pixelFormat ||
        temporalScalerDesc?.outputTextureFormat != dst.pixelFormat ||
        temporalScalerDesc?.motionTextureFormat != motion.pixelFormat
        if needsNew {
            let desc = MTLFXTemporalScalerDescriptor()
            desc.inputWidth = src.width
            desc.inputHeight = src.height
            desc.outputWidth = dst.width
            desc.outputHeight = dst.height
            desc.colorTextureFormat = src.pixelFormat
            desc.outputTextureFormat = dst.pixelFormat
            desc.motionTextureFormat = motion.pixelFormat
            desc.depthTextureFormat = .r32Float
            desc.isAutoExposureEnabled = false
            desc.isInputContentPropertiesEnabled = false
            desc.isReactiveMaskTextureEnabled = false
            desc.reactiveMaskTextureFormat = .invalid
            desc.requiresSynchronousInitialization = false
            desc.inputContentMinScale = minScale
            desc.inputContentMaxScale = maxScale
            temporalScalerDesc = desc
            guard let created = desc.makeTemporalScaler(device: device) else {
                fatalError("Failed to create MetalFX Temporal Scaler.")
            }
            temporalScaler = created
            scalerHistoryNeedsReset = true
        }
        guard let scaler = temporalScaler else {
            fatalError("MetalFX Temporal Scaler is not available.")
        }
        return scaler
    }
    
    func setupTemporalDenoisedScaler(src: MTLTexture, dst: MTLTexture, motion: MTLTexture) -> MTLFXTemporalDenoisedScaler {
        guard MTLFXTemporalDenoisedScalerDescriptor.supportsDevice(device) else {
            fatalError("MetalFX Temporal Denoised Scaler is not supported on this device.")
        }
        let needsNew = temporalDenoisedScaler == nil ||
        temporalDenoisedScalerDesc?.inputWidth != src.width ||
        temporalDenoisedScalerDesc?.inputHeight != src.height ||
        temporalDenoisedScalerDesc?.outputWidth != dst.width ||
        temporalDenoisedScalerDesc?.outputHeight != dst.height ||
        temporalDenoisedScalerDesc?.colorTextureFormat != src.pixelFormat ||
        temporalDenoisedScalerDesc?.outputTextureFormat != dst.pixelFormat ||
        temporalDenoisedScalerDesc?.motionTextureFormat != motion.pixelFormat
        if needsNew {
            let desc = MTLFXTemporalDenoisedScalerDescriptor()
            desc.inputWidth = src.width
            desc.inputHeight = src.height
            desc.outputWidth = dst.width
            desc.outputHeight = dst.height
            desc.colorTextureFormat = src.pixelFormat
            desc.depthTextureFormat = .r32Float
            desc.motionTextureFormat = motion.pixelFormat
            desc.outputTextureFormat = dst.pixelFormat
            desc.diffuseAlbedoTextureFormat = .invalid
            desc.specularAlbedoTextureFormat = .invalid
            desc.normalTextureFormat = .invalid
            desc.roughnessTextureFormat = .invalid
            desc.specularHitDistanceTextureFormat = .invalid
            desc.denoiseStrengthMaskTextureFormat = .invalid
            desc.transparencyOverlayTextureFormat = .invalid
            desc.isSpecularHitDistanceTextureEnabled = false
            desc.isDenoiseStrengthMaskTextureEnabled = false
            desc.isTransparencyOverlayTextureEnabled = false
            desc.isReactiveMaskTextureEnabled = false
            desc.reactiveMaskTextureFormat = .invalid
            desc.isAutoExposureEnabled = false
            temporalDenoisedScalerDesc = desc
            guard let created = desc.makeTemporalDenoisedScaler(device: device) else {
                fatalError("Failed to create MetalFX Temporal Denoised Scaler.")
            }
            temporalDenoisedScaler = created
            scalerHistoryNeedsReset = true
        }
        guard let scaler = temporalDenoisedScaler else {
            fatalError("MetalFX Temporal Denoised Scaler is not available.")
        }
        return scaler
    }
    
    func encodeFrameInterpolation(current: MTLTexture, previous: MTLTexture, motion: MTLTexture, commandBuffer: MTLCommandBuffer) -> MTLTexture {
        let interpolator = setupFrameInterpolator(src: current, dstWidth: current.width, dstHeight: current.height, motion: motion)
        let outTex = ensureIntermediateTexture(width: current.width, height: current.height, pixelFormat: current.pixelFormat)
        interpolator.colorTexture = current
        interpolator.prevColorTexture = previous
        interpolator.motionTexture = motion
        interpolator.depthTexture = ensureFlatDepthTexture(width: current.width, height: current.height)
        interpolator.outputTexture = outTex
        interpolator.motionVectorScaleX = 1.0
        interpolator.motionVectorScaleY = 1.0
        interpolator.deltaTime = 1.0 / Float(settings.frameGenMode == .x2 ? 2 : 3)
        interpolator.shouldResetHistory = interpolatorHistoryNeedsReset
        interpolator.fieldOfView = 60.0
        interpolator.aspectRatio = Float(current.width) / Float(current.height)
        interpolator.nearPlane = 0.1
        interpolator.farPlane = 1000
        interpolator.encode(commandBuffer: commandBuffer)
        interpolatorHistoryNeedsReset = false
        return outTex
    }
    
    func setupFrameInterpolator(src: MTLTexture, dstWidth: Int, dstHeight: Int, motion: MTLTexture) -> MTLFXFrameInterpolator {
        guard MTLFXFrameInterpolatorDescriptor.supportsDevice(device) else {
            fatalError("MetalFX Frame Interpolator is not supported on this device.")
        }
        let needsNew = frameInterpolator == nil ||
        frameInterpolatorDesc?.inputWidth != src.width ||
        frameInterpolatorDesc?.inputHeight != src.height ||
        frameInterpolatorDesc?.outputWidth != dstWidth ||
        frameInterpolatorDesc?.outputHeight != dstHeight ||
        frameInterpolatorDesc?.colorTextureFormat != src.pixelFormat ||
        frameInterpolatorDesc?.motionTextureFormat != motion.pixelFormat
        if needsNew {
            let desc = MTLFXFrameInterpolatorDescriptor()
            desc.inputWidth = src.width
            desc.inputHeight = src.height
            desc.outputWidth = dstWidth
            desc.outputHeight = dstHeight
            desc.colorTextureFormat = src.pixelFormat
            desc.outputTextureFormat = src.pixelFormat
            desc.motionTextureFormat = motion.pixelFormat
            desc.depthTextureFormat = .r32Float
            desc.uiTextureFormat = .invalid
            desc.scaler = nil
            frameInterpolatorDesc = desc
            guard let created = desc.makeFrameInterpolator(device: device) else {
                fatalError("Failed to create MetalFX Frame Interpolator.")
            }
            frameInterpolator = created
            interpolatorHistoryNeedsReset = true
        }
        guard let interpolator = frameInterpolator else {
            fatalError("MetalFX Frame Interpolator is not available.")
        }
        return interpolator
    }
    
    func ensureZeroMotionTexture(width: Int, height: Int) -> MTLTexture {
        if zeroMotionTexture == nil || zeroMotionTexture?.width != width || zeroMotionTexture?.height != height {
            zeroMotionTexture = makeConstantTexture(width: width, height: height, pixelFormat: .rg32Float, components: 2, values: [0.0, 0.0])
        }
        guard let texture = zeroMotionTexture else {
            fatalError("Failed to create zero-motion texture.")
        }
        return texture
    }
    
    func ensureFlatDepthTexture(width: Int, height: Int) -> MTLTexture {
        if flatDepthTexture == nil || flatDepthTexture?.width != width || flatDepthTexture?.height != height {
            flatDepthTexture = makeConstantTexture(width: width, height: height, pixelFormat: .r32Float, components: 1, values: [1.0])
        }
        guard let texture = flatDepthTexture else {
            fatalError("Failed to create flat depth texture.")
        }
        return texture
    }
    
    private func makeConstantTexture(width: Int, height: Int, pixelFormat: MTLPixelFormat, components: Int, values: [Float]) -> MTLTexture? {
        precondition(values.count == components, "Values count komponent sayisiyla eslesmeli")
        let desc = MTLTextureDescriptor.texture2DDescriptor(pixelFormat: pixelFormat, width: width, height: height, mipmapped: false)
        desc.usage = [.shaderRead]
        desc.storageMode = .shared
        guard let texture = device.makeTexture(descriptor: desc) else { return nil }
        let total = width * height * components
        var data = [Float](repeating: 0, count: total)
        for c in 0..<components {
            let value = values[c]
            var idx = c
            while idx < total {
                data[idx] = value
                idx += components
            }
        }
        let bytesPerRow = width * components * MemoryLayout<Float>.size
        let region = MTLRegion(origin: MTLOrigin(x: 0, y: 0, z: 0), size: MTLSize(width: width, height: height, depth: 1))
        data.withUnsafeBytes { ptr in
            texture.replace(region: region, mipmapLevel: 0, withBytes: ptr.baseAddress!, bytesPerRow: bytesPerRow)
        }
        return texture
    }
    
    private func setupHUDLayerIfNeeded(on view: MTKView) {
        guard settings.showFPS else {
            fpsLayer?.removeFromSuperlayer()
            fpsLayer = nil
            return
        }
        if fpsLayer == nil {
            let layer = CATextLayer()
            layer.contentsScale = view.window?.backingScaleFactor ?? (view.layer?.contentsScale ?? 2.0)
            layer.font = NSFont.monospacedDigitSystemFont(ofSize: 12, weight: .regular)
            layer.fontSize = 12
            layer.alignmentMode = .left
            layer.foregroundColor = NSColor.white.cgColor
            layer.backgroundColor = NSColor.black.withAlphaComponent(0.55).cgColor
            layer.cornerRadius = 4
            layer.masksToBounds = true
            layer.string = "FPS: --"
            let h: CGFloat = 20
            layer.frame = CGRect(x: 8, y: (view.bounds.height - h - 8), width: 120, height: h)
            view.layer?.addSublayer(layer)
            fpsLayer = layer
        }
    }
    
    private func createTexture(from buffer: CVPixelBuffer) -> MTLTexture? {
        var cvTex: CVMetalTexture?
        let w = CVPixelBufferGetWidth(buffer)
        let h = CVPixelBufferGetHeight(buffer)
        guard let cache = textureCache else { return nil }
        let cvFormat = CVPixelBufferGetPixelFormatType(buffer)
        let metalFormat: MTLPixelFormat
        switch cvFormat {
        case kCVPixelFormatType_32BGRA:
            metalFormat = .bgra8Unorm
        case kCVPixelFormatType_TwoComponent32Float:
            metalFormat = .rg32Float
        default:
            metalFormat = .bgra8Unorm
        }
        CVMetalTextureCacheCreateTextureFromImage(kCFAllocatorDefault, cache, buffer, nil, metalFormat, w, h, 0, &cvTex)
        return cvTex.flatMap { CVMetalTextureGetTexture($0) }
    }
    
    private func makeIntermediateTextureLike(_ reference: MTLTexture) -> MTLTexture? {
        let desc = MTLTextureDescriptor.texture2DDescriptor(pixelFormat: reference.pixelFormat,
                                                            width: reference.width,
                                                            height: reference.height,
                                                            mipmapped: false)
        desc.usage = [.shaderRead, .shaderWrite]
        return device.makeTexture(descriptor: desc)
    }
    
    private func ensureIntermediateTexture(width: Int, height: Int, pixelFormat: MTLPixelFormat) -> MTLTexture {
        let desc = MTLTextureDescriptor.texture2DDescriptor(pixelFormat: pixelFormat, width: width, height: height, mipmapped: false)
        desc.usage = [.shaderRead, .shaderWrite]
        guard let tex = device.makeTexture(descriptor: desc) else {
            fatalError("Failed to create intermediate texture.")
        }
        return tex
    }
    
    private func ensurePrivateOutputTexture(matching texture: MTLTexture) -> MTLTexture {
        if privateOutputTexture == nil ||
            privateOutputTexture?.width != texture.width ||
            privateOutputTexture?.height != texture.height ||
            privateOutputTexture?.pixelFormat != texture.pixelFormat {
            let desc = MTLTextureDescriptor.texture2DDescriptor(pixelFormat: texture.pixelFormat,
                                                                width: texture.width,
                                                                height: texture.height,
                                                                mipmapped: false)
            desc.storageMode = .private
            desc.usage = [.shaderRead, .shaderWrite, .renderTarget]
            privateOutputTexture = device.makeTexture(descriptor: desc)
        }
        guard let outTex = privateOutputTexture else {
            fatalError("Failed to create private output texture.")
        }
        return outTex
    }
    
    private func copy(texture source: MTLTexture, to destination: MTLTexture, using commandBuffer: MTLCommandBuffer) {
        guard let blit = commandBuffer.makeBlitCommandEncoder() else {
            fatalError("Failed to create blit encoder.")
        }
        blit.copy(from: source, to: destination)
        blit.endEncoding()
    }
    
    // Tracking & Overlay
    func createOverlayWindow(targetFrame: CGRect) -> NSWindow {
        let panel = NSPanel(contentRect: targetFrame,
                            styleMask: [.nonactivatingPanel, .borderless],
                            backing: .buffered,
                            defer: false)
        panel.backgroundColor = .clear
        panel.isOpaque = false
        panel.hasShadow = false
        panel.level = .floating
        panel.ignoresMouseEvents = true
        panel.collectionBehavior = [.canJoinAllSpaces, .fullScreenAuxiliary, .ignoresCycle, .stationary]
        panel.hidesOnDeactivate = false
        panel.becomesKeyOnlyIfNeeded = false
        return panel
    }
    
    func startTracking(windowID: CGWindowID, pid: Int32, overlay: NSWindow) {
        self.trackedWindowID = windowID
        self.trackedPID = pid
        self.overlayWindow = overlay

        // Build AX window reference for faster tracking
        let appElement = AXUIElementCreateApplication(pid)
        var windowsRef: CFTypeRef?
        if AXUIElementCopyAttributeValue(appElement, kAXWindowsAttribute as CFString, &windowsRef) == .success,
           let windows = windowsRef as? [AXUIElement] {
            self.trackedAXWindow = windows.first
        } else {
            self.trackedAXWindow = nil
        }

        DispatchQueue.main.async {
            self.trackingTimer?.invalidate()
            self.trackingTimer = Timer.scheduledTimer(withTimeInterval: 0.016, repeats: true) { [weak self] _ in
                guard let self = self else { return }
                if self.trackedAXWindow != nil {
                    self.updateOverlayPositionAX()
                } else {
                    self.updateOverlayPosition()
                }
            }
        }
    }
    
    func stopTracking() {
        trackingTimer?.invalidate()
        trackingTimer = nil
    }
    
    deinit {
        trackingTimer?.invalidate()
    }
    
    private func updateOverlayPositionAX() {
        guard let overlay = overlayWindow, let axWin = trackedAXWindow else { return }
        var posValue: CFTypeRef?
        var sizeValue: CFTypeRef?
        AXUIElementCopyAttributeValue(axWin, kAXPositionAttribute as CFString, &posValue)
        AXUIElementCopyAttributeValue(axWin, kAXSizeAttribute as CFString, &sizeValue)

        var pos = CGPoint.zero
        var size = CGSize.zero
        if let p = posValue, CFGetTypeID(p) == AXValueGetTypeID() {
            AXValueGetValue(p as! AXValue, .cgPoint, &pos)
        }
        if let s = sizeValue, CFGetTypeID(s) == AXValueGetTypeID() {
            AXValueGetValue(s as! AXValue, .cgSize, &size)
        }
        guard size.width > 0, size.height > 0 else { return }

        // Convert to Cocoa coordinates using union of screens
        let union = NSScreen.screens.reduce(NSRect.null) { $0.union($1.frame) }
        let cocoaX = union.minX + pos.x
        let cocoaY = union.maxY - (pos.y + size.height)
        let newFrame = CGRect(x: cocoaX, y: cocoaY, width: size.width, height: size.height)
        if overlay.frame != newFrame {
            overlay.setFrame(newFrame, display: true, animate: false)
        }
    }
    
    private func updateOverlayPosition() {
        guard let overlay = overlayWindow else { return }
        guard let list = CGWindowListCopyWindowInfo([.optionIncludingWindow], trackedWindowID) as? [[String: Any]],
              let info = list.first,
              let bounds = info[kCGWindowBounds as String] as? [String: CGFloat] else {
            stopTracking()
            overlay.orderOut(nil)
            return
        }

        let x = bounds["X"] ?? 0
        let y = bounds["Y"] ?? 0
        let w = bounds["Width"] ?? 100
        let h = bounds["Height"] ?? 100

        // Compute union rect of all screens in Cocoa coordinates
        let union = NSScreen.screens.reduce(NSRect.null) { $0.union($1.frame) }
        let cocoaX = union.minX + x
        let cocoaY = union.maxY - (y + h)
        let newFrame = CGRect(x: cocoaX, y: cocoaY, width: w, height: h)

        if overlay.frame != newFrame {
            overlay.setFrame(newFrame, display: true, animate: false)
        }
    }
    
    // Resource cleanup for safe shutdown
    func cleanup() {
        stopTracking()
        prevTexture = nil
        prevBuffer = nil
        privateOutputTexture = nil
        spatialScaler = nil
        spatialScalerDesc = nil
        temporalScaler = nil
        temporalScalerDesc = nil
        temporalDenoisedScaler = nil
        temporalDenoisedScalerDesc = nil
        frameInterpolator = nil
        frameInterpolatorDesc = nil
        zeroMotionTexture = nil
        flatDepthTexture = nil
        markAllTemporalHistoriesDirty()
    }
    
    // Delegate Stub
    func mtkView(_ view: MTKView, drawableSizeWillChange size: CGSize) {
        spatialScaler = nil
        spatialScalerDesc = nil
        temporalScaler = nil
        temporalScalerDesc = nil
        temporalDenoisedScaler = nil
        temporalDenoisedScalerDesc = nil
        frameInterpolator = nil
        frameInterpolatorDesc = nil
        zeroMotionTexture = nil
        flatDepthTexture = nil
        markAllTemporalHistoriesDirty()
        privateOutputTexture = nil
        if let cache = textureCache { CVMetalTextureCacheFlush(cache, 0) }
    }
    func draw(in view: MTKView) {}
}

