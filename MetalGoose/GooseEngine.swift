import Foundation
import AppKit
@preconcurrency import Metal
@preconcurrency import MetalKit
@preconcurrency import MetalFX
import IOSurface
import QuartzCore

struct PipelineStats {
    var captureFPS: Float = 0
    var outputFPS: Float = 0
    var interpolatedFPS: Float = 0
    var frameTime: Float = 0
    var gpuTime: Float = 0
    var captureLatency: Float = 0
    var frameCount: UInt64 = 0
    var outputFrameCount: UInt64 = 0
    var droppedFrames: UInt64 = 0
    var interpolatedFrameCount: UInt64 = 0
    var passthroughFrameCount: UInt64 = 0
    var gpuMemoryUsed: UInt64 = 0
    var gpuMemoryTotal: UInt64 = 0
    var isUsingVirtualDisplay: Bool = false
    var virtualResolution: CGSize = .zero
    var outputResolution: CGSize = .zero
}

@available(macOS 26.0, *)
@MainActor
final class GooseEngine: NSObject, ObservableObject, MTKViewDelegate {
    
    @Published private(set) var isCapturing: Bool = false
    @Published private(set) var stats: PipelineStats = PipelineStats()
    @Published private(set) var lastError: String?
    
    var deviceName: String { device.name }
    
    var onFrameReady: ((MTLTexture) -> Void)?
    private let device: MTLDevice
    private let commandQueue: MTLCommandQueue
    
    // MetalFX Spatial Scaler
    private var spatialScaler: MTLFXSpatialScaler?
    private var spatialScalerInputSize: (width: Int, height: Int) = (0, 0)
    private var spatialScalerOutputSize: (width: Int, height: Int) = (0, 0)
    
    // Retained pipelines
    private var scalePipeline: MTLComputePipelineState?
    private var casPipeline: MTLComputePipelineState?
    private var fxaaPipeline: MTLComputePipelineState?
    private var smaaEdgePipeline: MTLComputePipelineState?
    private var smaaWeightPipeline: MTLComputePipelineState?
    private var smaaBlendPipeline: MTLComputePipelineState?
    private var msaaPipeline: MTLComputePipelineState?
    private var temporalPipeline: MTLComputePipelineState?
    private var copyPipeline: MTLComputePipelineState?

    // Optical flow / frame generation pipelines
    private var lumaFromColorPipeline: MTLComputePipelineState?
    private var lumaDownsamplePipeline: MTLComputePipelineState?
    private var flowInitPipeline: MTLComputePipelineState?
    private var flowRefinePipeline: MTLComputePipelineState?
    private var flowUpsamplePipeline: MTLComputePipelineState?
    private var flowWarpPipeline: MTLComputePipelineState?
    private var flowComposePipeline: MTLComputePipelineState?
    private var flowOcclusionPipeline: MTLComputePipelineState?
    
    private var renderPipeline: MTLRenderPipelineState?
    private weak var mtkView: MTKView?
    
    private var renderTexture: MTLTexture?
    private var scaledTexture: MTLTexture?
    private var casTexture: MTLTexture?
    private var usmTexture: MTLTexture?
    private var fxaaTexture: MTLTexture?
    private var smaaEdgeTexture: MTLTexture?
    private var smaaWeightTexture: MTLTexture?
    private var smaaOutputTexture: MTLTexture?
    private var msaaTexture: MTLTexture?
    private var taaHistoryTexture: MTLTexture?
    private var taaOutputTexture: MTLTexture?
    
    // Optical flow resources
    private var lumaPrevPyramid: [MTLTexture] = []
    private var lumaNextPyramid: [MTLTexture] = []
    private var flowPyramid: [MTLTexture] = []
    private var flowScratchPyramid: [MTLTexture] = []
    private var flowPyramidLevels: Int = 0
    private var occlusionTexture: MTLTexture?
    private var warpedPrevTexture: MTLTexture?
    private var warpedNextTexture: MTLTexture?
    
    struct FrameHistory {
        let texture: MTLTexture
        let timestamp: CFTimeInterval
        let flowFromPrev: MTLTexture?
        let flowToPrev: MTLTexture?
    }
    
    private final class FrameRingBuffer: @unchecked Sendable {
        private var buffer: [FrameHistory] = []
        private let capacity = 4
        private let lock = NSLock()
        
        func push(_ frame: FrameHistory) {
            lock.lock()
            defer { lock.unlock() }
            buffer.append(frame)
            if buffer.count > capacity {
                buffer.removeFirst()
            }
            buffer.sort { $0.timestamp < $1.timestamp }
        }
        
        func getFramesForTime(_ targetTime: CFTimeInterval) -> (prev: FrameHistory, next: FrameHistory)? {
            lock.lock()
            defer { lock.unlock() }
            
            guard buffer.count >= 2 else { return nil }
            
            for i in 0..<(buffer.count - 1) {
                let prev = buffer[i]
                let next = buffer[i+1]
                if targetTime >= prev.timestamp && targetTime <= next.timestamp {
                    return (prev, next)
                }
            }
            
            if let last = buffer.last, targetTime > last.timestamp {
                return (buffer[buffer.count-2], last)
            }
            
            return (buffer[0], buffer[1])
        }
        
        var count: Int {
            lock.lock()
            defer { lock.unlock() }
            return buffer.count
        }
        
        var newestFrame: FrameHistory? {
            lock.lock()
            defer { lock.unlock() }
            return buffer.last
        }

        func clear() {
            lock.lock()
            defer { lock.unlock() }
            buffer.removeAll()
        }
    }
    
    private let frameBuffer = FrameRingBuffer()
    private var blendTexture: MTLTexture?
    private var hasTAAHistory: Bool = false
    private var lastProcessedSize: CGSize = .zero

    private var scalingType: CaptureSettings.ScalingType = .off
    private var qualityMode: CaptureSettings.QualityMode = .balanced
    private var aaMode: CaptureSettings.AAMode = .off
    private var renderScaleFactor: Float = 1.0
    private var scaleFactor: Float = 1.0
    private var sharpness: Float = 0.5
    private var temporalBlend: Float = 0.1
    private var motionScale: Float = 1.0
    private var captureCursor: Bool = true
    private var frameGenEnabled: Bool = false
    private var frameGenMode: CaptureSettings.FrameGenMode = .off
    private var frameGenType: CaptureSettings.FrameGenType = .adaptive
    private var targetFPS: Int = 120
    private var frameGenMultiplier: Int = 2
    private var adaptiveSync: Bool = true
    private var vsyncEnabled: Bool = true
    private var qualityProfile: QualityProfile = CaptureSettings.QualityMode.balanced.profile
    
    private let flowMaxLevels: Int = 5
    private let flowOcclusionThreshold: Float = 1.5

    private var estimatedCaptureInterval: Double = 0
    private var lastCaptureTimestamp: CFTimeInterval = 0
    private var lastPreferredFPS: Int = 0
    private var lastPreferredUpdateTime: CFTimeInterval = 0

    private var virtualDisplayManager: VirtualDisplayManager?
    private var captureRefreshRate: Int = 0
    
    private var lastFrameTime: CFTimeInterval = 0
    private var frameCount: Int = 0
    private var fpsStartTime: CFTimeInterval = 0
    
    private var outputSize: CGSize = .zero
    private var currentRefreshRate: Int = 0
    
    override init() {
        guard let dev = MTLCreateSystemDefaultDevice() else {
            fatalError("Error Code: MG-ENG-002 Metal device not available")
        }
        guard let queue = dev.makeCommandQueue() else {
            fatalError("Error Code: MG-ENG-003 Metal command queue not available")
        }
        
        self.device = dev
        self.commandQueue = queue
        
        super.init()
        setupPipelines()
    }
    
    private func setupPipelines() {
        guard let library = device.makeDefaultLibrary() else { return }
        
        func makeCompute(_ name: String) -> MTLComputePipelineState? {
            guard let function = library.makeFunction(name: name) else { return nil }
            return try? device.makeComputePipelineState(function: function)
        }

        // Core pipelines
        scalePipeline = makeCompute("blitScaleBilinear")
        casPipeline = makeCompute("contrastAdaptiveSharpening")
        copyPipeline = makeCompute("copyTexture")
        
        // Anti-aliasing pipelines
        fxaaPipeline = makeCompute("fxaa")
        smaaEdgePipeline = makeCompute("smaaEdgeDetection")
        smaaWeightPipeline = makeCompute("smaaBlendingWeights")
        smaaBlendPipeline = makeCompute("smaaBlend")
        msaaPipeline = makeCompute("msaa")
        temporalPipeline = makeCompute("temporalReproject")

        // Optical flow / frame generation pipelines
        lumaFromColorPipeline = makeCompute("lumaFromColor")
        lumaDownsamplePipeline = makeCompute("lumaDownsample2x")
        flowInitPipeline = makeCompute("flowInit")
        flowRefinePipeline = makeCompute("flowRefine")
        flowUpsamplePipeline = makeCompute("flowUpsample2x")
        flowWarpPipeline = makeCompute("flowWarp")
        flowComposePipeline = makeCompute("flowCompose")
        flowOcclusionPipeline = makeCompute("flowOcclusion")

        // Render pipeline
        do {
            if let vtx = library.makeFunction(name: "texture_vertex"),
               let frag = library.makeFunction(name: "texture_fragment") {
                let desc = MTLRenderPipelineDescriptor()
                desc.vertexFunction = vtx
                desc.fragmentFunction = frag
                desc.colorAttachments[0].pixelFormat = .bgra8Unorm
                renderPipeline = try device.makeRenderPipelineState(descriptor: desc)
            }
        } catch {
            lastError = "Error Code: MG-ENG-001 Pipeline setup failed: \(error)"
        }
        
    }
    
    private func ensureMetalFXSpatialScaler(inputWidth: Int, inputHeight: Int,
                                            outputWidth: Int, outputHeight: Int) -> MTLFXSpatialScaler? {
        // Reuse existing scaler if dimensions match
        if let scaler = spatialScaler,
           spatialScalerInputSize == (inputWidth, inputHeight),
           spatialScalerOutputSize == (outputWidth, outputHeight) {
            return scaler
        }
        
        // Create new scaler
        let descriptor = MTLFXSpatialScalerDescriptor()
        descriptor.inputWidth = inputWidth
        descriptor.inputHeight = inputHeight
        descriptor.outputWidth = outputWidth
        descriptor.outputHeight = outputHeight
        descriptor.colorTextureFormat = .bgra8Unorm
        descriptor.outputTextureFormat = .bgra8Unorm
        descriptor.colorProcessingMode = .perceptual
        
        guard let scaler = descriptor.makeSpatialScaler(device: device) else {
            lastError = "Error Code: MG-ENG-004 MetalFX Spatial Scaler creation failed"
            return nil
        }
        
        spatialScaler = scaler
        spatialScalerInputSize = (inputWidth, inputHeight)
        spatialScalerOutputSize = (outputWidth, outputHeight)
        
        return scaler
    }

    private func ensureTexture(_ texture: inout MTLTexture?, width: Int, height: Int,
                               pixelFormat: MTLPixelFormat = .bgra8Unorm,
                               usage: MTLTextureUsage = [.shaderRead, .shaderWrite]) -> MTLTexture? {
        if let tex = texture,
           tex.width == width,
           tex.height == height,
           tex.pixelFormat == pixelFormat {
            return tex
        }

        let desc = MTLTextureDescriptor.texture2DDescriptor(
            pixelFormat: pixelFormat,
            width: width,
            height: height,
            mipmapped: false
        )
        desc.usage = usage
        desc.storageMode = .private
        texture = device.makeTexture(descriptor: desc)
        return texture
    }

    private func makeFlowTexture(width: Int, height: Int) -> MTLTexture? {
        let desc = MTLTextureDescriptor.texture2DDescriptor(
            pixelFormat: .rg16Float,
            width: width,
            height: height,
            mipmapped: false
        )
        desc.usage = [.shaderRead, .shaderWrite]
        desc.storageMode = .private
        return device.makeTexture(descriptor: desc)
    }

    private func encodeCopy(from input: MTLTexture,
                            to output: MTLTexture,
                            commandBuffer: MTLCommandBuffer) -> Bool {
        guard let copyPipeline = copyPipeline,
              let encoder = commandBuffer.makeComputeCommandEncoder() else {
            lastError = "Error Code: MG-ENG-011 Optical flow pipeline unavailable"
            return false
        }
        encoder.setComputePipelineState(copyPipeline)
        encoder.setTexture(input, index: 0)
        encoder.setTexture(output, index: 1)
        dispatchThreads(pipeline: copyPipeline, encoder: encoder, width: output.width, height: output.height)
        encoder.endEncoding()
        return true
    }
    
    private func flowLevels(for width: Int, height: Int) -> Int {
        var w = width
        var h = height
        var levels = 1
        while w > 32 && h > 32 && levels < flowMaxLevels {
            w = max(1, w / 2)
            h = max(1, h / 2)
            levels += 1
        }
        return levels
    }

    private func flowRefineStopLevel(for width: Int, height: Int, levels: Int) -> Int {
        let maxDim = max(width, height)
        var stop: Int
        switch qualityMode {
        case .performance:
            stop = min(2, levels - 1)
        case .balanced:
            stop = min(1, levels - 1)
        case .ultra:
            stop = 0
        }
        if maxDim >= 3840 {
            stop = max(stop, 2)
        } else if maxDim >= 2560 {
            stop = max(stop, 1)
        }
        return min(stop, levels - 1)
    }

    private func effectiveFlowCoarseRadius() -> UInt32 {
        switch qualityMode {
        case .performance:
            return 2
        case .balanced:
            return 3
        case .ultra:
            return 4
        }
    }

    private func effectiveFlowRefineRadius() -> UInt32 {
        switch qualityMode {
        case .performance:
            return 1
        case .balanced:
            return 2
        case .ultra:
            return 2
        }
    }
    
    private func makePyramidTexture(width: Int, height: Int, pixelFormat: MTLPixelFormat) -> MTLTexture? {
        let desc = MTLTextureDescriptor.texture2DDescriptor(
            pixelFormat: pixelFormat,
            width: width,
            height: height,
            mipmapped: false
        )
        desc.usage = [.shaderRead, .shaderWrite]
        desc.storageMode = .private
        return device.makeTexture(descriptor: desc)
    }
    
    private func ensurePyramid(_ pyramid: inout [MTLTexture],
                               levels: Int,
                               width: Int,
                               height: Int,
                               pixelFormat: MTLPixelFormat) -> Bool {
        if pyramid.count == levels,
           let first = pyramid.first,
           first.width == width,
           first.height == height,
           first.pixelFormat == pixelFormat {
            return true
        }
        
        pyramid.removeAll(keepingCapacity: true)
        var w = width
        var h = height
        for _ in 0..<levels {
            guard let tex = makePyramidTexture(width: w, height: h, pixelFormat: pixelFormat) else { return false }
            pyramid.append(tex)
            w = max(1, w / 2)
            h = max(1, h / 2)
        }
        return pyramid.count == levels
    }
    
    private func ensureFlowResources(width: Int, height: Int) -> Bool {
        let levels = flowLevels(for: width, height: height)
        let lumaOK = ensurePyramid(&lumaPrevPyramid, levels: levels, width: width, height: height, pixelFormat: .r16Float) &&
                     ensurePyramid(&lumaNextPyramid, levels: levels, width: width, height: height, pixelFormat: .r16Float)
        let flowOK = ensurePyramid(&flowPyramid, levels: levels, width: width, height: height, pixelFormat: .rg16Float) &&
                     ensurePyramid(&flowScratchPyramid, levels: levels, width: width, height: height, pixelFormat: .rg16Float)
        if lumaOK && flowOK {
            flowPyramidLevels = levels
            return true
        }
        return false
    }

    private func dispatchThreads(pipeline: MTLComputePipelineState,
                                 encoder: MTLComputeCommandEncoder,
                                 width: Int,
                                 height: Int) {
        let threadW = pipeline.threadExecutionWidth
        let threadH = pipeline.maxTotalThreadsPerThreadgroup / threadW
        let threadsPerGroup = MTLSize(width: threadW, height: threadH, depth: 1)
        let grid = MTLSize(width: (width + threadW - 1) / threadW,
                           height: (height + threadH - 1) / threadH,
                           depth: 1)
        encoder.dispatchThreadgroups(grid, threadsPerThreadgroup: threadsPerGroup)
    }
    
    private func computeOpticalFlow(prev: MTLTexture,
                                    next: MTLTexture,
                                    commandBuffer: MTLCommandBuffer) -> MTLTexture? {
        guard let lumaFromColorPipeline = lumaFromColorPipeline,
              let lumaDownsamplePipeline = lumaDownsamplePipeline,
              let flowInitPipeline = flowInitPipeline,
              let flowRefinePipeline = flowRefinePipeline,
              let flowUpsamplePipeline = flowUpsamplePipeline else {
            lastError = "Error Code: MG-ENG-011 Optical flow pipeline unavailable"
            return nil
        }
        
        guard ensureFlowResources(width: prev.width, height: prev.height) else {
            lastError = "Error Code: MG-ENG-012 Optical flow resources unavailable"
            return nil
        }
        
        guard let lumaPrev0 = lumaPrevPyramid.first,
              let lumaNext0 = lumaNextPyramid.first else {
            lastError = "Error Code: MG-ENG-012 Optical flow resources unavailable"
            return nil
        }
        
        // Level 0 luma from color
        if let encoder = commandBuffer.makeComputeCommandEncoder() {
            encoder.setComputePipelineState(lumaFromColorPipeline)
            encoder.setTexture(prev, index: 0)
            encoder.setTexture(lumaPrev0, index: 1)
            dispatchThreads(pipeline: lumaFromColorPipeline, encoder: encoder, width: lumaPrev0.width, height: lumaPrev0.height)
            encoder.endEncoding()
        } else {
            lastError = "Error Code: MG-ENG-011 Optical flow pipeline unavailable"
            return nil
        }
        if let encoder = commandBuffer.makeComputeCommandEncoder() {
            encoder.setComputePipelineState(lumaFromColorPipeline)
            encoder.setTexture(next, index: 0)
            encoder.setTexture(lumaNext0, index: 1)
            dispatchThreads(pipeline: lumaFromColorPipeline, encoder: encoder, width: lumaNext0.width, height: lumaNext0.height)
            encoder.endEncoding()
        } else {
            lastError = "Error Code: MG-ENG-011 Optical flow pipeline unavailable"
            return nil
        }
        
        // Downsample pyramid
        if flowPyramidLevels > 1 {
            for level in 1..<flowPyramidLevels {
                let prevIn = lumaPrevPyramid[level - 1]
                let nextIn = lumaNextPyramid[level - 1]
                let prevOut = lumaPrevPyramid[level]
                let nextOut = lumaNextPyramid[level]
                guard let prevEncoder = commandBuffer.makeComputeCommandEncoder() else {
                    lastError = "Error Code: MG-ENG-011 Optical flow pipeline unavailable"
                    return nil
                }
                prevEncoder.setComputePipelineState(lumaDownsamplePipeline)
                prevEncoder.setTexture(prevIn, index: 0)
                prevEncoder.setTexture(prevOut, index: 1)
                dispatchThreads(pipeline: lumaDownsamplePipeline, encoder: prevEncoder, width: prevOut.width, height: prevOut.height)
                prevEncoder.endEncoding()
                guard let nextEncoder = commandBuffer.makeComputeCommandEncoder() else {
                    lastError = "Error Code: MG-ENG-011 Optical flow pipeline unavailable"
                    return nil
                }
                nextEncoder.setComputePipelineState(lumaDownsamplePipeline)
                nextEncoder.setTexture(nextIn, index: 0)
                nextEncoder.setTexture(nextOut, index: 1)
                dispatchThreads(pipeline: lumaDownsamplePipeline, encoder: nextEncoder, width: nextOut.width, height: nextOut.height)
                nextEncoder.endEncoding()
            }
        }
        
        // Coarse flow init
        let lastLevel = flowPyramidLevels - 1
        let lumaPrevCoarse = lumaPrevPyramid[lastLevel]
        let lumaNextCoarse = lumaNextPyramid[lastLevel]
        let flowCoarse = flowPyramid[lastLevel]
        var initParams = FlowInitParams(searchRadius: effectiveFlowCoarseRadius())
        guard let initEncoder = commandBuffer.makeComputeCommandEncoder() else {
            lastError = "Error Code: MG-ENG-011 Optical flow pipeline unavailable"
            return nil
        }
        initEncoder.setComputePipelineState(flowInitPipeline)
        initEncoder.setTexture(lumaPrevCoarse, index: 0)
        initEncoder.setTexture(lumaNextCoarse, index: 1)
        initEncoder.setTexture(flowCoarse, index: 2)
        initEncoder.setBytes(&initParams, length: MemoryLayout<FlowInitParams>.size, index: 0)
        dispatchThreads(pipeline: flowInitPipeline, encoder: initEncoder, width: flowCoarse.width, height: flowCoarse.height)
        initEncoder.endEncoding()
        
        if flowPyramidLevels > 1 {
            let refineStopLevel = flowRefineStopLevel(for: prev.width, height: prev.height, levels: flowPyramidLevels)
            let refineRadius = effectiveFlowRefineRadius()
            if flowPyramidLevels - 1 > refineStopLevel {
                for level in stride(from: flowPyramidLevels - 2, through: refineStopLevel, by: -1) {
                    let flowUp = flowScratchPyramid[level]
                    let flowOut = flowPyramid[level]
                    let lumaPrevLevel = lumaPrevPyramid[level]
                    let lumaNextLevel = lumaNextPyramid[level]
                    var refineParams = FlowRefineParams(searchRadius: refineRadius)

                    guard let upEncoder = commandBuffer.makeComputeCommandEncoder() else {
                        lastError = "Error Code: MG-ENG-011 Optical flow pipeline unavailable"
                        return nil
                    }

                    upEncoder.setComputePipelineState(flowUpsamplePipeline)
                    upEncoder.setTexture(flowPyramid[level + 1], index: 0)
                    upEncoder.setTexture(flowUp, index: 1)
                    dispatchThreads(pipeline: flowUpsamplePipeline, encoder: upEncoder, width: flowUp.width, height: flowUp.height)
                    upEncoder.endEncoding()
                    guard let refineEncoder = commandBuffer.makeComputeCommandEncoder() else {
                        lastError = "Error Code: MG-ENG-011 Optical flow pipeline unavailable"
                        return nil
                    }
                    refineEncoder.setComputePipelineState(flowRefinePipeline)
                    refineEncoder.setTexture(lumaPrevLevel, index: 0)
                    refineEncoder.setTexture(lumaNextLevel, index: 1)
                    refineEncoder.setTexture(flowUp, index: 2)
                    refineEncoder.setTexture(flowOut, index: 3)
                    refineEncoder.setBytes(&refineParams, length: MemoryLayout<FlowRefineParams>.size, index: 0)
                    dispatchThreads(pipeline: flowRefinePipeline, encoder: refineEncoder, width: flowOut.width, height: flowOut.height)
                    refineEncoder.endEncoding()
                }
            }

            if refineStopLevel > 0 {
                for level in stride(from: refineStopLevel - 1, through: 0, by: -1) {
                    guard let upEncoder = commandBuffer.makeComputeCommandEncoder() else {
                        lastError = "Error Code: MG-ENG-011 Optical flow pipeline unavailable"
                        return nil
                    }
                    upEncoder.setComputePipelineState(flowUpsamplePipeline)
                    upEncoder.setTexture(flowPyramid[level + 1], index: 0)
                    upEncoder.setTexture(flowPyramid[level], index: 1)
                    dispatchThreads(pipeline: flowUpsamplePipeline, encoder: upEncoder, width: flowPyramid[level].width, height: flowPyramid[level].height)
                    upEncoder.endEncoding()
                }
            }
        }
        
        return flowPyramid.first
    }

    private func resetProcessingState(clearFrames: Bool = true) {
        renderTexture = nil
        scaledTexture = nil
        casTexture = nil
        usmTexture = nil
        fxaaTexture = nil
        smaaEdgeTexture = nil
        smaaWeightTexture = nil
        smaaOutputTexture = nil
        msaaTexture = nil
        taaHistoryTexture = nil
        taaOutputTexture = nil
        lumaPrevPyramid.removeAll()
        lumaNextPyramid.removeAll()
        flowPyramid.removeAll()
        flowScratchPyramid.removeAll()
        flowPyramidLevels = 0
        occlusionTexture = nil
        warpedPrevTexture = nil
        warpedNextTexture = nil
        blendTexture = nil
        hasTAAHistory = false
        lastProcessedSize = .zero
        if clearFrames {
            frameBuffer.clear()
            stats.droppedFrames = 0
            stats.interpolatedFrameCount = 0
            stats.passthroughFrameCount = 0
        }
    }

    private func resetFrameCounters() {
        stats.frameCount = 0
        stats.outputFrameCount = 0
        stats.interpolatedFrameCount = 0
        stats.passthroughFrameCount = 0
        stats.droppedFrames = 0
        stats.captureFPS = 0
        stats.outputFPS = 0
        stats.interpolatedFPS = 0
        frameCount = 0
        renderFrameCount = 0
        interpolatedFrameCount = 0
        fpsStartTime = CACurrentMediaTime()
        renderFPSStartTime = CACurrentMediaTime()
    }

    private func effectiveSharpness() -> Float {
        return sharpness * qualityProfile.sharpnessScale
    }

    private func effectiveTemporalBlend() -> Float {
        return temporalBlend * qualityProfile.temporalBlendScale
    }

    private func desiredOutputFPS() -> Int {
        var target: Int
        let captureFPS: Double = {
            if estimatedCaptureInterval > 0 {
                return 1.0 / estimatedCaptureInterval
            }
            if stats.captureFPS > 0 {
                return Double(stats.captureFPS)
            }
            return 0
        }()
        if frameGenEnabled {
            switch frameGenType {
            case .adaptive:
                let maxGenFPS = captureFPS > 0 ? Int(round(captureFPS * Double(frameGenMultiplier))) : targetFPS
                target = min(targetFPS, maxGenFPS)
            case .fixed:
                let maxGenFPS = captureFPS > 0 ? Int(round(captureFPS * Double(frameGenMultiplier))) : targetFPS
                target = maxGenFPS
            }
        } else if adaptiveSync {
            let capture = Int(round(captureFPS))
            target = capture
        } else {
            target = currentRefreshRate
        }

        if currentRefreshRate > 0 {
            target = min(target, currentRefreshRate)
        }
        return target
    }

    private func applyFrameRatePreference(_ preferred: Int) {
        let target = preferred
        guard target > 0 else { return }
        let now = CACurrentMediaTime()
        if abs(target - lastPreferredFPS) < 3 { return }
        if now - lastPreferredUpdateTime < 0.5 { return }
        if target != lastPreferredFPS {
            mtkView?.preferredFramesPerSecond = target
            lastPreferredFPS = target
            lastPreferredUpdateTime = now
        }
    }
    
    private func applyDisplaySync(to view: MTKView) {
        guard let layer = view.layer as? CAMetalLayer else { return }
        if #available(macOS 10.13, *) {
            layer.displaySyncEnabled = vsyncEnabled
        }
        layer.presentsWithTransaction = false
    }

    private func interpolationDelay(for targetFPS: Int) -> Double {
        guard targetFPS > 0 else { return 0 }
        let outputInterval = 1.0 / Double(targetFPS)
        let captureInterval = estimatedCaptureInterval
        if frameGenEnabled {
            let genInterval = captureInterval / Double(frameGenMultiplier)
            return outputInterval >= genInterval ? outputInterval : genInterval
        }
        return captureInterval >= outputInterval ? captureInterval : outputInterval
    }
    
    func attachToView(_ view: MTKView, refreshRate: Int) {
        view.device = device
        view.delegate = self
        view.preferredFramesPerSecond = refreshRate
        view.isPaused = false
        view.enableSetNeedsDisplay = false
        view.colorPixelFormat = .bgra8Unorm
        view.framebufferOnly = false
        applyDisplaySync(to: view)
        self.mtkView = view
        self.currentRefreshRate = refreshRate
        applyFrameRatePreference(desiredOutputFPS())
    }
    
    func detachFromView() {
        mtkView?.delegate = nil
        mtkView?.isPaused = true
        mtkView = nil
    }
    
    nonisolated func draw(in view: MTKView) {
        Task { @MainActor in
            renderFrame(in: view)
        }
    }
    
    private var renderFrameCount: Int = 0
    private var renderFPSStartTime: CFTimeInterval = 0
    private var interpolatedFrameCount: Int = 0
    
    private func renderFrame(in view: MTKView) {
        guard let drawable = view.currentDrawable,
              let renderPipeline = renderPipeline,
              let commandBuffer = commandQueue.makeCommandBuffer() else { return }
        
        let currentTime = CACurrentMediaTime()
        
        if renderFPSStartTime == 0 { renderFPSStartTime = currentTime }
        let elapsed = currentTime - renderFPSStartTime

        let targetFPS = desiredOutputFPS()
        applyFrameRatePreference(targetFPS)
        let targetTime = currentTime - interpolationDelay(for: targetFPS)

        var outputTex: MTLTexture?
        let frameGenActive = frameGenEnabled
        var isInterpolated = false

        if frameGenEnabled {
            guard let (prev, next) = frameBuffer.getFramesForTime(targetTime) else {
                stats.droppedFrames += 1
                lastError = "Error Code: MG-ENG-006 Frame interpolation failed: missing frame pair"
                commandBuffer.commit()
                return
            }
            let sameSize = prev.texture.width == next.texture.width &&
                           prev.texture.height == next.texture.height
            guard sameSize else {
                stats.droppedFrames += 1
                lastError = "Error Code: MG-ENG-006 Frame interpolation failed: size mismatch"
                commandBuffer.commit()
                return
            }
            let duration = next.timestamp - prev.timestamp
            let timeSincePrev = targetTime - prev.timestamp
            let ratio = duration > 0 ? (timeSincePrev / duration) : 0
            let t = Float(min(max(ratio, 0), 1))
            guard let interpolated = interpolateFrame(prev: prev, next: next, t: t, commandBuffer: commandBuffer) else {
                stats.droppedFrames += 1
                lastError = "Error Code: MG-ENG-006 Frame interpolation failed: pipeline error"
                commandBuffer.commit()
                return
            }
            outputTex = interpolated
            isInterpolated = true
        } else {
            outputTex = frameBuffer.newestFrame?.texture
        }
        
        guard let finalTex = outputTex else {
            commandBuffer.commit()
            return
        }
        
        renderFrameCount += 1
        if isInterpolated { interpolatedFrameCount += 1 }
        if elapsed >= 1.0 {
            stats.outputFPS = Float(renderFrameCount) / Float(elapsed)
            stats.interpolatedFPS = Float(interpolatedFrameCount) / Float(elapsed)
            renderFrameCount = 0
            interpolatedFrameCount = 0
            renderFPSStartTime = currentTime
        }
        
        let renderPassDesc = MTLRenderPassDescriptor()
        renderPassDesc.colorAttachments[0].texture = drawable.texture
        renderPassDesc.colorAttachments[0].loadAction = .clear
        renderPassDesc.colorAttachments[0].storeAction = .store
        renderPassDesc.colorAttachments[0].clearColor = MTLClearColor(red: 0, green: 0, blue: 0, alpha: 1)
        
        guard let renEncoder = commandBuffer.makeRenderCommandEncoder(descriptor: renderPassDesc) else { return }
        renEncoder.setRenderPipelineState(renderPipeline)
        renEncoder.setFragmentTexture(finalTex, index: 0)
        renEncoder.drawPrimitives(type: .triangleStrip, vertexStart: 0, vertexCount: 4)
        renEncoder.endEncoding()
        
        commandBuffer.present(drawable)
        commandBuffer.addCompletedHandler { _ in
            Task { @MainActor in
                self.stats.outputFrameCount += 1
                if frameGenActive {
                    if isInterpolated {
                        self.stats.interpolatedFrameCount += 1
                    } else {
                        self.stats.passthroughFrameCount += 1
                    }
                } else {
                    self.stats.passthroughFrameCount += 1
                }
            }
        }
        commandBuffer.commit()
    }
    
    private func updateCaptureStats(currentTime: CFTimeInterval, captureTimestamp: CFTimeInterval?) {
        frameCount += 1
        stats.frameCount += 1
        stats.gpuMemoryUsed = UInt64(device.currentAllocatedSize)
        stats.gpuMemoryTotal = UInt64(device.recommendedMaxWorkingSetSize)

        if lastCaptureTimestamp > 0 {
            let interval = currentTime - lastCaptureTimestamp
            if interval > 0 {
                estimatedCaptureInterval = (estimatedCaptureInterval * 0.9) + (interval * 0.1)
            }
        }
        lastCaptureTimestamp = currentTime

        let elapsed = currentTime - fpsStartTime
        if elapsed >= 1.0 {
            stats.captureFPS = Float(frameCount) / Float(elapsed)
            frameCount = 0
            fpsStartTime = currentTime
            applyFrameRatePreference(desiredOutputFPS())
        }

        let delta = currentTime - lastFrameTime
        if lastFrameTime > 0 {
            stats.frameTime = Float(delta * 1000.0)
        }
        if let captureTimestamp, captureTimestamp > 0 {
            stats.captureLatency = Float((currentTime - captureTimestamp) * 1000.0)
        } else {
            stats.captureLatency = Float(delta * 1000.0)
        }
        lastFrameTime = currentTime
    }

    private func processSurface(_ surface: IOSurfaceRef, timestamp: CFTimeInterval) {
        let w = IOSurfaceGetWidth(surface)
        let h = IOSurfaceGetHeight(surface)

        let desc = MTLTextureDescriptor.texture2DDescriptor(
            pixelFormat: .bgra8Unorm,
            width: w,
            height: h,
            mipmapped: false
        )
        desc.usage = [.shaderRead]
        guard let inputTex = device.makeTexture(descriptor: desc, iosurface: surface, plane: 0) else {
            lastError = "Error Code: MG-ENG-010 IOSurface texture creation failed"
            return
        }

        processCapturedTexture(inputTex, timestamp: timestamp)
    }

    private func processCapturedTexture(_ inputTex: MTLTexture, timestamp: CFTimeInterval) {
        guard let commandBuffer = commandQueue.makeCommandBuffer() else { return }

        let inputWidth = inputTex.width
        let inputHeight = inputTex.height
        let shouldScale = scalingType != .off
        let renderScale = shouldScale ? renderScaleFactor : 1.0
        let renderWidth = Int(round(Float(inputWidth) * renderScale))
        let renderHeight = Int(round(Float(inputHeight) * renderScale))

        let targetScale = shouldScale ? scaleFactor : 1.0
        let targetWidth = Int(round(Float(renderWidth) * targetScale))
        let targetHeight = Int(round(Float(renderHeight) * targetScale))

        let targetSize = CGSize(width: targetWidth, height: targetHeight)
        if targetSize != lastProcessedSize {
            resetProcessingState(clearFrames: true)
            lastProcessedSize = targetSize
        }

        var workingTex = inputTex

        if renderWidth != inputWidth || renderHeight != inputHeight {
            guard let scalePipeline = scalePipeline,
                  let renderTex = ensureTexture(&renderTexture, width: renderWidth, height: renderHeight),
                  let encoder = commandBuffer.makeComputeCommandEncoder() else {
                lastError = "Error Code: MG-ENG-008 Scale pipeline unavailable"
                commandBuffer.commit()
                return
            }
            encoder.setComputePipelineState(scalePipeline)
            encoder.setTexture(inputTex, index: 0)
            encoder.setTexture(renderTex, index: 1)
            dispatchThreads(pipeline: scalePipeline, encoder: encoder, width: renderWidth, height: renderHeight)
            encoder.endEncoding()
            workingTex = renderTex
        }

        var scaledTex = workingTex
        if shouldScale || targetWidth != workingTex.width || targetHeight != workingTex.height {
            // MetalFX Spatial Scaler requires .renderTarget usage flag
            guard let outputTex = ensureTexture(&scaledTexture, width: targetWidth, height: targetHeight,
                                                 usage: [.shaderRead, .shaderWrite, .renderTarget]) else {
                lastError = "Error Code: MG-ENG-008 Scale pipeline unavailable"
                commandBuffer.commit()
                return
            }
            let baseSharpness = effectiveSharpness()

            switch scalingType {
            case .mgup1:
                // MGUP-1: MetalFX Spatial Scaler + quality-based CAS sharpening
                guard let scaler = ensureMetalFXSpatialScaler(
                    inputWidth: workingTex.width, inputHeight: workingTex.height,
                    outputWidth: targetWidth, outputHeight: targetHeight
                ) else {
                    lastError = "Error Code: MG-ENG-004 MetalFX Spatial Scaler creation failed"
                    commandBuffer.commit()
                    return
                }
                scaler.colorTexture = workingTex
                scaler.outputTexture = outputTex
                scaler.encode(commandBuffer: commandBuffer)
                scaledTex = outputTex

                // Apply CAS sharpening based on quality mode
                if baseSharpness > 0.01 {
                    guard let casPipeline = casPipeline,
                          let casOut = ensureTexture(&casTexture, width: targetWidth, height: targetHeight),
                          let encoder = commandBuffer.makeComputeCommandEncoder() else {
                        lastError = "Error Code: MG-ENG-009 CAS pipeline unavailable"
                        commandBuffer.commit()
                        return
                    }
                    var params = SharpenParams(sharpness: baseSharpness, radius: 1.0)
                    encoder.setComputePipelineState(casPipeline)
                    encoder.setTexture(scaledTex, index: 0)
                    encoder.setTexture(casOut, index: 1)
                    encoder.setBytes(&params, length: MemoryLayout<SharpenParams>.size, index: 0)
                    dispatchThreads(pipeline: casPipeline, encoder: encoder, width: targetWidth, height: targetHeight)
                    encoder.endEncoding()
                    scaledTex = casOut
                }

            case .off:
                scaledTex = workingTex
            }
        }

        guard let finalTex = applyAntiAliasing(to: scaledTex, commandBuffer: commandBuffer) else {
            stats.droppedFrames += 1
            commandBuffer.commit()
            return
        }

        stats.outputResolution = CGSize(width: finalTex.width, height: finalTex.height)

        var flowFromPrev: MTLTexture?
        var flowToPrev: MTLTexture?
        if frameGenEnabled, let prevFrame = frameBuffer.newestFrame {
            let prevTex = prevFrame.texture
            if prevTex.width == finalTex.width && prevTex.height == finalTex.height {
                if let forwardTmp = computeOpticalFlow(prev: prevTex, next: finalTex, commandBuffer: commandBuffer),
                   let forwardOut = makeFlowTexture(width: finalTex.width, height: finalTex.height),
                   encodeCopy(from: forwardTmp, to: forwardOut, commandBuffer: commandBuffer) {
                    if let backwardTmp = computeOpticalFlow(prev: finalTex, next: prevTex, commandBuffer: commandBuffer),
                       let backwardOut = makeFlowTexture(width: finalTex.width, height: finalTex.height),
                       encodeCopy(from: backwardTmp, to: backwardOut, commandBuffer: commandBuffer) {
                        flowFromPrev = forwardOut
                        flowToPrev = backwardOut
                    } else {
                        lastError = "Error Code: MG-ENG-013 Frame generation pipeline unavailable"
                    }
                } else {
                    lastError = "Error Code: MG-ENG-013 Frame generation pipeline unavailable"
                }
            } else {
                lastError = "Error Code: MG-ENG-006 Frame interpolation failed: size mismatch"
            }
        }

        commandBuffer.addCompletedHandler { [weak self] buffer in
            guard let self = self else { return }
            let gpuTime = buffer.gpuEndTime - buffer.gpuStartTime
            Task { @MainActor in
                self.stats.gpuTime = Float(gpuTime * 1000.0)
            }
        }
        commandBuffer.commit()

        frameBuffer.push(FrameHistory(texture: finalTex, timestamp: timestamp, flowFromPrev: flowFromPrev, flowToPrev: flowToPrev))
    }

    private struct UpscaleParams {
        var sharpness: Float
        var inputSize: SIMD2<UInt32>
        var outputSize: SIMD2<UInt32>
    }

    private struct SharpenParams {
        var sharpness: Float
        var radius: Float
    }

    private struct AntiAliasParams {
        var threshold: Float
        var depthThreshold: Float
        var maxSearchSteps: Int32
        var subpixelBlend: Float
    }

    private struct FlowInitParams {
        var searchRadius: UInt32
    }

    private struct FlowRefineParams {
        var searchRadius: UInt32
    }

    private struct FlowWarpParams {
        var scale: Float
    }

    private struct FlowComposeParams {
        var t: Float
        var errorThreshold: Float
    }

    private struct FlowOcclusionParams {
        var threshold: Float
    }
    
    private struct TemporalParams {
        var blendFactor: Float
    }

    private func applyAntiAliasing(to input: MTLTexture,
                                   commandBuffer: MTLCommandBuffer) -> MTLTexture? {
        switch aaMode {
        case .off:
            return input
        case .fxaa:
            guard let fxaaPipeline = fxaaPipeline,
                  let out = ensureTexture(&fxaaTexture, width: input.width, height: input.height),
                  let encoder = commandBuffer.makeComputeCommandEncoder() else {
                lastError = "Error Code: MG-ENG-007 Anti-aliasing pipeline unavailable (FXAA)"
                return nil
            }
            var threshold = qualityProfile.aaThreshold
            encoder.setComputePipelineState(fxaaPipeline)
            encoder.setTexture(input, index: 0)
            encoder.setTexture(out, index: 1)
            encoder.setBytes(&threshold, length: MemoryLayout<Float>.size, index: 0)
            dispatchThreads(pipeline: fxaaPipeline, encoder: encoder, width: input.width, height: input.height)
            encoder.endEncoding()
            return out
        case .smaa:
            guard let edgePipe = smaaEdgePipeline,
                  let weightPipe = smaaWeightPipeline,
                  let blendPipe = smaaBlendPipeline,
                  let edges = ensureTexture(&smaaEdgeTexture, width: input.width, height: input.height),
                  let weights = ensureTexture(&smaaWeightTexture, width: input.width, height: input.height),
                  let out = ensureTexture(&smaaOutputTexture, width: input.width, height: input.height) else {
                lastError = "Error Code: MG-ENG-007 Anti-aliasing pipeline unavailable (SMAA)"
                return nil
            }
            var params = AntiAliasParams(
                threshold: qualityProfile.aaThreshold,
                depthThreshold: 0.1,
                maxSearchSteps: Int32(qualityProfile.smaaSearchSteps),
                subpixelBlend: 0.75
            )
            guard let edgeEncoder = commandBuffer.makeComputeCommandEncoder() else {
                lastError = "Error Code: MG-ENG-007 Anti-aliasing pipeline unavailable (SMAA)"
                return nil
            }
            edgeEncoder.setComputePipelineState(edgePipe)
            edgeEncoder.setTexture(input, index: 0)
            edgeEncoder.setTexture(edges, index: 1)
            edgeEncoder.setBytes(&params, length: MemoryLayout<AntiAliasParams>.size, index: 0)
            dispatchThreads(pipeline: edgePipe, encoder: edgeEncoder, width: input.width, height: input.height)
            edgeEncoder.endEncoding()

            guard let weightEncoder = commandBuffer.makeComputeCommandEncoder() else {
                lastError = "Error Code: MG-ENG-007 Anti-aliasing pipeline unavailable (SMAA)"
                return nil
            }
            weightEncoder.setComputePipelineState(weightPipe)
            weightEncoder.setTexture(edges, index: 0)
            weightEncoder.setTexture(weights, index: 1)
            weightEncoder.setBytes(&params, length: MemoryLayout<AntiAliasParams>.size, index: 0)
            dispatchThreads(pipeline: weightPipe, encoder: weightEncoder, width: input.width, height: input.height)
            weightEncoder.endEncoding()

            guard let blendEncoder = commandBuffer.makeComputeCommandEncoder() else {
                lastError = "Error Code: MG-ENG-007 Anti-aliasing pipeline unavailable (SMAA)"
                return nil
            }
            blendEncoder.setComputePipelineState(blendPipe)
            blendEncoder.setTexture(input, index: 0)
            blendEncoder.setTexture(weights, index: 1)
            blendEncoder.setTexture(out, index: 2)
            dispatchThreads(pipeline: blendPipe, encoder: blendEncoder, width: input.width, height: input.height)
            blendEncoder.endEncoding()
            return out
        case .msaa:
            guard let msaaPipeline = msaaPipeline,
                  let out = ensureTexture(&msaaTexture, width: input.width, height: input.height),
                  let encoder = commandBuffer.makeComputeCommandEncoder() else {
                lastError = "Error Code: MG-ENG-007 Anti-aliasing pipeline unavailable (MSAA)"
                return nil
            }
            var params = AntiAliasParams(
                threshold: qualityProfile.aaThreshold,
                depthThreshold: 0.1,
                maxSearchSteps: 8,
                subpixelBlend: 0.5
            )
            encoder.setComputePipelineState(msaaPipeline)
            encoder.setTexture(input, index: 0)
            encoder.setTexture(out, index: 1)
            encoder.setBytes(&params, length: MemoryLayout<AntiAliasParams>.size, index: 0)
            dispatchThreads(pipeline: msaaPipeline, encoder: encoder, width: input.width, height: input.height)
            encoder.endEncoding()
            return out
        case .taa:
            guard let temporalPipeline = temporalPipeline,
                  let copyPipeline = copyPipeline,
                  let history = ensureTexture(&taaHistoryTexture, width: input.width, height: input.height),
                  let out = ensureTexture(&taaOutputTexture, width: input.width, height: input.height) else {
                lastError = "Error Code: MG-ENG-007 Anti-aliasing pipeline unavailable (TAA)"
                return nil
            }

            if !hasTAAHistory {
                guard let encoder = commandBuffer.makeComputeCommandEncoder() else {
                    lastError = "Error Code: MG-ENG-007 Anti-aliasing pipeline unavailable (TAA)"
                    return nil
                }
                encoder.setComputePipelineState(copyPipeline)
                encoder.setTexture(input, index: 0)
                encoder.setTexture(history, index: 1)
                dispatchThreads(pipeline: copyPipeline, encoder: encoder, width: input.width, height: input.height)
                encoder.endEncoding()
                hasTAAHistory = true
                return input
            }

            guard let flow = computeOpticalFlow(prev: input, next: history, commandBuffer: commandBuffer) else {
                return nil
            }
            var params = TemporalParams(blendFactor: effectiveTemporalBlend())
            guard let encoder = commandBuffer.makeComputeCommandEncoder() else {
                lastError = "Error Code: MG-ENG-007 Anti-aliasing pipeline unavailable (TAA)"
                return nil
            }
            encoder.setComputePipelineState(temporalPipeline)
            encoder.setTexture(input, index: 0)
            encoder.setTexture(history, index: 1)
            encoder.setTexture(flow, index: 2)
            encoder.setTexture(out, index: 3)
            encoder.setBytes(&params, length: MemoryLayout<TemporalParams>.size, index: 0)
            dispatchThreads(pipeline: temporalPipeline, encoder: encoder, width: input.width, height: input.height)
            encoder.endEncoding()

            guard let copyEncoder = commandBuffer.makeComputeCommandEncoder() else {
                lastError = "Error Code: MG-ENG-007 Anti-aliasing pipeline unavailable (TAA)"
                return nil
            }
            copyEncoder.setComputePipelineState(copyPipeline)
            copyEncoder.setTexture(out, index: 0)
            copyEncoder.setTexture(history, index: 1)
            dispatchThreads(pipeline: copyPipeline, encoder: copyEncoder, width: input.width, height: input.height)
            copyEncoder.endEncoding()
            return out
        }
    }

    private func interpolateFrame(prev: FrameHistory,
                                  next: FrameHistory,
                                  t: Float,
                                  commandBuffer: MTLCommandBuffer) -> MTLTexture? {
        let prevTex = prev.texture
        let nextTex = next.texture
        guard let output = ensureTexture(&blendTexture, width: prevTex.width, height: prevTex.height) else { return nil }

        switch frameGenMode {
        case .mgfg1:
            guard let flowWarpPipeline = flowWarpPipeline,
                  let flowComposePipeline = flowComposePipeline,
                  let flowOcclusionPipeline = flowOcclusionPipeline else {
                lastError = "Error Code: MG-ENG-013 Frame generation pipeline unavailable"
                return nil
            }
            guard let flowForward = next.flowFromPrev,
                  let flowBackward = next.flowToPrev else {
                lastError = "Error Code: MG-ENG-013 Frame generation pipeline unavailable"
                return nil
            }
            guard flowForward.width == prevTex.width,
                  flowForward.height == prevTex.height,
                  flowBackward.width == prevTex.width,
                  flowBackward.height == prevTex.height else {
                lastError = "Error Code: MG-ENG-006 Frame interpolation failed: size mismatch"
                return nil
            }
            guard let occlusion = ensureTexture(&occlusionTexture, width: prevTex.width, height: prevTex.height, pixelFormat: .r16Float),
                  let warpPrev = ensureTexture(&warpedPrevTexture, width: prevTex.width, height: prevTex.height),
                  let warpNext = ensureTexture(&warpedNextTexture, width: prevTex.width, height: prevTex.height) else {
                lastError = "Error Code: MG-ENG-013 Frame generation pipeline unavailable"
                return nil
            }

            var occParams = FlowOcclusionParams(threshold: flowOcclusionThreshold)
            guard let occEncoder = commandBuffer.makeComputeCommandEncoder() else {
                lastError = "Error Code: MG-ENG-013 Frame generation pipeline unavailable"
                return nil
            }
            occEncoder.setComputePipelineState(flowOcclusionPipeline)
            occEncoder.setTexture(flowForward, index: 0)
            occEncoder.setTexture(flowBackward, index: 1)
            occEncoder.setTexture(occlusion, index: 2)
            occEncoder.setBytes(&occParams, length: MemoryLayout<FlowOcclusionParams>.size, index: 0)
            dispatchThreads(pipeline: flowOcclusionPipeline, encoder: occEncoder, width: occlusion.width, height: occlusion.height)
            occEncoder.endEncoding()

            var warpPrevParams = FlowWarpParams(scale: t)
            guard let warpPrevEncoder = commandBuffer.makeComputeCommandEncoder() else {
                lastError = "Error Code: MG-ENG-013 Frame generation pipeline unavailable"
                return nil
            }
            warpPrevEncoder.setComputePipelineState(flowWarpPipeline)
            warpPrevEncoder.setTexture(prevTex, index: 0)
            warpPrevEncoder.setTexture(flowForward, index: 1)
            warpPrevEncoder.setTexture(warpPrev, index: 2)
            warpPrevEncoder.setBytes(&warpPrevParams, length: MemoryLayout<FlowWarpParams>.size, index: 0)
            dispatchThreads(pipeline: flowWarpPipeline, encoder: warpPrevEncoder, width: warpPrev.width, height: warpPrev.height)
            warpPrevEncoder.endEncoding()

            var warpNextParams = FlowWarpParams(scale: (1.0 - t))
            guard let warpNextEncoder = commandBuffer.makeComputeCommandEncoder() else {
                lastError = "Error Code: MG-ENG-013 Frame generation pipeline unavailable"
                return nil
            }
            warpNextEncoder.setComputePipelineState(flowWarpPipeline)
            warpNextEncoder.setTexture(nextTex, index: 0)
            warpNextEncoder.setTexture(flowBackward, index: 1)
            warpNextEncoder.setTexture(warpNext, index: 2)
            warpNextEncoder.setBytes(&warpNextParams, length: MemoryLayout<FlowWarpParams>.size, index: 0)
            dispatchThreads(pipeline: flowWarpPipeline, encoder: warpNextEncoder, width: warpNext.width, height: warpNext.height)
            warpNextEncoder.endEncoding()

            var composeParams = FlowComposeParams(t: t, errorThreshold: qualityProfile.frameGenGradientThreshold)
            guard let composeEncoder = commandBuffer.makeComputeCommandEncoder() else {
                lastError = "Error Code: MG-ENG-013 Frame generation pipeline unavailable"
                return nil
            }
            composeEncoder.setComputePipelineState(flowComposePipeline)
            composeEncoder.setTexture(warpPrev, index: 0)
            composeEncoder.setTexture(warpNext, index: 1)
            composeEncoder.setTexture(occlusion, index: 2)
            composeEncoder.setTexture(output, index: 3)
            composeEncoder.setBytes(&composeParams, length: MemoryLayout<FlowComposeParams>.size, index: 0)
            dispatchThreads(pipeline: flowComposePipeline, encoder: composeEncoder, width: output.width, height: output.height)
            composeEncoder.endEncoding()
            return output
        case .off:
            return nil
        }
    }
    
    func configure(
        virtualResolution: CGSize,
        outputSize: CGSize
    ) {
        if self.outputSize != outputSize {
            resetProcessingState(clearFrames: true)
        }
        self.outputSize = outputSize
        stats.virtualResolution = virtualResolution
        stats.outputResolution = outputSize
        stats.isUsingVirtualDisplay = true
    }
    
    func updateSettings(_ settings: CaptureSettings) {
        let cursorChanged = settings.captureCursor != captureCursor
        let newScalingType = settings.scalingType
        let newQualityMode = settings.qualityMode
        let newAAMode = settings.aaMode
        let shouldScale = newScalingType != .off
        let newRenderScale = shouldScale ? settings.renderScale.multiplier : 1.0
        let newScaleFactor = shouldScale ? settings.scaleFactor.floatValue : 1.0
        let newProfile = newQualityMode.profile
        
        let pipelineChanged =
            newScalingType != scalingType ||
            newQualityMode != qualityMode ||
            newAAMode != aaMode ||
            abs(newRenderScale - renderScaleFactor) > 0.001 ||
            abs(newScaleFactor - scaleFactor) > 0.001
        
        if pipelineChanged {
            resetProcessingState(clearFrames: true)
        }
        
        scalingType = newScalingType
        qualityMode = newQualityMode
        aaMode = newAAMode
        renderScaleFactor = newRenderScale
        scaleFactor = newScaleFactor
        qualityProfile = newProfile
        sharpness = settings.sharpening
        temporalBlend = settings.temporalBlend
        motionScale = settings.motionScale
        captureCursor = settings.captureCursor
        frameGenMode = settings.frameGenMode
        frameGenEnabled = settings.frameGenMode != .off
        frameGenType = settings.frameGenType
        targetFPS = settings.targetFPS.intValue
        frameGenMultiplier = settings.frameGenMultiplier.intValue
        adaptiveSync = settings.adaptiveSync
        vsyncEnabled = settings.vsync
        
        applyFrameRatePreference(desiredOutputFPS())
        if let view = mtkView {
            applyDisplaySync(to: view)
        }

        if cursorChanged, isCapturing {
            restartCaptureForCursorChange()
        }

        if !frameGenEnabled {
            stats.droppedFrames = 0
            stats.interpolatedFrameCount = 0
            stats.passthroughFrameCount = 0
        }
    }

    private func restartCaptureForCursorChange() {
        if let manager = virtualDisplayManager {
            Task { @MainActor in
                _ = await startCaptureFromVirtualDisplay(manager, refreshRate: captureRefreshRate)
            }
        }
    }
    
    func startCaptureFromVirtualDisplay(_ virtualDisplayManager: VirtualDisplayManager, refreshRate: Int) async -> Bool {
        guard virtualDisplayManager.isActive else {
            lastError = "Error Code: MG-VD-006 No active virtual display"
            return false
        }
        
        resetProcessingState(clearFrames: true)
        resetFrameCounters()
        self.virtualDisplayManager = virtualDisplayManager
        
        virtualDisplayManager.onFrameReceived = { [weak self] surface, timestamp in
            guard let self = self else { return }
            Task { @MainActor in
                self.processIOSurfaceFrame(surface: surface, timestamp: timestamp)
            }
        }
        
        guard await virtualDisplayManager.startFrameCapture(refreshRate: refreshRate, showsCursor: captureCursor) else {
            lastError = virtualDisplayManager.lastError!
            isCapturing = false
            return false
        }
        
        self.isCapturing = true
        self.captureRefreshRate = refreshRate
        self.frameCount = 0
        self.fpsStartTime = CACurrentMediaTime()
        self.lastCaptureTimestamp = 0
        
        return true
    }
    
    private func processIOSurfaceFrame(surface: IOSurfaceRef, timestamp: Double) {
        let currentTime = CACurrentMediaTime()
        updateCaptureStats(currentTime: currentTime, captureTimestamp: timestamp)
        processSurface(surface, timestamp: currentTime)
    }
    
    func stopCapture() async {
        isCapturing = false
        if let manager = virtualDisplayManager {
            await manager.stopFrameCapture()
            manager.onFrameReceived = nil
        }
        lastCaptureTimestamp = 0
    }
    
    nonisolated func mtkView(_ view: MTKView, drawableSizeWillChange size: CGSize) {}
}
