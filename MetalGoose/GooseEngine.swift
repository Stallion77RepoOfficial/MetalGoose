import Foundation
import AppKit
@preconcurrency import Metal
@preconcurrency import MetalKit
import ScreenCaptureKit
import IOSurface
import QuartzCore
import CoreMedia

struct PipelineStats {
    var captureFPS: Float = 0
    var outputFPS: Float = 0
    var interpolatedFPS: Float = 0
    var frameTime: Float = 0
    var gpuTime: Float = 0
    var captureLatency: Float = 0
    var frameCount: UInt64 = 0
    var droppedFrames: UInt64 = 0
    var interpolatedFrameCount: UInt64 = 0
    var gpuMemoryUsed: UInt64 = 0
    var gpuMemoryTotal: UInt64 = 0
    var isUsingVirtualDisplay: Bool = false
    var virtualResolution: CGSize = .zero
    var outputResolution: CGSize = .zero
}

@available(macOS 26.0, *)
@MainActor
final class GooseEngine: NSObject, ObservableObject, SCStreamDelegate, SCStreamOutput, MTKViewDelegate {
    
    @Published private(set) var isCapturing: Bool = false
    @Published private(set) var stats: PipelineStats = PipelineStats()
    @Published private(set) var lastError: String?
    
    var deviceName: String { device.name }
    
    var onFrameReady: ((MTLTexture) -> Void)?
    var onWindowLost: (() -> Void)?
    
    private var policyFrameGenEnabled: Bool { frameGenEnabled }
    
    private let device: MTLDevice
    private let commandQueue: MTLCommandQueue
    private var upscalePipeline: MTLComputePipelineState?
    private var blendPipeline: MTLComputePipelineState?
    private var fxaaPipeline: MTLComputePipelineState?
    private var passthroughPipeline: MTLComputePipelineState?
    
    private var renderPipeline: MTLRenderPipelineState?
    private weak var mtkView: MTKView?
    private var currentFrameTexture: MTLTexture?
    
    private var captureTexture: MTLTexture?
    private var upscaledTexture: MTLTexture?
    private var previousTexture: MTLTexture?
    private var outputTexture: MTLTexture?
    
    struct FrameHistory {
        let texture: MTLTexture
        let timestamp: CFTimeInterval
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
    }
    
    private let frameBuffer = FrameRingBuffer()
    private var blendTexture: MTLTexture?

    private var stream: SCStream?
    private var streamConfig: SCStreamConfiguration?
    private var contentFilter: SCContentFilter?
    
    private var virtualDisplayManager: VirtualDisplayManager?
    
    private var lastFrameTime: CFTimeInterval = 0
    private var frameCount: Int = 0
    private var fpsStartTime: CFTimeInterval = 0
    
    private var outputSize: CGSize = .zero
    private var sharpness: Float = 0.5
    private var frameGenEnabled: Bool = false
    
    init?(device: MTLDevice? = nil) {
        guard let dev = device ?? MTLCreateSystemDefaultDevice() else {
            return nil
        }
        guard let queue = dev.makeCommandQueue() else {
            return nil
        }
        
        self.device = dev
        self.commandQueue = queue
        
        super.init()
        setupPipelines()
    }
    
    private func setupPipelines() {
        guard let library = device.makeDefaultLibrary() else { return }
        
        do {
            if let upscaleFunc = library.makeFunction(name: "mgup1_upscale") {
                upscalePipeline = try device.makeComputePipelineState(function: upscaleFunc)
            }
            if let blendFunc = library.makeFunction(name: "mgfg1_blend") {
                blendPipeline = try device.makeComputePipelineState(function: blendFunc)
            }
            if let vtx = library.makeFunction(name: "texture_vertex"),
               let frag = library.makeFunction(name: "texture_fragment") {
                let desc = MTLRenderPipelineDescriptor()
                desc.vertexFunction = vtx
                desc.fragmentFunction = frag
                desc.colorAttachments[0].pixelFormat = .bgra8Unorm
                renderPipeline = try device.makeRenderPipelineState(descriptor: desc)
            }
        } catch {
            lastError = "Pipeline setup failed: \(error)"
        }
    }
    
    func attachToView(_ view: MTKView) {
        view.device = device
        view.delegate = self
        view.preferredFramesPerSecond = 120
        view.isPaused = false
        view.enableSetNeedsDisplay = false
        view.colorPixelFormat = .bgra8Unorm
        view.framebufferOnly = false
        self.mtkView = view
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
        renderFrameCount += 1
        
        let elapsed = currentTime - renderFPSStartTime
        if elapsed >= 1.0 {
            stats.outputFPS = Float(renderFrameCount) / Float(elapsed)
            stats.interpolatedFPS = Float(interpolatedFrameCount) / Float(elapsed)
            
            renderFrameCount = 0
            interpolatedFrameCount = 0
            renderFPSStartTime = currentTime
        }
        
        let targetTime = currentTime - 0.032
        
        var outputTex: MTLTexture?
        var isInterpolated = false
        
        if frameGenEnabled {
            if let (prev, next) = frameBuffer.getFramesForTime(targetTime) {
                let duration = next.timestamp - prev.timestamp
                if duration > 0.001 {
                    let timeSincePrev = targetTime - prev.timestamp
                    let t = Float(max(0, min(1, timeSincePrev / duration)))
                    
                    if let blendPipe = blendPipeline, let encoder = commandBuffer.makeComputeCommandEncoder() {
                        if blendTexture == nil || blendTexture?.width != prev.texture.width || blendTexture?.height != prev.texture.height {
                            let desc = MTLTextureDescriptor.texture2DDescriptor(
                                pixelFormat: .bgra8Unorm,
                                width: prev.texture.width,
                                height: prev.texture.height,
                                mipmapped: false
                            )
                            desc.usage = [.shaderRead, .shaderWrite]
                            blendTexture = device.makeTexture(descriptor: desc)
                        }
                        
                        if let blendOut = blendTexture {
                            var params = FrameBlendParams(t: t, textureSize: SIMD2(UInt32(blendOut.width), UInt32(blendOut.height)))
                            
                            encoder.setComputePipelineState(blendPipe)
                            encoder.setTexture(prev.texture, index: 0)
                            encoder.setTexture(next.texture, index: 1)
                            encoder.setTexture(blendOut, index: 2)
                            encoder.setBytes(&params, length: MemoryLayout<FrameBlendParams>.size, index: 0)
                            
                            let w = blendPipe.threadExecutionWidth
                            let h = blendPipe.maxTotalThreadsPerThreadgroup / w
                            let threadsPerGroup = MTLSize(width: w, height: h, depth: 1)
                            let grid = MTLSize(width: (blendOut.width + w - 1) / w,
                                             height: (blendOut.height + h - 1) / h,
                                             depth: 1)
                            
                            encoder.dispatchThreadgroups(grid, threadsPerThreadgroup: threadsPerGroup)
                            encoder.endEncoding()
                            
                            outputTex = blendOut
                            isInterpolated = true
                        }
                    }
                } else {
                    outputTex = prev.texture
                }
            } else {
                if let stale = frameBuffer.newestFrame {
                    outputTex = stale.texture
                    stats.droppedFrames += 1
                }
            }
        } else {
            outputTex = frameBuffer.newestFrame?.texture
            stats.outputFPS = stats.captureFPS
            stats.interpolatedFPS = 0
            stats.droppedFrames = 0
        }
        
        guard let finalTex = outputTex else {
            commandBuffer.commit()
            return
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
            if isInterpolated {
                Task { @MainActor in
                    self.stats.interpolatedFrameCount += 1
                }
            }
        }
        commandBuffer.commit()
    }

    struct FrameBlendParams {
        var t: Float
        var textureSize: SIMD2<UInt32>
    }
    
    nonisolated func stream(_ stream: SCStream, didOutputSampleBuffer sampleBuffer: CMSampleBuffer, of type: SCStreamOutputType) {
        guard type == .screen else { return }
        Task { @MainActor in
            processFrame(sampleBuffer)
        }
    }
    
    private func processFrame(_ sampleBuffer: CMSampleBuffer) {
        guard let imageBuffer = CMSampleBufferGetImageBuffer(sampleBuffer) else { return }
        let currentTime = CACurrentMediaTime()
        
        frameCount += 1
        
        stats.frameCount += 1
        
        stats.gpuMemoryUsed = UInt64(device.currentAllocatedSize)
        stats.gpuMemoryTotal = UInt64(device.recommendedMaxWorkingSetSize)
        
        let elapsed = currentTime - fpsStartTime
        if elapsed >= 1.0 {
            stats.captureFPS = Float(frameCount) / Float(elapsed)
            stats.frameTime = Float((currentTime - lastFrameTime) * 1000.0)
            
            stats.captureLatency = Float((currentTime - lastFrameTime) * 1000.0)
            
            frameCount = 0
            fpsStartTime = currentTime
        }
        lastFrameTime = currentTime
        
        guard let surface = CVPixelBufferGetIOSurface(imageBuffer)?.takeUnretainedValue() else { return }
        let w = IOSurfaceGetWidth(surface)
        let h = IOSurfaceGetHeight(surface)
        
        let desc = MTLTextureDescriptor.texture2DDescriptor(pixelFormat: .bgra8Unorm, width: w, height: h, mipmapped: false)
        desc.usage = [.shaderRead]
        guard let inputTex = device.makeTexture(descriptor: desc, iosurface: surface, plane: 0) else { return }
        
        guard let commandBuffer = commandQueue.makeCommandBuffer() else { return }
        
        let outW = Int(outputSize.width)
        let outH = Int(outputSize.height)
        
        let outDesc = MTLTextureDescriptor.texture2DDescriptor(pixelFormat: .bgra8Unorm, width: outW, height: outH, mipmapped: false)
        outDesc.usage = [.shaderRead, .shaderWrite]
        guard let newFrameTex = device.makeTexture(descriptor: outDesc) else { return }
        
        if let upscalePipe = upscalePipeline, let compute = commandBuffer.makeComputeCommandEncoder() {
             struct UpscaleParams {
                var sharpness: Float
                var inputSize: SIMD2<UInt32>
                var outputSize: SIMD2<UInt32>
            }
            var params = UpscaleParams(sharpness: sharpness, 
                                     inputSize: SIMD2(UInt32(w), UInt32(h)), 
                                     outputSize: SIMD2(UInt32(outW), UInt32(outH)))
                                     
            compute.setComputePipelineState(upscalePipe)
            compute.setTexture(inputTex, index: 0)
            compute.setTexture(newFrameTex, index: 1)
            compute.setBytes(&params, length: MemoryLayout<UpscaleParams>.size, index: 0)
            
            let threadW = upscalePipe.threadExecutionWidth
            let threadH = upscalePipe.maxTotalThreadsPerThreadgroup / threadW
            let grid = MTLSize(width: (outW + threadW - 1) / threadW, height: (outH + threadH - 1) / threadH, depth: 1)
            compute.dispatchThreadgroups(grid, threadsPerThreadgroup: MTLSize(width: threadW, height: threadH, depth: 1))
            compute.endEncoding()
        }
        
        commandBuffer.commit()
        
        frameBuffer.push(FrameHistory(texture: newFrameTex, timestamp: currentTime))
    }
    
    func configure(
        virtualResolution: CGSize?,
        outputSize: CGSize,
        sharpness: Float = 0.5,
        frameGenEnabled: Bool = false
    ) {
        self.outputSize = outputSize
        self.sharpness = sharpness
        self.frameGenEnabled = frameGenEnabled
        stats.virtualResolution = virtualResolution ?? outputSize
        stats.outputResolution = outputSize
        stats.isUsingVirtualDisplay = virtualResolution != nil
    }
    
    func startCapture(windowID: CGWindowID) async -> Bool {
        do {
            let content = try await SCShareableContent.excludingDesktopWindows(false, onScreenWindowsOnly: true)
            
            guard let window = content.windows.first(where: { $0.windowID == windowID }) else {
                lastError = "Window not found"
                return false
            }
            
            let filter = SCContentFilter(desktopIndependentWindow: window)
            self.contentFilter = filter
            
            let config = SCStreamConfiguration()
            config.width = Int(outputSize.width)
            config.height = Int(outputSize.height)
            config.minimumFrameInterval = CMTime(value: 1, timescale: 120)
            config.queueDepth = 3
            config.pixelFormat = kCVPixelFormatType_32BGRA
            config.showsCursor = true
            self.streamConfig = config
            
            let stream = SCStream(filter: filter, configuration: config, delegate: self)
            try stream.addStreamOutput(self, type: .screen, sampleHandlerQueue: .global(qos: .userInteractive))
            try await stream.startCapture()
            
            self.stream = stream
            self.isCapturing = true
            self.lastFrameTime = CACurrentMediaTime()
            self.fpsStartTime = CACurrentMediaTime()
            self.frameCount = 0
            
            self.frameCount = 0
            
            return true
            
        } catch {
            lastError = "Capture failed: \(error)"
            return false
        }
    }
    
    func startCaptureFromDisplay(displayID: CGDirectDisplayID) async -> Bool {
        do {
            let content = try await SCShareableContent.excludingDesktopWindows(false, onScreenWindowsOnly: false)
            guard let display = content.displays.first(where: { $0.displayID == displayID }) else {
                lastError = "Display \(displayID) not found"
                return false
            }
            
            let filter = SCContentFilter(display: display, excludingWindows: [])
            let config = SCStreamConfiguration()
            config.width = display.width
            config.height = display.height
            config.minimumFrameInterval = CMTime(value: 1, timescale: 120)
            config.queueDepth = 5
            config.pixelFormat = kCVPixelFormatType_32BGRA
            config.showsCursor = true
            
            let stream = SCStream(filter: filter, configuration: config, delegate: self)
            try stream.addStreamOutput(self, type: .screen, sampleHandlerQueue: .global(qos: .userInteractive))
            try await stream.startCapture()
            
            self.stream = stream
            self.isCapturing = true
            self.frameCount = 0
            self.fpsStartTime = CACurrentMediaTime()
            
            return true
        } catch {
            lastError = "\(error)"
            return false
        }
    }
    
    func startCaptureFromVirtualDisplay(_ virtualDisplayManager: VirtualDisplayManager, refreshRate: Int = 60) -> Bool {
        guard virtualDisplayManager.isActive else {
            lastError = "Virtual display is not active"
            return false
        }
        
        self.virtualDisplayManager = virtualDisplayManager
        
        virtualDisplayManager.onFrameReceived = { [weak self] surface, timestamp in
            guard let self = self else { return }
            Task { @MainActor in
                self.processIOSurfaceFrame(surface: surface, timestamp: timestamp)
            }
        }
        
        guard virtualDisplayManager.startFrameCapture(refreshRate: refreshRate) else {
            lastError = virtualDisplayManager.lastError ?? "Failed to start frame capture"
            return false
        }
        
        self.isCapturing = true
        self.frameCount = 0
        self.fpsStartTime = CACurrentMediaTime()
        
        return true
    }
    
    private func processIOSurfaceFrame(surface: IOSurfaceRef, timestamp: Double) {
        let currentTime = CACurrentMediaTime()
        
        frameCount += 1
        stats.frameCount += 1
        stats.gpuMemoryUsed = UInt64(device.currentAllocatedSize)
        stats.gpuMemoryTotal = UInt64(device.recommendedMaxWorkingSetSize)
        
        let elapsed = currentTime - fpsStartTime
        if elapsed >= 1.0 {
            stats.captureFPS = Float(frameCount) / Float(elapsed)
            stats.frameTime = Float((currentTime - lastFrameTime) * 1000.0)
            stats.captureLatency = Float((currentTime - timestamp) * 1000.0)
            
            frameCount = 0
            fpsStartTime = currentTime
        }
        lastFrameTime = currentTime
        
        let w = IOSurfaceGetWidth(surface)
        let h = IOSurfaceGetHeight(surface)
        
        let desc = MTLTextureDescriptor.texture2DDescriptor(pixelFormat: .bgra8Unorm, width: w, height: h, mipmapped: false)
        desc.usage = [.shaderRead]
        guard let inputTex = device.makeTexture(descriptor: desc, iosurface: surface, plane: 0) else { return }
        
        guard let commandBuffer = commandQueue.makeCommandBuffer() else { return }
        
        let outW = Int(outputSize.width)
        let outH = Int(outputSize.height)
        
        let outDesc = MTLTextureDescriptor.texture2DDescriptor(pixelFormat: .bgra8Unorm, width: outW, height: outH, mipmapped: false)
        outDesc.usage = [.shaderRead, .shaderWrite]
        guard let newFrameTex = device.makeTexture(descriptor: outDesc) else { return }
        
        if let upscalePipe = upscalePipeline, let compute = commandBuffer.makeComputeCommandEncoder() {
             struct UpscaleParams {
                var sharpness: Float
                var inputSize: SIMD2<UInt32>
                var outputSize: SIMD2<UInt32>
            }
            var params = UpscaleParams(sharpness: sharpness, 
                                     inputSize: SIMD2(UInt32(w), UInt32(h)), 
                                     outputSize: SIMD2(UInt32(outW), UInt32(outH)))
                                     
            compute.setComputePipelineState(upscalePipe)
            compute.setTexture(inputTex, index: 0)
            compute.setTexture(newFrameTex, index: 1)
            compute.setBytes(&params, length: MemoryLayout<UpscaleParams>.size, index: 0)
            
            let threadW = upscalePipe.threadExecutionWidth
            let threadH = upscalePipe.maxTotalThreadsPerThreadgroup / threadW
            let grid = MTLSize(width: (outW + threadW - 1) / threadW, height: (outH + threadH - 1) / threadH, depth: 1)
            compute.dispatchThreadgroups(grid, threadsPerThreadgroup: MTLSize(width: threadW, height: threadH, depth: 1))
            compute.endEncoding()
        }
        
        commandBuffer.commit()
        
        frameBuffer.push(FrameHistory(texture: newFrameTex, timestamp: currentTime))
    }
    
    func stopCapture() {
        isCapturing = false
        
        if let stream = stream {
            Task { try? await stream.stopCapture() }
        }
        stream = nil
        
        virtualDisplayManager?.stopFrameCapture()
        virtualDisplayManager?.onFrameReceived = nil
    }
    
    nonisolated func mtkView(_ view: MTKView, drawableSizeWillChange size: CGSize) {}
    
    nonisolated func stream(_ stream: SCStream, didStopWithError error: Error) {
        Task { @MainActor in
            self.lastError = "Stream stopped: \(error)"
            self.isCapturing = false
            self.onWindowLost?()
        }
    }
}
