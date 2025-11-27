// OpticalFlowEngine.swift
// Vision + Metal based optical flow and interpolation engine

import Foundation
import Vision
import Metal
import CoreVideo
import CoreImage

final class OpticalFlowEngine {
    private let device: MTLDevice
    private let commandQueue: MTLCommandQueue
    private let library: MTLLibrary
    private let pipeline: MTLComputePipelineState
    private let ciContext: CIContext

    init?(device: MTLDevice? = MTLCreateSystemDefaultDevice()) {
        guard let device = device, let commandQueue = device.makeCommandQueue() else { return nil }
        self.device = device
        self.commandQueue = commandQueue
        self.ciContext = CIContext(mtlDevice: device)
        do {
            self.library = try device.makeDefaultLibrary(bundle: .main)
            guard let fn = library.makeFunction(name: "flowWarpKernel") else { return nil }
            self.pipeline = try device.makeComputePipelineState(function: fn)
        } catch {
            return nil
        }
    }

    // Public API used by Renderer 2.swift
    func interpolate(previous prev: CVPixelBuffer, current cur: CVPixelBuffer, t: Float) -> CVPixelBuffer? {
        guard let flow = try? computeOpticalFlow(from: prev, to: cur) else { return nil }
        return warp(previous: prev, current: cur, flow: flow, t: t)
    }

    // Legacy placeholder kept for compatibility; forwards to interpolate with t = 0.5
    func intermediateFrame(previous: CVPixelBuffer, current: CVPixelBuffer) -> CVPixelBuffer? {
        return interpolate(previous: previous, current: current, t: 0.5)
    }

    private func computeOpticalFlow(from: CVPixelBuffer, to: CVPixelBuffer) throws -> VNPixelBufferObservation {
        let request = VNGenerateOpticalFlowRequest(targetedCVPixelBuffer: to, options: [:])
        request.computationAccuracy = .medium
        request.outputPixelFormat = kCVPixelFormatType_TwoComponent32Float
        let handler = VNImageRequestHandler(cvPixelBuffer: from, orientation: .up)
        try handler.perform([request])
        guard let obs = request.results?.first as? VNPixelBufferObservation else {
            throw NSError(domain: "OpticalFlowEngine", code: -1)
        }
        return obs
    }

    private func warp(previous: CVPixelBuffer, current: CVPixelBuffer, flow: VNPixelBufferObservation, t: Float) -> CVPixelBuffer? {
        let width = CVPixelBufferGetWidth(current)
        let height = CVPixelBufferGetHeight(current)

        guard let cmd = commandQueue.makeCommandBuffer(),
              let encoder = cmd.makeComputeCommandEncoder() else { return nil }

        let ciPrev = CIImage(cvPixelBuffer: previous)
        let ciCur = CIImage(cvPixelBuffer: current)

        guard let prevTex = makeTexture(width: width, height: height),
              let curTex = makeTexture(width: width, height: height),
              let flowTex = makeFlowTexture(from: flow.pixelBuffer),
              let outTex = makeTexture(width: width, height: height) else { return nil }

        ciContext.render(ciPrev, to: prevTex, commandBuffer: cmd, bounds: ciPrev.extent, colorSpace: CGColorSpaceCreateDeviceRGB())
        ciContext.render(ciCur, to: curTex, commandBuffer: cmd, bounds: ciCur.extent, colorSpace: CGColorSpaceCreateDeviceRGB())

        encoder.setComputePipelineState(pipeline)
        encoder.setTexture(prevTex, index: 0)
        encoder.setTexture(curTex, index: 1)
        encoder.setTexture(flowTex, index: 2)
        encoder.setTexture(outTex, index: 3)

        var tValue = t
        encoder.setBytes(&tValue, length: MemoryLayout<Float>.size, index: 0)

        let w = pipeline.threadExecutionWidth
        let h = max(1, pipeline.maxTotalThreadsPerThreadgroup / w)
        let threadsPerThreadgroup = MTLSize(width: w, height: h, depth: 1)
        let threadsPerGrid = MTLSize(width: width, height: height, depth: 1)
        encoder.dispatchThreads(threadsPerGrid, threadsPerThreadgroup: threadsPerThreadgroup)
        encoder.endEncoding()

        cmd.commit()
        cmd.waitUntilCompleted()

        var pb: CVPixelBuffer?
        let attrs: [String: Any] = [
            kCVPixelBufferMetalCompatibilityKey as String: true,
            kCVPixelBufferCGImageCompatibilityKey as String: true,
            kCVPixelBufferCGBitmapContextCompatibilityKey as String: true,
        ]
        CVPixelBufferCreate(kCFAllocatorDefault, width, height, kCVPixelFormatType_32BGRA, attrs as CFDictionary, &pb)
        if let pb = pb, let outImage = CIImage(mtlTexture: outTex, options: [.colorSpace: CGColorSpaceCreateDeviceRGB()]) {
            ciContext.render(outImage, to: pb)
        }
        return pb
    }

    private func makeTexture(width: Int, height: Int) -> MTLTexture? {
        let desc = MTLTextureDescriptor.texture2DDescriptor(pixelFormat: .bgra8Unorm, width: width, height: height, mipmapped: false)
        desc.usage = [.shaderRead, .shaderWrite]
        return device.makeTexture(descriptor: desc)
    }

    private func makeFlowTexture(from pixelBuffer: CVPixelBuffer) -> MTLTexture? {
        let width = CVPixelBufferGetWidth(pixelBuffer)
        let height = CVPixelBufferGetHeight(pixelBuffer)
        let desc = MTLTextureDescriptor.texture2DDescriptor(pixelFormat: .rg32Float, width: width, height: height, mipmapped: false)
        desc.usage = [.shaderRead]
        guard let tex = device.makeTexture(descriptor: desc) else { return nil }

        CVPixelBufferLockBaseAddress(pixelBuffer, .readOnly)
        defer { CVPixelBufferUnlockBaseAddress(pixelBuffer, .readOnly) }
        if let ba = CVPixelBufferGetBaseAddress(pixelBuffer) {
            let bytesPerRow = CVPixelBufferGetBytesPerRow(pixelBuffer)
            let region = MTLRegionMake2D(0, 0, width, height)
            tex.replace(region: region, mipmapLevel: 0, withBytes: ba, bytesPerRow: bytesPerRow)
        }
        return tex
    }
}
