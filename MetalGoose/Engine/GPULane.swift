import Foundation
import QuartzCore
import os
@preconcurrency import Metal

/// What every lane reports to: GPU time used, command buffers that failed, and the errors to show.
final class GPUReports: Sendable {
    let errors = ErrorLog()
    private let busy = OSAllocatedUnfairLock(initialState: 0.0)
    private let failures = OSAllocatedUnfairLock(initialState: 0)

    /// Seconds of GPU time the pipeline has used since it started.
    var busyTime: Double { busy.withLock { $0 } }
    /// Bumped by every command buffer that did not complete.
    var failureGeneration: Int { failures.withLock { $0 } }

    fileprivate func record(_ label: String, feedback: any MTL4CommitFeedback) {
        if let error = feedback.error {
            failures.withLock { $0 &+= 1 }
            errors.report(.gpuExecutionFailed(stage: label, detail: error.localizedDescription))
            return
        }
        let elapsed = feedback.gpuEndTime - feedback.gpuStartTime
        if elapsed > 0 { busy.withLock { $0 += elapsed } }
    }
}

/// One Metal 4 command queue, and what its command buffers are made with.
///
/// The pipeline runs on three of them — the capture path, MetalFX interpolation, and presentation — so that a MetalFX
/// interpolation that takes several milliseconds never holds up a capture or the image a refresh is waiting for. Metal 4
/// tracks nothing on its own, so the order of the work is spelt out:
///
/// - On one lane, every pass waits for everything committed before it, and every step of a pass for the step before it
///   (`GPUCommand.makeComputePass`). The work of a lane is small; nothing is gained by letting it overlap, and every
///   scratch texture is safe to reuse in the next command buffer.
/// - Across lanes, a command buffer waits on the GPU for the work of another lane it reads (`GPUCommand.wait(for:value:)`),
///   which each lane counts in `progress`.
/// - A texture that one lane reads and another writes again later — a capture in the ring, a MetalFX image, a motion
///   field — is a leased slot, and the lease is held by every command buffer that reads it until that has run.
///
/// Each lane is encoded from one thread only: the processing queue for capture and interpolation, the render thread
/// for presentation. What a command buffer used is made resident for it and held until it has run, so a texture can be
/// let go of at any time without a command buffer on the GPU losing it.
final class GPULane: @unchecked Sendable {

    let queue: any MTL4CommandQueue
    let label: String

    /// Signalled with the number of command buffers committed so far, after each of them.
    let progress: any MTLSharedEvent

    /// Command buffers committed so far. Belongs to the thread that encodes on this lane.
    private(set) var submitted: UInt64 = 0

    fileprivate let table: any MTL4ArgumentTable
    fileprivate let reports: GPUReports
    private let device: any MTLDevice
    /// Where the command buffers' completions arrive. Metal does not keep it alive, so the lane does.
    private let feedback: DispatchQueue
    private let idle = OSAllocatedUnfairLock<[Slot]>(uncheckedState: [])

    /// What one command buffer is encoded with, kept for the next once the GPU has finished with it.
    fileprivate final class Slot: @unchecked Sendable {
        let commandBuffer: any MTL4CommandBuffer
        let allocator: any MTL4CommandAllocator
        /// Everything the command buffer uses, made resident for it and held by the set until it has run.
        let residency: any MTLResidencySet
        /// The values passed by `ComputePass.setBytes`, which Metal 4 binds as buffer addresses.
        let constants: any MTLBuffer

        static let constantsLength = 16 * 1024

        init?(device: any MTLDevice) {
            guard let commandBuffer = device.makeCommandBuffer(),
                  let allocator = device.makeCommandAllocator(),
                  let residency = try? device.makeResidencySet(descriptor: MTLResidencySetDescriptor()),
                  let constants = device.makeBuffer(length: Self.constantsLength, options: .storageModeShared) else { return nil }
            self.commandBuffer = commandBuffer
            self.allocator = allocator
            self.residency = residency
            self.constants = constants
        }
    }

    init?(device: any MTLDevice, label: String, reports: GPUReports) {
        let feedback = DispatchQueue(label: "com.metalgoose.gpu.\(label)", qos: .userInteractive)
        let descriptor = MTL4CommandQueueDescriptor()
        descriptor.label = "MetalGoose \(label)"
        descriptor.feedbackQueue = feedback
        let tableDescriptor = MTL4ArgumentTableDescriptor()
        // The kernels bind at most five textures and two buffers.
        tableDescriptor.maxTextureBindCount = 8
        tableDescriptor.maxBufferBindCount = 4
        guard let queue = try? device.makeMTL4CommandQueue(descriptor: descriptor),
              let progress = device.makeSharedEvent(),
              let table = try? device.makeArgumentTable(descriptor: tableDescriptor) else { return nil }
        self.queue = queue
        self.label = label
        self.progress = progress
        self.table = table
        self.reports = reports
        self.device = device
        self.feedback = feedback
    }

    /// A command buffer to encode on this lane, labelled for the GPU's error reports and captures.
    func makeCommand(_ label: String) -> GPUCommand? {
        guard let slot = idle.withLockUnchecked({ $0.popLast() }) ?? Slot(device: device) else { return nil }
        slot.commandBuffer.beginCommandBuffer(allocator: slot.allocator)
        slot.commandBuffer.label = label
        slot.commandBuffer.useResidencySet(slot.residency)
        return GPUCommand(lane: self, slot: slot, label: label)
    }

    fileprivate func committed() -> UInt64 {
        submitted += 1
        queue.signalEvent(progress, value: submitted)
        return submitted
    }

    fileprivate func recycle(_ slot: Slot) {
        slot.allocator.reset()
        slot.residency.removeAllAllocations()
        slot.residency.commit()
        idle.withLockUnchecked { $0.append(slot) }
    }
}

/// How a command buffer ended, for the work that waited on it.
struct GPUCompletion: Sendable {
    let succeeded: Bool
    /// Seconds the GPU spent on it.
    let gpuTime: CFTimeInterval
}

/// One Metal 4 command buffer on a lane: its passes, the work of other lanes it waits for, what it holds until it has run,
/// and what runs when it has.
final class GPUCommand: @unchecked Sendable {

    let lane: GPULane
    let label: String
    fileprivate let slot: GPULane.Slot

    /// Objects the GPU does not see that have to live as long as the command buffer: leases, pixel buffers.
    private var retained: [AnyObject] = []
    private var handlers: [@Sendable (GPUCompletion) -> Void] = []
    private var waits: [(event: any MTLSharedEvent, value: UInt64)] = []
    private var constantsOffset = 0
    private var usesConstants = false

    fileprivate init(lane: GPULane, slot: GPULane.Slot, label: String) {
        self.lane = lane
        self.slot = slot
        self.label = label
    }

    /// Makes `allocation` resident for this command buffer and holds it until the command buffer has run.
    func use(_ allocation: any MTLAllocation) {
        slot.residency.addAllocation(allocation)
    }

    /// Holds `object` until the command buffer has run.
    func retain(_ object: AnyObject) {
        retained.append(object)
    }

    /// Runs `handler` once the command buffer has run, on the lane's completion queue.
    func onCompleted(_ handler: @escaping @Sendable (GPUCompletion) -> Void) {
        handlers.append(handler)
    }

    /// Has the GPU wait for the work of `other` up to `value` (`GPULane.submitted` after it) before this command buffer.
    func wait(for other: GPULane, value: UInt64) {
        guard other !== lane, value > 0 else { return }
        if let index = waits.firstIndex(where: { $0.event === other.progress }) {
            waits[index].value = max(waits[index].value, value)
        } else {
            waits.append((other.progress, value))
        }
    }

    /// A compute pass, in which copies run too. It starts once everything committed to the lane before it has finished.
    func makeComputePass() -> ComputePass? {
        guard let encoder = slot.commandBuffer.makeComputeCommandEncoder() else { return nil }
        encoder.barrier(afterQueueStages: .all, beforeStages: [.dispatch, .blit], visibilityOptions: .device)
        encoder.setArgumentTable(lane.table)
        return ComputePass(encoder: encoder, command: self)
    }

    /// A render pass into `target`: cleared to `clearColor`, or with every pixel drawn by the pass where there is none. It
    /// starts once everything committed to the lane before it has finished.
    func makeRenderPass(target: any MTLTexture, clearColor: MTLClearColor? = nil) -> RenderPass? {
        let descriptor = MTL4RenderPassDescriptor()
        descriptor.colorAttachments[0].texture = target
        descriptor.colorAttachments[0].loadAction = clearColor == nil ? .dontCare : .clear
        if let clearColor { descriptor.colorAttachments[0].clearColor = clearColor }
        descriptor.colorAttachments[0].storeAction = .store
        guard let encoder = slot.commandBuffer.makeRenderCommandEncoder(descriptor: descriptor) else { return nil }
        encoder.barrier(afterQueueStages: .all, beforeStages: [.vertex, .fragment], visibilityOptions: .device)
        encoder.setArgumentTable(lane.table, stages: .fragment)
        return RenderPass(encoder: encoder, command: self)
    }

    /// For a MetalFX effect, which makes passes of its own: between passes of this command buffer, each of which waits
    /// for everything before it and holds back everything after it.
    var commandBuffer: any MTL4CommandBuffer { slot.commandBuffer }

    /// Copies `length` bytes into the command buffer's constants and returns their GPU address.
    fileprivate func constant(_ bytes: UnsafeRawPointer, length: Int) -> MTLGPUAddress {
        // Bound as `constant` buffers, whose offsets are kept to 256 bytes.
        let offset = (constantsOffset + 255) & ~255
        precondition(offset + length <= GPULane.Slot.constantsLength, "\(label): more constants than a command buffer holds")
        slot.constants.contents().advanced(by: offset).copyMemory(from: bytes, byteCount: length)
        constantsOffset = offset + length
        if !usesConstants {
            usesConstants = true
            use(slot.constants)
        }
        return slot.constants.gpuAddress + UInt64(offset)
    }

    /// Commits the command buffer and returns its place on the lane (`GPULane.submitted`). With `drawable`, the command
    /// buffer draws into it, and it is presented once that is done.
    @discardableResult
    func commit(presenting drawable: (any CAMetalDrawable)? = nil) -> UInt64 {
        slot.residency.commit()
        slot.commandBuffer.endCommandBuffer()
        for wait in waits { lane.queue.waitForEvent(wait.event, value: wait.value) }
        if let drawable { lane.queue.waitForDrawable(drawable) }
        let options = MTL4CommitOptions()
        options.addFeedbackHandler { [self] feedback in finished(feedback) }
        lane.queue.commit([slot.commandBuffer], options: options)
        let value = lane.committed()
        if let drawable {
            lane.queue.signalDrawable(drawable)
            drawable.present()
        }
        return value
    }

    private func finished(_ feedback: any MTL4CommitFeedback) {
        lane.reports.record(label, feedback: feedback)
        let completion = GPUCompletion(succeeded: feedback.error == nil,
                                       gpuTime: max(0, feedback.gpuEndTime - feedback.gpuStartTime))
        for handler in handlers { handler(completion) }
        handlers.removeAll()
        retained.removeAll()
        lane.recycle(slot)
    }
}

/// A compute pass of a `GPUCommand`, bound as a Metal 3 encoder is: textures and buffers by index, small values by bytes.
/// Its steps run in the order they are encoded, each once the one before it has finished, as a serial Metal 3 encoder's
/// did; and what follows the pass waits for all of it.
final class ComputePass {
    private let encoder: any MTL4ComputeCommandEncoder
    private let command: GPUCommand
    private var steps = 0

    fileprivate init(encoder: any MTL4ComputeCommandEncoder, command: GPUCommand) {
        self.encoder = encoder
        self.command = command
    }

    func setComputePipelineState(_ pipeline: any MTLComputePipelineState) {
        encoder.setComputePipelineState(pipeline)
    }

    func setTexture(_ texture: any MTLTexture, index: Int) {
        command.use(texture)
        command.lane.table.setTexture(texture.gpuResourceID, index: index)
    }

    func setBuffer(_ buffer: any MTLBuffer, index: Int) {
        command.use(buffer)
        command.lane.table.setAddress(buffer.gpuAddress, index: index)
    }

    func setBytes(_ bytes: UnsafeRawPointer, length: Int, index: Int) {
        command.lane.table.setAddress(command.constant(bytes, length: length), index: index)
    }

    func dispatchThreadgroups(_ threadgroups: MTLSize, threadsPerThreadgroup: MTLSize) {
        afterPreviousStep()
        encoder.dispatchThreadgroups(threadgroupsPerGrid: threadgroups, threadsPerThreadgroup: threadsPerThreadgroup)
    }

    func copy(from source: any MTLTexture, to destination: any MTLTexture) {
        command.use(source)
        command.use(destination)
        afterPreviousStep()
        encoder.copy(sourceTexture: source, destinationTexture: destination)
    }

    /// Copies `source` into a drawable of its size, which its layer makes resident.
    func copy(from source: any MTLTexture, toDrawable drawable: any MTLTexture) {
        command.use(source)
        afterPreviousStep()
        encoder.copy(sourceTexture: source, destinationTexture: drawable)
    }

    /// Copies the top-left `size` of `source` into the top-left of `destination`.
    func copy(from source: any MTLTexture, size: MTLSize, to destination: any MTLTexture) {
        command.use(source)
        command.use(destination)
        afterPreviousStep()
        encoder.copy(sourceTexture: source, sourceSlice: 0, sourceLevel: 0, sourceOrigin: MTLOrigin(x: 0, y: 0, z: 0),
                     sourceSize: size, destinationTexture: destination, destinationSlice: 0, destinationLevel: 0,
                     destinationOrigin: MTLOrigin(x: 0, y: 0, z: 0))
    }

    func endEncoding() {
        encoder.barrier(afterStages: [.dispatch, .blit], beforeQueueStages: .all, visibilityOptions: .device)
        encoder.endEncoding()
    }

    private func afterPreviousStep() {
        if steps > 0 {
            encoder.barrier(afterEncoderStages: [.dispatch, .blit], beforeEncoderStages: [.dispatch, .blit],
                            visibilityOptions: .device)
        }
        steps += 1
    }
}

/// A render pass of a `GPUCommand`: one full-target draw, for resampling into a drawable.
final class RenderPass {
    private let encoder: any MTL4RenderCommandEncoder
    private let command: GPUCommand

    fileprivate init(encoder: any MTL4RenderCommandEncoder, command: GPUCommand) {
        self.encoder = encoder
        self.command = command
    }

    func setRenderPipelineState(_ pipeline: any MTLRenderPipelineState) {
        encoder.setRenderPipelineState(pipeline)
    }

    func setFragmentTexture(_ texture: any MTLTexture, index: Int) {
        command.use(texture)
        command.lane.table.setTexture(texture.gpuResourceID, index: index)
    }

    func drawFullTarget() {
        encoder.drawPrimitives(primitiveType: .triangleStrip, vertexStart: 0, vertexCount: 4)
    }

    func endEncoding() {
        encoder.barrier(afterStages: .fragment, beforeQueueStages: .all, visibilityOptions: .device)
        encoder.endEncoding()
    }
}
