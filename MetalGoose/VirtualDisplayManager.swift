import Foundation
import AppKit
import CoreGraphics
import ScreenCaptureKit
import CoreMedia
import IOSurface

@available(macOS 26.0, *)
@MainActor
final class VirtualDisplayManager: NSObject, ObservableObject, SCStreamDelegate, SCStreamOutput {
    
    @Published private(set) var isActive: Bool = false
    @Published private(set) var displayID: CGDirectDisplayID = 0
    @Published private(set) var currentResolution: CGSize = .zero
    @Published private(set) var lastError: String?
    @Published private(set) var isCapturing: Bool = false
    @Published private(set) var capturedFrameCount: UInt64 = 0
    
    private var virtualDisplay: CGVirtualDisplay?
    private var terminationHandler: (() -> Void)?
    private var captureStream: SCStream?
    private var stopTask: Task<Void, Never>?
    private var streamStartTime: CFTimeInterval = 0
    private var captureShowsCursor: Bool = true
    private var lastDestroyedDisplayID: CGDirectDisplayID = 0
    
    var onFrameReceived: ((_ surface: IOSurfaceRef, _ timestamp: Double) -> Void)?
    
    struct DisplayConfig {
        var width: UInt32
        var height: UInt32
        var refreshRate: Double
        var name: String
        var hiDPI: Bool
        
        static let r720p = DisplayConfig(width: 1280, height: 720, refreshRate: 60.0, name: "MetalGoose Virtual 720p", hiDPI: false)
        static let r900p = DisplayConfig(width: 1600, height: 900, refreshRate: 60.0, name: "MetalGoose Virtual 900p", hiDPI: false)
        static let r1080p = DisplayConfig(width: 1920, height: 1080, refreshRate: 60.0, name: "MetalGoose Virtual 1080p", hiDPI: false)
        static let r1440p = DisplayConfig(width: 2560, height: 1440, refreshRate: 60.0, name: "MetalGoose Virtual 1440p", hiDPI: false)
        
        static func custom(width: UInt32, height: UInt32, refreshRate: Double) -> DisplayConfig {
            DisplayConfig(width: width, height: height, refreshRate: refreshRate, name: "MetalGoose Virtual \(width)x\(height)", hiDPI: false)
        }
    }
    
    override init() {}
    
    @discardableResult
    func createDisplay(config: DisplayConfig) async -> CGDirectDisplayID? {
        await destroyDisplay()
        
        guard let descriptor = CGVirtualDisplayDescriptor() else {
            lastError = "Error Code: MG-VD-001 Failed to create CGVirtualDisplayDescriptor"
            return nil
        }
        
        descriptor.name = config.name
        descriptor.maxPixelsWide = config.width
        descriptor.maxPixelsHigh = config.height
        
        descriptor.sizeInMillimeters = CGSize(width: 530, height: 300)
        
        descriptor.vendorID = 0x1337
        descriptor.productID = 0x0001
        descriptor.serialNum = 0x0001
        
        descriptor.redPrimary = CGPoint(x: 0.64, y: 0.33)
        descriptor.greenPrimary = CGPoint(x: 0.30, y: 0.60)
        descriptor.bluePrimary = CGPoint(x: 0.15, y: 0.06)

        
        descriptor.queue = DispatchQueue.main
        
        descriptor.terminationHandler = { [weak self] in
            Task { @MainActor in
                await self?.handleTermination()
            }
        }
        
        guard let display = CGVirtualDisplay(descriptor: descriptor) else {
            lastError = "Error Code: MG-VD-002 Failed to create CGVirtualDisplay"
            return nil
        }
        
        guard let mode = CGVirtualDisplayMode(width: config.width, height: config.height, refreshRate: config.refreshRate) else {
            lastError = "Error Code: MG-VD-003 Failed to create CGVirtualDisplayMode"
            return nil
        }
        
        guard let settings = CGVirtualDisplaySettings() else {
            lastError = "Error Code: MG-VD-004 Failed to create CGVirtualDisplaySettings"
            return nil
        }
        settings.modes = [mode]
        settings.hiDPI = config.hiDPI ? 1 : 0
        
        guard display.applySettings(settings) else {
            lastError = "Error Code: MG-VD-005 Failed to apply display settings"
            return nil
        }
        
        self.virtualDisplay = display
        self.displayID = display.displayID
        self.currentResolution = CGSize(width: CGFloat(config.width), height: CGFloat(config.height))
        self.isActive = true
        self.lastError = nil
        
        return display.displayID
    }
    
    func startFrameCapture(refreshRate: Int, showsCursor: Bool) async -> Bool {
        guard isActive, displayID != 0 else {
            lastError = "Error Code: MG-VD-006 No active virtual display"
            return false
        }
        
        await stopFrameCapture()
        captureShowsCursor = showsCursor
        
        do {
            let availableDisplays = try await SCShareableContent.excludingDesktopWindows(
                false,
                onScreenWindowsOnly: true
            )
            
            guard let targetDisplay = availableDisplays.displays.first(where: { $0.displayID == displayID }) else {
                lastError = "Error Code: MG-VD-007 Virtual display not found in ScreenCaptureKit"
                isCapturing = false
                return false
            }
            
            let config = SCStreamConfiguration()
            config.width = Int(currentResolution.width)
            config.height = Int(currentResolution.height)
            config.minimumFrameInterval = CMTime(value: 1, timescale: CMTimeScale(refreshRate))
            config.pixelFormat = kCVPixelFormatType_32BGRA
            config.showsCursor = showsCursor
            config.scalesToFit = false
            config.captureResolution = .automatic
            config.shouldBeOpaque = false
            config.backgroundColor = .clear
            
            // Latency optimization settings
            config.queueDepth = 3  // Lower = less latency (default is 8)
            config.streamName = "MetalGoose"
            config.capturesAudio = false
            
            let filter = SCContentFilter(display: targetDisplay, excludingWindows: [])
            let stream = SCStream(filter: filter, configuration: config, delegate: self)
            
            try stream.addStreamOutput(self, type: .screen, sampleHandlerQueue: DispatchQueue.main)
            try await stream.startCapture()
            
            captureStream = stream
            isCapturing = true
            capturedFrameCount = 0
            streamStartTime = CACurrentMediaTime()
            lastError = nil
            return true
        } catch {
            lastError = "Error Code: MG-VD-008 ScreenCaptureKit error: \(error.localizedDescription)"
            isCapturing = false
            return false
        }
    }
    
    
    func stopFrameCapture() async {
        if let stopTask {
            await stopTask.value
            return
        }
        
        guard let stream = captureStream else {
            isCapturing = false
            return
        }
        
        captureStream = nil
        isCapturing = false
        
        let task = Task { @MainActor in
            do {
                try await stream.stopCapture()
            } catch {
                let nsError = error as NSError
                if nsError.domain == SCStreamErrorDomain && nsError.code == -3808 {
                    return
                }
                lastError = "Error Code: MG-VD-009 ScreenCaptureKit stop error: \(error.localizedDescription)"
            }
        }
        
        stopTask = task
        await task.value
        stopTask = nil
    }
    
    func destroyDisplay() async {
        await stopFrameCapture()
        
        guard let display = virtualDisplay else { return }
        let oldID = display.displayID

        virtualDisplay = nil
        lastDestroyedDisplayID = oldID

        DispatchQueue.main.async {
            _ = display
        }

        displayID = 0
        currentResolution = .zero
        isActive = false

        await waitForDisplayRemoval(oldID)
    }
    
    func changeResolution(width: UInt32, height: UInt32, refreshRate: Double) async -> Bool {
        guard let display = virtualDisplay else {
            lastError = "Error Code: MG-VD-006 No active virtual display"
            return false
        }
        
        let wasCapturing = isCapturing
        await stopFrameCapture()
        
        guard let mode = CGVirtualDisplayMode(width: width, height: height, refreshRate: refreshRate) else {
            lastError = "Error Code: MG-VD-003 Failed to create CGVirtualDisplayMode"
            return false
        }
        
        guard let settings = CGVirtualDisplaySettings() else {
            lastError = "Error Code: MG-VD-004 Failed to create CGVirtualDisplaySettings"
            return false
        }
        settings.modes = [mode]
        settings.hiDPI = 0
        
        guard display.applySettings(settings) else {
            lastError = "Error Code: MG-VD-005 Failed to apply display settings"
            return false
        }
        
        currentResolution = CGSize(width: CGFloat(width), height: CGFloat(height))
        
        if wasCapturing {
            _ = await startFrameCapture(refreshRate: Int(refreshRate), showsCursor: captureShowsCursor)
        }
        
        return true
    }
    
    func getScreen() -> NSScreen? {
        guard isActive, displayID != 0 else { return nil }
        return NSScreen.screens.first { screen in
            let screenID = screen.deviceDescription[NSDeviceDescriptionKey("NSScreenNumber")] as? CGDirectDisplayID
            return screenID == displayID
        }
    }
    
    func getDisplayOrigin() -> CGPoint? {
        return getScreen()?.frame.origin
    }
    
    static var availablePresets: [DisplayConfig] {
        [.r720p, .r900p, .r1080p, .r1440p]
    }
    
    private func handleTermination() async {
        await stopFrameCapture()
        isActive = false
        displayID = 0
        currentResolution = .zero
        virtualDisplay = nil
        terminationHandler?()
    }

    private func waitForDisplayRemoval(_ id: CGDirectDisplayID) async {
        guard id != 0 else { return }
        for _ in 0..<10 {
            let stillPresent = NSScreen.screens.contains { screen in
                let screenID = screen.deviceDescription[NSDeviceDescriptionKey("NSScreenNumber")] as? CGDirectDisplayID
                return screenID == id
            }
            if !stillPresent { return }
            try? await Task.sleep(nanoseconds: 100_000_000)
        }
    }
    
    
    func onTermination(_ handler: @escaping () -> Void) {
        self.terminationHandler = handler
    }
    
    nonisolated func stream(_ stream: SCStream, didStopWithError error: Error) {
        Task { @MainActor in
            if self.stopTask != nil { return }
            let nsError = error as NSError
            if nsError.domain == SCStreamErrorDomain && nsError.code == -3808 { return }
            self.lastError = "Error Code: MG-VD-010 Stream stopped with error: \(error.localizedDescription)"
            self.isCapturing = false
            self.captureStream = nil
        }
    }
    
    nonisolated func stream(_ stream: SCStream, didOutputSampleBuffer sampleBuffer: CMSampleBuffer, of type: SCStreamOutputType) {
        guard type == .screen else { return }
        
        guard let attachmentsArray = CMSampleBufferGetSampleAttachmentsArray(sampleBuffer, createIfNecessary: false) as? [[SCStreamFrameInfo: Any]],
              let attachments = attachmentsArray.first else {
            return
        }
        
        guard let statusRawValue = attachments[SCStreamFrameInfo.status] as? Int,
              let status = SCFrameStatus(rawValue: statusRawValue),
              status == .complete else {
            return
        }

        if let dirtyRects = attachments[SCStreamFrameInfo.dirtyRects] as? [NSValue],
           dirtyRects.isEmpty {
            return
        }
        
        guard let pixelBuffer = CMSampleBufferGetImageBuffer(sampleBuffer) else {
            return
        }
        
        guard let surface = CVPixelBufferGetIOSurface(pixelBuffer)?.takeUnretainedValue() else {
            return
        }
        
        let timestamp = CMTimeGetSeconds(CMSampleBufferGetPresentationTimeStamp(sampleBuffer))
        
        Task { @MainActor in
            self.capturedFrameCount += 1
            self.onFrameReceived?(surface, timestamp)
        }
    }
}
