import Foundation
import AppKit
import CoreGraphics

@available(macOS 26.0, *)
@MainActor
final class VirtualDisplayManager: ObservableObject {
    
    @Published private(set) var isActive: Bool = false
    @Published private(set) var displayID: CGDirectDisplayID = 0
    @Published private(set) var currentResolution: CGSize = .zero
    @Published private(set) var lastError: String?
    @Published private(set) var isCapturing: Bool = false
    @Published private(set) var capturedFrameCount: UInt64 = 0
    
    private var virtualDisplay: CGVirtualDisplay?
    private var terminationHandler: (() -> Void)?
    
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
        
        static func custom(width: UInt32, height: UInt32, refreshRate: Double = 60.0) -> DisplayConfig {
            DisplayConfig(width: width, height: height, refreshRate: refreshRate, name: "MetalGoose Virtual \(width)x\(height)", hiDPI: false)
        }
    }
    
    init() {}
    
    @discardableResult
    func createDisplay(config: DisplayConfig = .r720p) -> CGDirectDisplayID? {
        destroyDisplay()
        
        guard let descriptor = CGVirtualDisplayDescriptor() else {
            lastError = "Failed to create CGVirtualDisplayDescriptor"
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
                self?.handleTermination()
            }
        }
        
        guard let display = CGVirtualDisplay(descriptor: descriptor) else {
            lastError = "Failed to create CGVirtualDisplay"
            return nil
        }
        
        guard let mode = CGVirtualDisplayMode(width: config.width, height: config.height, refreshRate: config.refreshRate) else {
            lastError = "Failed to create CGVirtualDisplayMode"
            return nil
        }
        
        guard let settings = CGVirtualDisplaySettings() else {
            lastError = "Failed to create CGVirtualDisplaySettings"
            return nil
        }
        settings.modes = [mode]
        settings.hiDPI = config.hiDPI ? 1 : 0
        
        guard display.applySettings(settings) else {
            lastError = "Failed to apply display settings"
            return nil
        }
        
        self.virtualDisplay = display
        self.displayID = display.displayID
        self.currentResolution = CGSize(width: CGFloat(config.width), height: CGFloat(config.height))
        self.isActive = true
        self.lastError = nil
        
        return display.displayID
    }
    
    func startFrameCapture(refreshRate: Int = 60) -> Bool {
        guard isActive, displayID != 0 else {
            lastError = "No active virtual display"
            return false
        }
        
        stopFrameCapture()
        
        // Note: Frame capture for virtual displays should use ScreenCaptureKit via DirectEngineBridge
        // This method is kept for API compatibility but capture is handled externally
        isCapturing = true
        capturedFrameCount = 0
        return true
    }
    
    func stopFrameCapture() {
        isCapturing = false
    }
    
    func destroyDisplay() {
        stopFrameCapture()
        
        guard let display = virtualDisplay else { return }
        
        virtualDisplay = nil
        
        DispatchQueue.main.async {
            _ = display
        }
        
        displayID = 0
        currentResolution = .zero
        isActive = false
    }
    
    func changeResolution(width: UInt32, height: UInt32, refreshRate: Double = 60.0) -> Bool {
        guard let display = virtualDisplay else {
            lastError = "No active virtual display"
            return false
        }
        
        let wasCapturing = isCapturing
        stopFrameCapture()
        
        guard let mode = CGVirtualDisplayMode(width: width, height: height, refreshRate: refreshRate) else {
            lastError = "Failed to create CGVirtualDisplayMode"
            return false
        }
        
        guard let settings = CGVirtualDisplaySettings() else {
            lastError = "Failed to create CGVirtualDisplaySettings"
            return false
        }
        settings.modes = [mode]
        settings.hiDPI = 0
        
        guard display.applySettings(settings) else {
            lastError = "Failed to apply new resolution"
            return false
        }
        
        currentResolution = CGSize(width: CGFloat(width), height: CGFloat(height))
        
        if wasCapturing {
            _ = startFrameCapture(refreshRate: Int(refreshRate))
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
    
    private func handleTermination() {
        stopFrameCapture()
        isActive = false
        displayID = 0
        currentResolution = .zero
        virtualDisplay = nil
        terminationHandler?()
    }
    
    func onTermination(_ handler: @escaping () -> Void) {
        self.terminationHandler = handler
    }
}
