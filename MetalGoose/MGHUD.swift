import SwiftUI
import AppKit

@available(macOS 26.0, *)
@MainActor
final class MGHUDData: ObservableObject {
    @Published var deviceName: String = "Unknown GPU"
    @Published var driverVersion: String = ""
    
    @Published var captureResolution: String = "-"
    @Published var outputResolution: String = "-"
    @Published var scaleFactor: String = "1.0x"
    @Published var renderScale: String = "100%"
    
    @Published var captureFPS: Float = 0
    @Published var outputFPS: Float = 0
    @Published var interpolatedFPS: Float = 0
    @Published var targetFPS: Int = 120
    
    @Published var frameTime: Float = 0
    @Published var gpuTime: Float = 0
    @Published var captureLatency: Float = 0
    @Published var presentLatency: Float = 0
    
    @Published var gpuMemoryUsed: UInt64 = 0
    @Published var gpuMemoryTotal: UInt64 = 0
    @Published var textureMemory: UInt64 = 0
    
    @Published var framesProcessed: UInt64 = 0
    @Published var framesInterpolated: UInt64 = 0
    @Published var framesDropped: UInt64 = 0
    
    @Published var upscaleMode: String = "Off"
    @Published var frameGenMode: String = "Off"
    @Published var aaMode: String = "Off"
    
    @Published var computeEncoders: UInt32 = 0
    @Published var blitEncoders: UInt32 = 0
    @Published var commandBuffers: UInt32 = 0
    
    func update(from stats: DirectEngineStats, settings: CaptureSettings) {
        captureFPS = stats.captureFPS
        outputFPS = stats.fps
        interpolatedFPS = stats.interpolatedFPS
        
        frameTime = stats.frameTime
        gpuTime = stats.gpuTime
        captureLatency = stats.captureLatency
        presentLatency = stats.presentLatency
        
        gpuMemoryUsed = stats.gpuMemoryUsed
        gpuMemoryTotal = stats.gpuMemoryTotal
        textureMemory = stats.textureMemoryUsed
        
        framesProcessed = stats.frameCount
        framesInterpolated = stats.interpolatedFrameCount
        framesDropped = stats.droppedFrames
        
        computeEncoders = stats.computeEncoders
        blitEncoders = stats.blitEncoders
        commandBuffers = stats.commandBuffers
        
        upscaleMode = settings.scalingType.rawValue
        frameGenMode = settings.isFrameGenEnabled ? "MGFG-1 \(settings.frameGenMultiplier.rawValue)" : "Off"
        aaMode = settings.aaMode.rawValue
        scaleFactor = "\(settings.scaleFactor.rawValue)"
        renderScale = settings.renderScale.rawValue
    }
}

@available(macOS 26.0, *)
struct MGHUDView: View {
    @ObservedObject var data: MGHUDData
    let isCompact: Bool
    
    init(data: MGHUDData, compact: Bool = false) {
        self.data = data
        self.isCompact = compact
    }
    
    var body: some View {
        mainContent
            .padding(isCompact ? 6 : 10)
            .background(hudBackground)
            .foregroundColor(.white)
    }
    
    @ViewBuilder
    private var mainContent: some View {
        VStack(alignment: .leading, spacing: isCompact ? 2 : 4) {
            headerSection
            divider(opacity: 0.3)
            HUDRow(label: "GPU", value: data.deviceName, compact: isCompact)
            divider()
            frameRateSection
            divider()
            timingSection
            if !isCompact {
                extendedContent
            }
        }
    }
    
    @ViewBuilder
    private var headerSection: some View {
        HStack {
            Image(systemName: "gamecontroller.fill")
                .foregroundColor(.green)
            Text("MetalGoose")
                .font(.system(size: isCompact ? 10 : 12, weight: .bold, design: .monospaced))
            Spacer()
        }
    }
    
    private var frameRateSection: some View {
        let captureFPSText = "\(Int(data.captureFPS)) FPS"
        let outputFPSText = "\(Int(data.outputFPS)) FPS"
        let interpFPSText = "\(Int(data.interpolatedFPS)) FPS"
        
        return VStack(spacing: isCompact ? 2 : 3) {
            HUDRow(label: "Capture", value: captureFPSText, compact: isCompact, color: fpsColor(data.captureFPS, target: Float(data.targetFPS)))
            HUDRow(label: "Output", value: outputFPSText, compact: isCompact, color: fpsColor(data.outputFPS, target: Float(data.targetFPS)))
            
            if data.interpolatedFPS > data.captureFPS {
                HUDRow(label: "Interpolated", value: interpFPSText, compact: isCompact, color: .cyan)
            }
        }
    }
    
    private var timingSection: some View {
        let frameTimeText = String(format: "%.2f ms", data.frameTime)
        let gpuTimeText = String(format: "%.2f ms", data.gpuTime)
        let latencyText = String(format: "%.1f ms", data.captureLatency)
        
        return VStack(spacing: isCompact ? 2 : 3) {
            HUDRow(label: "Frame Time", value: frameTimeText, compact: isCompact)
            HUDRow(label: "GPU Time", value: gpuTimeText, compact: isCompact)
            HUDRow(label: "Latency", value: latencyText, compact: isCompact)
        }
    }
    
    @ViewBuilder
    private var extendedContent: some View {
        divider()
        memorySection
        divider()
        pipelineSection
        divider()
        frameStatsSection
    }
    
    private var memorySection: some View {
        let usedMB = Double(data.gpuMemoryUsed) / (1024.0 * 1024.0)
        let totalMB = Double(data.gpuMemoryTotal) / (1024.0 * 1024.0)
        let textureMB = Double(data.textureMemory) / (1024.0 * 1024.0)
        
        let effectiveUsed = usedMB > 0.1 ? usedMB : textureMB
        let effectiveTotal = totalMB > 0.1 ? totalMB : max(textureMB * 8, 1024.0)
        let percent = effectiveTotal > 0 ? (effectiveUsed / effectiveTotal) * 100 : 0
        
        let vramText: String = effectiveUsed >= 1.0
            ? String(format: "%.0f / %.0f MB (%.0f%%)", effectiveUsed, effectiveTotal, percent)
            : String(format: "%.1f KB", effectiveUsed * 1024.0)
        
        let memoryColor: Color = percent > 90 ? .red : (percent > 75 ? .orange : (percent > 50 ? .yellow : .white))
        
        return HUDRow(label: "VRAM", value: vramText, compact: isCompact, color: memoryColor)
    }
    
    private var pipelineSection: some View {
        let upscaleText = "\(data.upscaleMode) \(data.scaleFactor)"
        
        return VStack(spacing: isCompact ? 2 : 3) {
            HUDRow(label: "Upscale", value: upscaleText, compact: isCompact)
            HUDRow(label: "Render Scale", value: data.renderScale, compact: isCompact)
            HUDRow(label: "Frame Gen", value: data.frameGenMode, compact: isCompact)
            HUDRow(label: "AA", value: data.aaMode, compact: isCompact)
        }
    }
    
    @ViewBuilder
    private var frameStatsSection: some View {
        HUDRow(label: "Frames", value: "\(data.framesProcessed)", compact: isCompact)
        HUDRow(label: "Interpolated", value: "\(data.framesInterpolated)", compact: isCompact)
        if data.framesDropped > 0 {
            HUDRow(label: "Dropped", value: "\(data.framesDropped)", compact: isCompact, color: .red)
        }
    }
    
    private func divider(opacity: Double = 0.2) -> some View {
        Divider()
            .background(Color.white.opacity(opacity))
    }
    
    private var hudBackground: some View {
        RoundedRectangle(cornerRadius: 8)
            .fill(Color.black.opacity(0.75))
            .overlay(
                RoundedRectangle(cornerRadius: 8)
                    .stroke(Color.white.opacity(0.2), lineWidth: 1)
            )
    }
    
    private func fpsColor(_ fps: Float, target: Float) -> Color {
        if fps >= target * 0.95 { return .green }
        if fps >= target * 0.75 { return .yellow }
        if fps >= target * 0.5 { return .orange }
        return .red
    }
}

@available(macOS 26.0, *)
struct HUDRow: View {
    let label: String
    let value: String
    let compact: Bool
    var color: Color = .white
    
    var body: some View {
        HStack {
            Text(label)
                .font(.system(size: compact ? 9 : 10, weight: .regular, design: .monospaced))
                .foregroundColor(.gray)
            Spacer()
            Text(value)
                .font(.system(size: compact ? 9 : 10, weight: .medium, design: .monospaced))
                .foregroundColor(color)
        }
    }
}

@available(macOS 26.0, *)
class MGHUDOverlayView: NSView {
    private var hostingView: NSHostingView<MGHUDView>?
    private let hudData = MGHUDData()
    
    var isCompact: Bool = false {
        didSet {
            updateHostingView()
        }
    }
    
    override init(frame frameRect: NSRect) {
        super.init(frame: frameRect)
        setup()
    }
    
    required init?(coder: NSCoder) {
        super.init(coder: coder)
        setup()
    }
    
    private func setup() {
        wantsLayer = true
        layer?.backgroundColor = .clear
        updateHostingView()
    }
    
    private func updateHostingView() {
        hostingView?.removeFromSuperview()
        
        let hudView = MGHUDView(data: hudData, compact: isCompact)
        let hosting = NSHostingView(rootView: hudView)
        hosting.frame = bounds
        hosting.autoresizingMask = [.width, .height]
        
        addSubview(hosting)
        hostingView = hosting
    }
    
    func update(stats: DirectEngineStats, settings: CaptureSettings) {
        Task { @MainActor in
            hudData.update(from: stats, settings: settings)
        }
    }
    
    func setDeviceName(_ name: String) {
        Task { @MainActor in
            hudData.deviceName = name
        }
    }
    
    func setResolutions(capture: CGSize, output: CGSize) {
        Task { @MainActor in
            hudData.captureResolution = "\(Int(capture.width))x\(Int(capture.height))"
            hudData.outputResolution = "\(Int(output.width))x\(Int(output.height))"
        }
    }
}

@available(macOS 26.0, *)
@MainActor
final class MGHUDWindowController {
    private var hudWindow: NSWindow?
    private var hudView: MGHUDOverlayView?
    private var anchorCorner: Corner = .topLeft
    private var margin: CGFloat = 20
    
    enum Corner {
        case topLeft, topRight, bottomLeft, bottomRight
    }
    
    func show(on screen: NSScreen? = nil, corner: Corner = .topLeft, compact: Bool = false) {
        let targetScreen = screen ?? NSScreen.main ?? NSScreen.screens.first!
        
        let hudSize = compact ? CGSize(width: 180, height: 150) : CGSize(width: 220, height: 350)
        
        var origin: CGPoint
        
        let topOffset: CGFloat = (targetScreen == NSScreen.main) ? 24 : 0
        
        switch corner {
        case .topLeft:
            origin = CGPoint(x: targetScreen.frame.minX + margin,
                           y: targetScreen.frame.maxY - hudSize.height - margin - topOffset)
        case .topRight:
            origin = CGPoint(x: targetScreen.frame.maxX - hudSize.width - margin,
                           y: targetScreen.frame.maxY - hudSize.height - margin - topOffset)
        case .bottomLeft:
            origin = CGPoint(x: targetScreen.frame.minX + margin,
                           y: targetScreen.frame.minY + margin)
        case .bottomRight:
            origin = CGPoint(x: targetScreen.frame.maxX - hudSize.width - margin,
                           y: targetScreen.frame.minY + margin)
        }
        
        let window = NSWindow(
            contentRect: CGRect(origin: origin, size: hudSize),
            styleMask: [.borderless],
            backing: .buffered,
            defer: false
        )
        
        window.isReleasedWhenClosed = false
        window.level = NSWindow.Level(rawValue: Int(CGWindowLevelForKey(.maximumWindow)) + 1)
        window.backgroundColor = .clear
        window.isOpaque = false
        window.hasShadow = true
        window.ignoresMouseEvents = true
        window.collectionBehavior = [.canJoinAllSpaces, .stationary]
        
        let overlay = MGHUDOverlayView(frame: CGRect(origin: .zero, size: hudSize))
        overlay.isCompact = compact
        
        window.contentView = overlay
        window.orderFront(nil)
        
        hudWindow = window
        hudView = overlay
        anchorCorner = corner
    }
    
    func hide() {
        hudWindow?.orderOut(nil)
        hudWindow = nil
        hudView = nil
    }
    
    func update(stats: DirectEngineStats, settings: CaptureSettings) {
        hudView?.update(stats: stats, settings: settings)
    }
    
    func setDeviceName(_ name: String) {
        hudView?.setDeviceName(name)
    }
    
    func setResolutions(capture: CGSize, output: CGSize) {
        hudView?.setResolutions(capture: capture, output: output)
    }
    
    var isVisible: Bool {
        hudWindow?.isVisible ?? false
    }
}
