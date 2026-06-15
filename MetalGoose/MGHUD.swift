import SwiftUI
import AppKit

@MainActor
final class MGHUDData: ObservableObject {
    @Published var deviceName: String = "Unknown GPU"
    @Published var pid: Int32 = 0

    @Published var captureResolution: String = "-"
    @Published var outputResolution: String = "-"
    @Published var scaleFactor: String = "-"
    @Published var renderScale: String = "-"

    @Published var captureFPS: Float = 0
    @Published var outputFPS: Float = 0
    @Published var interpolatedFPS: Float = 0
    @Published var screenRefreshRate: Int = 0
    @Published var targetOutputFPS: Int = 0

    @Published var frameTime: Float = 0
    @Published var avgFrameTime: Float = 0
    @Published var gpuTime: Float = 0
    @Published var captureLatency: Float = 0
    @Published var presentLatency: Float = 0
    @Published var endToEndLatency: Float = 0
    @Published var framePacingScore: Float = 0

    @Published var gpuMemoryUsed: UInt64 = 0
    @Published var gpuMemoryTotal: UInt64 = 0

    @Published var framesProcessed: UInt64 = 0
    @Published var framesPresented: UInt64 = 0
    @Published var framesInterpolated: UInt64 = 0
    @Published var framesPassthrough: UInt64 = 0
    @Published var framesDropped: UInt64 = 0

    @Published var upscaleMode: String = "Off"
    @Published var frameGenMode: String = "Off"
    @Published var aaMode: String = "Off"
    @Published var vsyncMode: String = "On"
}

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
            if data.pid != 0 {
                HUDRow(label: "PID", value: "\(data.pid)", compact: isCompact)
            }
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
            HUDRow(label: "Capture", value: captureFPSText, compact: isCompact, color: fpsColor(data.captureFPS, target: Float(data.targetOutputFPS)))
            HUDRow(label: "Output", value: outputFPSText, compact: isCompact, color: fpsColor(data.outputFPS, target: Float(data.targetOutputFPS)))

            if data.frameGenMode != "Off" || data.interpolatedFPS > 0 {
                HUDRow(label: "Interpolated", value: interpFPSText, compact: isCompact, color: .cyan)
            }

            if !isCompact {
                HUDRow(label: "Screen Refresh", value: "\(data.screenRefreshRate) Hz", compact: isCompact)
                HUDRow(label: "Render Target", value: "\(data.targetOutputFPS) FPS", compact: isCompact)
            }
        }
    }
    
    private var timingSection: some View {
        let captureTimeText = String(format: "%.2f ms", data.frameTime)
        let outputFrameTimeText = String(format: "%.2f ms", data.avgFrameTime)
        let gpuTimeText = String(format: "%.2f ms", data.gpuTime)
        let latencyText = String(format: "%.1f ms", data.captureLatency)
        let presentText = String(format: "%.1f ms", data.presentLatency)
        let endToEndText = String(format: "%.1f ms", data.endToEndLatency)
        let pacingText = String(format: "%.0f", data.framePacingScore)
        let pacingColor: Color = data.framePacingScore >= 90 ? .green : (data.framePacingScore >= 70 ? .yellow : (data.framePacingScore >= 40 ? .orange : .red))

        return VStack(spacing: isCompact ? 2 : 3) {
            HUDRow(label: "Capture Time", value: captureTimeText, compact: isCompact)
            HUDRow(label: "GPU Time", value: gpuTimeText, compact: isCompact)
            HUDRow(label: "Latency", value: latencyText, compact: isCompact)
            if !isCompact {
                HUDRow(label: "Output Frame Time", value: outputFrameTimeText, compact: isCompact)
                HUDRow(label: "Present", value: presentText, compact: isCompact)
                HUDRow(label: "End-to-End", value: endToEndText, compact: isCompact)
                HUDRow(label: "Pacing", value: pacingText, compact: isCompact, color: pacingColor)
            }
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
        let percent = totalMB > 0 ? (usedMB / totalMB) * 100 : 0
        let vramText: String = usedMB >= 1.0
            ? String(format: "%.0f / %.0f MB (%.0f%%)", usedMB, totalMB, percent)
            : String(format: "%.1f KB", usedMB * 1024.0)
        
        let memoryColor: Color = percent > 90 ? .red : (percent > 75 ? .orange : (percent > 50 ? .yellow : .white))
        
        return HUDRow(label: "VRAM", value: vramText, compact: isCompact, color: memoryColor)
    }
    
    private var pipelineSection: some View {
        let upscaleText = "\(data.upscaleMode) \(data.scaleFactor)"
        
        return VStack(spacing: isCompact ? 2 : 3) {
            HUDRow(label: "Capture Res", value: data.captureResolution, compact: isCompact)
            HUDRow(label: "Output Res", value: data.outputResolution, compact: isCompact)
            HUDRow(label: "Upscale", value: upscaleText, compact: isCompact)
            HUDRow(label: "Render Scale", value: data.renderScale, compact: isCompact)
            HUDRow(label: "Frame Gen", value: data.frameGenMode, compact: isCompact)
            HUDRow(label: "AA", value: data.aaMode, compact: isCompact)
            HUDRow(label: "VSync", value: data.vsyncMode, compact: isCompact)
        }
    }
    
    @ViewBuilder
    private var frameStatsSection: some View {
        HUDRow(label: "Captured", value: "\(data.framesProcessed)", compact: isCompact)
        HUDRow(label: "Presented", value: "\(data.framesPresented)", compact: isCompact)
        HUDRow(label: "Interpolated", value: "\(data.framesInterpolated)", compact: isCompact)
        HUDRow(label: "Passthrough", value: "\(data.framesPassthrough)", compact: isCompact)
        HUDRow(label: "Dropped", value: "\(data.framesDropped)", compact: isCompact, color: data.framesDropped > 0 ? .red : .white)
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
    
    func setDeviceName(_ name: String) {
        Task { @MainActor in
            hudData.deviceName = name
        }
    }

    func setPID(_ pid: Int32) {
        Task { @MainActor in
            hudData.pid = pid
        }
    }
    
    func setResolutions(capture: CGSize, output: CGSize) {
        Task { @MainActor in
            hudData.captureResolution = "\(Int(capture.width))x\(Int(capture.height))"
            hudData.outputResolution = "\(Int(output.width))x\(Int(output.height))"
        }
    }
    
    func updateFromGooseEngine(stats: PipelineStats, settings: CaptureSettings) {
        Task { @MainActor in
            hudData.captureFPS = stats.captureFPS
            hudData.outputFPS = stats.outputFPS
            hudData.interpolatedFPS = stats.interpolatedFPS
            hudData.frameTime = stats.frameTime
            hudData.avgFrameTime = stats.avgFrameTime
            hudData.gpuTime = stats.gpuTime
            hudData.captureLatency = stats.captureLatency
            hudData.presentLatency = stats.presentLatency
            hudData.endToEndLatency = stats.endToEndLatency
            hudData.framePacingScore = stats.framePacingScore
            hudData.screenRefreshRate = stats.screenRefreshRate
            hudData.targetOutputFPS = stats.targetOutputFPS
            hudData.framesProcessed = stats.frameCount
            hudData.framesDropped = stats.droppedFrames
            hudData.framesInterpolated = stats.interpolatedFrameCount
            hudData.framesPresented = stats.outputFrameCount
            hudData.framesPassthrough = stats.passthroughFrameCount
            
            hudData.gpuMemoryUsed = stats.gpuMemoryUsed
            hudData.gpuMemoryTotal = stats.gpuMemoryTotal
            
            if stats.outputResolution.width > 0 && stats.outputResolution.height > 0 {
                hudData.outputResolution = "\(Int(stats.outputResolution.width))x\(Int(stats.outputResolution.height))"
            }

            hudData.upscaleMode = settings.scalingType.rawValue
            hudData.frameGenMode = settings.isFrameGenEnabled ? "MGFG-1 (2x)" : "Off"
            hudData.aaMode = settings.aaMode.rawValue
            hudData.scaleFactor = "\(settings.scaleFactor.rawValue)"
            hudData.renderScale = settings.renderScale.rawValue
            hudData.vsyncMode = settings.vsync ? "On" : "Off"
        }
    }
}

@MainActor
final class MGHUDWindowController {
    private var hudWindow: NSWindow?
    private var hudView: MGHUDOverlayView?
    private var margin: CGFloat = 20
    
    enum Corner {
        case topLeft, topRight, bottomLeft, bottomRight
    }
    
    func show(on screen: NSScreen, corner: Corner = .topLeft, compact: Bool = false) {
        let targetScreen = screen
        
        let hudSize = compact ? CGSize(width: 180, height: 150) : CGSize(width: 220, height: 535)
        
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
    }
    
    func hide() {
        hudWindow?.orderOut(nil)
        hudWindow = nil
        hudView = nil
    }

    func setDeviceName(_ name: String) {
        hudView?.setDeviceName(name)
    }

    func setPID(_ pid: Int32) {
        hudView?.setPID(pid)
    }
    
    func setResolutions(capture: CGSize, output: CGSize) {
        hudView?.setResolutions(capture: capture, output: output)
    }
    
    func updateFromGooseEngine(stats: PipelineStats, settings: CaptureSettings) {
        hudView?.updateFromGooseEngine(stats: stats, settings: settings)
    }
}
