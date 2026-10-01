import SwiftUI

/// What the HUD says about the session that does not change from frame to frame.
struct HUDInfo: Equatable {
    var deviceName = "Unknown GPU"
    var pid: Int32 = 0
    var captureResolution = "-"
    var upscale = "Off"
    var renderScale = "-"
    var frameGeneration = "Off"
    var antiAliasing = "Off"
    var vsync = "On"
}

@MainActor
final class HUDModel: ObservableObject {
    @Published var stats = PipelineStats()
    @Published var info = HUDInfo()
}

struct HUDView: View {
    @ObservedObject var model: HUDModel

    var body: some View {
        let stats = model.stats
        let info = model.info

        VStack(alignment: .leading, spacing: 4) {
            // No icon: the overlay attaches to whatever window was picked, and the name is the whole label.
            Text("MetalGoose")
                .font(.system(size: 12, weight: .bold, design: .monospaced))
            divider(opacity: 0.3)
            HUDRow(label: "GPU", value: info.deviceName)
            if info.pid != 0 {
                HUDRow(label: "PID", value: "\(info.pid)")
            }
            divider()
            frameRates(stats, info)
            divider()
            timings(stats)
            divider()
            memory(stats)
            divider()
            pipeline(stats, info)
            divider()
            counters(stats)
        }
        .padding(10)
        .background(
            RoundedRectangle(cornerRadius: 8)
                .fill(Color.black.opacity(0.75))
                .overlay(RoundedRectangle(cornerRadius: 8).stroke(Color.white.opacity(0.2), lineWidth: 1))
        )
        .foregroundColor(.white)
    }

    private func frameRates(_ stats: PipelineStats, _ info: HUDInfo) -> some View {
        let target = Float(stats.targetOutputFPS)
        return VStack(spacing: 3) {
            HUDRow(label: "Capture", value: "\(Int(stats.captureFPS)) FPS", color: fpsColor(stats.captureFPS, target: target))
            // Images the screen was given per second. The panel repeats whatever it last showed, so
            // its refresh rate is not a measure of this — new images are.
            HUDRow(label: "Output", value: "\(Int(stats.outputFPS)) FPS", color: fpsColor(stats.outputFPS, target: target))
            if info.frameGeneration != "Off" || stats.generatedFPS > 0 {
                HUDRow(label: "Generated", value: "\(Int(stats.generatedFPS)) FPS", color: .cyan)
            }
            HUDRow(label: "Screen Refresh", value: "\(stats.screenRefreshRate) Hz")
            HUDRow(label: "ProMotion", value: stats.isProMotion ? "On" : "Off")
            HUDRow(label: "Render Target", value: "\(stats.targetOutputFPS) FPS")
        }
    }

    private func timings(_ stats: PipelineStats) -> some View {
        let pacingColor: Color = stats.framePacingScore >= 90 ? .green
            : stats.framePacingScore >= 70 ? .yellow
            : stats.framePacingScore >= 40 ? .orange : .red
        return VStack(spacing: 3) {
            HUDRow(label: "Capture Time", value: String(format: "%.2f ms", stats.frameTime))
            HUDRow(label: "GPU Time", value: String(format: "%.2f ms", stats.gpuTime))
            HUDRow(label: "GPU Load", value: String(format: "%.1f%%", stats.gpuLoad), color: loadColor(Double(stats.gpuLoad)))
            HUDRow(label: "Latency", value: String(format: "%.1f ms", stats.captureLatency))
            HUDRow(label: "Output Frame Time", value: String(format: "%.2f ms", stats.avgFrameTime))
            HUDRow(label: "Present", value: String(format: "%.1f ms", stats.presentLatency))
            HUDRow(label: "End-to-End", value: String(format: "%.1f ms", stats.endToEndLatency))
            HUDRow(label: "Pacing", value: String(format: "%.0f", stats.framePacingScore), color: pacingColor)
        }
    }

    private func memory(_ stats: PipelineStats) -> some View {
        let megabyte = 1024.0 * 1024.0
        let gpuUsed = Double(stats.gpuMemoryUsed) / megabyte
        let gpuTotal = Double(stats.gpuMemoryTotal) / megabyte
        let gpuPercent = gpuTotal > 0 ? gpuUsed / gpuTotal * 100 : 0
        let vram = gpuUsed >= 1.0
            ? String(format: "%.0f / %.0f MB (%.0f%%)", gpuUsed, gpuTotal, gpuPercent)
            : String(format: "%.1f KB", gpuUsed * 1024.0)

        let processMB = Double(stats.processMemoryUsed) / megabyte
        let physicalMB = Double(ProcessInfo.processInfo.physicalMemory) / megabyte
        let ramPercent = physicalMB > 0 ? processMB / physicalMB * 100 : 0

        let cpu = Double(stats.cpuUsage)
        return VStack(spacing: 3) {
            HUDRow(label: "VRAM", value: vram, color: loadColor(gpuPercent))
            HUDRow(label: "Memory", value: String(format: "%.0f / %.0f MB (%.0f%%)", processMB, physicalMB, ramPercent),
                   color: loadColor(ramPercent))
            // A core is 100%, so a pipeline that spreads over several reads above it.
            HUDRow(label: "CPU", value: String(format: "%.1f%%", cpu),
                   color: cpu > 200 ? .red : cpu > 100 ? .orange : cpu > 50 ? .yellow : .white)
        }
    }

    private func pipeline(_ stats: PipelineStats, _ info: HUDInfo) -> some View {
        let output = stats.outputResolution
        return VStack(spacing: 3) {
            HUDRow(label: "Capture Res", value: info.captureResolution)
            HUDRow(label: "Output Res", value: output.width > 0 ? "\(Int(output.width))x\(Int(output.height))" : "-")
            HUDRow(label: "Upscale", value: info.upscale)
            HUDRow(label: "Render Scale", value: info.renderScale)
            HUDRow(label: "Frame Gen", value: info.frameGeneration)
            HUDRow(label: "AA", value: info.antiAliasing)
            HUDRow(label: "VSync", value: info.vsync)
        }
    }

    /// `Generated + Passthrough = Presented`: every image on screen is either one the generator
    /// synthesised or a captured frame shown as captured.
    private func counters(_ stats: PipelineStats) -> some View {
        VStack(spacing: 3) {
            HUDRow(label: "Captured", value: "\(stats.frameCount)")
            HUDRow(label: "Presented", value: "\(stats.outputFrameCount)")
            HUDRow(label: "Generated", value: "\(stats.generatedFrameCount)")
            HUDRow(label: "Passthrough", value: "\(stats.passthroughFrameCount)")
            HUDRow(label: "Dropped", value: "\(stats.droppedFrames)", color: stats.droppedFrames > 0 ? .red : .white)
        }
    }

    private func divider(opacity: Double = 0.2) -> some View {
        Divider().background(Color.white.opacity(opacity))
    }

    private func fpsColor(_ fps: Float, target: Float) -> Color {
        if fps >= target * 0.95 { return .green }
        if fps >= target * 0.75 { return .yellow }
        if fps >= target * 0.5 { return .orange }
        return .red
    }

    private func loadColor(_ percent: Double) -> Color {
        percent > 90 ? .red : percent > 75 ? .orange : percent > 50 ? .yellow : .white
    }
}

struct HUDRow: View {
    let label: String
    let value: String
    var color: Color = .white

    var body: some View {
        HStack {
            Text(verbatim: label)
                .font(.system(size: 10, weight: .regular, design: .monospaced))
                .foregroundColor(.gray)
            Spacer()
            Text(verbatim: value)
                .font(.system(size: 10, weight: .medium, design: .monospaced))
                .foregroundColor(color)
        }
    }
}
