import SwiftUI
import MetalKit
import ScreenCaptureKit
import Vision
import AppKit
import Combine

struct RootView: View {
    @StateObject private var settings = CaptureSettings()
    @StateObject private var engine: CaptureEngine
    @State private var showingSettings = false
    @State private var isCountingDown = false
    @State private var secondsLeft = 5

    @State private var fpsValue: Double = 0
    @State private var fpsFrameCount: Int = 0
    @State private var fpsWindowStart: CFAbsoluteTime = CFAbsoluteTimeGetCurrent()

    // Permissions gate
    @State private var accessibilityGranted = PermissionsChecker.isAccessibilityGranted
    @State private var screenRecordingGranted = PermissionsChecker.isScreenRecordingGranted

    init() {
        let settings = CaptureSettings()
        _settings = StateObject(wrappedValue: settings)
        _engine = StateObject(wrappedValue: CaptureEngine(settings: settings))
    }

    var body: some View {
        Group {
            if accessibilityGranted && screenRecordingGranted {
                ZStack {
                    MetalView(engine: engine)
                        .environmentObject(settings)
                        .ignoresSafeArea()
                    if !engine.isCapturing {
                        Color.black.opacity(0.7).ignoresSafeArea()
                        VStack {
                            Text("Press Run to start")
                                .font(.headline)
                                .foregroundColor(.white)
                        }
                    }
                    if isCountingDown {
                        Color.black.opacity(0.4).ignoresSafeArea()
                        Text("\(secondsLeft)")
                            .font(.system(size: 160, weight: .bold))
                            .foregroundColor(.white)
                            .shadow(radius: 10)
                    }
                    if settings.showFPSCounter && engine.isCapturing {
                        VStack {
                            HStack {
                                Text(String(format: "FPS: %.1f", fpsValue))
                                    .font(.system(size: 12, weight: .semibold, design: .monospaced))
                                    .foregroundColor(.white)
                                    .padding(6)
                                    .background(Color.black.opacity(0.5))
                                    .clipShape(RoundedRectangle(cornerRadius: 6, style: .continuous))
                                Spacer()
                            }
                            Spacer()
                        }
                        .padding()
                        .ignoresSafeArea()
                    }
                }
                .onReceive(NotificationCenter.default.publisher(for: Notification.Name("FrameDidRender"))) { _ in
                    fpsFrameCount += 1
                    let now = CFAbsoluteTimeGetCurrent()
                    let elapsed = now - fpsWindowStart
                    if elapsed >= 0.5 { // update twice per second for stability
                        fpsValue = Double(fpsFrameCount) / elapsed
                        fpsFrameCount = 0
                        fpsWindowStart = now
                    }
                }
                .onChange(of: settings.sharpenAmount) { _, newValue in
                    NotificationCenter.default.post(name: Notification.Name("PostProcessUpdate"), object: NSNumber(value: newValue))
                }
                .toolbar {
                    ToolbarItem(placement: .automatic) {
                        Button {
                            showingSettings.toggle()
                        } label: {
                            Image(systemName: "gear")
                        }
                    }
                    ToolbarItem(placement: .automatic) {
                        Button {
                            secondsLeft = 5
                            isCountingDown = true
                            Task {
                                while secondsLeft > 0 {
                                    try? await Task.sleep(nanoseconds: 1_000_000_000)
                                    await MainActor.run { secondsLeft -= 1 }
                                }
                                await MainActor.run { isCountingDown = false }
                                await engine.start()
                            }
                        } label: {
                            Label("Run", systemImage: "play.fill")
                        }
                    }
                }
                .sheet(isPresented: $showingSettings) {
                    CaptureSettingsView()
                        .environmentObject(settings)
                }
            } else {
                PermissionsView()
            }
        }
        .onReceive(NotificationCenter.default.publisher(for: .permissionsShouldRefresh)) { _ in
            refreshPermissions()
        }
        .onAppear { refreshPermissions() }
    }

    private func refreshPermissions() {
        accessibilityGranted = PermissionsChecker.isAccessibilityGranted
        screenRecordingGranted = PermissionsChecker.isScreenRecordingGranted
    }
}

struct MetalView: NSViewRepresentable {
    @ObservedObject var engine: CaptureEngine
    @EnvironmentObject var settings: CaptureSettings

    class Coordinator {
        var renderer: Renderer?
    }

    func makeCoordinator() -> Coordinator {
        Coordinator()
    }

    func makeNSView(context: Context) -> MTKView {
        let view = MTKView()
        view.device = MTLCreateSystemDefaultDevice()
        view.enableSetNeedsDisplay = false
        view.isPaused = true
        view.colorPixelFormat = .bgra8Unorm
        view.framebufferOnly = true

        if let renderer = Renderer(view: view) {
            view.delegate = renderer
            context.coordinator.renderer = renderer

            NotificationCenter.default.addObserver(forName: Notification.Name("PostProcessUpdate"), object: nil, queue: .main) { note in
                if let num = note.object as? NSNumber {
                    context.coordinator.renderer?.updatePostProcess(sharpen: num.floatValue)
                }
            }
        }

        return view
    }

    func updateNSView(_ nsView: MTKView, context: Context) {
        if let frame = engine.currentFrame {
            context.coordinator.renderer?.update(with: frame)
            NotificationCenter.default.post(name: Notification.Name("FrameDidRender"), object: nil)
            nsView.draw()
        }
    }
}

struct CaptureSettingsView: View {
    @EnvironmentObject var settings: CaptureSettings
    @Environment(\.dismiss) private var dismiss

    var body: some View {
        let isMetalFXSupported: Bool = {
            if #available(macOS 26, *) { return true }
            return false
        }()

        VStack(alignment: .leading, spacing: 16) {
            Text("Capture Settings").font(.headline)
            Group {
                Picker("Scaling", selection: $settings.scalingMode) {
                    ForEach(CaptureSettings.ScalingMode.allCases) { mode in
                        Text(mode.rawValue.capitalized).tag(mode)
                    }
                }
                Stepper(value: $settings.integerScale, in: 1...8) {
                    Text("Integer Scale: \(settings.integerScale)x")
                }
            }
            Group {
                VStack {
                    HStack(alignment: .firstTextBaseline, spacing: 8) {
                        Toggle("Enable Frame Generation", isOn: $settings.enableFrameGeneration)
                            .disabled(!isMetalFXSupported)
                        if !isMetalFXSupported {
                            Text("Unsupported OS level")
                                .font(.footnote)
                                .foregroundStyle(.secondary)
                        }
                    }
                }
                HStack {
                    Text("Frame Gen Ratio")
                    Slider(value: $settings.frameGenerationRatio, in: 0...2, step: 0.5)
                    Text(String(format: "%.1fx", settings.frameGenerationRatio + 1.0))
                }
                Toggle("Show Fullscreen Overlay on Run", isOn: $settings.showFullscreenOverlayOnRun)
                HStack {
                    Text("Downscale")
                    Slider(value: $settings.downscaleFactor, in: 0.5...1.0, step: 0.05)
                    Text(String(format: "%.2fx", settings.downscaleFactor))
                }
                HStack {
                    Text("Sharpen")
                    Slider(value: $settings.sharpenAmount, in: 0...1.0, step: 0.05)
                    Text(String(format: "%.2f", settings.sharpenAmount))
                }
                Toggle("Show FPS Counter", isOn: $settings.showFPSCounter)
            }
            HStack {
                Spacer()
                Button("Done") { dismiss() }
            }
        }
        .padding()
        .frame(minWidth: 420)
    }
}
