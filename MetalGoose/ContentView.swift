import SwiftUI
import MetalKit
import ApplicationServices

// DARK THEME PALETTE
let BG_COLOR = Color(red: 0.1, green: 0.1, blue: 0.12)
let PANEL_COLOR = Color(red: 0.15, green: 0.15, blue: 0.18)
let ACCENT_RED = Color(red: 0.8, green: 0.2, blue: 0.2)
let TEXT_COLOR = Color.white.opacity(0.9)

@available(macOS 26.0, *)
struct ContentView: View {
    @StateObject var settings = CaptureSettings.shared
    @StateObject var engine = CaptureEngine()
    
    @State private var countdown = 5
    @State private var isCountingDown = false
    @State private var isOverlayActive = false
    @State private var overlayWindow: NSWindow?
    @State private var renderer: Renderer?
    @State private var connectedProcessName: String = "-"
    @State private var connectedPID: Int32 = 0
    @State private var connectedWindowID: CGWindowID = 0
    @State private var connectedSize: CGSize = .zero
    @State private var showAlert = false
    @State private var alertMessage = ""
    
    // Permissions state
    @State private var axGranted: Bool = AXIsProcessTrusted()
    @State private var recGranted: Bool = CGPreflightScreenCaptureAccess()
    @State private var permTimer: Timer?
    private var permissionsGranted: Bool { axGranted && recGranted }
    
    @State private var targetDisplayID: CGDirectDisplayID?
    @State private var mtkView: MTKView = MTKView()
    private var macOSVersionString: String {
        let version = ProcessInfo.processInfo.operatingSystemVersion
        return "macOS \(version.majorVersion).\(version.minorVersion).\(version.patchVersion)"
    }
    
    var body: some View {
        ZStack(alignment: .bottomLeading) {
            HStack(spacing: 0) {
            // SIDEBAR
            VStack(alignment: .leading) {
                VStack(alignment: .leading, spacing: 12) {
                    Text("Overlay Info")
                        .font(.headline)
                        .padding([.top, .horizontal])

                    VStack(alignment: .leading, spacing: 8) {
                        InfoRow(label: "Status", value: isOverlayActive ? "Active" : "Idle")
                        InfoRow(label: "Process", value: connectedProcessName)
                        InfoRow(label: "PID", value: String(connectedPID))
                        InfoRow(label: "Window ID", value: String(connectedWindowID))
                        InfoRow(label: "Frame", value: connectedSize.width > 0 ? "\(Int(connectedSize.width)) x \(Int(connectedSize.height))" : "-")
                        InfoRow(label: "Display ID", value: targetDisplayID.map { String($0) } ?? "-")
                    }
                    .padding(.horizontal)
                    .padding(.bottom)

                    Spacer()

                    HStack {
                        Spacer()
                        Menu {
                            Button("About") {
                                let v = Bundle.main.infoDictionary?["CFBundleShortVersionString"] as? String ?? ""
                                let b = Bundle.main.infoDictionary?["CFBundleVersion"] as? String ?? ""
                                alertMessage = "MetalGoose v\(v) (\(b))"
                                showAlert = true
                            }
                            Button("Check for Updates") {
                                alertMessage = "You're up to date."
                                showAlert = true
                            }
                        } label: {
                            Image(systemName: "gearshape")
                        }
                    }
                    .padding()
                }
            }
            .frame(width: 200)
            .background(Color.black.opacity(0.3))
            .disabled(!permissionsGranted)
            
            // MAIN DASHBOARD
            ScrollView {
                VStack(alignment: .leading, spacing: 20) {
                    
                    // PERMISSIONS BANNER
                    if !permissionsGranted {
                        PermissionBanner(
                            axGranted: axGranted,
                            recGranted: recGranted,
                            requestAX: {
                                let options = [kAXTrustedCheckOptionPrompt.takeUnretainedValue() as String: true] as CFDictionary
                                AXIsProcessTrustedWithOptions(options)
                            },
                            requestREC: {
                                _ = CGRequestScreenCaptureAccess()
                            }
                        )
                        .padding(.bottom, 8)
                    }
                    
                    // HEADER
                    HStack {
                        Text("Profile: \"Default\"")
                            .font(.largeTitle).bold()
                        Spacer()
                        
                        if isOverlayActive {
                            Button("STOP SCALING") { stop() }
                                .buttonStyle(ActionButtonStyle(color: .red))
                        } else if isCountingDown {
                            Text("\(countdown)").font(.title2).foregroundColor(ACCENT_RED)
                        } else {
                            Button("SCALE") { startCountdown() }
                                .buttonStyle(ActionButtonStyle(color: .blue))
                        }
                    }
                    .padding(.bottom)
                    .disabled(!permissionsGranted)
                    
                    // GRID LAYOUT
                    HStack(alignment: .top, spacing: 20) {
                        
                        // LEFT COLUMN
                        VStack(spacing: 20) {
                            // FRAME GEN PANEL
                            ConfigPanel(title: "Frame Generation") {
                                PickerRow(label: "Type", selection: $settings.frameGenMode)
                                if settings.frameGenMode != .off {
                                    PickerRow(label: "Backend", selection: $settings.frameGenBackend)
                                }
                                Text(settings.frameGenBackend == .vision ? "Vision Optical Flow" : "MetalFX Frame Interpolator")
                                    .font(.caption)
                                    .foregroundColor(.gray)
                            }
                            
                            // CAPTURE PANEL
                            ConfigPanel(title: "Capture") {
                                ToggleRow(label: "Capture Cursor", isOn: $settings.captureCursor)
                                ToggleRow(label: "Clip Cursor (Experimental)", isOn: $settings.clipCursor)
                            }
                        }
                        .frame(maxWidth: .infinity, alignment: .topLeading)
                        
                        // RIGHT COLUMN
                        VStack(spacing: 20) {
                            // SCALING PANEL
                            ConfigPanel(title: "Scaling") {
                                PickerRow(label: "Type", selection: $settings.scalingType)
                                
                                if settings.scalingType == .metalFX {
                                    PickerRow(label: "MetalFX Mode", selection: $settings.metalFXMode)
                                    PickerRow(label: "Quality", selection: $settings.qualityMode)
                                } else if settings.scalingType == .integer {
                                    HStack {
                                        Text("Factor")
                                        Slider(value: $settings.scaleFactor, in: 1.0...5.0, step: 1.0)
                                        Text("\(Int(settings.scaleFactor))x")
                                    }
                                } else {
                                    HStack {
                                        Text("Factor")
                                        Slider(value: $settings.scaleFactor, in: 1.0...3.0, step: 0.1)
                                        Text(String(format: "%.1f", settings.scaleFactor))
                                    }
                                }
                            }
                            
                            // RENDERING PANEL
                            ConfigPanel(title: "Rendering") {
                                ToggleRow(label: "Draw FPS", isOn: $settings.showFPS)
                                ToggleRow(label: "VSync", isOn: $settings.vsync)
                                HStack {
                                    Text("Max Latency")
                                    Spacer()
                                    Picker("", selection: $settings.maxLatency) {
                                        Text("1").tag(1); Text("2").tag(2); Text("3").tag(3)
                                    }
                                    .frame(width: 50)
                                }
                            }
                        }
                        .frame(maxWidth: .infinity, alignment: .topLeading)
                    }
                    .disabled(!permissionsGranted)
                }
                .padding(30)
            }
            }
            Text(macOSVersionString)
                .font(.caption)
                .foregroundColor(.gray)
                .padding(16)
        }
        .background(BG_COLOR)
        .preferredColorScheme(.dark)
        .onAppear {
            permTimer = Timer.scheduledTimer(withTimeInterval: 1.0, repeats: true) { _ in
                axGranted = AXIsProcessTrusted()
                recGranted = CGPreflightScreenCaptureAccess()
            }
        }
        .onDisappear {
            permTimer?.invalidate()
            permTimer = nil
        }
        .onChange(of: settings.vsync, initial: false) { _, newValue in
            mtkView.preferredFramesPerSecond = newValue ? 60 : 0
        }
        .frame(width: 900, height: 600)
        .onChange(of: engine.currentFrame, initial: false) { _, newFrame in
            if let frame = newFrame {
                renderer?.processAndRender(buffer: frame, view: mtkView)
            }
        }
        .onReceive(NotificationCenter.default.publisher(for: NSApplication.willTerminateNotification)) { _ in
            stop()
        }
        .alert("Warning", isPresented: $showAlert) {
            Button("OK", role: .cancel) { }
        } message: {
            Text(alertMessage)
        }
    }
    
    // --- LOGIC ---
    func startCountdown() {
        isCountingDown = true
        countdown = 5
        Timer.scheduledTimer(withTimeInterval: 1.0, repeats: true) { timer in
            if countdown > 1 { countdown -= 1 } else {
                timer.invalidate()
                isCountingDown = false
                activateOverlay()
            }
        }
    }
    
    func activateOverlay() {
        guard let app = NSWorkspace.shared.frontmostApplication,
              app.processIdentifier != NSRunningApplication.current.processIdentifier else {
            alertMessage = "Unable to locate the foreground application. Please select a game or window."
            showAlert = true
            return
        }
        
        if settings.showFPS { setenv("MTL_HUD_ENABLED", "1", 1) } else { unsetenv("MTL_HUD_ENABLED") }
        
        let options = CGWindowListOption(arrayLiteral: .optionOnScreenOnly, .excludeDesktopElements)
        guard let list = CGWindowListCopyWindowInfo(options, kCGNullWindowID) as? [[String: Any]],
              let targetInfo = list.first(where: { ($0[kCGWindowOwnerPID as String] as? Int32) == app.processIdentifier }),
              let wid = targetInfo[kCGWindowNumber as String] as? CGWindowID,
              let bounds = targetInfo[kCGWindowBounds as String] as? [String: CGFloat] else {
            alertMessage = "Target window not found. Ensure the window is visible."
            showAlert = true
            return
        }
        
        let frame = CGRect(x: bounds["X"] ?? 0, y: bounds["Y"] ?? 0, width: bounds["Width"] ?? 100, height: bounds["Height"] ?? 100)
        let screenH = NSScreen.main?.frame.height ?? 1080
        let nsRect = CGRect(x: frame.origin.x, y: screenH - (frame.origin.y + frame.height), width: frame.width, height: frame.height)
        
        let overlayScreen = NSScreen.screens.first(where: { $0.frame.contains(nsRect.origin) }) ?? NSScreen.main
        let scale = overlayScreen?.backingScaleFactor ?? 1.0
        
        connectedProcessName = app.localizedName ?? "Unknown"
        connectedPID = app.processIdentifier
        connectedWindowID = wid
        connectedSize = nsRect.size
        
        mtkView = MTKView()
        mtkView.isPaused = false
        mtkView.enableSetNeedsDisplay = true
        mtkView.preferredFramesPerSecond = settings.vsync ? 60 : 0
        mtkView.autoResizeDrawable = false
        
        if overlayWindow != nil {
            mtkView.frame = CGRect(origin: .zero, size: nsRect.size)
            mtkView.autoresizingMask = [.width, .height]
        }
        
        let safeScale: CGFloat = (scale.isFinite && scale > 0) ? scale : 1.0
        mtkView.layer?.contentsScale = safeScale
        let dw = nsRect.width * safeScale
        let dh = nsRect.height * safeScale
        if dw.isFinite && dh.isFinite && dw > 0 && dh > 0 {
            mtkView.drawableSize = CGSize(width: dw, height: dh)
        }
        
        renderer = Renderer(metalKitView: mtkView, settings: settings)
        guard renderer != nil else {
            alertMessage = "Renderer failed to initialize. Your Metal device may be unsupported."
            showAlert = true
            return
        }
        overlayWindow = renderer?.createOverlayWindow(targetFrame: nsRect)
        if overlayWindow != nil {
            mtkView.frame = CGRect(origin: .zero, size: nsRect.size)
            mtkView.autoresizingMask = [.width, .height]
        }
        overlayWindow?.contentView = mtkView
        if let contentBounds = overlayWindow?.contentView?.bounds {
            mtkView.frame = contentBounds
        }
        overlayWindow?.orderFrontRegardless()
        NSApp.setActivationPolicy(.accessory); NSApp.deactivate()
        renderer?.startTracking(windowID: wid, pid: app.processIdentifier, overlay: overlayWindow!)
        
        Task { @MainActor in
            // Determine target display for overlay
            if let screen = NSScreen.screens.first(where: { $0.frame.contains(nsRect.origin) }) {
                targetDisplayID = screen.deviceDescription[NSDeviceDescriptionKey("NSScreenNumber")] as? CGDirectDisplayID
            }
            do {
                try await engine.startCapture(targetWindowID: wid, displayID: targetDisplayID, captureCursor: settings.captureCursor, queueDepth: settings.maxLatency * 2)
                isOverlayActive = true
            } catch {
                alertMessage = "Failed to start capture: \(error.localizedDescription)"
                showAlert = true
                stop()
            }
        }
    }
    
    func stop() {
        engine.stopCapture()
        renderer?.stopTracking()
        overlayWindow?.orderOut(nil)
        overlayWindow = nil
        renderer = nil
        isOverlayActive = false
        connectedProcessName = "-"
        connectedPID = 0
        connectedWindowID = 0
        connectedSize = .zero
        NSApp.setActivationPolicy(.regular)
    }
}

// --- UI COMPONENTS ---
struct ConfigPanel<Content: View>: View {
    let title: String
    let content: Content
    
    init(title: String, @ViewBuilder content: () -> Content) {
        self.title = title
        self.content = content()
    }
    
    var body: some View {
        VStack(alignment: .leading, spacing: 15) {
            Text(title).font(.title3).bold().foregroundColor(.white)
            Divider().background(Color.gray)
            content
        }
        .padding()
        .frame(maxWidth: .infinity, alignment: .leading)
        .background(PANEL_COLOR)
        .cornerRadius(10)
    }
}

// FIX: T generic type must be CaseIterable to use T.allCases
struct PickerRow<T: Hashable & Identifiable & RawRepresentable & CaseIterable>: View where T.RawValue == String {
    let label: String
    @Binding var selection: T
    
    var body: some View {
        HStack {
            Text(label).foregroundColor(.gray)
            Spacer()
            Picker("", selection: $selection) {
                ForEach(Array(T.allCases), id: \.id) { item in
                    Text(item.rawValue).tag(item)
                }
            }
            .labelsHidden()
            .frame(minWidth: 160, maxWidth: 220)
        }
    }
}

struct ToggleRow: View {
    let label: String
    @Binding var isOn: Bool
    
    var body: some View {
        HStack {
            Text(label).foregroundColor(.gray)
            Spacer()
            Toggle("", isOn: $isOn).labelsHidden()
        }
    }
}

struct InfoRow: View {
    let label: String
    let value: String
    var body: some View {
        HStack {
            Text(label).foregroundColor(.gray)
            Spacer()
            Text(value).foregroundColor(.white)
                .font(.system(.body, design: .monospaced))
        }
    }
}

struct ActionButtonStyle: ButtonStyle {
    let color: Color
    func makeBody(configuration: Configuration) -> some View {
        configuration.label
            .padding()
            .frame(minWidth: 120)
            .frame(height: 36)
            .background(color.opacity(configuration.isPressed ? 0.7 : 1.0))
            .foregroundColor(.white).cornerRadius(8)
            .fontWeight(.bold)
    }
}

struct PermissionBanner: View {
    let axGranted: Bool
    let recGranted: Bool
    let requestAX: () -> Void
    let requestREC: () -> Void

    var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            HStack(spacing: 12) {
                StatusPill(label: "Accessibility", ok: axGranted, action: requestAX)
                StatusPill(label: "Screen Recording", ok: recGranted, action: requestREC)
                Spacer()
            }
        }
        .padding(12)
        .background(Color.yellow.opacity(0.15))
        .overlay(
            RoundedRectangle(cornerRadius: 8).stroke(Color.yellow.opacity(0.4), lineWidth: 1)
        )
        .cornerRadius(8)
    }
}

struct StatusPill: View {
    let label: String
    let ok: Bool
    let action: () -> Void

    var body: some View {
        HStack(spacing: 8) {
            Text(ok ? "[ PASS ]" : "[ REQUIRED ]")
                .foregroundColor(ok ? .green : .orange)
                .font(.system(.caption, design: .monospaced))
            Text(label)
                .foregroundColor(.white)
                .font(.caption)
            if !ok {
                Button("GRANT") { action() }
                    .buttonStyle(.borderedProminent)
                    .tint(.orange)
            }
        }
        .padding(.horizontal, 10)
        .padding(.vertical, 6)
        .background(Color.black.opacity(0.4))
        .cornerRadius(6)
    }
}

