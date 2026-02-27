import SwiftUI
import MetalKit
import ApplicationServices

@available(macOS 26.0, *)
struct ContentView: View {

    @StateObject var settings = CaptureSettings.shared

    @State private var countdown = 5
    @State private var isCountingDown = false
    @State private var isScalingActive = false


    
    @State private var gooseEngine: GooseEngine?
    @State private var windowCaptureManager: WindowCaptureManager?
    @State private var overlayManager: OverlayWindowManager?
    @State private var gooseMtkView: MTKView?

    @State private var connectedProcessName: String = "-"
    @State private var connectedPID: Int32 = 0
    @State private var connectedWindowID: CGWindowID = 0
    @State private var connectedSize: CGSize = .zero

    @State private var showAlert = false
    @State private var alertMessage = ""

    @State private var currentFPS: Float = 0.0
    @State private var interpolatedFPS: Float = 0.0
    @State private var processingTime: Double = 0.0
    @State private var isTransitioning: Bool = false
    @State private var lastHotkeyTime: CFTimeInterval = 0

    @State private var axGranted: Bool = AXIsProcessTrusted()
    @State private var recGranted: Bool = CGPreflightScreenCaptureAccess()

    @State private var permTimer: Timer?

    private var permissionsGranted: Bool { axGranted && recGranted }

    @State private var targetDisplayID: CGDirectDisplayID?
    @State private var activeOutputScreen: NSScreen?

    @State private var statsTimer: Timer?

    @State private var hotkeyMonitor: Any?
    @State private var localHotkeyMonitor: Any?

    @State private var hudController = MGHUDWindowController()

    private var macOSVersionString: String {
        let v = ProcessInfo.processInfo.operatingSystemVersion
        return "macOS \(v.majorVersion).\(v.minorVersion).\(v.patchVersion)"
    }
    
    private var scalingStatusText: String {
        if isScalingActive {
            return String(localized: "Active", defaultValue: "Active")
        }
        return String(localized: "Idle", defaultValue: "Idle")
    }
    
    private var fpsText: String {
        return currentFPS > 0 ? String(format: "%.1f", currentFPS) : "-"
    }
    
    private var interpFPSText: String {
        return String(format: "%.1f", interpolatedFPS)
    }
    
    private var latencyText: String {
        return processingTime > 0 ? String(format: "%.2f ms", processingTime) : "-"
    }
    
    private var frameSizeText: String {
        return connectedSize.width > 0 ? "\(Int(connectedSize.width)) x \(Int(connectedSize.height))" : "-"
    }
    
    private var displayIDText: String {
        if let id = targetDisplayID {
            return String(id)
        }
        return "-"
    }
    
    private var sidebarHeader: some View {
        HStack {
            Image("GooseLogo")
                .resizable()
                .aspectRatio(contentMode: .fit)
                .frame(width: 40, height: 40)
                .cornerRadius(8)
            Text("MetalGoose")
                .font(.headline)
        }
        .padding([.top, .horizontal])
    }
    
    private var scalingInfoSection: some View {
        VStack(alignment: .leading, spacing: 12) {
            Text(String(localized: "Scaling Info", defaultValue: "Scaling Info"))
                .font(.headline)
                .padding([.top, .horizontal])
            
            VStack(alignment: .leading, spacing: 8) {
                InfoRow(label: String(localized: "Status", defaultValue: "Status"),
                        value: scalingStatusText)
                InfoRow(label: String(localized: "FPS", defaultValue: "FPS"),
                        value: fpsText)
                
                if interpolatedFPS > currentFPS {
                    InfoRow(label: String(localized: "Interp FPS", defaultValue: "Interp FPS"),
                            value: interpFPSText)
                }
                
                InfoRow(label: String(localized: "Latency", defaultValue: "Latency"),
                        value: latencyText)
                
                InfoRow(label: String(localized: "Process", defaultValue: "Process"),
                        value: connectedProcessName)
                InfoRow(label: String(localized: "PID", defaultValue: "PID"),
                        value: String(connectedPID))
                InfoRow(label: String(localized: "Window ID", defaultValue: "Window ID"),
                        value: String(connectedWindowID))
                
                InfoRow(label: String(localized: "Frame", defaultValue: "Frame"),
                        value: frameSizeText)
                
                InfoRow(label: String(localized: "Display ID", defaultValue: "Display ID"),
                        value: displayIDText)
            }
            .padding(.horizontal)
            .padding(.bottom)
        }
    }
    
    private var sidebarMenu: some View {
        HStack {
            Spacer()
            Menu {
                Button("About") {
                    let v: String
                    if let version = Bundle.main.infoDictionary?["CFBundleShortVersionString"] as? String {
                        v = version
                    } else {
                        v = ""
                    }
                    let b: String
                    if let build = Bundle.main.infoDictionary?["CFBundleVersion"] as? String {
                        b = build
                    } else {
                        b = ""
                    }
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
    
    private var sidebarView: some View {
        VStack(alignment: .leading) {
            sidebarHeader
            scalingInfoSection
            Spacer()
            sidebarMenu
        }
        .frame(minWidth: 200)
        .disabled(!permissionsGranted)
        .navigationTitle("MetalGoose")
    }
    
    private var detailView: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 20) {
                
                if !permissionsGranted {
                    PermissionBanner(
                        axGranted: axGranted,
                        recGranted: recGranted,
                        requestAX: {
                            let opts = [
                                kAXTrustedCheckOptionPrompt.takeUnretainedValue() as String: true
                            ] as CFDictionary
                            AXIsProcessTrustedWithOptions(opts)
                        },
                        requestREC: {
                            _ = CGRequestScreenCaptureAccess()
                        }
                    )
                    .padding(.bottom, 8)
                }
                
                headerSection
                
                HStack(alignment: .top, spacing: 20) {
                    leftConfigColumn
                    rightConfigColumn
                }
                .disabled(!permissionsGranted)
                .opacity(permissionsGranted ? 1.0 : 0.5)
                
            }
            .padding(24)
        }
    }

    var body: some View {
        NavigationSplitView {
            sidebarView
        } detail: {
            detailView
        }
        .overlay(alignment: .bottomLeading) {
            Text(macOSVersionString)
            .font(.caption2)
            .foregroundColor(.gray.opacity(0.5))
            .padding(6)
        }
        .onAppear {
            startPermissionTimer()
            initializeGooseEngine()
            setupHotkeys()
        }
        .onDisappear {
            permTimer?.invalidate()
            statsTimer?.invalidate()
            if let monitor = hotkeyMonitor {
                NSEvent.removeMonitor(monitor)
            }
            if let local = localHotkeyMonitor {
                NSEvent.removeMonitor(local)
            }
        }
        .onReceive(settings.objectWillChange) { _ in
            gooseEngine?.updateSettings(settings)
        }
        .onChange(of: settings.showMGHUD, initial: false) { _, newValue in
            if newValue && isScalingActive {
                if let screen = activeOutputScreen {
                    hudController.show(on: screen, compact: false)
                }
            } else {
                hudController.hide()
            }
        }
        .onReceive(NotificationCenter.default.publisher(for: NSApplication.willTerminateNotification)) { _ in
            stop()
        }
        .alert("Version Info", isPresented: $showAlert) {
            Button("OK", role: .cancel) {}
        } message: {
            Text(alertMessage)
        }
    }
    
    private var headerSection: some View {
        HStack {
            Text("Profile: \"\(settings.selectedProfile)\"")
                .font(.largeTitle).bold()
            Spacer()

            if isScalingActive {
                Button("STOP SCALING") { stop() }
                    .buttonStyle(ActionButtonStyle(color: .red))

            } else if isCountingDown {
                Text("\(countdown)")
                    .font(.title2)
                    .foregroundColor(.red)

            } else {
                Button("START SCALING") { startCountdown() }
                    .buttonStyle(ActionButtonStyle(color: .green))
                    .disabled(!permissionsGranted)
                    .opacity(permissionsGranted ? 1.0 : 0.5)
            }
        }
        .padding(.bottom, 10)
    }

    private var leftConfigColumn: some View {
        VStack(spacing: 16) {
            
            Group {
                ConfigPanel(title: String(localized: "Upscaling", defaultValue: "Upscaling")) {
                        PickerRow(label: String(localized: "Method", defaultValue: "Method"),
                                  selection: $settings.scalingType,
                                  helpText: String(localized: "Upscaling mode:\n• Off\n• MGUP-1 / Fast / Quality",
                                                    defaultValue: "Upscaling mode:\n• Off\n• MGUP-1 / Fast / Quality"))

                        if settings.scalingType != .off {
                            PickerRow(label: String(localized: "Scale Factor", defaultValue: "Scale Factor"),
                                      selection: $settings.scaleFactor,
                                      helpText: String(localized: "Upscale multiplier (1.5x – 10x).",
                                                        defaultValue: "Upscale multiplier (1.5x – 10x)."))

                            PickerRow(label: String(localized: "Render Scale", defaultValue: "Render Scale"),
                                      selection: $settings.renderScale,
                                      helpText: String(localized: "Internal capture resolution %.",
                                                        defaultValue: "Internal capture resolution %."))
                        }
                    }

                ConfigPanel(title: String(localized: "Frame Generation", defaultValue: "Frame Generation")) {
                       PickerRow(label: String(localized: "Mode", defaultValue: "Mode"),
                                 selection: $settings.frameGenMode,
                                 helpText: String(localized: "• Off: lowest latency\n• MGFG-1: motion-aware interpolation",
                                                   defaultValue: "• Off: lowest latency\n• MGFG-1: motion-aware interpolation"))

                       if settings.frameGenMode != .off {
                           Text(settings.frameGenMode.description)
                               .font(.caption)
                               .foregroundColor(.secondary)
                               .padding(.leading, 4)

                           PickerRow(label: String(localized: "Type", defaultValue: "Type"),
                                     selection: $settings.frameGenType,
                                     helpText: String(localized: "Adaptive or Fixed", defaultValue: "Adaptive or Fixed"))

                           if settings.frameGenType == .adaptive {
                               PickerRow(label: String(localized: "Target FPS", defaultValue: "Target FPS"),
                                         selection: $settings.targetFPS,
                                         helpText: String(localized: "Target FPS.", defaultValue: "Target FPS."))
                           } else {
                               PickerRow(label: String(localized: "Multiplier", defaultValue: "Multiplier"),
                                         selection: $settings.frameGenMultiplier,
                                         helpText: String(localized: "2× / 3× / 4×", defaultValue: "2× / 3× / 4×"))
                           }
                       }
                   }

                   ConfigPanel(title: String(localized: "Anti-Aliasing", defaultValue: "Anti-Aliasing")) {
                       PickerRow(label: String(localized: "Mode", defaultValue: "Mode"),
                                 selection: $settings.aaMode,
                                 helpText: String(localized: "FXAA / SMAA / MSAA-like / TAA",
                                                   defaultValue: "FXAA / SMAA / MSAA-like / TAA"))

                       if settings.aaMode != .off {
                           Text(settings.aaMode.description)
                               .font(.caption)
                               .foregroundColor(.secondary)
                           }
                   }
            }
        }
    }

    private var rightConfigColumn: some View {
        VStack(spacing: 16) {

            if settings.scalingType == .mgup1 {
                ConfigPanel(title: String(localized: "MGUP-1 Settings", comment: "Panel title: MGUP-1 settings")) {
                    PickerRow(label: String(localized: "Quality", comment: "Label: Quality"),
                              selection: $settings.qualityMode,
                              helpText: String(localized: "MetalFX + CAS", comment: "Help text: MetalFX + CAS"))

                    Text(String(localized: "MetalGoose Upscaler - MetalFX Spatial + CAS", comment: "Description text"))
                        .font(.caption)
                        .foregroundColor(.secondary)
                }
            }

            Group {
                ConfigPanel(title: String(localized: "Display Settings", comment: "Panel title: Display settings")) {
                    ToggleRow(label: String(localized: "Show MG HUD", comment: "Toggle label"), isOn: $settings.showMGHUD,
                              helpText: String(localized: "Overlay", comment: "Toggle help text"))

                    ToggleRow(label: String(localized: "Capture Cursor", comment: "Toggle label"), isOn: $settings.captureCursor,
                              helpText: String(localized: "Include cursor", comment: "Toggle help text"))

                    ToggleRow(label: String(localized: "VSync", comment: "Toggle label"), isOn: $settings.vsync,
                              helpText: String(localized: "Sync to display", comment: "Toggle help text"))

                    ToggleRow(label: String(localized: "Adaptive Sync", comment: "Toggle label"), isOn: $settings.adaptiveSync,
                              helpText: String(localized: "Auto pacing", comment: "Toggle help text"))

                    SliderRow(label: String(localized: "Sharpness", comment: "Slider label"), value: $settings.sharpening, range: 0...1,
                              helpText: String(localized: "CAS intensity", comment: "Slider help text"))
                }
            }
        }
    }
    
    private func initializeGooseEngine() {
        guard gooseEngine == nil else { return }

        let engine = GooseEngine()
        gooseEngine = engine
        windowCaptureManager = WindowCaptureManager()
        engine.updateSettings(settings)
        
    }
    
    private func startGooseCapture() {
        if isTransitioning { return }
        isTransitioning = true
        Task { @MainActor in
            defer { isTransitioning = false }
            await startGooseCaptureAsync()
        }
    }

    @MainActor
    private func startGooseCaptureAsync() async {
        guard let app = NSWorkspace.shared.frontmostApplication,
              app.processIdentifier != NSRunningApplication.current.processIdentifier else {
            alertMessage = "Error Code: MG-UI-002 Please switch to the target window before starting."
            showAlert = true
            return
        }
        
        let opts: CGWindowListOption = [.optionOnScreenOnly, .excludeDesktopElements]
        guard let list = CGWindowListCopyWindowInfo(opts, kCGNullWindowID) as? [[String: Any]],
              let targetInfo = list.first(where: { ($0[kCGWindowOwnerPID as String] as? Int32) == app.processIdentifier }),
              let wid = targetInfo[kCGWindowNumber as String] as? CGWindowID else {
            alertMessage = "Error Code: MG-UI-003 Target window not found."
            showAlert = true
            return
        }
        
        let windowTitle = targetInfo[kCGWindowName as String] as? String
        
        guard let boundsDict = targetInfo[kCGWindowBounds as String] as? [String: CGFloat],
              let boundX = boundsDict["X"],
              let boundY = boundsDict["Y"],
              let boundW = boundsDict["Width"],
              let boundH = boundsDict["Height"] else {
            alertMessage = "Error Code: MG-UI-006 Window bounds unavailable."
            showAlert = true
            return
        }
        
        let cgFrame = CGRect(x: boundX, y: boundY, width: boundW, height: boundH)
        
        if gooseEngine == nil { initializeGooseEngine() }
        if windowCaptureManager == nil { windowCaptureManager = WindowCaptureManager() }
        if overlayManager == nil { overlayManager = OverlayWindowManager() }
        
        guard let engine = gooseEngine,
              let overlay = overlayManager else { return }
        
        engine.updateSettings(settings)
        
        guard let outputScreen = NSScreen.main else {
            alertMessage = "Error Code: MG-UI-004 No display found."
            showAlert = true
            return
        }

        let outputSize = cgFrame.size

        guard let displayID = outputScreen.deviceDescription[NSDeviceDescriptionKey("NSScreenNumber")] as? CGDirectDisplayID else {
            alertMessage = "Error Code: MG-UI-007 Display ID not found."
            showAlert = true
            return
        }
        let targetRefreshRate = Double(settings.targetFPS.intValue)
        let captureRefreshRate = Int(round(targetRefreshRate))
        
        guard let captureManager = windowCaptureManager else { return }
        
        let sourceRes = cgFrame.size
        let scaledOutputSize = outputScreen.frame.size
        
        engine.configure(sourceResolution: sourceRes, outputSize: scaledOutputSize)
        
        let success = await captureManager.startCapture(windowID: wid, refreshRate: captureRefreshRate)
        if success {
            await engine.startCaptureFromWindow(captureManager, refreshRate: captureRefreshRate)
            
            if let name = app.localizedName {
                connectedProcessName = name
            } else {
                connectedProcessName = ""
            }
            connectedPID = app.processIdentifier
            connectedWindowID = wid
            connectedSize = sourceRes
            targetDisplayID = displayID
            activeOutputScreen = outputScreen
            isScalingActive = true
            
            let config = OverlayWindowConfig(
                targetScreen: outputScreen,
                windowFrame: cgFrame,
                size: scaledOutputSize,
                refreshRate: targetRefreshRate,
                vsyncEnabled: settings.vsync,
                adaptiveSyncEnabled: settings.adaptiveSync,
                passThrough: true,
                scaleFactor: 1.0,
                captureCursor: settings.captureCursor
            )
            
            if overlay.createOverlay(config: config) {
                let mtkView = MTKView(frame: CGRect(origin: .zero, size: scaledOutputSize))
                overlay.setMTKView(mtkView)
                engine.attachToView(mtkView, refreshRate: captureRefreshRate)
                gooseMtkView = mtkView
                
                overlay.setTargetWindow(wid, pid: app.processIdentifier)
                overlay.updateWindowPosition()
            }
            
            startStatsTimer()
            
            if settings.showMGHUD {
                hudController.show(on: outputScreen, compact: false)
                hudController.setDeviceName(engine.deviceName)
                hudController.setResolutions(capture: sourceRes, output: scaledOutputSize)
            }
            
            DispatchQueue.main.asyncAfter(deadline: .now() + 0.3) {
                let targetApp = NSWorkspace.shared.runningApplications.first { $0.processIdentifier == app.processIdentifier }
                targetApp?.activate()
            }
        } else {
            alertMessage = captureManager.lastError ?? "Unknown Capture Error"
            showAlert = true
            await captureManager.stopCapture()
        }
    }
    
    private func stopGooseCapture() {
        if isTransitioning { return }
        isTransitioning = true
        Task { @MainActor in
            defer { isTransitioning = false }
            await stopGooseCaptureAsync()
        }
    }

    @MainActor
    private func stopGooseCaptureAsync() async {
        statsTimer?.invalidate()
        statsTimer = nil
        
        gooseEngine?.detachFromView()
        overlayManager?.destroyOverlay()
        gooseMtkView = nil
        activeOutputScreen = nil
        
        if let engine = gooseEngine {
            await engine.stopCapture()
        }

        if let captureManager = windowCaptureManager {
            await captureManager.stopCapture()
        }
        
        isScalingActive = false
        currentFPS = 0
        interpolatedFPS = 0
        processingTime = 0
        connectedProcessName = "-"
        connectedPID = 0
        connectedWindowID = 0
        connectedSize = .zero
        targetDisplayID = nil
        
        hudController.hide()
    }

    private func setupHotkeys() {
        hotkeyMonitor = NSEvent.addGlobalMonitorForEvents(matching: .keyDown) { event in
            if event.modifierFlags.contains([.command, .shift]),
               event.charactersIgnoringModifiers?.lowercased() == "t" {
                Task { @MainActor in handleHotkeyToggle() }
            }
        }
        localHotkeyMonitor = NSEvent.addLocalMonitorForEvents(matching: .keyDown) { event in
            if event.modifierFlags.contains([.command, .shift]),
               event.charactersIgnoringModifiers?.lowercased() == "t" {
                Task { @MainActor in handleHotkeyToggle() }
                return event
            }
            return event
        }
    }
    
    private func handleHotkeyToggle() {
        let now = CACurrentMediaTime()
        if now - lastHotkeyTime < 0.4 { return }
        lastHotkeyTime = now
        toggleScaling()
    }

    private func toggleScaling() {
        guard permissionsGranted else { return }
        if isTransitioning { return }
        if isScalingActive { stopGooseCapture() } else { startGooseCapture() }
    }

    private func startPermissionTimer() {
        permTimer?.invalidate()
        permTimer = Timer.scheduledTimer(withTimeInterval: 1.0, repeats: true) { _ in
            axGranted = AXIsProcessTrusted()
            recGranted = CGPreflightScreenCaptureAccess()
        }
    }

    private func startStatsTimer() {
        statsTimer?.invalidate()
        statsTimer = Timer.scheduledTimer(withTimeInterval: 1.0, repeats: true) { [self] _ in
            Task { @MainActor in
                if let engine = self.gooseEngine {
                    let stats = engine.stats
                    self.currentFPS = stats.outputFPS
                    self.interpolatedFPS = stats.interpolatedFPS
                    self.processingTime = Double(stats.frameTime)
                    
                    self.hudController.updateFromGooseEngine(stats: stats, settings: self.settings)
                }
                self.overlayManager?.setCaptureCursorEnabled(self.settings.captureCursor)
                self.overlayManager?.updateWindowPosition()
            }
        }
    }

    func startCountdown() {
        isCountingDown = true
        countdown = 5
        Timer.scheduledTimer(withTimeInterval: 1.0, repeats: true) { timer in
            if countdown > 1 { countdown -= 1 }
            else {
                timer.invalidate()
                isCountingDown = false
                
                if let window = NSApp.windows.first(where: { $0.isVisible && $0.title.contains("MetalGoose") || $0.contentViewController != nil }) {
                    window.orderOut(nil)
                }
                
                startGooseCapture()
            }
        }
    }

    func stop() {
        stopGooseCapture()
        statsTimer?.invalidate()
        statsTimer = nil
        hudController.hide()
        isScalingActive = false
        currentFPS = 0.0
        interpolatedFPS = 0.0
        processingTime = 0.0
        connectedProcessName = "-"
        connectedPID = 0
        connectedWindowID = 0
        connectedSize = .zero
        targetDisplayID = nil
    }

}

struct ConfigPanel<Content: View>: View {
    let title: String
    let content: Content
    init(title: String, @ViewBuilder content: () -> Content) {
        self.title = title
        self.content = content()
    }
    var body: some View {
        VStack(alignment: .leading, spacing: 15) {
            Text(title).font(.title3).bold()
            Divider().background(Color.gray)
            content
        }
        .padding()
        .frame(maxWidth: .infinity, alignment: .leading)
        .background(Color(NSColor.windowBackgroundColor		))
        .cornerRadius(10)
    }
}

struct PickerRow<T: Hashable & Identifiable & RawRepresentable & CaseIterable>: View where T.RawValue == String {
    let label: String
    @Binding var selection: T
    var helpText: String? = nil
    var body: some View {
        let row = HStack {
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
        if let helpText { row.help(helpText) } else { row }
    }
}

struct ToggleRow: View {
    let label: String
    @Binding var isOn: Bool
    var helpText: String? = nil
    var body: some View {
        let row = HStack {
            Text(label).foregroundColor(.gray)
            Spacer()
            Toggle("", isOn: $isOn).labelsHidden()
        }
        if let helpText { row.help(helpText) } else { row }
    }
}

struct SliderRow: View {
    let label: String
    @Binding var value: Float
    var range: ClosedRange<Float> = 0...1
    var helpText: String? = nil
    var body: some View {
        let row = HStack {
            Text(label).foregroundColor(.gray)
            Spacer()
            Slider(value: $value, in: range)
                .frame(width: 120)
            Text(String(format: "%.2f", value))
                .font(.system(.caption, design: .monospaced))
                .frame(width: 40, alignment: .trailing)
        }
        if let helpText { row.help(helpText) } else { row }
    }
}

struct InfoRow: View {
    let label: String
    let value: String
    var body: some View {
        HStack {
            Text(label).foregroundColor(.gray)
            Spacer()
            Text(value)
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
            .cornerRadius(8)
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
        .overlay(RoundedRectangle(cornerRadius: 8).stroke(Color.yellow.opacity(0.4), lineWidth: 1))
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
                .font(.caption)
            if !ok {
                Button("GRANT") { action() }
                    .buttonStyle(.borderedProminent)
                    .tint(.orange)
            }
        }
        .padding(.horizontal, 10)
        .padding(.vertical, 6)
        .cornerRadius(6)
    }
}
