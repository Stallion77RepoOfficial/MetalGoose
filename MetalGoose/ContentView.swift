import SwiftUI
import MetalKit
import ApplicationServices
import Carbon.HIToolbox

struct ContentView: View {

    @StateObject var settings = CaptureSettings.shared
    @StateObject private var updater = AutoUpdater.shared

    @State private var countdown = 5
    @State private var isCountingDown = false
    @State private var isScalingActive = false


    
    @State private var gooseEngine: GooseEngine?
    @State private var windowCaptureManager: WindowCaptureManager?
    @State private var overlayManager: OverlayWindowManager?

    @State private var connectedPID: Int32 = 0

    @State private var showAlert = false
    @State private var alertMessage = ""

    @State private var isTransitioning: Bool = false
    @State private var lastHotkeyTime: CFTimeInterval = 0
    @State private var lastCursorHotkeyTime: CFTimeInterval = 0

    @State private var axGranted: Bool = AXIsProcessTrusted()
    @State private var recGranted: Bool = CGPreflightScreenCaptureAccess()

    @State private var permTimer: Timer?

    private var permissionsGranted: Bool { axGranted && recGranted }

    @State private var activeOutputScreen: NSScreen?

    @State private var statsTimer: Timer?
    @State private var countdownTimer: Timer?

    // Consecutive stats-timer ticks where the target appears to be in an unreachable
    // (native fullscreen) Space. Must reach the threshold before we act, so the brief
    // fullscreen-transition animation never triggers a false positive.
    @State private var fullscreenStrikes = 0
    private let fullscreenStrikeThreshold = 3

    @State private var hudController = MGHUDWindowController()

    private var macOSVersionString: String {
        let v = ProcessInfo.processInfo.operatingSystemVersion
        return "macOS \(v.majorVersion).\(v.minorVersion).\(v.patchVersion)"
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
                    updater.checkForUpdates()
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
                                "AXTrustedCheckOptionPrompt": true
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
            .padding(.leading, 16)
            .padding(.bottom, 12)
        }
        .onAppear {
            startPermissionTimer()
            initializeGooseEngine()
            setupHotkeys()
        }
        .onDisappear {
            permTimer?.invalidate()
            statsTimer?.invalidate()
            GlobalHotkeyManager.shared.unregisterAll()
        }
        .onReceive(settings.objectWillChange) { _ in
            DispatchQueue.main.async {
                gooseEngine?.updateSettings(settings)
            }
        }
        .onChange(of: settings.showMGHUD, initial: false) { _, newValue in
            if newValue && isScalingActive {
                if let screen = activeOutputScreen {
                    hudController.show(on: screen, compact: false)
                    if let engine = gooseEngine {
                        hudController.setDeviceName(engine.deviceName)
                    }
                    hudController.setPID(connectedPID)
                }
            } else {
                hudController.hide()
            }
        }
        .onReceive(NotificationCenter.default.publisher(for: NSApplication.willTerminateNotification)) { _ in
            stop()
        }
        .alert("MetalGoose", isPresented: $showAlert) {
            Button("OK", role: .cancel) {}
        } message: {
            Text(alertMessage)
        }
        // Update: already up to date
        .alert(String(localized: "Already up to date", comment: "Update alert title"),
               isPresented: Binding(
                get: { if case .upToDate = updater.state { return true }; return false },
                set: { if !$0 { updater.state = .idle } }
               )) {
            Button("OK", role: .cancel) {}
        } message: {
            Text(String(localized: "MetalGoose is already up to date.", comment: "Update alert body"))
        }
        // Update: new version available
        .alert(String(localized: "Update Available", comment: "Update alert title"),
               isPresented: Binding(
                get: { if case .available = updater.state { return true }; return false },
                set: { if !$0 { updater.state = .idle } }
               )) {
            if case .available(let release) = updater.state {
                Button(String(localized: "Download & Install", comment: "Update button")) {
                    updater.downloadAndInstall(release: release)
                }
            }
            Button(String(localized: "Later", comment: "Dismiss update"), role: .cancel) {
                updater.state = .idle
            }
        } message: {
            if case .available(let release) = updater.state {
                Text(String(format: String(localized: "A new version is available: %@\nWould you like to download and install it now?", comment: "Update body"), release.tagName))
            }
        }
        // Update: error
        .alert(String(localized: "Update Failed", comment: "Update error title"),
               isPresented: Binding(
                get: { if case .failed = updater.state { return true }; return false },
                set: { if !$0 { updater.state = .idle } }
               )) {
            Button("OK", role: .cancel) { updater.state = .idle }
        } message: {
            if case .failed(let msg) = updater.state {
                Text(msg)
            }
        }
        // Update: downloading / installing progress sheet
        .sheet(isPresented: Binding(
            get: {
                switch updater.state {
                case .checking, .downloading, .installing, .done: return true
                default: return false
                }
            },
            set: { _ in }
        )) {
            UpdateProgressSheet(state: updater.state)
        }
    }
    
    private var headerSection: some View {
        HStack {
            Text("MetalGoose")
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
                                  selection: $settings.scalingType)

                        if settings.scalingType != .off {
                            PickerRow(label: String(localized: "Scale Factor", defaultValue: "Scale Factor"),
                                      selection: $settings.scaleFactor)

                            PickerRow(label: String(localized: "Render Scale", defaultValue: "Render Scale"),
                                      selection: $settings.renderScale)
                        }
                    }

                ConfigPanel(title: String(localized: "Frame Generation", defaultValue: "Frame Generation")) {
                       PickerRow(label: String(localized: "Mode", defaultValue: "Mode"),
                                 selection: $settings.frameGenMode)

                   }

                   ConfigPanel(title: String(localized: "Anti-Aliasing", defaultValue: "Anti-Aliasing")) {
                       PickerRow(label: String(localized: "Mode", defaultValue: "Mode"),
                                 selection: $settings.aaMode)
                   }
            }
        }
    }

    private var rightConfigColumn: some View {
        VStack(spacing: 16) {

            if settings.scalingType == .mgup1 {
                ConfigPanel(title: String(localized: "MGUP-1 Settings", comment: "Panel title: MGUP-1 settings")) {
                    PickerRow(label: String(localized: "Quality", comment: "Label: Quality"),
                              selection: $settings.qualityMode)

                }
            }

            Group {
                ConfigPanel(title: String(localized: "Display Settings", comment: "Panel title: Display settings")) {
                    ToggleRow(label: String(localized: "Show MG HUD", comment: "Toggle label"), isOn: $settings.showMGHUD)

                    ToggleRow(label: String(localized: "Capture Cursor", comment: "Toggle label"), isOn: $settings.captureCursor)

                    ToggleRow(label: String(localized: "VSync", comment: "Toggle label"), isOn: $settings.vsync)

                    StepperSliderRow(label: String(localized: "Frame Buffering", comment: "Slider label: pipeline buffer depth"),
                                     value: $settings.bufferCount,
                                     range: CaptureSettings.minBufferCount...CaptureSettings.maxBufferCount)
                        .disabled(isScalingActive)
                }

                ConfigPanel(title: String(localized: "Maintenance", comment: "Panel title: Maintenance")) {
                    Button {
                        clearMetalCache()
                    } label: {
                        Text(String(localized: "Clear Metal Cache", comment: "Button: clear Metal shader cache"))
                            .frame(maxWidth: .infinity)
                    }
                    .disabled(isScalingActive)
                }
            }
        }
    }

    private func clearMetalCache() {
        let fm = FileManager.default
        let bundleID = Bundle.main.bundleIdentifier ?? "com.MetalGoose"
        var targets: [URL] = []

        // The real compiled-shader / pipeline cache lives in the Darwin user cache
        // directory (DARWIN_USER_CACHE_DIR = /var/folders/.../C/), NOT ~/Library/Caches.
        // temporaryDirectory is .../T/, so its sibling "C" is that cache root.
        let darwinCache = fm.temporaryDirectory
            .deletingLastPathComponent()
            .appendingPathComponent("C", isDirectory: true)
        for name in [bundleID, "com.apple.metal", "com.apple.metalfx", "com.apple.metalfe"] {
            targets.append(darwinCache.appendingPathComponent(name, isDirectory: true))
        }

        // Secondary: app-specific cache under ~/Library/Caches (if present).
        if let caches = fm.urls(for: .cachesDirectory, in: .userDomainMask).first {
            targets.append(caches.appendingPathComponent(bundleID, isDirectory: true))
        }

        var removed = 0
        for url in targets where fm.fileExists(atPath: url.path) {
            do {
                try fm.removeItem(at: url)
                removed += 1
            } catch {
                // A locked/in-use cache entry is non-fatal; the rest still clear.
            }
        }

        alertMessage = removed > 0
            ? String(localized: "Metal cache cleared (\(removed) location(s)). Restart MetalGoose so shaders rebuild cleanly.", comment: "Alert: cache cleared")
            : String(localized: "No Metal cache found to clear.", comment: "Alert: nothing to clear")
        showAlert = true
    }
    
    private func initializeGooseEngine() {
        guard gooseEngine == nil else { return }

        guard let engine = GooseEngine.make() else {
            alertMessage = GooseEngine.lastInitError ?? "Error Code: MG-ENG-002 Metal device not available."
            showAlert = true
            return
        }
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
            alertMessage = "Error Code: MG-UI-001 Please switch to the target window before starting."
            showAlert = true
            return
        }
        
        let opts: CGWindowListOption = [.optionOnScreenOnly, .excludeDesktopElements]
        guard let list = CGWindowListCopyWindowInfo(opts, kCGNullWindowID) as? [[String: Any]],
              let targetInfo = list.first(where: { ($0[kCGWindowOwnerPID as String] as? Int32) == app.processIdentifier }),
              let wid = targetInfo[kCGWindowNumber as String] as? CGWindowID else {
            alertMessage = "Error Code: MG-UI-002 Target window not found."
            showAlert = true
            return
        }
        
        guard let boundsDict = targetInfo[kCGWindowBounds as String] as? [String: CGFloat],
              let boundX = boundsDict["X"],
              let boundY = boundsDict["Y"],
              let boundW = boundsDict["Width"],
              let boundH = boundsDict["Height"] else {
            alertMessage = "Error Code: MG-UI-003 Window bounds unavailable."
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
        
        let primaryHeight = NSScreen.screens.first?.frame.height ?? 0
        let cocoaWindowFrame = CGRect(
            x: cgFrame.origin.x,
            y: primaryHeight - cgFrame.maxY,
            width: cgFrame.width,
            height: cgFrame.height
        )
        guard let outputScreen = NSScreen.screens.first(where: { $0.frame.intersects(cocoaWindowFrame) }) ?? NSScreen.main else {
            alertMessage = "Error Code: MG-UI-004 No display found."
            showAlert = true
            return
        }

        guard let displayID = outputScreen.deviceDescription[NSDeviceDescriptionKey("NSScreenNumber")] as? CGDirectDisplayID else {
            alertMessage = "Error Code: MG-UI-005 Display ID not found."
            showAlert = true
            return
        }

        let displayMaxFPS = outputScreen.maximumFramesPerSecond > 0 ? outputScreen.maximumFramesPerSecond : 60

        guard let captureManager = windowCaptureManager else { return }

        let sourceRes = cgFrame.size
        let shouldFullScreen = settings.scalingType != .off
        let scaledOutputSize = shouldFullScreen ? outputScreen.frame.size : sourceRes

        engine.configure(outputSize: scaledOutputSize)

        let success = await captureManager.startCapture(windowID: wid, maxFPS: displayMaxFPS, showsCursor: false)
        if success {
            await engine.startCaptureFromWindow(captureManager, refreshRate: displayMaxFPS)

            connectedPID = app.processIdentifier
            activeOutputScreen = outputScreen
            isScalingActive = true
            fullscreenStrikes = 0

            let config = OverlayWindowConfig(
                targetScreen: outputScreen,
                windowFrame: cgFrame,
                size: scaledOutputSize,
                captureCursor: settings.captureCursor,
                displayBounds: CGDisplayBounds(displayID),
                fullScreenOutput: shouldFullScreen
            )

            guard overlay.createOverlay(config: config) else {
                alertMessage = overlay.lastError ?? "Error Code: MG-OV-001 Overlay creation failed."
                showAlert = true
                await stopGooseCaptureAsync()
                return
            }

            let mtkView = MTKView(frame: CGRect(origin: .zero, size: scaledOutputSize))
            overlay.setMTKView(mtkView)
            engine.attachToView(mtkView, displayRefreshRate: displayMaxFPS)

            overlay.setTargetWindow(wid, pid: app.processIdentifier)
            overlay.updateWindowPosition()

            startStatsTimer()
            
            if settings.showMGHUD {
                hudController.show(on: outputScreen, compact: false)
                hudController.setDeviceName(engine.deviceName)
                hudController.setPID(connectedPID)

                let captureScale = outputScreen.backingScaleFactor
                let capturePixelSize = CGSize(width: sourceRes.width * captureScale,
                                               height: sourceRes.height * captureScale)
                hudController.setResolutions(capture: capturePixelSize, output: scaledOutputSize)
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
        activeOutputScreen = nil
        
        if let engine = gooseEngine {
            await engine.stopCapture()
        }

        if let captureManager = windowCaptureManager {
            await captureManager.stopCapture()
        }
        
        isScalingActive = false
        connectedPID = 0

        hudController.hide()
    }

    private func setupHotkeys() {
        GlobalHotkeyManager.shared.register(keyCode: UInt32(kVK_ANSI_T), modifiers: UInt32(cmdKey | shiftKey)) {
            Task { @MainActor in handleHotkeyToggle() }
        }
        GlobalHotkeyManager.shared.register(keyCode: UInt32(kVK_ANSI_C), modifiers: UInt32(cmdKey | shiftKey)) {
            Task { @MainActor in handleCursorHotkeyToggle() }
        }
    }

    private func handleHotkeyToggle() {
        let now = CACurrentMediaTime()
        if now - lastHotkeyTime < 0.4 { return }
        lastHotkeyTime = now
        toggleScaling()
    }

    private func handleCursorHotkeyToggle() {
        let now = CACurrentMediaTime()
        if now - lastCursorHotkeyTime < 0.3 { return }
        lastCursorHotkeyTime = now
        MouseConstraintManager.shared.toggleCursorSpriteVisible()
    }

    private func toggleScaling() {
        guard permissionsGranted else { return }
        if isTransitioning { return }
        if isScalingActive { stopGooseCapture() } else { startGooseCapture() }
    }

    private func startPermissionTimer() {
        permTimer?.invalidate()
        permTimer = Timer.scheduledTimer(withTimeInterval: 1.0, repeats: true) { _ in
            MainActor.assumeIsolated {
                axGranted = AXIsProcessTrusted()
                recGranted = CGPreflightScreenCaptureAccess()
            }
        }
    }

    private func startStatsTimer() {
        statsTimer?.invalidate()
        statsTimer = Timer.scheduledTimer(withTimeInterval: 1.0, repeats: true) { [self] _ in
            Task { @MainActor in
                if let engine = self.gooseEngine {
                    let stats = engine.stats
                    self.hudController.updateFromGooseEngine(stats: stats, settings: self.settings)
                    if let engineError = engine.consumePendingError() {
                        self.alertMessage = engineError
                        self.showAlert = true
                    }
                }
                self.overlayManager?.setCaptureCursorEnabled(self.settings.captureCursor)
                self.overlayManager?.updateWindowPosition()

                // Native (Spaces) fullscreen can't be overlaid. Detect it conservatively
                // and stop with guidance instead of leaving a silently-hidden overlay.
                if self.isScalingActive, self.overlayManager?.isTargetInUnreachableSpace() == true {
                    self.fullscreenStrikes += 1
                    if self.fullscreenStrikes >= self.fullscreenStrikeThreshold {
                        self.fullscreenStrikes = 0
                        await self.stopGooseCaptureAsync()
                        // Our window was hidden when scaling started and the target now
                        // owns a fullscreen Space, so pull MetalGoose back to the front
                        // (switching Spaces) — otherwise the alert is stuck behind the game.
                        self.bringAppToFront()
                        self.alertMessage = "Error Code: MG-CAP-005 Target entered macOS fullscreen, which cannot be captured with the overlay. Please use windowed or borderless (windowed fullscreen) mode."
                        self.showAlert = true
                    }
                } else {
                    self.fullscreenStrikes = 0
                }
            }
        }
    }

    func startCountdown() {
        isCountingDown = true
        countdown = 5
        countdownTimer?.invalidate()
        countdownTimer = Timer.scheduledTimer(withTimeInterval: 1.0, repeats: true) { _ in
            MainActor.assumeIsolated {
                if countdown > 1 { countdown -= 1 }
                else {
                    countdownTimer?.invalidate()
                    countdownTimer = nil
                    isCountingDown = false

                    if let window = NSApp.windows.first(where: { $0.isVisible && $0.title.contains("MetalGoose") }) {
                        window.orderOut(nil)
                    }

                    startGooseCapture()
                }
            }
        }
    }

    private func bringAppToFront() {
        // The main window is ordered out while scaling; order it back (which also
        // switches to its Space) and activate the app so a follow-up alert is visible.
        if let window = NSApp.windows.first(where: { $0.title.contains("MetalGoose") }) {
            window.makeKeyAndOrderFront(nil)
        }
        NSApp.activate(ignoringOtherApps: true)
    }

    func stop() {
        stopGooseCapture()
        statsTimer?.invalidate()
        statsTimer = nil
        hudController.hide()
        isScalingActive = false
        connectedPID = 0
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

struct StepperSliderRow: View {
    let label: String
    @Binding var value: Int
    let range: ClosedRange<Int>
    var body: some View {
        HStack {
            Text(label).foregroundColor(.gray)
            Spacer()
            Slider(
                value: Binding(
                    get: { Double(value) },
                    set: { value = min(range.upperBound, max(range.lowerBound, Int($0.rounded()))) }
                ),
                in: Double(range.lowerBound)...Double(range.upperBound),
                step: 1
            )
            .frame(minWidth: 120, maxWidth: 180)
            Text("\(value)")
                .font(.system(.caption, design: .monospaced))
                .frame(width: 16, alignment: .trailing)
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

// MARK: - Update Progress Sheet

struct UpdateProgressSheet: View {
    let state: UpdateState

    var body: some View {
        VStack(spacing: 20) {
            Image(systemName: iconName)
                .font(.system(size: 40))
                .foregroundColor(iconColor)

            Text(title)
                .font(.headline)

            if case .downloading(let progress) = state {
                ProgressView(value: progress)
                    .progressViewStyle(.linear)
                    .frame(width: 260)
                Text(String(format: "%.0f%%", progress * 100))
                    .font(.caption)
                    .foregroundColor(.secondary)
            } else if case .checking = state {
                ProgressView()
                    .progressViewStyle(.circular)
            } else if case .installing = state {
                ProgressView()
                    .progressViewStyle(.circular)
            } else if case .done = state {
                Text(String(localized: "Relaunching MetalGoose…", comment: "Update done label"))
                    .font(.caption)
                    .foregroundColor(.secondary)
            }
        }
        .padding(32)
        .frame(width: 320)
    }

    private var title: String {
        switch state {
        case .checking:   return String(localized: "Checking for Updates…", comment: "Update sheet title")
        case .downloading: return String(localized: "Downloading Update…", comment: "Update sheet title")
        case .installing: return String(localized: "Installing Update…", comment: "Update sheet title")
        case .done:       return String(localized: "Update Installed!", comment: "Update sheet title")
        default:          return ""
        }
    }

    private var iconName: String {
        switch state {
        case .done: return "checkmark.circle.fill"
        default:    return "arrow.down.circle"
        }
    }

    private var iconColor: Color {
        switch state {
        case .done: return .green
        default:    return .accentColor
        }
    }
}
