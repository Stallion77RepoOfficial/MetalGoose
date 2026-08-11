import SwiftUI
import MetalKit
import ApplicationServices
import Carbon.HIToolbox
import ScreenCaptureKit

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
                        requestAX: { requestAccessibilityAccess() },
                        requestREC: { requestScreenRecordingAccess() }
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
            permTimer = nil
            statsTimer?.invalidate()
            statsTimer = nil
            countdownTimer?.invalidate()
            countdownTimer = nil
            isCountingDown = false
            // The hotkeys deliberately outlive the window. Closing it with Cmd+W
            // leaves the overlay and the capture running, and tearing the hotkeys
            // down here left no way to stop either: no window to click Stop in,
            // and Cmd+Shift+T dead. They cost nothing while the process lives and
            // die with it.
        }
        .onReceive(settings.objectWillChange) { _ in
            DispatchQueue.main.async {
                gooseEngine?.updateSettings(settings)

                let upscaling = settings.scalingType != .off
                overlayManager?.setOutputScale(upscaling ? CGFloat(settings.scaleFactor.floatValue) : 1.0,
                                               fillsScreen: upscaling && settings.scaleFactor.fillsScreen)

                if isScalingActive, let captureManager = windowCaptureManager {
                    let renderScale = upscaling ? settings.renderScale.multiplier : 1.0
                    Task { await captureManager.updateRenderScale(renderScale) }
                }
            }
        }
        .onChange(of: settings.showMGHUD, initial: false) { _, newValue in
            if newValue && isScalingActive {
                if let screen = activeOutputScreen {
                    hudController.show(on: screen)
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
        .alert(String(localized: "Already up to date", comment: "Update alert title"),
               isPresented: Binding(
                get: { if case .upToDate = updater.state { return true }; return false },
                set: { if !$0 { updater.state = .idle } }
               )) {
            Button("OK", role: .cancel) {}
        } message: {
            Text(String(localized: "MetalGoose is already up to date.", comment: "Update alert body"))
        }
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
                Button("Stop Scaling", role: .destructive) { stop() }
                    .buttonStyle(.bordered)
                    .controlSize(.large)

            } else if isCountingDown {
                Text("\(countdown)")
                    .font(.title2.monospacedDigit())

            } else {
                Button("Start Scaling") { startCountdown() }
                    .buttonStyle(.borderedProminent)
                    .controlSize(.large)
                    .disabled(!permissionsGranted)
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

                       // Only extrapolation has a multiplier to choose. MetalFX
                       // interpolation is 2x by construction, and a row that
                       // reports a figure nothing can change reads as a control
                       // that stopped responding.
                       if settings.frameGenMode == .extrapolation {
                           StepperSliderRow(
                               label: String(localized: "Multiplier", defaultValue: "Multiplier"),
                               value: $settings.frameGenMultiplier,
                               range: CaptureSettings.minFrameGenMultiplier
                                   ... CaptureSettings.maxFrameGenMultiplier(for: .extrapolation),
                               format: { "\($0)x" })
                       }
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

                    ToggleRow(label: String(localized: "Triple Buffering", comment: "Toggle label: pipeline buffer depth"),
                              isOn: $settings.tripleBuffering)
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

        let darwinCache = fm.temporaryDirectory
            .deletingLastPathComponent()
            .appendingPathComponent("C", isDirectory: true)
        for name in [bundleID, "com.apple.metal", "com.apple.metalfx", "com.apple.metalfe"] {
            targets.append(darwinCache.appendingPathComponent(name, isDirectory: true))
        }

        if let caches = fm.urls(for: .cachesDirectory, in: .userDomainMask).first {
            targets.append(caches.appendingPathComponent(bundleID, isDirectory: true))
        }

        var removed = 0
        for url in targets where fm.fileExists(atPath: url.path) {
            do {
                try fm.removeItem(at: url)
                removed += 1
            } catch {
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

        // Two independent readings of the same panel rather than a hardcoded
        // fallback: AppKit reports the mode's rate, CoreGraphics reports the
        // active display mode. Some virtual and captured displays leave one of
        // the two at zero, and no real display leaves both there.
        let modeFPS = CGDisplayCopyDisplayMode(displayID)?.refreshRate ?? 0
        let displayMaxFPS = max(outputScreen.maximumFramesPerSecond, Int(modeFPS.rounded()))
        guard displayMaxFPS > 0 else {
            alertMessage = "Error Code: MG-UI-006 Display refresh rate unavailable."
            showAlert = true
            return
        }

        // The panel's own floor. On a fixed-refresh display it equals the
        // ceiling, which is what tells the engine there is no variable range.
        let maxInterval = outputScreen.maximumRefreshInterval
        let displayMinFPS = maxInterval > 0 ? Int((1.0 / maxInterval).rounded()) : displayMaxFPS

        guard let captureManager = windowCaptureManager else { return }

        let upscaling = settings.scalingType != .off
        let fillsScreen = upscaling && settings.scaleFactor.fillsScreen
        let outputScale = upscaling ? CGFloat(settings.scaleFactor.floatValue) : 1.0
        let renderScale = upscaling ? settings.renderScale.multiplier : 1.0

        let success = await captureManager.startCapture(windowID: wid, maxFPS: displayMaxFPS,
                                                        showsCursor: false, renderScale: renderScale,
                                                        queueDepth: settings.bufferCount)
        if success {
            await engine.startCaptureFromWindow(captureManager)

            connectedPID = app.processIdentifier
            activeOutputScreen = outputScreen
            isScalingActive = true
            fullscreenStrikes = 0

            let config = OverlayWindowConfig(
                targetScreen: outputScreen,
                windowFrame: cgFrame,
                captureCursor: settings.captureCursor,
                outputScale: outputScale,
                fillsScreen: fillsScreen
            )

            guard overlay.createOverlay(config: config) else {
                alertMessage = overlay.lastError ?? "Error Code: MG-OV-001 Overlay creation failed."
                showAlert = true
                await stopGooseCaptureAsync()
                return
            }

            // setMTKView sizes the view and its drawable from the overlay itself.
            let mtkView = MTKView(frame: .zero)
            overlay.setMTKView(mtkView)
            engine.attachToView(mtkView, displayRefreshRate: displayMaxFPS, minRefreshRate: displayMinFPS)

            overlay.setTargetWindow(wid, pid: app.processIdentifier)
            overlay.updateWindowPosition()

            startStatsTimer()
            
            if settings.showMGHUD {
                hudController.show(on: outputScreen)
                hudController.setDeviceName(engine.deviceName)
                hudController.setPID(connectedPID)
                hudController.setCaptureResolution(captureManager.capturePixelSize)
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
        
        gooseEngine?.stopCapture()

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

    private func requestAccessibilityAccess() {
        guard !AXIsProcessTrusted() else { return }
        // Raw value of kAXTrustedCheckOptionPrompt. The imported constant is a
        // global var, which Swift 6 concurrency checking rejects.
        let options = ["AXTrustedCheckOptionPrompt": true] as CFDictionary
        _ = AXIsProcessTrustedWithOptions(options)
    }

    /// Mirrors the Accessibility path: check state, then let the framework ask.
    /// The request has to go through ScreenCaptureKit — any material use of it
    /// raises the TCC prompt and registers the app under Screen Recording, while
    /// CGRequestScreenCaptureAccess is the legacy CoreGraphics entry point and
    /// is known to stay silent on macOS 26 (FB22261705). The CoreGraphics
    /// preflight check is unaffected and still reports the correct state.
    private func requestScreenRecordingAccess() {
        guard !CGPreflightScreenCaptureAccess() else { return }
        Task { _ = try? await SCShareableContent.excludingDesktopWindows(false, onScreenWindowsOnly: true) }
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
                    // Render scale is applied at the source now, so the capture
                    // resolution changes at runtime and cannot be set once.
                    if let capture = self.windowCaptureManager?.capturePixelSize, capture != .zero {
                        self.hudController.setCaptureResolution(capture)
                    }
                    if let engineError = engine.consumePendingError() {
                        self.alertMessage = engineError
                        self.showAlert = true
                    }
                }
                self.overlayManager?.setCaptureCursorEnabled(self.settings.captureCursor)
                self.overlayManager?.updateWindowPosition()

                if self.isScalingActive, self.overlayManager?.isTargetInUnreachableSpace() == true {
                    self.fullscreenStrikes += 1
                    if self.fullscreenStrikes >= self.fullscreenStrikeThreshold {
                        self.fullscreenStrikes = 0
                        await self.stopGooseCaptureAsync()
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
        if let window = NSApp.windows.first(where: { $0.title.contains("MetalGoose") }) {
            window.makeKeyAndOrderFront(nil)
        }
        NSApp.activate(ignoringOtherApps: true)
    }

    /// Also reached from `willTerminateNotification`, where the async teardown may
    /// not get a chance to run. The mouse constraint installs an event tap and
    /// hides the system cursor, and a hidden cursor outlives the process, so that
    /// one part is undone synchronously; the rest is idempotent.
    func stop() {
        MouseConstraintManager.shared.stopConstraining()
        stopGooseCapture()
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
        .background(Color(NSColor.windowBackgroundColor))
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

struct StepperSliderRow: View {
    let label: String
    @Binding var value: Int
    let range: ClosedRange<Int>
    var format: (Int) -> String = { "\($0)" }

    /// The displayed value is the binding clamped to the range, so a stored
    /// value the current range cannot reach never shows a figure the pipeline
    /// is not running at.
    private var clamped: Int {
        min(range.upperBound, max(range.lowerBound, value))
    }

    var body: some View {
        HStack {
            Text(label).foregroundColor(.gray)
            Spacer()
            // A range with a single value would make the slider divide by its
            // own zero width, so it is reported as text instead.
            if range.lowerBound < range.upperBound {
                Slider(
                    value: Binding(
                        get: { Double(clamped) },
                        set: { value = min(range.upperBound, max(range.lowerBound, Int($0.rounded()))) }
                    ),
                    in: Double(range.lowerBound)...Double(range.upperBound),
                    step: 1
                )
                .frame(minWidth: 110, maxWidth: 160)
            }
            Text(format(clamped))
                .font(.system(.caption, design: .monospaced))
                .frame(width: 28, alignment: .trailing)
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

struct PermissionBanner: View {
    let axGranted: Bool
    let recGranted: Bool
    let requestAX: () -> Void
    let requestREC: () -> Void
    var body: some View {
        // The panel reads as a section like any other rather than a tinted
        // warning box. What is missing is already stated in words and marked by
        // the one control that does anything about it.
        VStack(alignment: .leading, spacing: 12) {
            Text("Permissions Required")
                .font(.title3).bold()
            Divider()
            StatusRow(label: "Accessibility", ok: axGranted, action: requestAX)
            StatusRow(label: "Screen Recording", ok: recGranted, action: requestREC)
        }
        .padding()
        .frame(maxWidth: .infinity, alignment: .leading)
        .background(Color(NSColor.windowBackgroundColor))
        .cornerRadius(10)
    }
}

struct StatusRow: View {
    let label: String
    let ok: Bool
    let action: () -> Void
    var body: some View {
        HStack(spacing: 8) {
            Text(label)
            Spacer()
            if ok {
                Text("Granted")
                    .foregroundStyle(.secondary)
            } else {
                Button("Grant") { action() }
            }
        }
    }
}

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
        case .done:       return String(localized: "Update Installed", comment: "Update sheet title")
        default:          return ""
        }
    }

    private var iconName: String {
        switch state {
        case .done: return "checkmark.circle.fill"
        default:    return "arrow.down.circle"
        }
    }

    private var iconColor: Color { .accentColor }
}
