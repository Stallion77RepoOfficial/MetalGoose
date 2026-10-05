import SwiftUI

struct ContentView: View {
    @ObservedObject var settings: CaptureSettings
    @ObservedObject var permissions: PermissionManager
    @ObservedObject var controller: ScalingController
    @StateObject private var updater = AutoUpdater.shared

    private var versionString: String {
        let info = Bundle.main.infoDictionary
        return "v\(info?["CFBundleShortVersionString"] as? String ?? "") (\(info?["CFBundleVersion"] as? String ?? ""))"
    }

    private var macOSVersionString: String {
        let v = ProcessInfo.processInfo.operatingSystemVersion
        return "macOS \(v.majorVersion).\(v.minorVersion).\(v.patchVersion)"
    }

    private var permissionsGranted: Bool { controller.permissionsAllowScaling }

    var body: some View {
        NavigationSplitView {
            sidebar
        } detail: {
            detail
        }
        .overlay(alignment: .bottomLeading) {
            Text(verbatim: macOSVersionString)
                .font(.caption2)
                .foregroundColor(.gray.opacity(0.5))
                .padding(.leading, 16)
                .padding(.bottom, 12)
        }
        .background(WindowReader { controller.mainWindow = $0 })
        .onAppear { permissions.startMonitoring() }
        .onDisappear { permissions.stopMonitoring() }
        .alert("MetalGoose", isPresented: Binding(
            get: { controller.alertMessage != nil },
            set: { if !$0 { controller.alertMessage = nil } }),
               presenting: controller.alertMessage
        ) { _ in
            Button("OK", role: .cancel) { controller.alertMessage = nil }
        } message: { message in
            Text(verbatim: message)
        }
        .modifier(UpdateAlerts(updater: updater))
    }

    // MARK: - Sidebar

    private var sidebar: some View {
        VStack(alignment: .leading) {
            HStack {
                Image("GooseLogo")
                    .resizable()
                    .aspectRatio(contentMode: .fit)
                    .frame(width: 40, height: 40)
                    .cornerRadius(8)
                Text("MetalGoose").font(.headline)
            }
            .padding([.top, .horizontal])

            Spacer()

            HStack {
                Spacer()
                Menu {
                    Button("About") { controller.alertMessage = "MetalGoose \(versionString)" }
                    Button("Check for Updates") { updater.checkForUpdates() }
                } label: {
                    Image(systemName: "gearshape")
                }
            }
            .padding()
        }
        .frame(minWidth: 200)
        .disabled(!permissionsGranted || controller.isTransitioning)
        .navigationTitle("MetalGoose")
    }

    // MARK: - Detail

    private var detail: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 20) {
                if !permissionsGranted {
                    PermissionBanner(permissions: permissions)
                        .padding(.bottom, 8)
                }

                header

                HStack(alignment: .top, spacing: 20) {
                    leftColumn
                    rightColumn
                }
                .disabled(!permissionsGranted || controller.isTransitioning)
                .opacity(permissionsGranted ? 1.0 : 0.5)
            }
            .padding(24)
        }
    }

    private var header: some View {
        HStack {
            Text("MetalGoose").font(.largeTitle).bold()
            Spacer()

            switch controller.phase {
            case .active:
                Button("Stop Scaling", role: .destructive) { controller.stop() }
                    .buttonStyle(.bordered)
                    .controlSize(.large)
            case .countingDown(let remaining):
                Text(verbatim: "\(remaining)").font(.title2.monospacedDigit())
            case .idle:
                Button("Start Scaling") { controller.startCountdown() }
                    .buttonStyle(.borderedProminent)
                    .controlSize(.large)
                    .disabled(!permissionsGranted || controller.isTransitioning)
            }
        }
        .padding(.bottom, 10)
    }

    private var leftColumn: some View {
        VStack(spacing: 16) {
            ConfigPanel(title: "Upscaling") {
                PickerRow(label: "Method", selection: $settings.scalingMethod)
                if settings.isUpscaling {
                    PickerRow(label: "Scale Factor", selection: $settings.scaleFactor)
                    PickerRow(label: "Render Scale", selection: $settings.renderScale)
                }
            }

            ConfigPanel(title: "Frame Generation") {
                PickerRow(label: "Mode", selection: $settings.frameGenMode)

                // A mode with one multiplier has nothing to choose, and a row that reports a figure
                // nothing can change reads as a control that stopped responding.
                if settings.frameGenMode != .off {
                    SliderRow(label: "Multiplier", value: $settings.frameGenMultiplier,
                              values: settings.frameGenMode.multipliers)
                }
            }

            ConfigPanel(title: "Anti-Aliasing") {
                PickerRow(label: "Mode", selection: $settings.aaMode)
            }
        }
    }

    private var rightColumn: some View {
        VStack(spacing: 16) {
            if settings.isUpscaling {
                ConfigPanel(title: "MGUP-1 Settings") {
                    PickerRow(label: "Sharpening", selection: $settings.sharpening)
                }
            }

            ConfigPanel(title: "Display Settings") {
                ToggleRow(label: "Show MG HUD", isOn: $settings.showMGHUD)
                ToggleRow(label: "Align Pointer", isOn: $settings.alignPointer)
                ToggleRow(label: "VSync", isOn: $settings.vsync)
                ToggleRow(label: "Triple Buffering", isOn: $settings.tripleBuffering)
                    .disabled(controller.isActive)
            }

            ConfigPanel(title: "Maintenance") {
                Button {
                    controller.alertMessage = MetalCache.clear()
                } label: {
                    Text("Clear Metal Cache").frame(maxWidth: .infinity)
                }
                .disabled(controller.isActive)
            }
        }
    }
}

/// The shader caches MetalGoose has built, which can go stale after a toolchain change and make a
/// correct shader misbehave.
enum MetalCache {
    /// Removes MetalGoose's own entries from the system's shader caches. Other apps keep theirs: the
    /// parent directories are shared, so only the subfolder named for this bundle is removed.
    static func clear() -> String {
        let fileManager = FileManager.default
        let bundleID = Bundle.main.bundleIdentifier ?? "com.MetalGoose"

        // The per-user cache directory sits next to the temporary one.
        let userCache = fileManager.temporaryDirectory.deletingLastPathComponent()
            .appendingPathComponent("C", isDirectory: true)
        var targets = ["com.apple.metal", "com.apple.metalfx", "com.apple.metalfe"].map {
            userCache.appendingPathComponent($0, isDirectory: true).appendingPathComponent(bundleID, isDirectory: true)
        }
        targets.append(userCache.appendingPathComponent(bundleID, isDirectory: true))
        if let caches = fileManager.urls(for: .cachesDirectory, in: .userDomainMask).first {
            targets.append(caches.appendingPathComponent(bundleID, isDirectory: true))
        }

        let removed = targets.filter { url in
            fileManager.fileExists(atPath: url.path) && (try? fileManager.removeItem(at: url)) != nil
        }.count

        return removed > 0
            ? String(localized: "Metal cache cleared (\(removed) location(s)). Restart MetalGoose so shaders rebuild cleanly.")
            : String(localized: "No Metal cache found to clear.")
    }
}
