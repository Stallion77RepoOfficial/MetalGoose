import SwiftUI
import Combine
import MetalKit

// GHIDRA/DRAGON THEME CONSTANTS
let DRAGON_RED = Color(red: 0.8, green: 0.05, blue: 0.05)
let TERMINAL_BG = Color(red: 0.05, green: 0.05, blue: 0.05)
let TERMINAL_TEXT = Color(red: 0.9, green: 0.9, blue: 0.9)

struct ContentView: View {
    @StateObject var settings = CaptureSettings.shared
    @StateObject var engine = CaptureEngine()
    
    @State private var countdown = 5
    @State private var isCountingDown = false
    @State private var isOverlayActive = false
    
    @State private var overlayWindow: NSWindow?
    @State private var renderer: Renderer?
    
    var body: some View {
        ZStack {
            TERMINAL_BG.ignoresSafeArea()
            
            VStack(alignment: .leading, spacing: 0) {
                // Header
                HStack {
                    Text("METALGOOSE_KERNEL_V1")
                        .font(.system(.headline, design: .monospaced))
                        .foregroundColor(DRAGON_RED)
                    Spacer()
                    Text("SYSTEM_READY")
                        .font(.system(.caption, design: .monospaced))
                        .foregroundColor(.green)
                }
                .padding()
                .border(DRAGON_RED, width: 1)
                .background(Color.black)
                
                if isOverlayActive {
                    // ACTIVE STATE UI
                    VStack(spacing: 20) {
                        Spacer()
                        Text(">> OVERLAY_INJECTED <<")
                            .font(.system(.title, design: .monospaced))
                            .foregroundColor(.green)
                            .padding()
                            .border(Color.green, width: 2)
                        
                        Text("PROCESS: ACTIVE\nRENDERER: METALFX_SPATIAL\nFRAME_GEN: VISION_OPTICAL_FLOW")
                            .font(.system(.body, design: .monospaced))
                            .foregroundColor(TERMINAL_TEXT)
                            .multilineTextAlignment(.center)
                        
                        Button(action: stop) {
                            Text("[ ABORT_SEQUENCE ]")
                                .font(.system(.title2, design: .monospaced))
                                .fontWeight(.bold)
                                .foregroundColor(DRAGON_RED)
                                .padding()
                                .background(Color.black)
                                .overlay(Rectangle().stroke(DRAGON_RED, lineWidth: 2))
                        }
                        .buttonStyle(.plain)
                        Spacer()
                    }
                    .frame(maxWidth: .infinity)
                    
                } else if isCountingDown {
                    // COUNTDOWN UI
                    VStack {
                        Spacer()
                        Text("INITIALIZING_CAPTURE...")
                            .font(.system(.body, design: .monospaced))
                            .foregroundColor(DRAGON_RED)
                        
                        Text("\(countdown)")
                            .font(.system(size: 100, weight: .bold, design: .monospaced))
                            .foregroundColor(DRAGON_RED)
                            .id(countdown)
                        
                        Text("SWITCH_TO_TARGET_WINDOW_NOW")
                            .font(.system(.caption, design: .monospaced))
                            .foregroundColor(TERMINAL_TEXT)
                        Spacer()
                    }
                    .frame(maxWidth: .infinity)
                    
                } else {
                    // SETTINGS UI
                    VStack(alignment: .leading, spacing: 20) {
                        
                        // Scale Control
                        VStack(alignment: .leading) {
                            Text("> SET_SCALE_FACTOR:")
                                .font(.system(.caption, design: .monospaced))
                                .foregroundColor(DRAGON_RED)
                            
                            HStack {
                                Text("1.0x")
                                    .font(.system(.caption, design: .monospaced))
                                    .foregroundColor(.gray)
                                Slider(value: $settings.scaleFactor, in: 1.0...40.0, step: 0.1)
                                    .accentColor(DRAGON_RED)
                                Text(String(format: "%.1fx", settings.scaleFactor))
                                    .font(.system(.body, design: .monospaced))
                                    .foregroundColor(DRAGON_RED)
                                    .frame(width: 50)
                            }
                        }
                        .padding()
                        .border(Color.gray.opacity(0.3), width: 1)
                        
                        // Quality Control
                        VStack(alignment: .leading) {
                            Text("> RENDER_CONFIG:")
                                .font(.system(.caption, design: .monospaced))
                                .foregroundColor(DRAGON_RED)
                            
                            Picker("", selection: $settings.qualityMode) {
                                ForEach(CaptureSettings.QualityMode.allCases) { mode in
                                    Text(mode.rawValue).tag(mode)
                                }
                            }
                            .pickerStyle(.segmented)
                            .labelsHidden()
                        }
                        .padding()
                        .border(Color.gray.opacity(0.3), width: 1)

                        // FPS Overlay Toggle
                        VStack(alignment: .leading) {
                            Text("> SHOW_FPS_OVERLAY:")
                                .font(.system(.caption, design: .monospaced))
                                .foregroundColor(DRAGON_RED)
                            Toggle("", isOn: $settings.showFPSOverlay)
                                .toggleStyle(SwitchToggleStyle(tint: DRAGON_RED))
                                .labelsHidden()
                        }
                        .padding()
                        .border(Color.gray.opacity(0.3), width: 1)
                        
                        Spacer()
                        
                        // Start Button
                        Button(action: startCountdown) {
                            HStack {
                                Spacer()
                                Text("[ EXECUTE_INJECTION ]")
                                    .font(.system(.title3, design: .monospaced))
                                    .fontWeight(.bold)
                                    .foregroundColor(.black)
                                Spacer()
                            }
                            .padding()
                            .background(DRAGON_RED)
                        }
                        .buttonStyle(.plain)
                    }
                    .padding()
                }
            }
        }
        .frame(width: 450, height: 480)
        .onReceive(engine.$currentFrame.compactMap { $0 }) { buffer in
            guard isOverlayActive else { return }
            renderer?.enqueue(buffer: buffer)
        }
        .onChange(of: settings.showFPSOverlay) { newValue in
            renderer?.setFPSOverlay(enabled: newValue)
        }
    }
    
    func startCountdown() {
        isCountingDown = true
        countdown = 5
        
        Timer.scheduledTimer(withTimeInterval: 1.0, repeats: true) { timer in
            if countdown > 1 {
                countdown -= 1
            } else {
                timer.invalidate()
                isCountingDown = false
                activateOverlay()
            }
        }
    }
    
    func activateOverlay() {
        guard let app = NSWorkspace.shared.frontmostApplication else { return }
        
        if app.processIdentifier == NSRunningApplication.current.processIdentifier {
            return
        }
        
        let options = CGWindowListOption(arrayLiteral: .optionOnScreenOnly, .excludeDesktopElements)
        guard let list = CGWindowListCopyWindowInfo(options, kCGNullWindowID) as? [[String: Any]] else { return }
        
        guard let targetWinInfo = list.first(where: { ($0[kCGWindowOwnerPID as String] as? Int32) == app.processIdentifier }),
              let boundsDict = targetWinInfo[kCGWindowBounds as String] as? [String: CGFloat]
        else { return }
        
        // Target Window ID
        let wid = targetWinInfo[kCGWindowNumber as String] as? CGWindowID ?? 0
        if wid == 0 { return }
        
        // Calculate Frame
        let targetFrame = CGRect(x: boundsDict["X"] ?? 0, y: boundsDict["Y"] ?? 0, width: boundsDict["Width"] ?? 100, height: boundsDict["Height"] ?? 100)
        let screenH = NSScreen.main?.frame.height ?? 1080
        let nsRect = CGRect(x: targetFrame.origin.x, y: screenH - (targetFrame.origin.y + targetFrame.height), width: targetFrame.width, height: targetFrame.height)
        
        // Setup Renderer and Overlay
        let mtkView = MTKView()
        guard let renderer = Renderer(metalKitView: mtkView, settings: settings) else { return }
        self.renderer = renderer
        let overlay = renderer.createOverlayWindow(targetFrame: nsRect)
        overlay.orderFront(nil)
        self.overlayWindow = overlay
        
        // START TRACKING
        renderer.startTracking(windowID: wid, overlay: overlay)
        renderer.setFPSOverlay(enabled: settings.showFPSOverlay)
        
        Task {
            try? await engine.startCapture(targetWindowID: wid, scaleFactor: settings.scaleFactor)
            isOverlayActive = true
        }
    }
    
    func stop() {
        engine.stopCapture()
        renderer?.stopTracking() // Stop tracking timer
        overlayWindow?.orderOut(nil)
        overlayWindow = nil
        renderer = nil
        isOverlayActive = false
    }
}
