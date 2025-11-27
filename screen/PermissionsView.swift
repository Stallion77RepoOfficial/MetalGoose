import SwiftUI
import AppKit
import CoreGraphics // Request için gerekli

struct PermissionsView: View {
    @State private var axGranted = AXIsProcessTrusted()
    @State private var recGranted = CGPreflightScreenCaptureAccess()
    @State private var timer: Timer?

    // Theme Constants
    let DRAGON_RED = Color(red: 0.8, green: 0.05, blue: 0.05)
    let TERMINAL_BG = Color(red: 0.05, green: 0.05, blue: 0.05)

    var body: some View {
        ZStack {
            TERMINAL_BG.ignoresSafeArea()
            
            VStack(spacing: 0) {
                // Header
                HStack {
                    Text("SYSTEM_BOOT_SEQUENCE")
                        .font(.system(.headline, design: .monospaced))
                        .foregroundColor(DRAGON_RED)
                    Spacer()
                    Text("ERR_PERM_DENIED")
                        .font(.system(.caption, design: .monospaced))
                        .foregroundColor(DRAGON_RED)
                }
                .padding()
                .border(DRAGON_RED, width: 1)
                .background(Color.black)
                
                VStack(spacing: 30) {
                    Text(">> CRITICAL_FAILURE: MISSING_PRIVILEGES <<")
                        .font(.system(.title3, design: .monospaced))
                        .fontWeight(.bold)
                        .foregroundColor(DRAGON_RED)
                        .padding(.top, 20)
                    
                    VStack(alignment: .leading, spacing: 15) {
                        // 1. Accessibility
                        PermissionRow(
                            title: "ACCESSIBILITY_DAEMON",
                            isGranted: axGranted,
                            color: DRAGON_RED
                        ) {
                            let options = [kAXTrustedCheckOptionPrompt.takeUnretainedValue() as String: true] as CFDictionary
                            AXIsProcessTrustedWithOptions(options)
                        }
                        
                        // 2. Screen Recording (FIXED LOGIC)
                        PermissionRow(
                            title: "SCREEN_RECORD_SERVICE",
                            isGranted: recGranted,
                            color: DRAGON_RED
                        ) {
                            // First, try to trigger the system prompt
                            if !CGRequestScreenCaptureAccess() {
                                // If it fails (already denied), open Settings
                                if let url = URL(string: "x-apple.systempreferences:com.apple.preference.security?Privacy_ScreenCapture") {
                                    NSWorkspace.shared.open(url)
                                }
                            }
                        }
                    }
                    .padding()
                    
                    if axGranted && recGranted {
                        Text("[ REBOOT_REQUIRED_TO_APPLY_PATCH ]")
                            .font(.system(.body, design: .monospaced))
                            .foregroundColor(.green)
                            .padding()
                            .border(Color.green, width: 1)
                    } else {
                        Text("AWAITING_USER_INPUT...")
                            .font(.system(.caption, design: .monospaced))
                            .foregroundColor(.gray)
                    }
                }
                .padding()
                
                Spacer()
            }
        }
        .frame(width: 500, height: 400)
        .onAppear { startPolling() }
        .onDisappear { timer?.invalidate() }
    }
    
    func startPolling() {
        timer = Timer.scheduledTimer(withTimeInterval: 1.0, repeats: true) { _ in
            self.axGranted = AXIsProcessTrusted()
            self.recGranted = CGPreflightScreenCaptureAccess()
        }
    }
}

struct PermissionRow: View {
    let title: String
    let isGranted: Bool
    let color: Color
    let action: () -> Void
    
    var body: some View {
        HStack {
            Text(isGranted ? "[ PASS ]" : "[ FAIL ]")
                .font(.system(.body, design: .monospaced))
                .foregroundColor(isGranted ? .green : color)
                .frame(width: 80, alignment: .leading)
            
            Text(title)
                .font(.system(.body, design: .monospaced))
                .foregroundColor(.white)
            
            Spacer()
            
            if !isGranted {
                Button(action: action) {
                    Text("< GRANT >")
                        .font(.system(.caption, design: .monospaced))
                        .fontWeight(.bold)
                        .foregroundColor(.black)
                        .padding(.vertical, 4)
                        .padding(.horizontal, 8)
                        .background(color)
                }
                .buttonStyle(.plain)
            }
        }
        .padding(12)
        .border(Color.gray.opacity(0.3), width: 1)
    }
}
