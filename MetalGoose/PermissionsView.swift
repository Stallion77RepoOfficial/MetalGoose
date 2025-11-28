import SwiftUI
import AppKit
import CoreGraphics

struct PermissionsView: View {
    @State private var axGranted = AXIsProcessTrusted()
    @State private var recGranted = CGPreflightScreenCaptureAccess()
    @State private var timer: Timer?
    
    let DRAGON_RED = Color(red: 0.8, green: 0.05, blue: 0.05)
    
    var body: some View {
        ZStack {
            Color.black.ignoresSafeArea()
            VStack(spacing: 30) {
                Text("SYSTEM_BOOT_SEQUENCE")
                    .font(.system(.headline, design: .monospaced))
                    .foregroundColor(DRAGON_RED)
                    .padding().border(DRAGON_RED, width: 1)
                
                VStack(alignment: .leading, spacing: 20) {
                    PermissionRow(title: "ACCESSIBILITY_DAEMON", isGranted: axGranted, color: DRAGON_RED) {
                        let options = [kAXTrustedCheckOptionPrompt.takeUnretainedValue() as String: true] as CFDictionary
                        AXIsProcessTrustedWithOptions(options)
                    }
                    PermissionRow(title: "SCREEN_RECORD_SERVICE", isGranted: recGranted, color: DRAGON_RED) {
                        if !CGRequestScreenCaptureAccess() {
                            if let url = URL(string: "x-apple.systempreferences:com.apple.preference.security?Privacy_ScreenCapture") {
                                NSWorkspace.shared.open(url)
                            }
                        }
                    }
                }
                .padding(40)
            }
        }
        .onAppear {
            timer = Timer.scheduledTimer(withTimeInterval: 1.0, repeats: true) { _ in
                self.axGranted = AXIsProcessTrusted()
                self.recGranted = CGPreflightScreenCaptureAccess()
            }
        }
        .onDisappear {
            timer?.invalidate()
            timer = nil
        }
    }
}

struct PermissionRow: View {
    let title: String; let isGranted: Bool; let color: Color; let action: () -> Void
    var body: some View {
        HStack {
            Text(isGranted ? "[ PASS ]" : "[ FAIL ]").foregroundColor(isGranted ? .green : color).font(.system(.body, design: .monospaced))
            Text(title).foregroundColor(.white).font(.system(.body, design: .monospaced))
            Spacer()
            if !isGranted { Button("GRANT") { action() }.buttonStyle(.borderedProminent).tint(color) }
        }
        .padding().border(Color.gray.opacity(0.3))
    }
}
