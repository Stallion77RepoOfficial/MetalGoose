import SwiftUI
import AppKit

struct PermissionsView: View {
    @State private var axGranted = AXIsProcessTrusted()
    @State private var recGranted = false // SCKit needs explicit check flow

    var body: some View {
        VStack(spacing: 30) {
            Text("Required Permissions")
                .font(.system(size: 24, weight: .bold, design: .rounded))
            
            VStack(spacing: 15) {
                PermissionRow(
                    title: "Accessibility",
                    description: "Required to forward mouse and keyboard inputs to the game.",
                    isGranted: axGranted
                ) {
                    let options = [kAXTrustedCheckOptionPrompt.takeUnretainedValue() as String: true] as CFDictionary
                    AXIsProcessTrustedWithOptions(options)
                }
                
                PermissionRow(
                    title: "Screen Recording",
                    description: "Required to capture the game window.",
                    isGranted: recGranted
                ) {
                    // KRİTİK DÜZELTME: Yerel izin penceresini tetikler.
                    if #available(macOS 10.15, *) {
                        CGRequestScreenCaptureAccess()
                    }
                    
                    // Kullanıcıya rehberlik etmesi için Ayarlar penceresini aç
                    if let url = URL(string: "x-apple.systempreferences:com.apple.preference.security?Privacy_ScreenRecording") {
                        NSWorkspace.shared.open(url)
                    }
                }
            }
            .padding()
            .background(Color.black.opacity(0.2))
            .cornerRadius(12)
            
            if axGranted && recGranted {
                Text("Press Run to Start")
                    .font(.title2)
                    .fontWeight(.heavy)
                    .foregroundColor(.green)
                    .padding()
                    .transition(.scale)
            } else {
                Text("Please grant permissions to continue")
                    .font(.caption)
                    .foregroundColor(.secondary)
            }
        }
        .padding(40)
        .frame(width: 500, height: 400)
        .onAppear { checkPermissions() }
        .onReceive(NotificationCenter.default.publisher(for: NSApplication.didBecomeActiveNotification)) { _ in
            checkPermissions()
        }
    }
    
    func checkPermissions() {
        axGranted = AXIsProcessTrusted()
        // Ekran Kaydı iznini kontrol et
        recGranted = CGPreflightScreenCaptureAccess()
    }
}

struct PermissionRow: View {
    let title: String
    let description: String
    let isGranted: Bool
    let action: () -> Void
    
    var body: some View {
        HStack {
            Image(systemName: isGranted ? "checkmark.circle.fill" : "xmark.circle.fill")
                .foregroundColor(isGranted ? .green : .red)
                .font(.title2)
            
            VStack(alignment: .leading) {
                Text(title).font(.headline)
                Text(description).font(.caption).foregroundColor(.gray)
            }
            Spacer()
            
            if !isGranted {
                Button("Grant") { action() }
                    .buttonStyle(.borderedProminent)
            }
        }
        .padding(10)
        .background(Color.white.opacity(0.05))
        .cornerRadius(8)
    }
}
