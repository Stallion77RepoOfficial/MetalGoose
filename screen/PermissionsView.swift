// PermissionsView.swift
// A simple macOS SwiftUI view to request Accessibility and Screen Recording permissions

import SwiftUI
import AppKit
import ScreenCaptureKit

#if os(macOS)
struct PermissionsView: View {
    @State private var accessibilityGranted: Bool = PermissionsChecker.isAccessibilityGranted
    @State private var screenRecordingGranted: Bool = PermissionsChecker.isScreenRecordingGranted

    var body: some View {
        VStack(alignment: .leading, spacing: 16) {
            Text("Gerekli İzinler")
                .font(.title2)
                .bold()

            PermissionRow(title: "Erişilebilirlik", subtitle: "Klavye ve fare ile hedef pencereye tıklamak için gereklidir.", granted: accessibilityGranted) {
                PermissionsChecker.requestAccessibility()
                DispatchQueue.main.asyncAfter(deadline: .now() + 1.0) {
                    accessibilityGranted = PermissionsChecker.isAccessibilityGranted
                    NotificationCenter.default.post(name: .permissionsShouldRefresh, object: nil)
                }
            }

            PermissionRow(title: "Ekran Kaydı", subtitle: "Yakalama için zorunludur. macOS yeniden başlatma isteyebilir.", granted: screenRecordingGranted) {
                PermissionsChecker.requestScreenRecording()
                DispatchQueue.main.asyncAfter(deadline: .now() + 1.0) {
                    screenRecordingGranted = PermissionsChecker.isScreenRecordingGranted
                    NotificationCenter.default.post(name: .permissionsShouldRefresh, object: nil)
                }
            }

            if accessibilityGranted && screenRecordingGranted {
                HStack {
                    Spacer()
                    Text("Press Run to Start")
                        .font(.headline)
                        .foregroundStyle(.secondary)
                    Spacer()
                }
                .transition(.opacity)
            }
        }
        .padding(20)
        .frame(minWidth: 360)
        .onAppear {
            refreshStates()
        }
    }

    private func refreshStates() {
        accessibilityGranted = PermissionsChecker.isAccessibilityGranted
        screenRecordingGranted = PermissionsChecker.isScreenRecordingGranted
    }
}

private struct PermissionRow: View {
    var title: String
    var subtitle: String? = nil
    var granted: Bool
    var action: () -> Void

    var body: some View {
        HStack(spacing: 12) {
            StatusIcon(granted: granted)
            VStack(alignment: .leading, spacing: 2) {
                Text(title)
                    .font(.headline)
                if let subtitle, !subtitle.isEmpty {
                    Text(subtitle)
                        .font(.footnote)
                        .foregroundStyle(.secondary)
                }
            }
            Spacer()
            Button(granted ? "Verildi" : "Ayarları Aç") {
                action()
            }
            .buttonStyle(.borderedProminent)
            .tint(granted ? .green : .red)
            .disabled(granted)
        }
        .padding(10)
        .background(Color(nsColor: .windowBackgroundColor).opacity(0.6))
        .clipShape(RoundedRectangle(cornerRadius: 10, style: .continuous))
    }
}

private struct StatusIcon: View {
    var granted: Bool
    var body: some View {
        ZStack {
            Circle()
                .fill(granted ? Color.green : Color.red)
                .frame(width: 20, height: 20)
            Image(systemName: granted ? "checkmark" : "xmark")
                .font(.system(size: 10, weight: .bold))
                .foregroundStyle(.white)
        }
        .accessibilityHidden(true)
    }
}

enum PermissionsChecker {
    static var isAccessibilityGranted: Bool {
        // AXIsProcessTrustedWithOptions does not provide status without prompting.
        // For status-only, use AXIsProcessTrusted; returns true if already granted.
        AXIsProcessTrusted()
    }

    static var isScreenRecordingGranted: Bool {
        // ScreenCaptureKit doesn't expose a direct status API. The canonical way
        // is to check if the app is listed as allowed via CGPreflightScreenCaptureAccess
        // (10.15+) or try to create a stream and handle failure. Use CG API here.
        if #available(macOS 10.15, *) {
            return CGPreflightScreenCaptureAccess()
        } else {
            return false
        }
    }

    static func requestAccessibility() {
        // Open System Settings -> Privacy & Security -> Accessibility
        let url = URL(string: "x-apple.systempreferences:com.apple.preference.security?Privacy_Accessibility")!
        NSWorkspace.shared.open(url)
        // Optionally request with options (shows prompt if possible)
        let options = [kAXTrustedCheckOptionPrompt.takeUnretainedValue() as String: true] as CFDictionary
        AXIsProcessTrustedWithOptions(options)
    }

    static func requestScreenRecording() {
        // Open System Settings -> Privacy & Security -> Screen Recording
        if let url = URL(string: "x-apple.systempreferences:com.apple.preference.security?Privacy_ScreenRecording") {
            NSWorkspace.shared.open(url)
        }
        if #available(macOS 10.15, *) {
            CGRequestScreenCaptureAccess()
        }
    }
}
#endif
