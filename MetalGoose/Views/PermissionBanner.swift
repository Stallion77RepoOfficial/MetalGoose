import SwiftUI

/// What is missing before scaling can start, and the way to get it.
struct PermissionBanner: View {
    @ObservedObject var permissions: PermissionManager

    var body: some View {
        VStack(alignment: .leading, spacing: 12) {
            Text("Permissions Required")
                .font(.title3).bold()
            Divider()

            StatusRow(label: "Screen Recording", granted: permissions.screenRecordingGranted,
                      grant: permissions.requestScreenRecording,
                      openSettings: permissions.openScreenRecordingSettings)
            StatusRow(label: "Accessibility", granted: permissions.accessibilityGranted,
                      grant: permissions.requestAccessibility,
                      openSettings: permissions.openAccessibilitySettings)
        }
        .padding()
        .frame(maxWidth: .infinity, alignment: .leading)
        .background(Color(NSColor.windowBackgroundColor))
        .cornerRadius(10)
    }
}

private struct StatusRow: View {
    let label: LocalizedStringKey
    let granted: Bool
    let grant: () -> Void
    let openSettings: () -> Void

    var body: some View {
        HStack(spacing: 8) {
            Text(label)
            Spacer()
            if granted {
                Text("Granted").foregroundStyle(.secondary)
            } else {
                Button("Open Settings", action: openSettings)
                    .buttonStyle(.link)
                Button("Grant", action: grant)
            }
        }
    }
}
