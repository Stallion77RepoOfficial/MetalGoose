import SwiftUI

struct UpdateProgressSheet: View {
    let state: UpdateState

    var body: some View {
        VStack(spacing: 20) {
            Image(systemName: state.isDone ? "checkmark.circle.fill" : "arrow.down.circle")
                .font(.system(size: 40))
                .foregroundColor(.accentColor)

            Text(title).font(.headline)

            switch state {
            case .downloading(let progress):
                ProgressView(value: progress)
                    .progressViewStyle(.linear)
                    .frame(width: 260)
                Text(verbatim: String(format: "%.0f%%", progress * 100))
                    .font(.caption)
                    .foregroundColor(.secondary)
            case .checking, .installing:
                ProgressView().progressViewStyle(.circular)
            case .done:
                Text("Relaunching MetalGoose…")
                    .font(.caption)
                    .foregroundColor(.secondary)
            default:
                EmptyView()
            }
        }
        .padding(32)
        .frame(width: 320)
    }

    private var title: LocalizedStringKey {
        switch state {
        case .checking:    return "Checking for Updates…"
        case .downloading: return "Downloading Update…"
        case .installing:  return "Installing Update…"
        case .done:        return "Update Installed"
        default:           return ""
        }
    }
}

/// The alerts and the progress sheet that go with the update flow.
struct UpdateAlerts: ViewModifier {
    @ObservedObject var updater: AutoUpdater

    /// An alert is showing while the state matches, and dismissing it resets the state.
    private func isShowing(_ matches: @escaping (UpdateState) -> Bool) -> Binding<Bool> {
        Binding(get: { matches(updater.state) }, set: { if !$0 { updater.state = .idle } })
    }

    func body(content: Content) -> some View {
        content
            .alert("Already up to date", isPresented: isShowing { if case .upToDate = $0 { true } else { false } }) {
                Button("OK", role: .cancel) {}
            } message: {
                Text("MetalGoose is already up to date.")
            }
            .alert("Update Available", isPresented: isShowing { if case .available = $0 { true } else { false } }) {
                if case .available(let release) = updater.state {
                    Button("Download & Install") { updater.downloadAndInstall(release: release) }
                }
                Button("Later", role: .cancel) { updater.state = .idle }
            } message: {
                if case .available(let release) = updater.state {
                    Text("A new version is available: \(release.tagName)\nWould you like to download and install it now?")
                }
            }
            .alert("Update Failed", isPresented: isShowing { if case .failed = $0 { true } else { false } }) {
                Button("OK", role: .cancel) { updater.state = .idle }
            } message: {
                if case .failed(let message) = updater.state { Text(verbatim: message) }
            }
            .sheet(isPresented: Binding(
                get: {
                    switch updater.state {
                    case .checking, .downloading, .installing, .done: return true
                    default: return false
                    }
                },
                set: { _ in })
            ) {
                UpdateProgressSheet(state: updater.state)
            }
    }
}
