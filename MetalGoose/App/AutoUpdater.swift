import AppKit
import CryptoKit
import Foundation

struct GitHubRelease: Decodable, Sendable {
    let tagName: String
    let name: String
    let assets: [GitHubAsset]

    enum CodingKeys: String, CodingKey {
        case tagName = "tag_name"
        case name
        case assets
    }
}

struct GitHubAsset: Decodable, Sendable {
    let name: String
    let browserDownloadURL: String
    /// `sha256:<hex>`, computed by GitHub when the asset was uploaded. Absent on releases that predate it.
    let digest: String?

    enum CodingKeys: String, CodingKey {
        case name
        case browserDownloadURL = "browser_download_url"
        case digest
    }
}

enum UpdateState {
    case idle
    case checking
    case upToDate
    case available(release: GitHubRelease)
    case downloading(progress: Double)
    case installing
    case done
    case failed(String)

    var isDone: Bool {
        if case .done = self { return true }
        return false
    }
}

private enum UpdateError: LocalizedError {
    case http(Int)
    case noAsset
    case untrustedSource
    case checksumMismatch
    case extractionFailed(Int32)
    case noAppInArchive

    var errorDescription: String? {
        switch self {
        case .http(let code):        return "GitHub answered with status \(code)."
        case .noAsset:               return "No downloadable .zip asset found in release."
        case .untrustedSource:       return "Release asset is not served from GitHub over HTTPS; refusing to install it."
        case .checksumMismatch:      return "The downloaded update does not match the checksum GitHub published for it; refusing to install it."
        case .extractionFailed(let status): return "Extracting the update failed (status \(status))."
        case .noAppInArchive:        return "No .app bundle found in the update."
        }
    }
}

@MainActor
final class AutoUpdater: ObservableObject {

    static let shared = AutoUpdater()

    private let repository = "Stallion77RepoOfficial/MetalGoose"

    @Published var state: UpdateState = .idle

    private init() {}

    func checkForUpdates() {
        Task { await check() }
    }

    func downloadAndInstall(release: GitHubRelease) {
        Task { await install(release) }
    }

    // MARK: - Checking

    private func check() async {
        state = .checking
        do {
            let release = try await fetchLatestRelease()
            let latest = SemanticVersion(release.tagName)
            let current = SemanticVersion(Bundle.main.infoDictionary?["CFBundleShortVersionString"] as? String ?? "")
            state = latest > current ? .available(release: release) : .upToDate
        } catch {
            state = .failed(error.localizedDescription)
        }
    }

    private func fetchLatestRelease() async throws -> GitHubRelease {
        var request = URLRequest(url: URL(string: "https://api.github.com/repos/\(repository)/releases/latest")!)
        request.timeoutInterval = 15
        request.setValue("application/vnd.github+json", forHTTPHeaderField: "Accept")
        request.setValue("2022-11-28", forHTTPHeaderField: "X-GitHub-Api-Version")
        request.setValue("MetalGoose-Updater", forHTTPHeaderField: "User-Agent")

        let (data, response) = try await URLSession.shared.data(for: request)
        if let http = response as? HTTPURLResponse, http.statusCode != 200 { throw UpdateError.http(http.statusCode) }
        return try JSONDecoder().decode(GitHubRelease.self, from: data)
    }

    // MARK: - Installing

    private func install(_ release: GitHubRelease) async {
        do {
            guard let asset = release.assets.first(where: { $0.name.hasSuffix(".zip") }),
                  let url = URL(string: asset.browserDownloadURL) else { throw UpdateError.noAsset }

            // The installer replaces the running bundle and strips quarantine, so the download has to
            // come from GitHub over TLS and nowhere else. A free Apple ID Personal Team cannot issue a
            // Developer ID certificate, so this and the published checksum are the only verification
            // available.
            guard Self.isGitHubHTTPS(url) else { throw UpdateError.untrustedSource }

            state = .downloading(progress: 0)
            let archive = try await download(url)
            defer { try? FileManager.default.removeItem(at: archive) }

            if let expected = asset.digest?.lowercased().replacingOccurrences(of: "sha256:", with: ""),
               try await Self.sha256(of: archive) != expected {
                throw UpdateError.checksumMismatch
            }

            state = .installing
            let installed = try await Task.detached { try Self.extractAndReplace(archive) }.value
            Self.removeQuarantine(at: installed)
            state = .done
            Self.relaunch(installed)
        } catch {
            state = .failed(error.localizedDescription)
        }
    }

    private static func isGitHubHTTPS(_ url: URL) -> Bool {
        guard url.scheme == "https", let host = url.host()?.lowercased() else { return false }
        return host == "github.com" || host.hasSuffix(".github.com")
            || host == "githubusercontent.com" || host.hasSuffix(".githubusercontent.com")
    }

    private func download(_ url: URL) async throws -> URL {
        var request = URLRequest(url: url)
        request.timeoutInterval = 180
        request.setValue("MetalGoose-Updater", forHTTPHeaderField: "User-Agent")

        let progress = DownloadProgress { [weak self] fraction in
            Task { @MainActor in self?.state = .downloading(progress: fraction) }
        }
        let (file, response) = try await URLSession.shared.download(for: request, delegate: progress)
        if let http = response as? HTTPURLResponse, http.statusCode != 200 {
            try? FileManager.default.removeItem(at: file)
            throw UpdateError.http(http.statusCode)
        }
        return file
    }

    /// Reports the real byte count from the transfer instead of guessing it from the size of the file
    /// on disk.
    private final class DownloadProgress: NSObject, URLSessionDownloadDelegate, @unchecked Sendable {
        private let report: @Sendable (Double) -> Void
        init(report: @escaping @Sendable (Double) -> Void) { self.report = report }

        func urlSession(_ session: URLSession, downloadTask: URLSessionDownloadTask, didWriteData bytesWritten: Int64,
                        totalBytesWritten: Int64, totalBytesExpectedToWrite: Int64) {
            guard totalBytesExpectedToWrite > 0 else { return }
            report(min(0.99, Double(totalBytesWritten) / Double(totalBytesExpectedToWrite)))
        }

        func urlSession(_ session: URLSession, downloadTask: URLSessionDownloadTask, didFinishDownloadingTo location: URL) {}
    }

    private static func sha256(of file: URL) async throws -> String {
        try await Task.detached {
            let handle = try FileHandle(forReadingFrom: file)
            defer { try? handle.close() }
            var hasher = SHA256()
            while let chunk = try handle.read(upToCount: 1 << 20), !chunk.isEmpty { hasher.update(data: chunk) }
            return hasher.finalize().map { String(format: "%02x", $0) }.joined()
        }.value
    }

    /// Unpacks the archive and swaps the result in for the running bundle. The swap is one atomic
    /// replace, so a failure part-way leaves the old app where it was.
    private nonisolated static func extractAndReplace(_ archive: URL) throws -> URL {
        let fileManager = FileManager.default
        let staging = fileManager.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        try fileManager.createDirectory(at: staging, withIntermediateDirectories: true)
        defer { try? fileManager.removeItem(at: staging) }

        // ditto, not unzip: it restores the symlinks and extended attributes an app bundle depends on.
        let ditto = Process()
        ditto.executableURL = URL(fileURLWithPath: "/usr/bin/ditto")
        ditto.arguments = ["-x", "-k", archive.path, staging.path]
        try ditto.run()
        ditto.waitUntilExit()
        guard ditto.terminationStatus == 0 else { throw UpdateError.extractionFailed(ditto.terminationStatus) }

        // The bundle sits at the top of the archive, or one folder down.
        func app(in directory: URL) -> URL? {
            (try? fileManager.contentsOfDirectory(at: directory, includingPropertiesForKeys: nil,
                                                  options: [.skipsHiddenFiles]))?.first { $0.pathExtension == "app" }
        }
        let folders = (try? fileManager.contentsOfDirectory(at: staging, includingPropertiesForKeys: nil,
                                                            options: [.skipsHiddenFiles])) ?? []
        guard let newApp = app(in: staging) ?? folders.lazy.compactMap({ app(in: $0) }).first else {
            throw UpdateError.noAppInArchive
        }

        let current = Bundle.main.bundleURL
        _ = try fileManager.replaceItemAt(current, withItemAt: newApp)
        return current
    }

    private nonisolated static func removeQuarantine(at url: URL) {
        let xattr = Process()
        xattr.executableURL = URL(fileURLWithPath: "/usr/bin/xattr")
        xattr.arguments = ["-dr", "com.apple.quarantine", url.path]
        try? xattr.run()
        xattr.waitUntilExit()
    }

    /// Waits for this process to exit, then opens the new bundle.
    private static func relaunch(_ app: URL) {
        let pid = ProcessInfo.processInfo.processIdentifier
        let quotedPath = app.path.replacingOccurrences(of: "'", with: "'\\''")
        let shell = Process()
        shell.executableURL = URL(fileURLWithPath: "/bin/sh")
        shell.arguments = ["-c", "while /bin/kill -0 \(pid) 2>/dev/null; do sleep 0.1; done; open -n '\(quotedPath)'"]
        shell.standardOutput = FileHandle.nullDevice
        shell.standardError = FileHandle.nullDevice
        try? shell.run()
        NSApplication.shared.terminate(nil)
    }
}
