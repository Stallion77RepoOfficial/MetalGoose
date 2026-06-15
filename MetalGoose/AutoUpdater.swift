import Foundation
import AppKit

struct GitHubRelease: Codable {
    let tagName: String
    let name: String
    let assets: [GitHubAsset]

    enum CodingKeys: String, CodingKey {
        case tagName = "tag_name"
        case name
        case assets
    }
}

struct GitHubAsset: Codable {
    let name: String
    let browserDownloadURL: String

    enum CodingKeys: String, CodingKey {
        case name
        case browserDownloadURL = "browser_download_url"
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
}

@MainActor
class AutoUpdater: NSObject, ObservableObject, URLSessionDownloadDelegate {

    static let shared = AutoUpdater()

    private let repoOwner = "Stallion77RepoOfficial"
    private let repoName  = "MetalGoose"

    @Published var state: UpdateState = .idle

    private var downloadContinuation: CheckedContinuation<URL, Error>?
    private lazy var session: URLSession = {
        let config = URLSessionConfiguration.default
        return URLSession(configuration: config, delegate: self, delegateQueue: nil)
    }()

    func checkForUpdates() {
        Task { await _checkForUpdates() }
    }

    func downloadAndInstall(release: GitHubRelease) {
        Task { await _downloadAndInstall(release: release) }
    }

    private func _checkForUpdates() async {
        state = .checking
        do {
            let release = try await fetchLatestRelease()
            let latestTag = release.tagName
                .trimmingCharacters(in: .whitespacesAndNewlines)
                .lowercased()
                .replacingOccurrences(of: "v", with: "")
            let currentVersion = (Bundle.main.infoDictionary?["CFBundleShortVersionString"] as? String ?? "")
                .trimmingCharacters(in: .whitespacesAndNewlines)
                .lowercased()

            if latestTag == currentVersion {
                state = .upToDate
            } else {
                state = .available(release: release)
            }
        } catch {
            state = .failed(error.localizedDescription)
        }
    }

    private func _downloadAndInstall(release: GitHubRelease) async {
        guard let asset = release.assets.first(where: { $0.name.hasSuffix(".zip") }),
              let downloadURL = URL(string: asset.browserDownloadURL) else {
            state = .failed("No downloadable .zip asset found in release.")
            return
        }

        state = .downloading(progress: 0)

        do {
            let tempZip = try await downloadFile(from: downloadURL)
            state = .installing
            let appURL = try installFromZip(tempZip)
            removeQuarantine(at: appURL)
            state = .done
            relaunch(newAppURL: appURL)
        } catch {
            state = .failed(error.localizedDescription)
        }
    }

    private func fetchLatestRelease() async throws -> GitHubRelease {
        let url = URL(string: "https://api.github.com/repos/\(repoOwner)/\(repoName)/releases/latest")!
        var request = URLRequest(url: url)
        request.setValue("application/vnd.github+json", forHTTPHeaderField: "Accept")
        request.setValue("2022-11-28", forHTTPHeaderField: "X-GitHub-Api-Version")

        let (data, response) = try await URLSession.shared.data(for: request)
        guard let http = response as? HTTPURLResponse, http.statusCode == 200 else {
            throw URLError(.badServerResponse)
        }
        return try JSONDecoder().decode(GitHubRelease.self, from: data)
    }

    private func downloadFile(from url: URL) async throws -> URL {
        try await withCheckedThrowingContinuation { continuation in
            self.downloadContinuation = continuation
            let task = session.downloadTask(with: url)
            task.resume()
        }
    }

    nonisolated func urlSession(_ session: URLSession,
                                downloadTask: URLSessionDownloadTask,
                                didFinishDownloadingTo location: URL) {
        do {
            let dest = FileManager.default.temporaryDirectory
                .appendingPathComponent(UUID().uuidString + ".zip")
            try FileManager.default.moveItem(at: location, to: dest)
            Task { @MainActor in self.downloadContinuation?.resume(returning: dest) }
        } catch {
            Task { @MainActor in self.downloadContinuation?.resume(throwing: error) }
        }
        Task { @MainActor in self.downloadContinuation = nil }
    }

    nonisolated func urlSession(_ session: URLSession,
                                downloadTask: URLSessionDownloadTask,
                                didWriteData bytesWritten: Int64,
                                totalBytesWritten: Int64,
                                totalBytesExpectedToWrite: Int64) {
        guard totalBytesExpectedToWrite > 0 else { return }
        let progress = Double(totalBytesWritten) / Double(totalBytesExpectedToWrite)
        Task { @MainActor in
            self.state = .downloading(progress: progress)
        }
    }

    nonisolated func urlSession(_ session: URLSession,
                                task: URLSessionTask,
                                didCompleteWithError error: Error?) {
        guard let error else { return }
        Task { @MainActor in
            self.downloadContinuation?.resume(throwing: error)
            self.downloadContinuation = nil
        }
    }

    private func installFromZip(_ zipURL: URL) throws -> URL {
        let fm = FileManager.default
        let unzipDir = fm.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        try fm.createDirectory(at: unzipDir, withIntermediateDirectories: true)

        let process = Process()
        process.executableURL = URL(fileURLWithPath: "/usr/bin/unzip")
        process.arguments = ["-q", zipURL.path, "-d", unzipDir.path]
        try process.run()
        process.waitUntilExit()
        guard process.terminationStatus == 0 else {
            throw NSError(domain: "AutoUpdater", code: 1,
                          userInfo: [NSLocalizedDescriptionKey: "unzip failed with code \(process.terminationStatus)"])
        }

        let contents = try fm.contentsOfDirectory(at: unzipDir,
                                                   includingPropertiesForKeys: nil,
                                                   options: [.skipsHiddenFiles])
        if let newApp = contents.first(where: { $0.pathExtension == "app" }) {
            return try replaceCurrentApp(with: newApp)
        }

        for item in contents {
            var isDir: ObjCBool = false
            if fm.fileExists(atPath: item.path, isDirectory: &isDir), isDir.boolValue {
                let sub = try fm.contentsOfDirectory(at: item, includingPropertiesForKeys: nil, options: [.skipsHiddenFiles])
                if let found = sub.first(where: { $0.pathExtension == "app" }) {
                    return try replaceCurrentApp(with: found)
                }
            }
        }
        throw NSError(domain: "AutoUpdater", code: 2,
                      userInfo: [NSLocalizedDescriptionKey: "No .app bundle found in zip"])
    }

    private func replaceCurrentApp(with newApp: URL) throws -> URL {
        let currentAppURL = Bundle.main.bundleURL
        let fm = FileManager.default

        let backupURL = currentAppURL.deletingLastPathComponent()
            .appendingPathComponent(currentAppURL.lastPathComponent + ".bak")
        if fm.fileExists(atPath: backupURL.path) {
            try fm.removeItem(at: backupURL)
        }
        try fm.moveItem(at: currentAppURL, to: backupURL)

        do {
            try fm.copyItem(at: newApp, to: currentAppURL)
            try? fm.removeItem(at: backupURL)
        } catch {
            try? fm.moveItem(at: backupURL, to: currentAppURL)
            throw error
        }
        return currentAppURL
    }

    private func removeQuarantine(at url: URL) {
        let p = Process()
        p.executableURL = URL(fileURLWithPath: "/usr/bin/xattr")
        p.arguments = ["-dr", "com.apple.quarantine", url.path]
        try? p.run()
        p.waitUntilExit()
    }

    private func relaunch(newAppURL: URL) {
        let script = "sleep 1 && open \"\(newAppURL.path)\""
        let p = Process()
        p.executableURL = URL(fileURLWithPath: "/bin/sh")
        p.arguments = ["-c", script]
        try? p.run()
        NSApplication.shared.terminate(nil)
    }
}
