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
class AutoUpdater: ObservableObject {

    static let shared = AutoUpdater()

    private let repoOwner = "Stallion77RepoOfficial"
    private let repoName  = "MetalGoose"

    @Published var state: UpdateState = .idle

    private init() {}

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

            if Self.isVersion(latestTag, newerThan: currentVersion) {
                state = .available(release: release)
            } else {
                state = .upToDate
            }
        } catch {
            state = .failed(error.localizedDescription)
        }
    }

    // Version ordering: numeric components compare first (1.2 < 1.2.1). For equal
    // numeric components, a trailing letter marks a pre-release of that version
    // (1.2a < 1.2b < 1.2c < 1.2, and 1.2.1b < 1.2.1).
    private static func isVersion(_ lhs: String, newerThan rhs: String) -> Bool {
        let (lhsNumbers, lhsSuffix) = splitVersion(lhs)
        let (rhsNumbers, rhsSuffix) = splitVersion(rhs)

        let count = max(lhsNumbers.count, rhsNumbers.count)
        for i in 0..<count {
            let l = i < lhsNumbers.count ? lhsNumbers[i] : 0
            let r = i < rhsNumbers.count ? rhsNumbers[i] : 0
            if l != r { return l > r }
        }

        switch (lhsSuffix, rhsSuffix) {
        case (nil, nil): return false
        case (nil, .some): return true
        case (.some, nil): return false
        case (.some(let l), .some(let r)): return l > r
        }
    }

    private static func splitVersion(_ version: String) -> ([Int], String?) {
        var version = version
        var suffix: String? = nil
        if let last = version.last, last.isLetter {
            suffix = String(last)
            version.removeLast()
        }
        let numbers = version.split(separator: ".").map { Int($0) ?? 0 }
        return (numbers, suffix)
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

    // On some networks CFNetwork/URLSession's TCP connection to api.github.com hangs
    // until timeout (ECH/HTTP3 path probing), even though curl reaches the same host
    // instantly. Shelling out to curl avoids that path entirely and is reliable here.
    private func fetchLatestRelease() async throws -> GitHubRelease {
        let url = "https://api.github.com/repos/\(repoOwner)/\(repoName)/releases/latest"

        let task = Process()
        task.executableURL = URL(fileURLWithPath: "/usr/bin/curl")
        task.arguments = [
            "-sS", "-m", "15",
            "-H", "Accept: application/vnd.github+json",
            "-H", "X-GitHub-Api-Version: 2022-11-28",
            "-H", "User-Agent: MetalGoose-Updater",
            url
        ]
        let outPipe = Pipe()
        let errPipe = Pipe()
        task.standardOutput = outPipe
        task.standardError = errPipe

        try task.run()
        while task.isRunning {
            try await Task.sleep(nanoseconds: 50_000_000)
        }

        let data = outPipe.fileHandleForReading.readDataToEndOfFile()
        guard task.terminationStatus == 0 else {
            let errText = String(data: errPipe.fileHandleForReading.readDataToEndOfFile(), encoding: .utf8)
                .map { $0.trimmingCharacters(in: .whitespacesAndNewlines) } ?? ""
            throw NSError(domain: "AutoUpdater", code: Int(task.terminationStatus),
                          userInfo: [NSLocalizedDescriptionKey: errText.isEmpty
                              ? "Update check failed (curl exit \(task.terminationStatus))."
                              : "Update check failed: \(errText)"])
        }
        return try JSONDecoder().decode(GitHubRelease.self, from: data)
    }

    private func downloadFile(from url: URL) async throws -> URL {
        let fm = FileManager.default
        let dest = fm.temporaryDirectory.appendingPathComponent(UUID().uuidString + ".zip")
        let expectedSize = headContentLength(url)

        let task = Process()
        task.executableURL = URL(fileURLWithPath: "/usr/bin/curl")
        task.arguments = ["-sS", "-L", "-m", "180", "-H", "User-Agent: MetalGoose-Updater", "-o", dest.path, url.absoluteString]
        try task.run()

        while task.isRunning {
            if expectedSize > 0,
               let attrs = try? fm.attributesOfItem(atPath: dest.path),
               let size = attrs[.size] as? Int64 {
                state = .downloading(progress: min(0.99, Double(size) / Double(expectedSize)))
            }
            try await Task.sleep(nanoseconds: 150_000_000)
        }

        guard task.terminationStatus == 0 else {
            try? fm.removeItem(at: dest)
            throw NSError(domain: "AutoUpdater", code: Int(task.terminationStatus),
                          userInfo: [NSLocalizedDescriptionKey: "Download failed (curl exit \(task.terminationStatus))."])
        }
        state = .downloading(progress: 1.0)
        return dest
    }

    private func headContentLength(_ url: URL) -> Int64 {
        let task = Process()
        task.executableURL = URL(fileURLWithPath: "/usr/bin/curl")
        task.arguments = ["-sS", "-I", "-L", "-m", "10", "-H", "User-Agent: MetalGoose-Updater", url.absoluteString]
        let pipe = Pipe()
        task.standardOutput = pipe
        do {
            try task.run()
        } catch {
            return 0
        }
        let data = pipe.fileHandleForReading.readDataToEndOfFile()
        task.waitUntilExit()
        guard let text = String(data: data, encoding: .utf8) else { return 0 }

        var length: Int64 = 0
        for line in text.components(separatedBy: "\n") {
            let lower = line.lowercased()
            if lower.hasPrefix("content-length:") {
                let parts = line.split(separator: ":", maxSplits: 1)
                if parts.count == 2, let value = Int64(parts[1].trimmingCharacters(in: .whitespacesAndNewlines)) {
                    length = value
                }
            }
        }
        return length
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
