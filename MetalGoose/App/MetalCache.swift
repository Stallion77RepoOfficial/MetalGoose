import Foundation

/// The shader caches MetalGoose has built, which can go stale after a toolchain change and make a
/// correct shader misbehave. MetalGoose's Clear Metal Cache action, taken from beside its `ContentView`.
enum MetalCache {
    /// Removes MetalGoose's own entries from the system's shader caches. Other apps keep theirs: the
    /// parent directories are shared, so only the subfolder named for this bundle is removed.
    static func clear() -> String {
        let fileManager = FileManager.default
        let bundleID = bundleIdentifier

        // The per-user cache directory sits next to the temporary one.
        let userCache = fileManager.temporaryDirectory.deletingLastPathComponent()
            .appendingPathComponent("C", isDirectory: true)
        var targets = ["com.apple.metal", "com.apple.metalfx", "com.apple.metalfe"].map {
            userCache.appendingPathComponent($0, isDirectory: true).appendingPathComponent(bundleID, isDirectory: true)
        }
        targets.append(userCache.appendingPathComponent(bundleID, isDirectory: true))
        if let caches = fileManager.urls(for: .cachesDirectory, in: .userDomainMask).first {
            targets.append(caches.appendingPathComponent(bundleID, isDirectory: true))
        }

        let removed = targets.filter { url in
            fileManager.fileExists(atPath: url.path) && (try? fileManager.removeItem(at: url)) != nil
        }.count

        return removed > 0
            ? "Metal cache cleared (\(removed) location(s)). MetalGoose rebuilds its shaders for the next game."
            : "No Metal cache found to clear."
    }

    /// The backend has no bundle of its own; the system files its caches under the app it sits in,
    /// …/Goose HUB.app/Contents/Resources/Engines/MetalGoose/bin.
    private static var bundleIdentifier: String {
        if let identifier = Bundle.main.bundleIdentifier { return identifier }
        var url = URL(fileURLWithPath: CommandLine.arguments[0]).resolvingSymlinksInPath()
        while url.pathComponents.count > 1 {
            url.deleteLastPathComponent()
            if url.pathExtension == "app", let identifier = Bundle(url: url)?.bundleIdentifier { return identifier }
        }
        return "com.MetalGoose"
    }
}
