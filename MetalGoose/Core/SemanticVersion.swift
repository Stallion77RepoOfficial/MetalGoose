import Foundation

/// A release version as this project writes them: dotted numbers, optionally followed by
/// either a hyphenated pre-release identifier (`1.3.0-beta`) or a bare trailing letter that
/// marks a later revision of the same numbers (`1.2.2a`).
///
/// The two suffixes order in opposite directions around the plain version, which is why
/// they are kept apart: a pre-release is older than the release it precedes, a letter
/// revision is newer than the release it follows.
struct SemanticVersion: Comparable, Equatable, Sendable {
    let numbers: [Int]
    let prerelease: String?
    let revision: String?

    /// Accepts a git tag or a bundle version: surrounding whitespace and a leading
    /// `v` are decoration. Only the tag's `v` prefix is dropped — removing every `v` would
    /// mangle suffixes like `preview`.
    init(_ text: String) {
        var core = Substring(text.trimmingCharacters(in: .whitespacesAndNewlines))
        if core.hasPrefix("v") || core.hasPrefix("V") { core.removeFirst() }
        // Build metadata after `+` carries no precedence at all.
        if let plus = core.firstIndex(of: "+") { core = core[..<plus] }

        var prerelease: String?
        if let dash = core.firstIndex(of: "-") {
            let identifier = core[core.index(after: dash)...]
            if !identifier.isEmpty { prerelease = String(identifier) }
            core = core[..<dash]
        }

        var numbers: [Int] = []
        var revision: String?
        for component in core.split(separator: ".") {
            let digits = component.prefix { $0.isNumber }
            numbers.append(Int(digits) ?? 0)
            let trailing = component.dropFirst(digits.count)
            if revision == nil, !trailing.isEmpty { revision = trailing.lowercased() }
        }
        self.numbers = numbers
        self.prerelease = prerelease
        self.revision = revision
    }

    /// Versions are equal when neither precedes the other, so `1.2` and `1.2.0` are the same release.
    /// The synthesised comparison of the stored fields would call them different while the ordering
    /// calls them equal, and `Comparable` requires the two to agree.
    static func == (lhs: SemanticVersion, rhs: SemanticVersion) -> Bool {
        !(lhs < rhs) && !(rhs < lhs)
    }

    static func < (lhs: SemanticVersion, rhs: SemanticVersion) -> Bool {
        let count = max(lhs.numbers.count, rhs.numbers.count)
        for i in 0..<count {
            let l = i < lhs.numbers.count ? lhs.numbers[i] : 0
            let r = i < rhs.numbers.count ? rhs.numbers[i] : 0
            if l != r { return l < r }
        }
        // Same numbers: a pre-release sorts below the release, and below another pre-release
        // that compares greater.
        switch (lhs.prerelease, rhs.prerelease) {
        case (nil, nil):             break
        case (.some, nil):           return true
        case (nil, .some):           return false
        case (.some(let l), .some(let r)) where l != r:
            let left = l.split(separator: "."), right = r.split(separator: ".")
            for (a, b) in zip(left, right) where a != b {
                let numericA = a.utf8.allSatisfy { (48...57).contains($0) }
                let numericB = b.utf8.allSatisfy { (48...57).contains($0) }
                if numericA != numericB { return numericA }
                if numericA, a.count != b.count { return a.count < b.count }
                return a < b
            }
            return left.count < right.count
        default:                     break
        }
        switch (lhs.revision, rhs.revision) {
        case (nil, .some):           return true
        case (.some(let l), .some(let r)): return l < r
        default:                     return false
        }
    }
}
