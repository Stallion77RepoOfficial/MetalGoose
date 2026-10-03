// What the logic names from Apple's frameworks and nothing else provides, for building it where they are not: Linux CI
// and toolchains without the macOS SDK. On macOS this file is empty. The other sources here are links to the app's own.

#if !canImport(Darwin)
import Foundation

typealias CFTimeInterval = Double

/// Only ever written as a literal in the logic, for a title the app shows.
struct LocalizedStringResource: ExpressibleByStringLiteral, ExpressibleByStringInterpolation, Sendable, Hashable {
    let key: String
    init(stringLiteral value: String) { key = value }
}
#endif
