import Foundation

/// The panel's refresh capabilities.
struct DisplayRate: Equatable, Sendable {
    /// The panel's refresh rate, or the top of its range.
    var maximum: Int
    /// The panel's own floor. A panel whose floor equals its ceiling is fixed-refresh; anything else is
    /// variable, so ProMotion needs no separate flag and cannot disagree with the two rates it is
    /// derived from.
    var minimum: Int

    init(maximum: Int, minimum: Int) {
        self.maximum = max(0, maximum)
        self.minimum = minimum > 0 ? min(minimum, self.maximum) : self.maximum
    }

    var isVariable: Bool { minimum < maximum }
}
