import Foundation

/// How far behind the newest capture the frame schedule runs, as it changes.
///
/// The delay is most of a capture interval plus the time the first image of a pair takes to arrive
/// (`FramePlanner.interpolationDelay`), and both move: one engine takes over from another that is quicker or slower,
/// a GPU that the captured app has got busy stretches the time an image takes, and a pair cut into four steps is wanted
/// sooner than one cut into two. The schedule samples the captures at `now - delay`, so a delay that drops by 10 ms puts
/// what is on the screen 10 ms ahead at once, and one that rises puts it 10 ms back: each is a step in the motion of the
/// picture, which is the one thing frame generation is there to remove.
///
/// A rise has to be taken at once, because an image that is wanted before it exists is not shown, and the pair goes by
/// without it. A fall need not be, since the images exist either way: it is given up at a small share of real time, so
/// that the picture runs a little fast while the schedule catches up, where a step would have been seen.
struct DelayGovernor {

    /// How much faster than real time the schedule may run while it catches up: at 5% a delay that fell by 10 ms
    /// is given back over a fifth of a second, and no motion on the screen is 5% off for long enough to read as anything.
    static let catchUp = 0.05

    private(set) var value: CFTimeInterval = 0
    private var updated: CFTimeInterval = 0

    /// The delay to use now, given the one that is needed.
    mutating func apply(_ needed: CFTimeInterval, now: CFTimeInterval) -> CFTimeInterval {
        defer { updated = now }
        guard value > 0, updated > 0 else {
            value = needed
            return value
        }
        value = needed >= value ? needed : max(needed, value - Self.catchUp * max(0, now - updated))
        return value
    }

    mutating func reset() {
        value = 0
        updated = 0
    }
}
