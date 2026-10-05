import AppKit
import SwiftUI

/// The statistics panel pinned to a corner of the screen the overlay is on.
@MainActor
final class HUDWindowController {
    private var window: NSWindow?
    private var hosting: NSHostingView<HUDView>?
    private let model = HUDModel()
    private let margin: CGFloat = 20

    var isShowing: Bool { window != nil }

    func show(on screen: NSScreen) {
        hide()

        // The HUD adds and removes rows at runtime, so its height is whatever SwiftUI lays the
        // content out to rather than a number kept in step by hand. `visibleFrame` already excludes
        // the menu bar and Dock, so no per-screen offset has to be guessed either.
        let hosting = NSHostingView(rootView: HUDView(model: model))
        let size = hosting.fittingSize
        let bounds = screen.visibleFrame
        let origin = CGPoint(x: bounds.minX + margin, y: bounds.maxY - size.height - margin)

        let window = NSWindow(contentRect: CGRect(origin: origin, size: size), styleMask: [.borderless],
                              backing: .buffered, defer: false)
        window.isReleasedWhenClosed = false
        window.level = NSWindow.Level(rawValue: Int(CGWindowLevelForKey(.maximumWindow)) + 1)
        window.backgroundColor = .clear
        window.isOpaque = false
        window.hasShadow = true
        window.ignoresMouseEvents = true
        window.collectionBehavior = [.canJoinAllSpaces, .stationary]
        window.contentView = hosting
        window.orderFront(nil)

        self.window = window
        self.hosting = hosting
    }

    func hide() {
        window?.orderOut(nil)
        window = nil
        hosting = nil
    }

    func update(stats: PipelineStats, info: HUDInfo) {
        model.stats = stats
        model.info = info
        fit()
    }

    /// Rows appear and disappear with the active mode, and values grow as the numbers do, so the
    /// window tracks the content instead of being sized once. Anchored at the top-left corner so
    /// growth extends downwards.
    private func fit() {
        guard let window, let hosting else { return }
        let size = hosting.fittingSize
        guard size.width > 0, size.height > 0, size != window.frame.size else { return }
        window.setFrame(CGRect(origin: CGPoint(x: window.frame.minX, y: window.frame.maxY - size.height), size: size),
                        display: true)
    }
}
