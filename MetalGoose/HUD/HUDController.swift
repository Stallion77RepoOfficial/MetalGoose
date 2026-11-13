import AppKit
import SwiftUI

/// The statistics panel, which the engine draws into the overlay's picture at its top-left corner.
///
/// It was a window of its own over the overlay, which the window server composites, and while it was there the display
/// stayed on its fixed refresh however the overlay presented (VRR), measured. Drawn into the picture, it reaches the screen
/// with each image, and goes with the overlay when the user is in another app.
@MainActor
final class HUDController {
    private let model = HUDModel()
    private lazy var renderer = ImageRenderer(content: HUDView(model: model))
    private var engine: GooseEngine?
    private var scale: CGFloat = 1
    private let margin: CGFloat = 20

    var isShowing: Bool { engine != nil }

    /// At `scale` pixels a point: the overlay's.
    func show(scale: CGFloat, engine: GooseEngine) {
        self.engine = engine
        self.scale = scale
        render()
    }

    /// Takes the HUD out of the picture for good, as a session ends or the setting goes off.
    func hide() {
        engine?.setHUD(nil, origin: .zero)
        engine = nil
    }

    func update(stats: PipelineStats, info: HUDInfo) {
        model.stats = stats
        model.info = info
        render()
    }

    /// Lays the panel out at its own size — rows appear and disappear with the mode, and values grow as the numbers do — and
    /// hands the engine the image, at the screen's pixels a point. One renderer, given the view again each time, draws the
    /// new values at once in 0.4 to 0.6 ms on the main thread, where a new renderer took 5 to 15, measured.
    private func render() {
        guard let engine else { return }
        renderer.content = HUDView(model: model)
        renderer.scale = scale
        engine.setHUD(renderer.cgImage, origin: CGPoint(x: margin * scale, y: margin * scale))
    }
}
