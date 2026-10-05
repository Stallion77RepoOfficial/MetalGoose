import AppKit
import QuartzCore

/// The overlay's content: a Metal layer the engine presents into, with the pointer drawn above it.
///
/// The pointer is a layer of its own rather than a draw call in the Metal pass, so the window
/// server composites it. Its position follows the mouse event tap directly instead of waiting for
/// the next frame to come out of the pipeline, which makes it as responsive as a system cursor
/// however many frames the pipeline is holding — and costs the GPU nothing.
final class OverlayView: NSView {

    /// What the engine presents into.
    let metalLayer = CAMetalLayer()

    private let cursorLayer = CALayer()

    override init(frame frameRect: NSRect) {
        super.init(frame: frameRect)
        wantsLayer = true
        metalLayer.addSublayer(cursorLayer)
        cursorLayer.isHidden = true
        cursorLayer.zPosition = 1
        cursorLayer.actions = ["position": NSNull(), "hidden": NSNull(), "contents": NSNull(), "bounds": NSNull()]
        updateCursorImage()
    }

    required init?(coder: NSCoder) { fatalError("OverlayView is created in code") }

    override func makeBackingLayer() -> CALayer { metalLayer }

    // MARK: - Drawable size

    /// The drawable is exactly the view in pixels, so presentation is 1:1 — a drawable at any other
    /// size would be resampled by the compositor on top of the pipeline's own upscale.
    private func updateDrawableSize() {
        let scale = window?.backingScaleFactor ?? 1
        metalLayer.contentsScale = scale
        let size = CGSize(width: bounds.width * scale, height: bounds.height * scale)
        guard size.width > 0, size.height > 0, metalLayer.drawableSize != size else { return }
        metalLayer.drawableSize = size
    }

    override func setFrameSize(_ newSize: NSSize) {
        super.setFrameSize(newSize)
        updateDrawableSize()
    }

    override func viewDidChangeBackingProperties() {
        super.viewDidChangeBackingProperties()
        updateDrawableSize()
        updateCursorImage()
    }

    override func viewDidMoveToWindow() {
        super.viewDidMoveToWindow()
        updateDrawableSize()
        updateCursorImage()
    }

    // MARK: - Pointer

    private func updateCursorImage() {
        let scale = window?.backingScaleFactor ?? 1
        let cursor = NSCursor.arrow
        let size = cursor.image.size
        cursorLayer.contentsScale = scale
        cursorLayer.contents = cursor.image.layerContents(forContentsScale: scale)
        cursorLayer.bounds = CGRect(origin: .zero, size: size)
        // The hotspot is measured from the image's top-left; the anchor point from its bottom-left.
        let hotSpot = cursor.hotSpot
        cursorLayer.anchorPoint = CGPoint(x: size.width > 0 ? hotSpot.x / size.width : 0,
                                          y: size.height > 0 ? 1 - hotSpot.y / size.height : 1)
    }

    /// Places the pointer at a fraction of the view, measured from its top-left corner; `nil`
    /// hides it.
    func setPointer(_ fraction: CGPoint?) {
        guard let fraction else {
            cursorLayer.isHidden = true
            return
        }
        cursorLayer.position = CGPoint(x: fraction.x * bounds.width, y: (1 - fraction.y) * bounds.height)
        cursorLayer.isHidden = false
    }
}
