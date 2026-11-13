import AppKit
import QuartzCore

/// The overlay's content: a Metal layer the engine presents into, with the pointer drawn above it.
///
/// The pointer is a layer of its own rather than a draw call in the Metal pass, so the window
/// server composites it. Its position follows the mouse event tap directly instead of waiting for
/// the next frame to come out of the pipeline, which makes it as responsive as a system cursor
/// however many frames the pipeline is holding — and costs the GPU nothing. While images are presented at their own times
/// (VRR) the engine draws it into them instead (`drawsPointer` off): a layer moving over the picture keeps the display on its
/// fixed refresh, measured.
final class OverlayView: NSView {

    /// What the engine presents into.
    let metalLayer = CAMetalLayer()

    /// Whether the pointer is this view's layer; where not, the engine draws it.
    var drawsPointer = true {
        didSet { setPointer(pointer) }
    }

    /// How the captured app magnified its content, drawn at one pixel a point, to the window's pixels; nil where its content
    /// is the window's pixels. The drawable is then at the content's resolution, and the window server magnifies it the same
    /// way: the picture as the app's own window showed it, and measured, 2 to 10 points of the whole GPU less at 54 to 120
    /// images a second than drawing all the window's pixels, with variable refresh and pacing as they were.
    var contentMagnification: ContentResolution.Magnification? {
        didSet {
            guard contentMagnification != oldValue else { return }
            metalLayer.magnificationFilter = contentMagnification == .nearest ? .nearest : .linear
            updateDrawableSize()
        }
    }

    /// Pixels a point the engine draws at.
    var drawableScale: CGFloat { contentMagnification == nil ? window?.backingScaleFactor ?? 1 : 1 }

    private let cursorLayer = CALayer()
    private var pointer: CGPoint?

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
    /// size would be resampled by the compositor on top of the pipeline's own upscale; or its points, where the content is
    /// drawn at one pixel a point (`contentMagnification`).
    private func updateDrawableSize() {
        let scale = drawableScale
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

    /// The arrow as an image at `scale` pixels a point, and its hot spot in pixels from the image's top-left corner, for the
    /// engine to draw into the picture.
    static func pointerImage(scale: CGFloat) -> (image: CGImage, hotSpot: CGPoint)? {
        let cursor = NSCursor.arrow
        let size = cursor.image.size
        var rect = CGRect(origin: .zero, size: size)
        guard size.width > 0, size.height > 0,
              let image = cursor.image.cgImage(forProposedRect: &rect, context: nil,
                                               hints: [.ctm: AffineTransform(scale: scale)]) else { return nil }
        return (image, CGPoint(x: cursor.hotSpot.x * CGFloat(image.width) / size.width,
                               y: cursor.hotSpot.y * CGFloat(image.height) / size.height))
    }

    /// Places the pointer at a fraction of the view, measured from its top-left corner; `nil`
    /// hides it.
    func setPointer(_ fraction: CGPoint?) {
        pointer = fraction
        guard let fraction, drawsPointer else {
            cursorLayer.isHidden = true
            return
        }
        cursorLayer.position = CGPoint(x: fraction.x * bounds.width, y: (1 - fraction.y) * bounds.height)
        cursorLayer.isHidden = false
    }
}
