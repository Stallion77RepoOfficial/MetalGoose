import CoreGraphics
import Foundation
@preconcurrency import Metal
import os

/// What is drawn over the picture inside the drawable: the HUD, and the pointer while images are presented at their own
/// times (VRR).
///
/// Both were composited by the window server, the HUD as a window of its own and the pointer as a layer above the picture,
/// and anything it composites over the overlay keeps a panel that can vary its refresh on its fixed one, measured. Drawn into
/// the image, they reach the screen with it. Set from the main thread; read by the render thread for every image.
final class OverlayGraphics: @unchecked Sendable {

    /// A premultiplied image, drawn at its own size in the drawable's pixels.
    struct Sprite {
        let texture: any MTLTexture
        /// The point of the image that is placed, in pixels from its top-left corner: that corner for the HUD, the hot spot
        /// for the pointer.
        let anchor: CGPoint
    }

    /// What the render thread draws over one image.
    struct Snapshot {
        /// The HUD, and where its top-left corner goes in pixels from the drawable's.
        var hud: (sprite: Sprite, origin: CGPoint)?
        /// Moves on with every HUD image, so that a picture that is not changing is drawn again with the new one.
        var hudGeneration = 0
        /// The pointer, and where it is as a fraction of the overlay from its top-left corner.
        var pointer: (sprite: Sprite, fraction: CGPoint)?
    }

    private struct State {
        var hud: (sprite: Sprite, origin: CGPoint)?
        var hudGeneration = 0
        var pointer: Sprite?
        var position: (@Sendable () -> CGPoint?)?
    }

    private let device: any MTLDevice
    private let state = OSAllocatedUnfairLock<State>(uncheckedState: State())
    /// Where the HUD's images are turned into textures, in the order they were given, off the main thread.
    private let uploads = DispatchQueue(label: "com.metalgoose.graphics", qos: .userInitiated)

    init(device: any MTLDevice) {
        self.device = device
    }

    /// The HUD's image, with its top-left corner `origin` pixels from the drawable's; `nil` for none. It is drawn from the
    /// next image on whose texture is ready.
    func setHUD(_ image: CGImage?, origin: CGPoint) {
        uploads.async { [self] in
            let sprite = image.flatMap { Self.texture(from: $0, device: device) }.map { Sprite(texture: $0, anchor: .zero) }
            state.withLockUnchecked {
                $0.hud = sprite.map { ($0, origin) }
                $0.hudGeneration &+= 1
            }
        }
    }

    /// The pointer's image and hot spot, in pixels; and `position`, which says where it is as a fraction of the overlay from
    /// its top-left corner, or `nil` while none is drawn.
    func setPointer(_ image: CGImage?, hotSpot: CGPoint, position: (@Sendable () -> CGPoint?)?) {
        let sprite = image.flatMap { Self.texture(from: $0, device: device) }.map { Sprite(texture: $0, anchor: hotSpot) }
        state.withLockUnchecked {
            $0.pointer = sprite
            $0.position = position
        }
    }

    /// What to draw over the next image: the pointer too where `includesPointer`, where it is now.
    func snapshot(includesPointer: Bool) -> Snapshot {
        let (hud, generation, pointer, position) = state.withLockUnchecked { ($0.hud, $0.hudGeneration, $0.pointer, $0.position) }
        var snapshot = Snapshot(hud: hud, hudGeneration: generation)
        // Asked outside the lock: the answer takes a lock of its own.
        if includesPointer, let pointer, let fraction = position?() {
            snapshot.pointer = (pointer, fraction)
        }
        return snapshot
    }

    /// `image` in a texture of its own, premultiplied BGRA as the drawable is. Nothing writes it again: a new image is a new
    /// texture, so one that a command buffer is still reading is never changed under it.
    private static func texture(from image: CGImage, device: any MTLDevice) -> (any MTLTexture)? {
        let width = image.width, height = image.height
        let bytesPerRow = width * 4
        guard width > 0, height > 0, let space = CGColorSpace(name: CGColorSpace.sRGB) else { return nil }
        var pixels = [UInt8](repeating: 0, count: bytesPerRow * height)
        let drawn = pixels.withUnsafeMutableBytes { buffer -> Bool in
            guard let context = CGContext(data: buffer.baseAddress, width: width, height: height, bitsPerComponent: 8,
                                          bytesPerRow: bytesPerRow, space: space,
                                          bitmapInfo: CGImageAlphaInfo.premultipliedFirst.rawValue
                                              | CGBitmapInfo.byteOrder32Little.rawValue) else { return false }
            context.draw(image, in: CGRect(x: 0, y: 0, width: width, height: height))
            return true
        }
        let descriptor = MTLTextureDescriptor.texture2DDescriptor(pixelFormat: GPUContext.drawablePixelFormat,
                                                                  width: width, height: height, mipmapped: false)
        descriptor.usage = .shaderRead
        descriptor.storageMode = .shared
        guard drawn, let texture = device.makeTexture(descriptor: descriptor) else { return nil }
        pixels.withUnsafeBytes { bytes in
            guard let address = bytes.baseAddress else { return }
            texture.replace(region: MTLRegionMake2D(0, 0, width, height), mipmapLevel: 0, withBytes: address,
                            bytesPerRow: bytesPerRow)
        }
        return texture
    }
}
