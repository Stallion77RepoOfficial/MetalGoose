import AppKit
import SwiftUI
import CoreGraphics

final class OverlayWindowController: NSObject {
    static let shared = OverlayWindowController()
    public var windowID: CGWindowID? {
        guard let num = window?.windowNumber else { return nil }
        return CGWindowID(num)
    }
    private var window: NSWindow?

    func presentFullscreenOverlay(on screen: NSScreen? = NSScreen.main) {
        guard let screen = screen else { return }
        let overlayView = OverlayView()
        let hosting = NSHostingView(rootView: overlayView)
        let frame = screen.frame
        let window = NSWindow(contentRect: frame,
                              styleMask: [.borderless],
                              backing: .buffered,
                              defer: false,
                              screen: screen)
        window.isOpaque = false
        window.backgroundColor = .clear
        window.level = .screenSaver
        window.ignoresMouseEvents = true
        window.collectionBehavior = [.canJoinAllSpaces, .fullScreenAuxiliary]
        window.contentView = hosting
        window.orderFrontRegardless()
        self.window = window
    }

    func dismissOverlay() {
        window?.orderOut(nil)
        window = nil
    }
}

struct OverlayView: View {
    var body: some View {
        ZStack {
            // Remove fullscreen dim; keep fully transparent background
            Color.clear.ignoresSafeArea()

            VStack {
                HStack {
                    Text("Running")
                        .font(.system(size: 12, weight: .semibold, design: .rounded))
                        .foregroundStyle(.white)
                        .padding(.horizontal, 10)
                        .padding(.vertical, 6)
                        .background(Color.black.opacity(0.55))
                        .clipShape(Capsule())
                    Spacer()
                }
                .padding(.top, 8)
                .padding(.leading, 10)

                Spacer()
            }
        }
        .allowsHitTesting(false)
    }
}
