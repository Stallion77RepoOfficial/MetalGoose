// AppRootView.swift
// Shows PermissionsView until both permissions are granted, then shows the Metal renderer content.

import SwiftUI
import MetalKit

struct AppRootView: View {
    @State private var accessibilityGranted = PermissionsChecker.isAccessibilityGranted
    @State private var screenRecordingGranted = PermissionsChecker.isScreenRecordingGranted

    var body: some View {
        Group {
            if accessibilityGranted && screenRecordingGranted {
                RendererContainerView()
                    .background(Color.black)
            } else {
                PermissionsView()
            }
        }
        .onReceive(NotificationCenter.default.publisher(for: .permissionsShouldRefresh)) { _ in
            refresh()
        }
        .onAppear { refresh() }
    }

    private func refresh() {
        accessibilityGranted = PermissionsChecker.isAccessibilityGranted
        screenRecordingGranted = PermissionsChecker.isScreenRecordingGranted
    }
}

// A SwiftUI wrapper hosting MTKView + Renderer
struct RendererContainerView: NSViewRepresentable {
    func makeNSView(context: Context) -> MTKView {
        let mtk = MTKView()
        mtk.device = MTLCreateSystemDefaultDevice()
        mtk.framebufferOnly = false
        mtk.preferredFramesPerSecond = 60
        let _ = Renderer(view: mtk) // Renderer sets itself as delegate of overlay views; for main view we can set explicitly
        mtk.delegate = context.coordinator.renderer
        return mtk
    }

    func updateNSView(_ nsView: MTKView, context: Context) { }

    func makeCoordinator() -> Coordinator {
        Coordinator()
    }

    final class Coordinator {
        let renderer: Renderer
        init() {
            let mtk = MTKView()
            mtk.device = MTLCreateSystemDefaultDevice()
            renderer = Renderer(view: mtk)! // reuse init to build pipelines
        }
    }
}

extension Notification.Name {
    static let permissionsShouldRefresh = Notification.Name("permissionsShouldRefresh")
}

