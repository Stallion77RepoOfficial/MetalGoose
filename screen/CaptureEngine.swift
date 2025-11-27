import ScreenCaptureKit
import CoreVideo
import CoreMedia
import Foundation
import AppKit
import CoreGraphics

@MainActor
final class CaptureEngine: NSObject, ObservableObject, SCStreamOutput {
    @Published private(set) var isCapturing: Bool = false
    @Published private(set) var currentFrame: CVPixelBuffer?
    @Published var availableDisplays: [SCDisplay] = []
    @Published var availableWindows: [SCWindow] = []

    private var previousFrame: CVPixelBuffer?
    
    let settings: CaptureSettings

    private var stream: SCStream?

    @MainActor
    init(settings: CaptureSettings) {
        self.settings = settings
    }

    private func frontmostWindowID() -> CGWindowID? {
        guard let app = NSWorkspace.shared.frontmostApplication else { return nil }
        let opts: CGWindowListOption = [.optionOnScreenOnly, .excludeDesktopElements]
        guard let infoList = CGWindowListCopyWindowInfo(opts, kCGNullWindowID) as? [[String: Any]] else { return nil }
        let pid = app.processIdentifier
        for info in infoList {
            if let ownerPID = info[kCGWindowOwnerPID as String] as? pid_t,
               ownerPID == pid,
               let wid = info[kCGWindowNumber as String] as? CGWindowID {
                return wid
            }
        }
        return nil
    }

    private func configure(_ config: SCStreamConfiguration, withDownscaleFactor factor: Double, from sourceSize: CGSize) {
        let clamped = max(0.1, min(1.0, factor))
        let target = CGSize(width: max(1, Int(Double(sourceSize.width) * clamped)),
                            height: max(1, Int(Double(sourceSize.height) * clamped)))
        config.width = Int(target.width)
        config.height = Int(target.height)
        config.pixelFormat = kCVPixelFormatType_32BGRA
        config.colorSpaceName = CGColorSpace.sRGB
        config.queueDepth = 3
        config.scalesToFit = true
    }

    private func makeFilterExcludingWindows(forDisplay display: SCDisplay?, window: SCWindow?, excludedWindowIDs: [CGWindowID]) async -> SCContentFilter? {
        do {
            let content = try await SCShareableContent.excludingDesktopWindows(false, onScreenWindowsOnly: true)
            let excluded: [SCWindow] = content.windows.filter { excludedWindowIDs.contains($0.windowID) }
            if let display {
                return try SCContentFilter(display: display, excludingWindows: excluded)
            }
            if let win = window {
                return SCContentFilter(desktopIndependentWindow: win)
            }
            return nil
        } catch {
            return nil
        }
    }

    func start() async {
        guard !isCapturing else { return }

        do {
            let content = try await SCShareableContent.excludingDesktopWindows(false, onScreenWindowsOnly: true)
            self.availableDisplays = content.displays
            self.availableWindows = content.windows

            // Determine frontmost window once (single-shot capture target)
            let frontID = self.frontmostWindowID()
            let selectedWin: SCWindow? = content.windows.first(where: { $0.windowID == frontID })
            let selectedDisp: SCDisplay? = selectedWin == nil ? content.displays.first : nil

            // Exclude overlay-like windows by heuristic (visible, high level)
            let excludedIDs: [CGWindowID] = NSApp.windows
                .filter { $0.isVisible && $0.level.rawValue >= NSWindow.Level.screenSaver.rawValue }
                .map { CGWindowID($0.windowNumber) }

            guard let filter = await makeFilterExcludingWindows(forDisplay: selectedDisp,
                                                                window: selectedWin,
                                                                excludedWindowIDs: excludedIDs) else {
                return
            }

            let config = SCStreamConfiguration()
            let sourceSize = CGSize(width: filter.contentRect.width, height: filter.contentRect.height)
            let factor = settings.downscaleFactor
            configure(config, withDownscaleFactor: factor, from: sourceSize)
            config.minimumFrameInterval = CMTime(value: 1, timescale: 60)

            let stream = SCStream(filter: filter, configuration: config, delegate: nil)
            self.stream = stream

            try stream.addStreamOutput(self,
                                       type: SCStreamOutputType.screen,
                                       sampleHandlerQueue: DispatchQueue.global(qos: .userInteractive))

            try await stream.startCapture()
            isCapturing = true
            
            if settings.showFullscreenOverlayOnRun {
                OverlayWindowController.shared.presentFullscreenOverlay()
            }
        } catch {
            isCapturing = false
            stream = nil
        }
    }

    func stop() {
        guard let stream else { return }
        stream.stopCapture { [weak self] _ in
            Task { @MainActor in
                self?.isCapturing = false
                self?.stream = nil
                self?.currentFrame = nil
                self?.previousFrame = nil
                OverlayWindowController.shared.dismissOverlay()
            }
        }
    }

    nonisolated func stream(_ stream: SCStream,
                            didOutputSampleBuffer sampleBuffer: CMSampleBuffer,
                            of type: SCStreamOutputType) {
        guard type == SCStreamOutputType.screen,
              let pixelBuffer = sampleBuffer.imageBuffer else {
            return
        }

        Task { @MainActor [weak self] in
            guard let self else { return }
            self.currentFrame = pixelBuffer
            self.previousFrame = pixelBuffer
        }
    }
}

