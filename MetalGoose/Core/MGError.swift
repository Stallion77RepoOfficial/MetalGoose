import Foundation

/// A failure the user is shown. The code is stable — the README documents each one — and
/// the detail says what went wrong this time.
struct MGError: Error, Sendable, Equatable, CustomStringConvertible {
    let code: String
    let detail: String

    init(_ code: String, _ detail: String) {
        self.code = code
        self.detail = detail
    }

    var message: String { "Error Code: \(code) \(detail)" }
    var description: String { message }
}

extension MGError {
    // MARK: UI
    static let frontmostIsSelf = MGError("MG-UI-001", "Please switch to the target window before starting.")
    static let targetWindowNotFound = MGError("MG-UI-002", "Target window not found.")
    static let noDisplay = MGError("MG-UI-004", "No display found.")
    static let displayIDUnavailable = MGError("MG-UI-005", "Display ID not found.")
    static let refreshRateUnavailable = MGError("MG-UI-006", "Display refresh rate unavailable.")
    static func shortcutUnavailable(_ shortcuts: [String]) -> MGError {
        MGError("MG-UI-007", "Already in use by another app: \(shortcuts.joined(separator: ", ")).")
    }

    // MARK: Capture
    static let captureWindowNotFound = MGError("MG-CAP-001", "Target window not found.")
    static func captureStartFailed(_ error: Error) -> MGError {
        MGError("MG-CAP-002", "ScreenCaptureKit start error: \(error.localizedDescription)")
    }
    static func captureStopFailed(_ error: Error) -> MGError {
        MGError("MG-CAP-003", "ScreenCaptureKit stop error: \(error.localizedDescription)")
    }
    static func captureStreamStopped(_ error: Error) -> MGError {
        MGError("MG-CAP-004", "Stream stopped with error: \(error.localizedDescription)")
    }
    static let targetEnteredFullscreen = MGError(
        "MG-CAP-005",
        "Target entered macOS fullscreen, which cannot be captured with the overlay. Please use windowed or borderless (windowed fullscreen) mode.")
    static func captureReconfigurationFailed(_ error: Error) -> MGError {
        MGError("MG-CAP-007", "Capture reconfiguration failed: \(error.localizedDescription)")
    }

    // MARK: Engine
    static func pipelineSetupFailed(_ detail: String = "") -> MGError {
        MGError("MG-ENG-001", detail.isEmpty ? "Metal pipeline setup failed." : "Pipeline setup failed: \(detail)")
    }
    static let metalDeviceUnavailable = MGError("MG-ENG-002", "Metal device not available.")
    static let commandQueueUnavailable = MGError("MG-ENG-003", "Metal command queue not available.")
    static let spatialScalerFailed = MGError("MG-ENG-004", "MetalFX Spatial Scaler creation failed")
    static func antiAliasingUnavailable(_ mode: String) -> MGError {
        MGError("MG-ENG-005", "Anti-aliasing pipeline unavailable (\(mode))")
    }
    static let sharpeningUnavailable = MGError("MG-ENG-007", "CAS pipeline unavailable")
    static let surfaceTextureFailed = MGError("MG-ENG-008", "IOSurface texture creation failed")
    static let interpolatorFailed = MGError("MG-ENG-010", "MetalFX Frame Interpolator creation failed")
}
