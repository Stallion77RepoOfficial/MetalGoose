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

    var message: String { String(localized: "Error code: \(code) \(detail)") }
    var description: String { message }
}

extension MGError {
    // MARK: UI
    static let frontmostIsSelf = MGError("MG-UI-001", String(localized: "Please switch to the target window before starting."))
    static let targetWindowNotFound = MGError("MG-UI-002", String(localized: "Target window not found."))
    static let noDisplay = MGError("MG-UI-004", String(localized: "No display found."))
    static let displayIDUnavailable = MGError("MG-UI-005", String(localized: "Display ID not found."))
    static let refreshRateUnavailable = MGError("MG-UI-006", String(localized: "Display refresh rate unavailable."))
    static func shortcutUnavailable(_ shortcuts: [String]) -> MGError {
        MGError("MG-UI-007", String(localized: "Already in use by another app: \(shortcuts.joined(separator: ", "))."))
    }

    // MARK: Capture
    static let captureWindowNotFound = MGError("MG-CAP-001", String(localized: "Target window not found."))
    static func captureStartFailed(_ error: Error) -> MGError {
        MGError("MG-CAP-002", String(localized: "ScreenCaptureKit start error: \(error.localizedDescription)"))
    }
    static func captureStopFailed(_ error: Error) -> MGError {
        MGError("MG-CAP-003", String(localized: "ScreenCaptureKit stop error: \(error.localizedDescription)"))
    }
    static func captureStreamStopped(_ error: Error) -> MGError {
        MGError("MG-CAP-004", String(localized: "Stream stopped with error: \(error.localizedDescription)"))
    }
    static let targetEnteredFullscreen = MGError(
        "MG-CAP-005",
        String(localized: "Target entered macOS fullscreen, which cannot be captured with the overlay. Please use windowed or borderless (windowed fullscreen) mode."))
    static func captureReconfigurationFailed(_ error: Error) -> MGError {
        MGError("MG-CAP-007", String(localized: "Capture reconfiguration failed: \(error.localizedDescription)"))
    }

    // MARK: Engine
    static func pipelineSetupFailed(_ detail: String = "") -> MGError {
        MGError("MG-ENG-001", detail.isEmpty ? String(localized: "Metal pipeline setup failed.") : String(localized: "Pipeline setup failed: \(detail)"))
    }
    static let metalDeviceUnavailable = MGError("MG-ENG-002", String(localized: "Metal device not available."))
    static let commandQueueUnavailable = MGError("MG-ENG-003", String(localized: "Metal command queue not available."))
    static let spatialScalerFailed = MGError("MG-ENG-004", String(localized: "MetalFX Spatial Scaler creation failed"))
    static func antiAliasingUnavailable(_ mode: String) -> MGError {
        MGError("MG-ENG-005", String(localized: "Anti-aliasing pipeline unavailable (\(mode))"))
    }
    static let sharpeningUnavailable = MGError("MG-ENG-007", String(localized: "CAS pipeline unavailable"))
    static let surfaceTextureFailed = MGError("MG-ENG-008", String(localized: "IOSurface texture creation failed"))
    static let interpolatorFailed = MGError("MG-ENG-010", String(localized: "MetalFX Frame Interpolator creation failed"))
    static func gpuExecutionFailed(stage: String, detail: String) -> MGError {
        MGError("MG-ENG-011", String(localized: "GPU execution failed (\(stage)): \(detail)"))
    }
}
