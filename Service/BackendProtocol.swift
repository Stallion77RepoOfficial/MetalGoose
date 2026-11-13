// Goose HUB integration for MetalGoose. Distributed under GPL-3.0; see LICENSE.
import Foundation

struct BackendCommand: Decodable, Sendable {
    let id: String?
    let command: String
    let pid: Int32?
    let windowID: UInt32?
    let configuration: BackendConfiguration?
    let screenRecording: Bool?
    let accessibility: Bool?
}

/// MetalGoose's settings (`CaptureSettings`) as the hub sends them: each option by the raw value
/// MetalGoose stores it under. A missing field takes MetalGoose's default.
struct BackendConfiguration: Codable, Sendable, Equatable {
    var scalingMethod = ScalingMethod.off.rawValue
    var scaleFactor = ScaleFactor.x1.rawValue
    var renderScale = RenderScale.native.rawValue
    var sharpening = Sharpening.strong.rawValue
    var frameGenMode = FrameGenMode.off.rawValue
    var frameGenMultiplier = 2
    var aaMode = AAMode.off.rawValue
    var alignPointer = true
    var showMGHUD = true
    var vsync = true
    var vrr = false
    var tripleBuffering = true

    init() {}

    @MainActor init(_ settings: CaptureSettings) {
        scalingMethod = settings.scalingMethod.rawValue
        scaleFactor = settings.scaleFactor.rawValue
        renderScale = settings.renderScale.rawValue
        sharpening = settings.sharpening.rawValue
        frameGenMode = settings.frameGenMode.rawValue
        frameGenMultiplier = settings.frameGenMultiplier
        aaMode = settings.aaMode.rawValue
        alignPointer = settings.alignPointer
        showMGHUD = settings.showMGHUD
        vsync = settings.vsync
        vrr = settings.vrr
        tripleBuffering = settings.tripleBuffering
    }

    enum CodingKeys: String, CodingKey {
        case scalingMethod, scaleFactor, renderScale, sharpening, frameGenMode, frameGenMultiplier
        case aaMode, alignPointer, showMGHUD, vsync, vrr, tripleBuffering
    }

    init(from decoder: Decoder) throws {
        let values = try decoder.container(keyedBy: CodingKeys.self)
        let defaults = BackendConfiguration()
        scalingMethod = try values.decodeIfPresent(String.self, forKey: .scalingMethod) ?? defaults.scalingMethod
        scaleFactor = try values.decodeIfPresent(String.self, forKey: .scaleFactor) ?? defaults.scaleFactor
        renderScale = try values.decodeIfPresent(String.self, forKey: .renderScale) ?? defaults.renderScale
        sharpening = try values.decodeIfPresent(String.self, forKey: .sharpening) ?? defaults.sharpening
        frameGenMode = try values.decodeIfPresent(String.self, forKey: .frameGenMode) ?? defaults.frameGenMode
        frameGenMultiplier = try values.decodeIfPresent(Int.self, forKey: .frameGenMultiplier) ?? defaults.frameGenMultiplier
        aaMode = try values.decodeIfPresent(String.self, forKey: .aaMode) ?? defaults.aaMode
        alignPointer = try values.decodeIfPresent(Bool.self, forKey: .alignPointer) ?? defaults.alignPointer
        showMGHUD = try values.decodeIfPresent(Bool.self, forKey: .showMGHUD) ?? defaults.showMGHUD
        vsync = try values.decodeIfPresent(Bool.self, forKey: .vsync) ?? defaults.vsync
        vrr = try values.decodeIfPresent(Bool.self, forKey: .vrr) ?? defaults.vrr
        tripleBuffering = try values.decodeIfPresent(Bool.self, forKey: .tripleBuffering) ?? defaults.tripleBuffering
    }

    /// Sets MetalGoose's settings to this configuration, all or none: an option it does not know
    /// changes nothing. Only the options that differ are set, so an unchanged one causes no work.
    /// Triple buffering is fixed while a session runs, as MetalGoose's own toggle is.
    @MainActor func apply(to settings: CaptureSettings, sessionRunning: Bool) throws {
        let scalingMethod = try Self.option(ScalingMethod.self, self.scalingMethod, "scalingMethod")
        let scaleFactor = try Self.option(ScaleFactor.self, self.scaleFactor, "scaleFactor")
        let renderScale = try Self.option(RenderScale.self, self.renderScale, "renderScale")
        let sharpening = try Self.option(Sharpening.self, self.sharpening, "sharpening")
        guard let frameGenMode = FrameGenMode(storedValue: self.frameGenMode) else {
            throw Self.invalid("frameGenMode", FrameGenMode.self)
        }
        let aaMode = try Self.option(AAMode.self, self.aaMode, "aaMode")

        func set<Value: Equatable>(_ path: ReferenceWritableKeyPath<CaptureSettings, Value>, _ value: Value) {
            if settings[keyPath: path] != value { settings[keyPath: path] = value }
        }
        set(\.scalingMethod, scalingMethod)
        set(\.scaleFactor, scaleFactor)
        set(\.renderScale, renderScale)
        set(\.sharpening, sharpening)
        set(\.frameGenMode, frameGenMode)
        set(\.frameGenMultiplier, frameGenMultiplier)
        set(\.aaMode, aaMode)
        set(\.alignPointer, alignPointer)
        set(\.showMGHUD, showMGHUD)
        set(\.vsync, vsync)
        set(\.vrr, vrr)
        if !sessionRunning { set(\.tripleBuffering, tripleBuffering) }
    }

    private static func option<Option: SettingOption>(_ type: Option.Type, _ raw: String, _ name: String) throws -> Option {
        guard let value = Option(rawValue: raw) else { throw invalid(name, type) }
        return value
    }

    private static func invalid<Option: SettingOption>(_ name: String, _ type: Option.Type) -> MGError {
        MGError("HUB-MG-CONFIG", "\(name) must be one of: \(Option.allCases.map(\.rawValue).joined(separator: ", ")).")
    }
}

extension MGError {
    /// MetalGoose starts a session only with both permissions, as its Start Scaling button does.
    static let permissionsRequired = MGError("HUB-MG-PERMISSIONS",
                                             "MetalGoose needs Screen Recording and Accessibility permission.")
}

struct BackendMetrics: Encodable {
    let captureFPS: Float
    let outputFPS: Float
    let generatedFPS: Float
    let gpuTimeMilliseconds: Float
    let gpuBudgetPercent: Float
    let captureLatencyMilliseconds: Float
    let presentationLatencyMilliseconds: Float
    let endToEndLatencyMilliseconds: Float
    let framePacingScore: Float
    let captured: UInt64
    let presented: UInt64
    let generated: UInt64
    let passthrough: UInt64
    let dropped: UInt64
    let counterEpoch: Int
    let gpuMemoryBytes: UInt64
    let processMemoryBytes: UInt64
    let cpuPercent: Float
    let outputWidth: Double
    let outputHeight: Double
    let panelRefreshRate: Int
    let targetOutputFPS: Int
    let generationEngine: String?
    let activeMultiplier: Int
    let generationUnavailable: Bool

    init(_ stats: PipelineStats, generation: GenerationChoice) {
        func finite(_ value: Float) -> Float { value.isFinite ? value : 0 }
        captureFPS = finite(stats.captureFPS)
        outputFPS = finite(stats.outputFPS)
        generatedFPS = finite(stats.generatedFPS)
        gpuTimeMilliseconds = finite(stats.gpuTime)
        gpuBudgetPercent = finite(stats.gpuLoad)
        captureLatencyMilliseconds = finite(stats.captureLatency)
        presentationLatencyMilliseconds = finite(stats.presentLatency)
        endToEndLatencyMilliseconds = finite(stats.endToEndLatency)
        framePacingScore = finite(stats.framePacingScore)
        captured = stats.frameCount
        presented = stats.outputFrameCount
        generated = stats.generatedFrameCount
        passthrough = stats.passthroughFrameCount
        dropped = stats.droppedFrames
        counterEpoch = stats.counterEpoch
        gpuMemoryBytes = stats.gpuMemoryUsed
        processMemoryBytes = stats.processMemoryUsed
        cpuPercent = finite(stats.cpuUsage)
        outputWidth = stats.outputResolution.width.isFinite ? stats.outputResolution.width : 0
        outputHeight = stats.outputResolution.height.isFinite ? stats.outputResolution.height : 0
        panelRefreshRate = stats.screenRefreshRate
        targetOutputFPS = stats.targetOutputFPS
        switch generation.engine {
        case .neuralEngine: generationEngine = "neural-engine"
        case .metalFX: generationEngine = "metalfx"
        case nil: generationEngine = nil
        }
        activeMultiplier = generation.multiplier
        generationUnavailable = generation.unavailable
    }
}

struct BackendStatus: Encodable {
    let backend = "MetalGoose"
    let version = "1.2.4-goosehub.2"
    let architecture = "arm64"
    let minimumMacOS = "27.0"
    let screenRecordingGranted: Bool
    let accessibilityGranted: Bool
    let shaderLibrary: String?
    let device: String?
    let engineInitialized: Bool
    let phase: String
    let presenting: Bool
    let frontmostPID: Int32?
    let targetOnScreen: Bool?
    let suspensionReason: String?
    let pid: Int32?
    let windowID: UInt32?
    let configuration: BackendConfiguration
    let metrics: BackendMetrics?
}

struct BackendFailure: Encodable {
    let code: String
    let message: String
}

struct BackendResponse: Encodable {
    let protocolVersion = 2
    var id: String?
    var ok = true
    var event: String?
    /// Why a session started or stopped on its own: shortcut, capture, fullscreen or target-exited.
    var reason: String?
    var status: BackendStatus?
    var message: String?
    var error: BackendFailure?

    static func failure(_ error: Error, id: String? = nil) -> BackendResponse {
        let failure = error as? MGError ?? MGError("HUB-MG-ERROR", error.localizedDescription)
        return BackendResponse(id: id, ok: false,
                               error: BackendFailure(code: failure.code, message: failure.detail))
    }
}

/// Stdout is exclusively JSON lines; framework diagnostics continue to use stderr.
final class BackendOutput: @unchecked Sendable {
    private let lock = NSLock()
    private let encoder = JSONEncoder()

    init() { encoder.outputFormatting = [.sortedKeys, .withoutEscapingSlashes] }

    func send(_ response: BackendResponse) {
        lock.lock()
        defer { lock.unlock() }
        guard var data = try? encoder.encode(response) else { return }
        data.append(0x0a)
        try? FileHandle.standardOutput.write(contentsOf: data)
    }
}
