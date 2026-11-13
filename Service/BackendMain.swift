// Goose HUB integration for MetalGoose. Distributed under GPL-3.0; see LICENSE.
import AppKit
import Darwin
import Foundation

@MainActor
final class BackendApplication: NSObject, NSApplicationDelegate {
    let output = BackendOutput()
    lazy var session = BackendSession(output: output)
    private var signalSources: [DispatchSourceSignal] = []
    private var queue: [BackendCommand] = []
    private var processing = false
    private var quitting = false

    func applicationDidFinishLaunching(_ notification: Notification) {
        _ = session
        session.registerHotkeys()
        // The Neural Engine's models are built while nothing is being scaled yet, where the system does not have them:
        // a session that waits for one begins without its images (`NeuralModelCache`).
        NeuralModelCache.buildAtLaunch()
        for value in [SIGINT, SIGTERM] {
            signal(value, SIG_IGN)
            let source = DispatchSource.makeSignalSource(signal: value, queue: .main)
            source.setEventHandler { [weak self] in
                MainActor.assumeIsolated { self?.enqueueQuit() }
            }
            source.resume()
            signalSources.append(source)
        }
        output.send(BackendResponse(event: "ready", status: session.status()))
        DispatchQueue.global(qos: .userInitiated).async { [weak self] in
            while let line = readLine() {
                guard line.utf8.count <= 65536 else {
                    let response = BackendResponse.failure(MGError("HUB-MG-REQUEST", "A command may contain at most 64 KiB."))
                    DispatchQueue.main.async { MainActor.assumeIsolated { self?.output.send(response) } }
                    continue
                }
                do {
                    let command = try JSONDecoder().decode(BackendCommand.self, from: Data(line.utf8))
                    DispatchQueue.main.async { MainActor.assumeIsolated { self?.enqueue(command) } }
                } catch {
                    DispatchQueue.main.async { MainActor.assumeIsolated {
                        self?.output.send(.failure(MGError("HUB-MG-JSON", error.localizedDescription)))
                    }
                    }
                }
            }
            // Closing the parent's control pipe must release the overlay and mouse constraint.
            DispatchQueue.main.async { MainActor.assumeIsolated { self?.enqueueQuit() } }
        }
    }

    func applicationShouldTerminate(_ sender: NSApplication) -> NSApplication.TerminateReply {
        if quitting { return .terminateNow }
        enqueueQuit()
        return .terminateCancel
    }

    func applicationWillTerminate(_ notification: Notification) { session.restorePointer() }

    func enqueue(_ command: BackendCommand) {
        guard !quitting else { return }
        queue.append(command)
        guard !processing else { return }
        processing = true
        Task {
            while !queue.isEmpty && !quitting { await handle(queue.removeFirst()) }
            processing = false
        }
    }

    private func enqueueQuit() {
        enqueue(BackendCommand(id: nil, command: "quit", pid: nil, windowID: nil,
                               configuration: nil, screenRecording: nil, accessibility: nil))
    }

    private func handle(_ request: BackendCommand) async {
        var message: String?
        do {
            switch request.command {
            case "status": break
            case "start":
                guard let pid = request.pid, pid > 0 else { throw MGError("HUB-MG-TARGET", "start requires the game's pid.") }
                if let configuration = request.configuration { try session.configure(configuration) }
                try await session.start(pid: pid, windowID: request.windowID)
            case "configure":
                guard let configuration = request.configuration else {
                    throw MGError("HUB-MG-CONFIG", "configure requires a configuration object.")
                }
                try session.configure(configuration)
            case "stop": await session.stop()
            case "request-permissions":
                if request.screenRecording ?? false { session.permissions.requestScreenRecording() }
                if request.accessibility ?? false { session.permissions.requestAccessibility() }
            case "clear-metal-cache":
                guard session.phase == .idle else { throw MGError("HUB-MG-BUSY", "Stop scaling before clearing the Metal cache.") }
                message = MetalCache.clear()
            case "quit":
                quitting = true
                await session.stop()
                output.send(BackendResponse(id: request.id, event: "goodbye", status: session.status()))
                NSApplication.shared.terminate(nil)
                return
            default: throw MGError("HUB-MG-COMMAND", "Unknown command: \(request.command)")
            }
            output.send(BackendResponse(id: request.id, status: session.status(), message: message))
        } catch { output.send(.failure(error, id: request.id)) }
    }
}

@main
enum MetalGooseBackendMain {
    static func main() {
        let arguments = Array(CommandLine.arguments.dropFirst())
        if arguments.contains("--help") || arguments.isEmpty {
            print("""
            metalgoose-backend --status [--json]
            metalgoose-backend --serve

            --status reports device, shader path and existing privacy grants; it starts no capture.
            --serve accepts one JSON object per stdin line and writes JSON responses to stdout.
            Commands: status, start, configure, stop, request-permissions, clear-metal-cache, quit.
            See the adjacent backend README for the protocol.
            """)
            return
        }
        MainActor.assumeIsolated {
            if arguments.allSatisfy({ ["--status", "--json"].contains($0) }) && arguments.contains("--status") {
                let output = BackendOutput()
                let session = BackendSession(output: output)
                output.send(BackendResponse(status: session.status()))
                return
            }
            guard arguments == ["--serve"] else {
                BackendOutput().send(.failure(MGError("HUB-MG-ARGUMENT", "Use --status or --serve.")))
                Darwin.exit(2)
            }
            let app = NSApplication.shared
            app.setActivationPolicy(.accessory)
            let delegate = BackendApplication()
            app.delegate = delegate
            withExtendedLifetime(delegate) { app.run() }
        }
    }
}
