import Carbon.HIToolbox
import AppKit

final class GlobalHotkeyManager {
    nonisolated(unsafe) static let shared = GlobalHotkeyManager()

    private var hotKeyRefs: [EventHotKeyRef?] = []
    private var eventHandler: EventHandlerRef?
    private var callbacks: [UInt32: () -> Void] = [:]
    private var nextID: UInt32 = 1

    private init() {}

    func register(keyCode: UInt32, modifiers: UInt32, handler: @escaping () -> Void) {
        if eventHandler == nil {
            installHandler()
        }

        let hotKeyID = EventHotKeyID(signature: OSType(0x4D47_4B53), id: nextID)
        callbacks[nextID] = handler

        var hotKeyRef: EventHotKeyRef?
        RegisterEventHotKey(keyCode, modifiers, hotKeyID, GetApplicationEventTarget(), 0, &hotKeyRef)
        hotKeyRefs.append(hotKeyRef)

        nextID += 1
    }

    private func installHandler() {
        var eventType = EventTypeSpec(eventClass: OSType(kEventClassKeyboard), eventKind: UInt32(kEventHotKeyPressed))

        InstallEventHandler(GetApplicationEventTarget(), { (_, eventRef, userData) -> OSStatus in
            guard let eventRef, let userData else { return OSStatus(eventNotHandledErr) }

            var hotKeyID = EventHotKeyID()
            GetEventParameter(eventRef, EventParamName(kEventParamDirectObject), EventParamType(typeEventHotKeyID), nil, MemoryLayout<EventHotKeyID>.size, nil, &hotKeyID)

            let manager = Unmanaged<GlobalHotkeyManager>.fromOpaque(userData).takeUnretainedValue()
            manager.callbacks[hotKeyID.id]?()
            return noErr
        }, 1, &eventType, Unmanaged.passUnretained(self).toOpaque(), &eventHandler)
    }

    func unregisterAll() {
        for ref in hotKeyRefs {
            if let ref {
                UnregisterEventHotKey(ref)
            }
        }
        hotKeyRefs.removeAll()
        callbacks.removeAll()

        if let eventHandler {
            RemoveEventHandler(eventHandler)
            self.eventHandler = nil
        }
    }
}
