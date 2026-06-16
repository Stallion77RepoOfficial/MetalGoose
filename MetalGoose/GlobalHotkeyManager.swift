import Carbon.HIToolbox
import AppKit

final class GlobalHotkeyManager {
    nonisolated(unsafe) static let shared = GlobalHotkeyManager()

    private struct Entry {
        let ref: EventHotKeyRef?
        let id: UInt32
    }

    private var entries: [UInt64: Entry] = [:]
    private var eventHandler: EventHandlerRef?
    private var callbacks: [UInt32: () -> Void] = [:]
    private var nextID: UInt32 = 1

    private init() {}

    private func comboKey(_ keyCode: UInt32, _ modifiers: UInt32) -> UInt64 {
        (UInt64(keyCode) << 32) | UInt64(modifiers)
    }

    func register(keyCode: UInt32, modifiers: UInt32, handler: @escaping () -> Void) {
        if eventHandler == nil {
            installHandler()
        }

        let combo = comboKey(keyCode, modifiers)
        if let existing = entries[combo] {
            if let ref = existing.ref { UnregisterEventHotKey(ref) }
            callbacks[existing.id] = nil
            entries[combo] = nil
        }

        let id = nextID
        nextID += 1
        let hotKeyID = EventHotKeyID(signature: OSType(0x4D47_4B53), id: id)
        callbacks[id] = handler

        var hotKeyRef: EventHotKeyRef?
        RegisterEventHotKey(keyCode, modifiers, hotKeyID, GetApplicationEventTarget(), 0, &hotKeyRef)
        entries[combo] = Entry(ref: hotKeyRef, id: id)
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
        for (_, entry) in entries {
            if let ref = entry.ref {
                UnregisterEventHotKey(ref)
            }
        }
        entries.removeAll()
        callbacks.removeAll()

        if let eventHandler {
            RemoveEventHandler(eventHandler)
            self.eventHandler = nil
        }
    }
}
