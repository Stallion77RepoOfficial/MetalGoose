import Carbon.HIToolbox
import AppKit

/// System-wide keyboard shortcuts, through Carbon's hot-key API — the one that needs no Accessibility
/// permission and works whichever app is in front.
///
/// Shortcuts are registered once and live as long as the process: nothing ever needs them undone, and
/// the system drops them when the process ends.
final class GlobalHotkeyManager {
    nonisolated(unsafe) static let shared = GlobalHotkeyManager()

    private struct Entry {
        let ref: EventHotKeyRef
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

    /// Whether the shortcut is now live. It is not when another app has registered the same one, or the
    /// system refuses for another reason; nothing is kept for a shortcut that failed.
    @discardableResult
    func register(keyCode: UInt32, modifiers: UInt32, handler: @escaping () -> Void) -> Bool {
        guard installHandlerIfNeeded() else { return false }

        let combo = comboKey(keyCode, modifiers)
        if let existing = entries.removeValue(forKey: combo) {
            UnregisterEventHotKey(existing.ref)
            callbacks[existing.id] = nil
        }

        let id = nextID
        nextID += 1
        var hotKeyRef: EventHotKeyRef?
        let status = RegisterEventHotKey(keyCode, modifiers, EventHotKeyID(signature: OSType(0x4D47_4B53), id: id),
                                         GetApplicationEventTarget(), 0, &hotKeyRef)
        guard status == noErr, let hotKeyRef else { return false }

        callbacks[id] = handler
        entries[combo] = Entry(ref: hotKeyRef, id: id)
        return true
    }

    private func installHandlerIfNeeded() -> Bool {
        if eventHandler != nil { return true }

        var eventType = EventTypeSpec(eventClass: OSType(kEventClassKeyboard), eventKind: UInt32(kEventHotKeyPressed))
        let status = InstallEventHandler(GetApplicationEventTarget(), { (_, eventRef, userData) -> OSStatus in
            guard let eventRef, let userData else { return OSStatus(eventNotHandledErr) }

            var hotKeyID = EventHotKeyID()
            let status = GetEventParameter(eventRef, EventParamName(kEventParamDirectObject), EventParamType(typeEventHotKeyID),
                                           nil, MemoryLayout<EventHotKeyID>.size, nil, &hotKeyID)
            guard status == noErr else { return status }

            let manager = Unmanaged<GlobalHotkeyManager>.fromOpaque(userData).takeUnretainedValue()
            manager.callbacks[hotKeyID.id]?()
            return noErr
        }, 1, &eventType, Unmanaged.passUnretained(self).toOpaque(), &eventHandler)
        return status == noErr
    }
}
