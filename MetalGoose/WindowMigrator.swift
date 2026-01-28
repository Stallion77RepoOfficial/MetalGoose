import Foundation
import AppKit
import ApplicationServices

@available(macOS 26.0, *)
@MainActor
final class WindowMigrator: ObservableObject {
    
    @Published private(set) var isWindowMigrated: Bool = false
    @Published private(set) var lastError: String?
    
    private var originalPosition: CGPoint = .zero
    private var originalSize: CGSize = .zero
    private var hasOriginalSize: Bool = false
    private var targetPID: pid_t = 0
    private var targetWindowElement: AXUIElement?
    
    func moveWindow(pid: pid_t, windowID: CGWindowID, windowFrame: CGRect? = nil, windowTitle: String? = nil, to position: CGPoint) -> Bool {
        guard AXIsProcessTrusted() else {
            lastError = "Error Code: MG-AX-001 Accessibility permission required"
            return false
        }
        
        guard let windowElement = resolveWindowElement(pid: pid, windowID: windowID, windowFrame: windowFrame, windowTitle: windowTitle) else {
            if lastError == nil {
                lastError = "Error Code: MG-AX-003 No windows found for PID \(pid)"
            }
            return false
        }
        
        targetPID = pid
        targetWindowElement = windowElement
        
        var positionRef: CFTypeRef?
        if AXUIElementCopyAttributeValue(windowElement, kAXPositionAttribute as CFString, &positionRef) == .success,
           let posValue = positionRef {
            var point = CGPoint.zero
            AXValueGetValue(posValue as! AXValue, .cgPoint, &point)
            originalPosition = point
        }
        
        if let size = getWindowSize(windowElement) {
            originalSize = size
            hasOriginalSize = true
        }
        
        var newPosition = position
        guard let positionValue = AXValueCreate(.cgPoint, &newPosition) else {
            lastError = "Error Code: MG-AX-004 Failed to create position value"
            return false
        }
        
        let setResult = AXUIElementSetAttributeValue(windowElement, kAXPositionAttribute as CFString, positionValue)
        
        if setResult == .success {
            isWindowMigrated = true
            lastError = nil
            return true
        } else {
            lastError = "Error Code: MG-AX-005 Failed to set position: \(setResult.rawValue)"
            return false
        }
    }
    
    func moveToVirtualDisplay(pid: pid_t, windowID: CGWindowID, windowFrame: CGRect? = nil, windowTitle: String? = nil, displayID: CGDirectDisplayID) -> Bool {
        guard let screen = getScreen(for: displayID) else {
            lastError = "Error Code: MG-AX-009 Virtual display screen not found"
            return false
        }
        
        let targetOrigin = screen.frame.origin
        
        return moveWindow(pid: pid, windowID: windowID, windowFrame: windowFrame, windowTitle: windowTitle, to: targetOrigin)
    }

    func isWindowFullscreen(pid: pid_t, windowID: CGWindowID, windowFrame: CGRect? = nil, windowTitle: String? = nil) -> Bool {
        guard AXIsProcessTrusted() else { return false }
        guard let windowElement = resolveWindowElement(pid: pid, windowID: windowID, windowFrame: windowFrame, windowTitle: windowTitle) else { return false }
        return isFullscreen(windowElement)
    }

    func moveWindowToScreen(pid: pid_t,
                            windowID: CGWindowID,
                            screen: NSScreen,
                            targetSize: CGSize?,
                            windowFrame: CGRect? = nil,
                            windowTitle: String? = nil) -> Bool {
        guard AXIsProcessTrusted() else {
            lastError = "Error Code: MG-AX-001 Accessibility permission required"
            return false
        }
        
        guard let windowElement = resolveWindowElement(pid: pid, windowID: windowID, windowFrame: windowFrame, windowTitle: windowTitle) else {
            if lastError == nil {
                lastError = "Error Code: MG-AX-003 No windows found for PID \(pid)"
            }
            return false
        }
        
        if isFullscreen(windowElement) {
            lastError = "Error Code: MG-AX-006 Fullscreen window detected; cannot move to virtual display."
            return false
        }
        
        targetPID = pid
        targetWindowElement = windowElement
        
        if let pos = getWindowPosition(windowElement) {
            originalPosition = pos
        }
        if let size = getWindowSize(windowElement) {
            originalSize = size
            hasOriginalSize = true
        }
        
        let finalSize = targetSize!
        
        let visible = screen.visibleFrame
        
        if finalSize.width > 0 && finalSize.height > 0 {
            if !setWindowSize(windowElement, size: finalSize) {
                lastError = "Error Code: MG-AX-008 Failed to set window size."
                return false
            }
        }
        
        let origin = CGPoint(
            x: visible.origin.x + (visible.size.width - finalSize.width) * 0.5,
            y: visible.origin.y + (visible.size.height - finalSize.height) * 0.5
        )
        
        var newPosition = origin
        guard let positionValue = AXValueCreate(.cgPoint, &newPosition) else {
            lastError = "Error Code: MG-AX-004 Failed to create position value"
            return false
        }
        
        let setResult = AXUIElementSetAttributeValue(windowElement, kAXPositionAttribute as CFString, positionValue)
        if setResult == .success {
            isWindowMigrated = true
            lastError = nil
            return true
        } else {
            lastError = "Error Code: MG-AX-005 Failed to set position: \(setResult.rawValue)"
            return false
        }
    }
    
    func restoreWindow() {
        guard isWindowMigrated, let windowElement = targetWindowElement else { return }
        
        var position = originalPosition
        guard let positionValue = AXValueCreate(.cgPoint, &position) else { return }
        
        let result = AXUIElementSetAttributeValue(windowElement, kAXPositionAttribute as CFString, positionValue)
        
        if result != .success {
        }
        
        if hasOriginalSize {
            _ = setWindowSize(windowElement, size: originalSize)
            hasOriginalSize = false
        }
        
        isWindowMigrated = false
        targetWindowElement = nil
        targetPID = 0
    }

    private func resolveWindowElement(pid: pid_t, windowID: CGWindowID, windowFrame: CGRect?, windowTitle: String?) -> AXUIElement? {
        let appElement = AXUIElementCreateApplication(pid)
        
        func windowNumber(for element: AXUIElement) -> CGWindowID? {
            var numberRef: CFTypeRef?
            let attr = "AXWindowNumber" as CFString
            let result = AXUIElementCopyAttributeValue(element, attr, &numberRef)
            if result == .success, let number = numberRef as? NSNumber {
                return CGWindowID(number.uint32Value)
            }
            return nil
        }
        
        func matches(_ element: AXUIElement) -> Bool {
            if let number = windowNumber(for: element) {
                return number == windowID
            }
            return false
        }
        
        func titleMatches(_ element: AXUIElement) -> Bool {
            guard let title = windowTitle, !title.isEmpty else { return true }
            var titleRef: CFTypeRef?
            if AXUIElementCopyAttributeValue(element, kAXTitleAttribute as CFString, &titleRef) == .success,
               let value = titleRef as? String {
                return value == title
            }
            return false
        }
        
        func sizeMatches(_ element: AXUIElement) -> Bool {
            guard let frame = windowFrame else { return false }
            guard let size = getWindowSize(element) else { return false }
            let w = Int(round(size.width))
            let h = Int(round(size.height))
            let fw = Int(round(frame.size.width))
            let fh = Int(round(frame.size.height))
            return w == fw && h == fh
        }
        
        var windowsRef: CFTypeRef?
        let result = AXUIElementCopyAttributeValue(appElement, kAXWindowsAttribute as CFString, &windowsRef)
        if result == .success, let windows = windowsRef as? [AXUIElement], !windows.isEmpty {
            if let matching = windows.first(where: { matches($0) }) {
                return matching
            }
            if let matching = windows.first(where: { sizeMatches($0) && titleMatches($0) }) {
                return matching
            }
        } else if result != .success {
            lastError = "Error Code: MG-AX-002 Failed to get windows: \(result.rawValue)"
            return nil
        }
        
        var fallbackCandidates: [AXUIElement] = []
        
        var focusedRef: CFTypeRef?
        if AXUIElementCopyAttributeValue(appElement, kAXFocusedWindowAttribute as CFString, &focusedRef) == .success,
           let focused = focusedRef {
            let focusedElement = focused as! AXUIElement
            fallbackCandidates.append(focusedElement)
        }
        
        var mainRef: CFTypeRef?
        if AXUIElementCopyAttributeValue(appElement, kAXMainWindowAttribute as CFString, &mainRef) == .success,
           let main = mainRef {
            let mainElement = main as! AXUIElement
            if !fallbackCandidates.contains(where: { CFEqual($0, mainElement) }) {
                fallbackCandidates.append(mainElement)
            }
        }
        
        if let matching = fallbackCandidates.first(where: { matches($0) }) {
            return matching
        }
        if let matching = fallbackCandidates.first(where: { sizeMatches($0) && titleMatches($0) }) {
            return matching
        }
        
        lastError = "Error Code: MG-AX-010 Window ID not found"
        return nil
    }

    private func isFullscreen(_ element: AXUIElement) -> Bool {
        var fullscreenRef: CFTypeRef?
        let attr = "AXFullScreen" as CFString
        let result = AXUIElementCopyAttributeValue(element, attr, &fullscreenRef)
        if result == .success, let value = fullscreenRef as? NSNumber {
            return value.boolValue
        }
        return false
    }

    private func getWindowPosition(_ element: AXUIElement) -> CGPoint? {
        var positionRef: CFTypeRef?
        guard AXUIElementCopyAttributeValue(element, kAXPositionAttribute as CFString, &positionRef) == .success,
              let ref = positionRef else { return nil }
        let posValue = ref as! AXValue
        var point = CGPoint.zero
        AXValueGetValue(posValue, .cgPoint, &point)
        return point
    }

    private func getWindowSize(_ element: AXUIElement) -> CGSize? {
        var sizeRef: CFTypeRef?
        guard AXUIElementCopyAttributeValue(element, kAXSizeAttribute as CFString, &sizeRef) == .success,
              let ref = sizeRef else { return nil }
        let sizeValue = ref as! AXValue
        var size = CGSize.zero
        AXValueGetValue(sizeValue, .cgSize, &size)
        return size
    }

    private func setWindowSize(_ element: AXUIElement, size: CGSize) -> Bool {
        var target = size
        guard let sizeValue = AXValueCreate(.cgSize, &target) else {
            lastError = "Error Code: MG-AX-007 Failed to create size value"
            return false
        }
        let result = AXUIElementSetAttributeValue(element, kAXSizeAttribute as CFString, sizeValue)
        if result != .success {
            lastError = "Error Code: MG-AX-008 Failed to set window size: \(result.rawValue)"
            return false
        }
        return true
    }
    
    private func getScreen(for displayID: CGDirectDisplayID) -> NSScreen? {
        return NSScreen.screens.first { screen in
            let screenID = screen.deviceDescription[NSDeviceDescriptionKey("NSScreenNumber")] as? CGDirectDisplayID
            return screenID == displayID
        }
    }
}
