import Foundation
import AppKit
import CoreGraphics

@available(macOS 26.0, *)
@MainActor
final class MouseEventRouter: ObservableObject {
    
    @Published private(set) var isRouting: Bool = false
    @Published private(set) var lastError: String?
    
    private var eventMonitor: Any?
    private var localEventMonitor: Any?
    
    private var overlayFrame: CGRect = .zero
    private var overlayScreen: NSScreen?
    
    private var virtualDisplayID: CGDirectDisplayID = 0
    private var virtualDisplayOrigin: CGPoint = .zero
    private var virtualDisplaySize: CGSize = .zero
    
    private var scaleX: CGFloat = 1.0
    private var scaleY: CGFloat = 1.0
    
    init() {}
    
    func configure(
        overlayFrame: CGRect,
        overlayScreen: NSScreen?,
        virtualDisplayID: CGDirectDisplayID,
        virtualSize: CGSize
    ) {
        self.overlayFrame = overlayFrame
        self.overlayScreen = overlayScreen
        self.virtualDisplayID = virtualDisplayID
        self.virtualDisplaySize = virtualSize
        
        if let vScreen = getScreen(for: virtualDisplayID) {
            self.virtualDisplayOrigin = vScreen.frame.origin
        } else {
            let bounds = CGDisplayBounds(virtualDisplayID)
            self.virtualDisplayOrigin = bounds.origin
        }
        
        if overlayFrame.width > 0 && overlayFrame.height > 0 {
            scaleX = virtualSize.width / overlayFrame.width
            scaleY = virtualSize.height / overlayFrame.height
        }
    }
    
    func startRouting() {
        guard !isRouting else { return }
        guard virtualDisplayID != 0 else {
            lastError = "Virtual display not configured"
            return
        }
        
        eventMonitor = NSEvent.addGlobalMonitorForEvents(
            matching: [.mouseMoved, .leftMouseDown, .leftMouseUp, .rightMouseDown, .rightMouseUp, .leftMouseDragged, .rightMouseDragged, .scrollWheel]
        ) { [weak self] event in
            Task { @MainActor in
                self?.handleMouseEvent(event)
            }
        }
        
        localEventMonitor = NSEvent.addLocalMonitorForEvents(
            matching: [.mouseMoved, .leftMouseDown, .leftMouseUp, .rightMouseDown, .rightMouseUp, .leftMouseDragged, .rightMouseDragged, .scrollWheel]
        ) { [weak self] event in
            Task { @MainActor in
                self?.handleMouseEvent(event)
            }
            return event
        }
        
        isRouting = true
        lastError = nil
    }
    
    func stopRouting() {
        if let monitor = eventMonitor {
            NSEvent.removeMonitor(monitor)
            eventMonitor = nil
        }
        if let monitor = localEventMonitor {
            NSEvent.removeMonitor(monitor)
            localEventMonitor = nil
        }
        isRouting = false
    }
    
    private func handleMouseEvent(_ event: NSEvent) {
        let mouseLocation = NSEvent.mouseLocation
        
        guard let mainScreen = NSScreen.main else { return }
        let screenHeight = mainScreen.frame.height
        
        let overlayInCocoaCoords = CGRect(
            x: overlayFrame.origin.x,
            y: screenHeight - overlayFrame.origin.y - overlayFrame.height,
            width: overlayFrame.width,
            height: overlayFrame.height
        )
        
        guard overlayInCocoaCoords.contains(mouseLocation) else {
            return
        }
        
        let relativeX = (mouseLocation.x - overlayInCocoaCoords.origin.x) / overlayInCocoaCoords.width
        let relativeY = (mouseLocation.y - overlayInCocoaCoords.origin.y) / overlayInCocoaCoords.height
        
        let targetX = virtualDisplayOrigin.x + (relativeX * virtualDisplaySize.width)
        let targetY = virtualDisplayOrigin.y + ((1.0 - relativeY) * virtualDisplaySize.height)
        
        let targetPoint = CGPoint(x: targetX, y: targetY)
        
        CGWarpMouseCursorPosition(targetPoint)
        
        injectMouseEvent(event: event, at: targetPoint)
    }
    
    private func injectMouseEvent(event: NSEvent, at point: CGPoint) {
        var eventType: CGEventType
        var mouseButton: CGMouseButton = .left
        
        switch event.type {
        case .mouseMoved:
            eventType = .mouseMoved
        case .leftMouseDown:
            eventType = .leftMouseDown
            mouseButton = .left
        case .leftMouseUp:
            eventType = .leftMouseUp
            mouseButton = .left
        case .rightMouseDown:
            eventType = .rightMouseDown
            mouseButton = .right
        case .rightMouseUp:
            eventType = .rightMouseUp
            mouseButton = .right
        case .leftMouseDragged:
            eventType = .leftMouseDragged
            mouseButton = .left
        case .rightMouseDragged:
            eventType = .rightMouseDragged
            mouseButton = .right
        case .scrollWheel:
            injectScrollEvent(event: event, at: point)
            return
        default:
            return
        }
        
        guard let cgEvent = CGEvent(
            mouseEventSource: nil,
            mouseType: eventType,
            mouseCursorPosition: point,
            mouseButton: mouseButton
        ) else {
            return
        }
        
        cgEvent.setIntegerValueField(.mouseEventClickState, value: Int64(event.clickCount))
        
        cgEvent.post(tap: .cghidEventTap)
    }
    
    private func injectScrollEvent(event: NSEvent, at point: CGPoint) {
        CGWarpMouseCursorPosition(point)
        
        guard let scrollEvent = CGEvent(
            scrollWheelEvent2Source: nil,
            units: .pixel,
            wheelCount: 2,
            wheel1: Int32(event.scrollingDeltaY),
            wheel2: Int32(event.scrollingDeltaX),
            wheel3: 0
        ) else {
            return
        }
        
        scrollEvent.post(tap: .cghidEventTap)
    }
    
    func updateOverlayFrame(_ frame: CGRect) {
        overlayFrame = frame
        
        if frame.width > 0 && frame.height > 0 {
            scaleX = virtualDisplaySize.width / frame.width
            scaleY = virtualDisplaySize.height / frame.height
        }
    }
    
    private func getScreen(for displayID: CGDirectDisplayID) -> NSScreen? {
        return NSScreen.screens.first { screen in
            let screenID = screen.deviceDescription[NSDeviceDescriptionKey("NSScreenNumber")] as? CGDirectDisplayID
            return screenID == displayID
        }
    }
    
    deinit {
    }
}
