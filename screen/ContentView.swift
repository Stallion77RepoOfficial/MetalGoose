import SwiftUI
import MetalKit

struct RootView: View {
    @StateObject var settings = CaptureSettings()
    @StateObject var engine = CaptureEngine()
    
    // Renderer referansı
    @State private var renderer: Renderer?
    
    // UI Durumları
    @State private var isRunning = false
    @State private var isCountingDown = false // Geri sayım aktif mi?
    @State private var countdownValue = 5     // Geri sayım değeri
    @State private var showPermissions = true
    
    var body: some View {
        ZStack {
            // Arka plan
            Color(nsColor: .windowBackgroundColor).ignoresSafeArea()
            
            // İzin kontrolü
            if showPermissions {
                PermissionsView()
                    .onChange(of: AXIsProcessTrusted() && CGPreflightScreenCaptureAccess()) { granted in
                        if granted { showPermissions = false }
                    }
                    .onAppear {
                        if AXIsProcessTrusted() && CGPreflightScreenCaptureAccess() {
                            showPermissions = false
                        }
                    }
            } else {
                // Ana kontrol paneli
                MainControlView(
                    isRunning: $isRunning,
                    isCountingDown: $isCountingDown,
                    countdownValue: $countdownValue,
                    startAction: startSequence, // Geri sayımı başlatan fonksiyon
                    stopAction: stopScaling
                )
                .environmentObject(settings)
            }
            
            // Geri Sayım Ekranı (Tüm ekranı kaplar)
            if isCountingDown {
                Color.black.opacity(0.85).ignoresSafeArea()
                VStack {
                    Text("Switch to Game")
                        .font(.largeTitle)
                        .foregroundColor(.white)
                        .padding(.bottom, 20)
                    
                    Text("\(countdownValue)")
                        .font(.system(size: 120, weight: .bold, design: .rounded))
                        .foregroundColor(.white)
                        .transition(.scale)
                        .id(countdownValue) // Animasyon için ID
                }
            }
        }
        .frame(width: 400, height: 350)
    }
    
    // 1. Adım: Geri sayımı başlat
    func startSequence() {
        guard !isRunning else { return }
        
        isCountingDown = true
        countdownValue = 5
        
        // 5 saniyelik zamanlayıcı başlat
        Timer.scheduledTimer(withTimeInterval: 1.0, repeats: true) { timer in
            if self.countdownValue > 1 {
                self.countdownValue -= 1
            } else {
                timer.invalidate()
                self.isCountingDown = false
                // Süre bitti, asıl işleme başla
                self.startScaling()
            }
        }
    }
    
    // 2. Adım: Ölçeklemeyi Başlat (Süre bitince çalışır)
    func startScaling() {
        // En öndeki uygulamayı bul (Kullanıcı bu sırada oyuna geçmiş olmalı)
        guard let app = NSWorkspace.shared.frontmostApplication else { return }
        
        // KENDİNİ YAKALAMA KONTROLÜ: Eğer hala bizim uygulama öndeyse durdur.
        if app.processIdentifier == NSRunningApplication.current.processIdentifier {
            print("HATA: Kendini ölçekleyemezsin! Lütfen süre bitmeden oyuna geçiş yap.")
            // Geri sayım bitti ama oyun açılmadıysa resetle
            stopScaling()
            return
        }
        
        let pid = app.processIdentifier
        
        // Pencere listesini al
        let list = CGWindowListCopyWindowInfo([.optionOnScreenOnly, .excludeDesktopElements], kCGNullWindowID) as? [[String: Any]]
        
        // Hedef pencereyi bul
        guard let entry = list?.first(where: { ($0[kCGWindowOwnerPID as String] as? Int) == Int(pid) }),
              let wid = entry[kCGWindowNumber as String] as? CGWindowID else {
            print("Hedef pencere bulunamadı.")
            return
        }
        
        // Renderer'ı hazırla
        if renderer == nil {
            let mtk = MTKView()
            renderer = Renderer(view: mtk)
        }
        
        // Ayarları uygula
        renderer?.scalingMode = settings.scalingMode
        renderer?.sharpenAmount = Float(settings.sharpenAmount)
        
        // İşlemi başlat
        Task {
            // ScreenCaptureKit başlat
            await engine.startCapture(windowID: wid, displayID: CGMainDisplayID())
            
            // Overlay penceresini aç
            renderer?.startOverlay(for: wid, pid: pid)
            
            isRunning = true
            
            // Frame döngüsü
            Task.detached {
                while isRunning {
                    if let frame = await engine.currentFrame {
                        await renderer?.update(with: frame)
                    }
                    try? await Task.sleep(nanoseconds: 1_000_000_000 / UInt64(settings.targetFPS))
                }
            }
        }
    }
    
    func stopScaling() {
        isRunning = false
        engine.stopCapture()
        renderer?.stopOverlay()
    }
}

// Ana Arayüz Bileşeni
struct MainControlView: View {
    @Binding var isRunning: Bool
    @Binding var isCountingDown: Bool
    @Binding var countdownValue: Int
    
    var startAction: () -> Void
    var stopAction: () -> Void
    @EnvironmentObject var settings: CaptureSettings
    
    var body: some View {
        VStack(spacing: 20) {
            Text("MetalGoose")
                .font(.largeTitle)
                .bold()
                .opacity(0.8)
            
            if isRunning {
                Button(action: stopAction) {
                    Label("STOP SCALING", systemImage: "stop.fill")
                        .font(.headline)
                        .padding()
                        .frame(maxWidth: .infinity)
                }
                .buttonStyle(.borderedProminent)
                .tint(.red)
                .controlSize(.large)
            } else {
                // Başlat butonu (Geri sayım sırasında devre dışı)
                Button(action: startAction) {
                    HStack {
                        if isCountingDown {
                            Text("Switch to Game... \(countdownValue)")
                        } else {
                            Label("START SCALING", systemImage: "play.fill")
                        }
                    }
                    .font(.headline)
                    .padding()
                    .frame(maxWidth: .infinity)
                }
                .buttonStyle(.borderedProminent)
                .tint(.blue)
                .controlSize(.large)
                .disabled(isCountingDown)
                
                if !isCountingDown {
                    Text("Click Start, then switch to your game window immediately.")
                        .font(.caption)
                        .foregroundColor(.secondary)
                        .multilineTextAlignment(.center)
                }
            }
            
            Divider()
            
            Form {
                Picker("Scaling Type", selection: $settings.scalingMode) {
                    ForEach(CaptureSettings.ScalingMode.allCases) { mode in
                        Text(mode.rawValue).tag(mode)
                    }
                }
                
                HStack {
                    Text("Sharpness")
                    Slider(value: $settings.sharpenAmount, in: 0...1)
                }
                
                Toggle("Show FPS", isOn: $settings.showFPS)
            }
            .disabled(isRunning || isCountingDown)
        }
        .padding()
    }
}
