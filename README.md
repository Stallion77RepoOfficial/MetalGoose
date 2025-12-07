<div align="center">
  <img src="Assets/logo.png" alt="MetalGoose Logo" width="128" height="128">
  
  # MetalGoose
  
  **GPU-accelerated upscaling and frame generation for macOS**
  
  [![macOS](https://img.shields.io/badge/macOS-26.0%2B-blue?logo=apple)](https://www.apple.com/macos/)
  [![Metal](https://img.shields.io/badge/Metal-4.0-orange?logo=apple)](https://developer.apple.com/metal/)
  [![License](https://img.shields.io/badge/License-GPL--3.0-green)](LICENSE)
  [![Swift](https://img.shields.io/badge/Swift-6.2-FA7343?logo=swift)](https://swift.org)
  
  [Features](#features) • [Installation](#installation) • [Usage](#usage) • [Requirements](#requirements) • [Building](#building) • [License](#license)
</div>

---

## Overview

MetalGoose is a native macOS application that provides real-time upscaling and frame generation for games and applications. Built entirely with Apple's Metal framework, it delivers a smooth, high-FPS experience similar to NVIDIA DLSS or AMD FSR, but designed specifically for macOS.

## Features

### MGUP-1 Upscaling
- **Performance Mode** — Fastest upscaling with minimal latency
- **Balanced Mode** — Optimal quality/performance ratio
- **Quality Mode** — Maximum visual fidelity
- Multiple render scales: Native, 75%, 67%, 50%, 33%
- Contrast-adaptive sharpening (CAS)

### MGFG-1 Frame Generation
- **2x, 3x, 4x** frame multipliers
- **Adaptive** or **Fixed** frame generation modes
- Motion-compensated interpolation
- Optical flow-based motion estimation
- Quality modes: Performance, Balanced, Quality

### Anti-Aliasing
- **FXAA** — Fast approximate anti-aliasing
- **SMAA** — Enhanced subpixel morphological AA
- **MSAA** — Multi-sample anti-aliasing
- **TAA** — Temporal anti-aliasing with history

### Performance Monitoring
- Real-time HUD overlay
- Capture/Output/Interpolated FPS tracking
- GPU time and frame time metrics
- VRAM usage monitoring
- Frame statistics

## Requirements

| Component | Requirement |
|-----------|-------------|
| **macOS** | 26.0 (Tahoe) or later |
| **Chip** | Apple Silicon (M1/M2/M3/M4)
| **Xcode** | 26.0 or later |
| **RAM** | 8 GB minimum, 16 GB recommended |

## Installation

### Download Release
1. Download the latest release from [Releases](https://github.com/Stallion77RepoOfficial/MetalGoose/releases)
2. Move `MetalGoose.app` to `/Applications`
3. Grant Screen Recording and Accessibility permissions when prompted

### Build from Source
```bash
git clone https://github.com/Stallion77RepoOfficial/MetalGoose
cd MetalGoose
open MetalGoose.xcodeproj
```

## Usage

1. **Launch MetalGoose**
2. **Select Target**
   - Choose a window or display to capture
3. **Configure Settings**
   - Enable upscaling (MGUP-1)
   - Enable frame generation (MGFG-1) 
   - Select anti-aliasing mode
4. **Start Scaling**
   - Click "Start" to begin processing

### Keyboard Shortcuts

| Shortcut | Action |
|----------|--------|
| `⌘ + T` | Toggle Scale |

## Architecture

```
MetalGoose/
├── ContentView.swift      # Main SwiftUI interface
├── DirectRenderer.swift   # Metal rendering pipeline
├── DirectEngineBridge.mm  # Objective-C++ bridge layer
├── Engine.mm              # Core C++ processing engine
├── Shaders.metal          # GPU compute shaders
├── MGHUD.swift           # Performance overlay
└── CaptureSettings.swift  # Settings management
```

### Technology Stack
- **SwiftUI** — Modern declarative UI
- **Metal 3** — GPU-accelerated processing
- **ScreenCaptureKit** — Low-latency screen capture
- **MetalPerformanceShaders** — Optimized GPU operations

## Performance Tips

1. **Use Balanced mode** for the best quality/performance ratio
2. **Lower render scale** if experiencing lag
3. **Enable VSync** to reduce tearing
4. **Reduce Latency** option available for competitive gaming
5. **Adaptive frame gen** automatically adjusts to game FPS

## Troubleshooting

| Issue | Solution |
|-------|----------|
| Black screen | Grant Screen Recording permission in System Settings |
| Low FPS | Lower render scale or disable frame generation |
| High latency | Enable "Reduce Latency" option |
| App not detecting windows | Restart MetalGoose and grant Accessibility permission |

## Contributing

Contributions are welcome! Please read our [Contributing Guidelines](CONTRIBUTING.md) before submitting a pull request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the GNU General Public License v3.0 - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Apple for the Metal framework and documentation
- The macOS gaming community for feedback and testing
- Contributors who helped improve the project

---

<div align="center">
  <sub>Built with ❤️ using Metal for macOS</sub>
</div>
