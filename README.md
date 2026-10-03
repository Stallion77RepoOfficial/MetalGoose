<div align="center">
  <img src="Assets/logo.png" alt="MetalGoose Logo" width="128" height="128">
  
  # MetalGoose
  
  **GPU-accelerated upscaling and frame generation for macOS**
  
  [![macOS](https://img.shields.io/badge/macOS-27.0%2B-blue?logo=apple)](https://www.apple.com/macos/)
  [![Metal](https://img.shields.io/badge/Metal-4.1-orange?logo=apple)](https://developer.apple.com/metal/)
  [![License](https://img.shields.io/badge/License-GPL--3.0-green)](LICENSE)
  [![Swift](https://img.shields.io/badge/Swift-6.4-FA7343?logo=swift)](https://swift.org)
  
  [Features](#features) • [Installation](#installation) • [Usage](#usage) • [Requirements](#requirements) • [Building](#build-from-source) • [License](#license)
</div>

---

## Overview

MetalGoose captures a window with ScreenCaptureKit, runs MetalFX spatial
upscaling and frame generation over the captured frames, and presents the result
in a borderless overlay pinned to the source window. It works on any window, not
only games — anything that renders faster than it is being watched.

## Features

### MGUP-1 Upscaling
- MetalFX Spatial upscaling to the overlay's size. **Scale Factor** (1.0x–10.0x, or Fullscreen)
  sets the overlay size; **Render Scale** (Native, 75%, 67%, 50%, 33%) lowers the resolution
  ScreenCaptureKit delivers.
- **Sharpening** — Light, Balanced or Strong: contrast-adaptive sharpening (CAS) strength and
  anti-aliasing sensitivity.

### MGFG-1 Frame Generation
MGFG-1 makes the images between two captures by interpolation. The Neural Engine does it where it can, and MetalFX on
the GPU where it cannot, and a single choice is made for every capture so that the two work as one. The HUD's
**Frame Gen** row names the engine in use and what it delivers.

- **Neural Engine** — VideoToolbox low-latency frame interpolation, for 2x and 4x. It leaves the GPU to the captured
  app: the GPU converts the capture to 4:2:0 once, and turns an image into colour only as it is shown. It works at the
  largest size it takes (1920 px on a side, 2.07 MP): a larger capture is shrunk for it and its images are enlarged as
  they are blended with the captures, so a 1440p or 4K window is covered too. Where the capture rate leaves no time for
  that size, it works at a coarser one — 1280×720 or 960×540 — rather than giving way. A size it has not been
  used at takes a second or two to prepare, the first time; the session that is serving goes on until the new one has
  started, and MetalFX stands in meanwhile.
- **MetalFX** — interpolation with the media engine's motion field, for 2x where the Neural Engine cannot be used or
  cannot keep up. It is the more faithful of the two, makes the midpoint only, and uses GPU time.
- Where neither can make its images before the next capture is due, nothing is held back: the captures are shown as
  they arrive.
- **Multiplier** — images presented per captured frame: 2x (the midpoint of each pair) or 4x (its quarters, on the
  Neural Engine). A multiplier the panel cannot show is not made (4x at 30 captures a second on a 60 Hz panel is 2x),
  and 4x falls back to 2x where the quarters do not fit the time between captures.
- **Interface and text stay as captured** — where the two captures barely differ (an interface, text, a still
  background) they are blended back into the generated image, which keeps what did not move: the Neural Engine's
  images are lossy there, and MetalFX's gain a little.

Interpolation holds the newest capture back by most of a capture interval plus the time the first generated image
takes to make (some 40 to 50 ms at 30 captures a second). That delay is eased where it falls, so that a change of
engine or load does not step the motion on the screen.

An engine is left at once when it stops keeping up, and not tried again for 30 seconds; a better one is taken only when
it would keep up with room to spare and the choice has stood for 10 seconds, so a rate near a limit does not move the
engine back and forth. With Render Scale below 100%, generation works on the reduced capture and the final scale-up
treats captured and generated frames alike; only the Neural Engine takes the window's own size, when that fits it,
which measured closer to the real image. Scene-cut detection avoids generating across hard cuts.

An image is never presented twice, so **Generated + Passthrough = Presented** in the HUD.

### Anti-Aliasing
Post-process anti-aliasing that runs on the final captured image, with no need
for depth buffers or motion vectors:
- **FXAA** — Fast approximate anti-aliasing (relative edge threshold + subpixel pass)
- **SMAA** — Morphological AA: pattern-based edge blending with local contrast adaptation and sharp-corner preservation

### Performance Monitoring
A HUD overlay reports, live:
- **Capture / Output / Generated** frame rates, the **Target** output (capture rate × the multiplier
  in use, which Output is coloured against), and the panel's refresh rate
- Capture time, GPU time, **GPU load** (the pipeline's own share of the GPU), latency, present
  latency, end-to-end latency, and a frame-pacing score
- VRAM, process memory, and CPU
- Cumulative counters: Captured, Presented, Generated, Passthrough, Dropped

## Requirements

| Component | Requirement |
|-----------|-------------|
| **macOS** | 27.0 or later |
| **Chip** | Apple Silicon (M1/M2/M3/M4) |
| **Xcode** | 27 or later (macOS 27 SDK) |
| **Swift** | 6.4 toolchain, Swift 6 language mode |
| **RAM** | 8 GB minimum, 16 GB recommended |

## Installation

### Download Release
1. Download the latest release from [Releases](https://github.com/Stallion77RepoOfficial/MetalGoose/releases)
2. Move `MetalGoose.app` to `/Applications`
3. Open `Terminal` and type `xattr -dr com.apple.quarantine /Applications/MetalGoose.app`
4. Grant Screen Recording (and Accessibility, if Capture Cursor is on) when prompted

### Build from Source
```bash
git clone https://github.com/Stallion77RepoOfficial/MetalGoose
cd MetalGoose
open MetalGoose.xcodeproj
```

## Usage

1. Launch MetalGoose and grant Screen Recording access (and Accessibility while Capture Cursor is on).
2. Configure upscaling (MGUP-1), frame generation (MGFG-1), and anti-aliasing. Changes apply
   to a running session.
3. Switch to the window you want to capture — it has to be frontmost, since
   MetalGoose targets whichever app is in front when scaling starts.
4. Press `⌘⇧T`, or return to MetalGoose and click **Start Scaling**.

### Keyboard Shortcuts

| Shortcut | Action |
|----------|--------|
| `⌘ + ⇧ + T` | Start or stop scaling |
| `⌘ + ⇧ + C` | Show or hide the cursor sprite |

Both are global and outlive the main window, so closing it with `⌘W` leaves the
overlay running and `⌘⇧T` still stops it.

## Error Codes

All error codes are shown as an in-app alert.

### UI (MG-UI)
- MG-UI-001: Frontmost app is MetalGoose; user must switch to target window.
- MG-UI-002: Target window not found for the selected app.
- MG-UI-004: No display found.
- MG-UI-005: Display ID not found for target screen.
- MG-UI-006: Display refresh rate unavailable for target screen.
- MG-UI-007: A global shortcut (`⌘⇧T` or `⌘⇧C`) is already registered by another app.

### Capture (MG-CAP)
- MG-CAP-001: Target window not found by ScreenCaptureKit.
- MG-CAP-002: ScreenCaptureKit start error.
- MG-CAP-003: ScreenCaptureKit stop error.
- MG-CAP-004: Stream stopped with error.
- MG-CAP-005: Target entered macOS fullscreen — use windowed or borderless (windowed fullscreen) mode.
- MG-CAP-007: Capture reconfiguration failed when applying a new render scale.

### Engine (MG-ENG)
- MG-ENG-001: Metal pipeline setup failed.
- MG-ENG-002: Metal device not available.
- MG-ENG-003: Metal command queue not available.
- MG-ENG-004: MetalFX Spatial Scaler creation failed.
- MG-ENG-005: Anti-aliasing pipeline unavailable.
- MG-ENG-007: CAS pipeline unavailable.
- MG-ENG-008: IOSurface texture creation failed.
- MG-ENG-010: MetalFX Frame Interpolator creation failed.

Codes are identifiers and are not renumbered when one is retired, so the lists have gaps.

## License

This project is licensed under the GNU General Public License v3.0 - see the [LICENSE](LICENSE) file for details.

## References

Apple documentation this project was built against:

- [Metal](https://developer.apple.com/documentation/metal) and
  [compute passes](https://developer.apple.com/documentation/metal/compute-passes)
- [MetalFX](https://developer.apple.com/documentation/metalfx/) — spatial scaling
  and frame interpolation
- [ScreenCaptureKit](https://developer.apple.com/documentation/screencapturekit/)
  and [capturing screen content in macOS](https://developer.apple.com/documentation/ScreenCaptureKit/capturing-screen-content-in-macos)
- [MTLTexture](https://developer.apple.com/documentation/metal/mtltexture) and
  [CVPixelBuffer](https://developer.apple.com/documentation/corevideo/cvpixelbuffer)
  — the IOSurface-backed path between capture and render
- [CAMetalDisplayLink](https://developer.apple.com/documentation/quartzcore/cametaldisplaylink)
  — the frame clock the render thread runs on
- [VideoToolbox](https://developer.apple.com/documentation/videotoolbox) — motion estimation on
  the media engine and low-latency frame interpolation on the Neural Engine
- [AppKit](https://developer.apple.com/documentation/appkit) — the overlay window

<div align="center">
  <sub>Built with ❤️ using Metal for macOS</sub>
</div>
