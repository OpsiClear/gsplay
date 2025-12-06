# Changelog

All notable changes to gsplay are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.1] - 2025-12-06

### Added

- **GStream Viewer**: Minimal streaming viewer accessible at `http://host:viser_port+1/`
  - Rotation controls (CW/CCW/Stop) with keyboard shortcuts (Left/Right/Escape)
  - Play/Pause animation control with Space key
  - Fullscreen toggle with F key
  - Clean, distraction-free interface for presentations

- **HTTP Control Server**: Remote control API at `http://host:viser_port+2/`
  - `POST /rotate-cw` - Start clockwise camera rotation
  - `POST /rotate-ccw` - Start counter-clockwise camera rotation
  - `POST /rotate-stop` - Stop camera rotation
  - `POST /play` - Start animation playback
  - `POST /pause` - Pause animation playback
  - `POST /toggle-playback` - Toggle animation state
  - `POST /center-scene` - Center scene at origin
  - `POST /get-state` - Get current viewer state
  - `POST /set-translation` - Set scene translation

- **WebSocket Streaming**: Low-latency binary JPEG streaming (~100-150ms latency)
  - Push-based delivery (no polling)
  - Automatic reconnection on disconnect
  - Frame rate controlled by viewer settings

### Changed

- **Architecture Improvements** (SOLID refactoring):
  - Extracted `EventDispatcher` to `src/gsplay/dispatch/dispatcher.py` for centralized event routing
  - Extracted `UISetup` to `src/gsplay/initialization/ui_setup.py` for UI callback management
  - Extracted rotation utilities to `src/gsplay/config/rotation_conversions.py`
  - Reduced `app.py` from 1,746 to 1,544 lines (-12%)
  - Reduced `ui_handles.py` from 936 to 702 lines (-25%)

- **Camera Auto-Rotation**: Now properly updates streaming output when rotation is active

### Fixed

- Streaming not updating during auto-rotation when playback is paused
- Play/Pause API now correctly starts/stops animation via PlaybackController

## [0.1.0] - 2025-11-25

### Added

- Initial release with core functionality
- Real-time PLY sequence playback
- Web-based launcher for multi-instance management
- Interactive camera controls (rotate, pan, zoom)
- Color adjustment controls (brightness, contrast, saturation, etc.)
- Scene transformation controls (translate, rotate, scale)
- Volume filtering (opacity, scale, spatial filters)
- Frame export with edit baking
- gsmod integration for GPU-accelerated color/transform operations
- Clean Architecture with domain/infrastructure/models separation

### Technical Details

- Python 3.12+ with type hints
- PyTorch 2.9+ with CUDA 12.8 support
- viser for web-based 3D UI
- gsplat for GPU-accelerated Gaussian splatting
- WebSocket-based streaming with websockets library
