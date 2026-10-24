<<<<<<< HEAD
# Pose Avatar

A Python project that drives a 3D humanoid avatar using webcam-based pose estimation with MediaPipe and visualizes it in real-time using the Ursina 3D engine.

![python](https://img.shields.io/badge/python-3.8%2B-blue)
![mediapipe](https://img.shields.io/badge/mediapipe-0.10.x-orange)
![ursina](https://img.shields.io/badge/ursina-7.x-green)
![License](https://img.shields.io/badge/license-MIT-green)

## 📋 System Requirements

### Minimum Requirements
- **CPU**: Any modern processor (multi-core recommended for smooth performance)
- **Memory**: 4GB RAM minimum
- **Storage**: 500 MB free space
- **Graphics**: Built-in graphics adapter with OpenGL 3.3+ support

### Recommended Requirements
- **CPU**: 4+ cores for simultaneous processing of camera, pose estimation, and 3D rendering
- **Memory**: 8GB+ RAM
- **Storage**: SSD recommended
- **Graphics**: Dedicated GPU with WebGL 2.0+ support

### Operating System Support
- **Linux**: Ubuntu 18.04+, Debian 10+, or newer Linux distributions
  - Requires v4l2 camera module (standard in most Linux kernels)
- **macOS**: macOS 10.14+ (with camera permissions)
- **Windows**: Windows 10 or Windows 11

### Dependencies
| Component | Required Version | Purpose |
|-----------|-----------------|---------|
| **Python** | 3.8 - 3.12 | Core runtime |
| **OpenCV** | 4.8.0+ / 5.x | Camera capture and image processing |
| **MediaPipe** | 0.10.14 - 0.10.22 | Pose landmark detection |
| **Ursina** | 7.0.0 - 7.x | 3D rendering engine |
| **NumPy** | 1.24.0+ | Numerical operations |

### Graphics Card Requirements
- **OpenGL**: 3.3 or higher
- **WebGL**: 2.0 or higher (for browsers)
- **Vulkan**: 1.1+ (optional, for advanced rendering)

### Camera requirements
- Any USB webcam with 640x480 resolution or higher
- No special driver requirements (standard V4L2)
- Supports at least 30 FPS for smooth tracking

## 🌟 Features

- **Real-time 3D Avatar**: Displays a 3D humanoid avatar that mirrors your webcam poses
- **Dual-Mode Operation**:
  - `pose_avatar.py`: Full 3D avatar with optional camera preview window
  - `pose_webcam.py`: Standalone 2D skeleton overlay (no 3D graphics)
- **Smooth Animation**: Interpolated movement at 60 FPS rendering despite ~7.5 FPS camera input
- **Interactive Controls**: WASD/Arrow keys to move the avatar, R/F to raise/lower
- **Undertstanding HUD**: Visual feedback showing pose tracking status and body visibility

## 📋 System Requirements

### Hardware
- **Webcam**: Any standard USB webcam (capable of 640x480 resolution minimum)
- **Processor**: CPU-based pose estimation (optimized for multi-core CPUs)
- **Graphics**: Basic 3D graphics support (Ursina requires OpenGL 3.3+)
- **Memory**: 4GB+ RAM recommended

### Software
- **Python**: 3.8 or higher
- **Operating Systems**: Linux, macOS, Windows
- **Camera Access**: Camera permissions must be granted

## 🚀 Installation

### Step 1: Install Dependencies

Create a virtual environment and install required packages:

```bash
# Create virtual environment
python3 -m venv .venv

# Activate virtual environment
# On Linux/macOS:
source .venv/bin/activate
# On Windows:
.venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

**Note**: The project uses `mediapipe>=0.10.14,<0.11` which includes the `mp.solutions.pose` API that was removed in mediapipe 1.x. It also requires `ursina>=7.0.0,<8` for the 3D rendering engine.

## 📖 Usage

### Run 3D Avatar with Camera Preview
Runs the 3D avatar alongside a camera preview window:

```bash
./run.sh pose_avatar.py --show-video
```

This opens TWO windows:
- **"Pose Avatar"** - The Ursina 3D scene with avatar and HUD
- **"pose"** - Live camera feed with 2D skeleton overlay

### Run 3D Avatar Only
Displays only the 3D avatar without camera preview:

```bash
./run.sh pose_avatar.py
```

### Run Standalone 2D Skeleton
Displays only the 2D pose tracking overlay:

```bash
./run.sh pose_webcam.py
```

**Important**: This must run INSTEAD of `pose_avatar.py` because the webcam can only be opened by one process at a time.

## 🎮 Controls

In the **"Pose Avatar"** window:

| Key        | Action                               |
|------------|--------------------------------------|
| W / Up     | Move avatar away from camera         |
| S / Down   | Move avatar toward camera            |
| A / Left   | Move avatar left                     |
| D / Right  | Move avatar right                    |
| R / F      | Raise/lower avatar                   |
| Q / Esc    | Quit application                     |

## 🔧 Command-Line Options

| Option | Default | Description |
|--------|---------|-------------|
| `--show-video` | false | Opens camera preview window with skeleton overlay |
| `--window-width` | 1280 | Width of the 3D window in pixels |
| `--window-height` | 720 | Height of the 3D window in pixels |
| `--camera` | 0 | Camera device index |
| `--min-visibility` | 0.25 | Landmark confidence threshold (lower = more visible) |
| `--scale` | 1.0 | Avatar size multiplier |
| `--smooth-tau` | 0.15 | Landmark smoothing time constant (seconds) |
| `--model-complexity` | 0 | MediaPipe model size (0=fastest, 2=most accurate) |
| `--exit-if-lost` | 0 | Exit after tracking is lost for X seconds (0=never) |
| `--no-flip` | false | Disable camera mirroring |

## 💡 Tips

1. **Body Visibility**: This webcam only captures head + shoulders by default. Step back until your legs are visible to fill the "legs" meter in the HUD.

2. **Frame Rate**: The camera is capped at ~7.5 FPS, but the avatar interpolates between frames to maintain smooth 60 FPS rendering.

3. **Positioning**: Use window positioning flags to arrange multiple windows on your desktop:
   - `--window-x`, `--window-y`: 3D window position
   - `--preview-x`, `--preview-y`: Camera preview window position

4. **Full Body Detection**: Use `--require-full-body` flag to only activate the avatar once all body parts are visible.

## 🧪 Development & Testing

### Verify Installation

```bash
# Check Python installation
python --version

# Verify dependencies are installed
pip check

# Run compiled check (no actual execution)
python -m compileall -q .
```

### Debug Mode

Run with `--min-visibility` set to 0 to see all landmarks:
```bash
./run.sh pose_avatar.py --show-video --min-visibility 0
```

## 📝 Project Structure

```
.
├── pose_avatar.py          # Main 3D avatar program
├── pose_webcam.py          # Standalone 2D pose tracker
├── run.sh                  # Execution helper script
├── requirements.txt        # Python dependencies
├── .gitignore              # Git ignore rules
├── README.md               # This file
└── README.old              # Original documentation
```

## 🤝 Contributing

Contributions are welcome! Please ensure:
1. Your code passes `python -m compileall` checks
2. Variable names follow snake_case convention
3. Comments explain complex logic
4. All new features are documented

## 📄 License

This project is for educational and personal use. Please respect the licenses of:
- [MediaPipe](https://github.com/google/mediapipe)
- [Ursina Engine](https://github.com/panda3d/ursina)

## ❓ Troubleshooting

### Camera not found
- **Solution**: Check camera permissions and verify no other app is using the webcam
- **Alternative**: Use `--camera 1` to try another camera index

### Import errors
- **Solution**: Ensure you've activated the virtual environment and installed dependencies:
```bash
source .venv/bin/activate  # or .venv\Scripts\activate on Windows
pip install -r requirements.txt
```

### Avatar appears oversized/undersized
- **Solution**: Use `--scale` parameter to adjust avatar size

### Poor tracking performance
- **Solution**: Increase `--model-complexity` to 1 or 2 for more accurate detection, or use `--require-full-body` to ensure all joints are visible

## 📊 Performance Metrics

- **Camera FPS**: ~7.5 FPS
- **Render FPS**: 60 FPS (smooth interpolation)
- **Inference Time**: ~30-50ms (depending on CPU)
- **Memory Usage**: ~200-400 MB

## 🎯 Future Enhancements

Potential improvements for future versions:
- [ ] Support for multi-person tracking
- [ ] Export pose data to 3D animation files
- [ ] VR/AR device support
- [ ] Custom avatar models and textures
- [ ] Machine learning-based motion retargeting

## 📧 Support

For issues, questions, or suggestions, please submit an issue on the project repository.

---

**Enjoy creating your digital twin!** 🦾
=======
# Pose-avatar
A Python project that drives a 3D humanoid avatar using webcam-based pose estimation with MediaPipe and visualizes it in real-time using the Ursina 3D engine.
>>>>>>> fc1852787a8d25a3ebb13b40b43b82a134cf56d9
