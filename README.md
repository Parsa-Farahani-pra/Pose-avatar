# Pose Avatar

A Python project that drives a 3D humanoid avatar using webcam-based pose estimation with MediaPipe and visualizes it in real-time using the Ursina 3D engine.

![python](https://img.shields.io/badge/python-3.8+-blue)
![mediapipe](https://img.shields.io/badge/mediapipe-0.10.14-orange)
![ursina](https://img.shields.io/badge/ursina-7.x-green)
![License](https://img.shields.io/badge/license-MIT-green)

## 📋 System Requirements

### Minimum Requirements
- **CPU**: Any modern processor (multi-core recommended for smooth performance)
- **Memory**: 4GB RAM minimum
- **Storage**: 500 MB free space
- **Graphics**: Built-in graphics adapter with OpenGL 3.3+ support

### Recommended Requirements
- **CPU**: 4+ cores for simultaneous processing of camera, pose/face/hand tracking, and 3D rendering
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
| **Python** | 3.8 - 3.13 | Core runtime (3.14 not supported by MediaPipe) |
| **OpenCV** | 4.8.0+ / 5.x | Camera capture and image processing |
| **MediaPipe** | 0.10.14 (exact) | Pose, hand, and face mesh landmark detection (`mp.solutions`) |
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
- **Hand & Finger Tracking** (`--enable-hands`): MediaPipe Hands drives an articulated hand rig —
  a palm capsule from wrist to middle-finger base plus 3 slender phalanx segments per finger
  (21 landmarks per hand), rendered only while the hand is actually visible
- **Face Mesh Tracking** (`--enable-face`): Full 468-landmark MediaPipe Face Mesh overlay on the 2D preview,
  with color-coded regions (jawline, eyes, eyebrows, lips, nose) and debug bounds display
- **Emotion Detection** (`--detect-emotion`): Real-time classification of 5 core emotions
  (Happy, Sad, Angry, Surprised, Neutral) from Face Mesh geometry with temporal smoothing
  and probability bars in the 3D HUD
- **Dual-Mode Operation**:
  - `pose_avatar.py`: Full 3D avatar with articulated hands, optional face mesh, optional emotion HUD,
    and optional camera preview window
  - `pose_webcam.py`: Standalone 2D viewer with pose/hand/face overlay
- **Smart Face Activation**: Face Mesh auto-activates when face ≥ 25% of frame height;
  Pose's 11 face landmarks auto-hide when FaceMesh is active
- **Smooth Animation**: Interpolated movement at 60 FPS rendering despite ~7.5 FPS camera input
- **Interactive Controls**: WASD/Arrow keys to move the avatar, R/F to raise/lower
- **Understanding HUD**: Visual feedback showing pose tracking status, body visibility,
  and optional emotion probability bars

## 📋 System Requirements

### Hardware
- **Webcam**: Any standard USB webcam (capable of 640x480 resolution minimum)
- **Processor**: CPU-based pose/face/hand estimation (optimized for multi-core CPUs)
- **Graphics**: Basic 3D graphics support (Ursina requires OpenGL 3.3+)
- **Memory**: 4GB+ RAM recommended

### Software
- **Python**: 3.8 or higher
- **Operating Systems**: Linux, macOS, Windows
- **Camera Access**: Camera permissions must be granted

## 🚀 Installation

### Step 1: Install Dependencies

> **⚠️ Python 3.14 is not supported.** MediaPipe's `mp.solutions.pose` API (used by
> this project) is only available in MediaPipe 0.10.14, which has no wheel for
> Python 3.14. Use **Python 3.8–3.13** instead.

Create a virtual environment and install required packages:

```bash
# Create virtual environment (Python 3.12 recommended)
python3.12 -m venv .venv

# Activate virtual environment
# On Linux/macOS:
source .venv/bin/activate
# On Windows:
.venv\Scripts\activate

# Install pinned, reproducible dependencies
pip install -r requirements/requirements-lock.txt
```

**Note**: MediaPipe is pinned to exactly `0.10.14` — this is the last version that
includes the `mp.solutions` submodule (removed in later 0.10.x patches and in 1.x).
Ursina 7.x provides the 3D rendering engine.

## 📖 Usage

### Run 3D Avatar with Camera Preview
Runs the 3D avatar alongside a camera preview window:

```bash
./scripts/run.sh pose_avatar.py --show-video
```

This opens TWO windows:
- **"Pose Avatar"** — The Ursina 3D scene with avatar and HUD
- **"pose"** — Live camera feed with 2D skeleton overlay

### Run 3D Avatar with Hand Tracking
Adds articulated hands — a palm plus individually bending finger segments — to the 3D avatar:

```bash
./scripts/run.sh pose_avatar.py --enable-hands --show-video
```

Hands only appear while they are visible to the camera (landmark visibility ≥ 0.7), and each
finger bends naturally with your real fingers. Left and right hands use identical geometry and
are colour-coded (teal = left, pink = right).

### Run 3D Avatar with Face Mesh
Shows the full 468-landmark Face Mesh overlay on the 2D preview window:

```bash
./scripts/run.sh pose_avatar.py --enable-face --show-video
```

### Run 3D Avatar with Emotion Detection
Shows real-time emotion classification (Happy, Sad, Angry, Surprised, Neutral) with probability
bars in the 3D HUD (requires face tracking):

```bash
./scripts/run.sh pose_avatar.py --enable-face --detect-emotion --show-video
```

### Run 3D Avatar with All Features

```bash
./scripts/run.sh pose_avatar.py --enable-hands --enable-face --detect-emotion --show-video
```

### Run 3D Avatar Only
Displays only the 3D avatar without camera preview:

```bash
./scripts/run.sh pose_avatar.py
```

Hand tracking can also be toggled here:

```bash
./scripts/run.sh pose_avatar.py --enable-hands
```

### Run Standalone 2D Skeleton
Displays only the 2D pose tracking overlay:

```bash
./scripts/run.sh pose_webcam.py
```

Hand landmarks and face mesh can be overlaid on the 2D skeleton too:

```bash
./scripts/run.sh pose_webcam.py --enable-hands --enable-face
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
| `--enable-hands` | false | Enable hand & finger tracking (palm capsule + articulated fingers on the avatar, 0.7 visibility threshold) |
| `--enable-face` | false | Enable full 468-landmark MediaPipe Face Mesh tracking (auto-activates at ≥25% frame height; hides Pose's 11 face landmarks while active) |
| `--detect-emotion` | false | Enable emotion detection from Face Mesh (requires `--enable-face`); shows 5-emotion probability bars in 3D HUD with temporal smoothing (~1s) |
| `--show-joints` | false | Draw joint spheres at each pose landmark |
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

5. **Hand Tracking**: With `--enable-hands`, hold your hand open and facing the camera; hands activate only when
   landmark visibility reaches 0.7, so a blurry or partially cropped hand simply stays hidden. The switch from
   pose to hand tracking is automatic per arm — no extra configuration needed.

6. **Face Tracking & Emotion**: With `--enable-face`, the full 468-landmark Face Mesh is available. It automatically
   engages when the detected face occupies ≥25% of frame height. With `--detect-emotion`, a real-time
   5-emotion HUD appears at bottom-left (Happy/Sad/Angry/Surprised/Neutral probability bars).
   The HUD shows "Face too small" when the face is below the size threshold and "No face detected"
   when tracking is lost.

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
# Use scripts/run.sh which includes src/ path automatically
./scripts/run.sh pose_avatar.py --show-video --min-visibility 0
```

### Running Tests

Run the integration test for the full feature pipeline:
```bash
./scripts/run.sh tests/test_integration.py
```

Run individual hand tracking tests:
```bash
./scripts/run.sh tests/hand_analysis/test_hand_landmarks.py
./scripts/run.sh tests/hand_analysis/test_hands_compatibility.py
```

Run emotion detection debug/test:
```bash
./scripts/run.sh tests/emotion_detection/test_emotion_debug.py
```

## 📝 Project Structure

```
.
├── README.md                   # This file
├── .gitignore                  # Git ignore rules
├── requirements/               # 📦 Dependency files
│   ├── requirements.txt
│   └── requirements-lock.txt
├── src/                        # 🐍 Python source code
│   ├── pose_avatar.py          # Main 3D avatar program (pose + hands + face + emotion)
│   ├── pose_webcam.py          # Standalone 2D pose/hand/face tracker
│   ├── face_tracker.py         # MediaPipe Face Mesh tracking module
│   ├── hand_tracker.py         # MediaPipe Hands tracking module
│   └── emotion_detection.py    # 5-emotion classifier from FaceMesh landmarks
├── tests/                      # ✅ Test suite (push to main)
│   ├── test_integration.py                      # Full pipeline integration test
│   ├── hand_analysis/                           # Hand tracking tests
│   │   ├── test_hand_landmarks.py              # Visualize 21 hand landmarks & finger connections
│   │   └── test_hands_compatibility.py         # Verify mp.solutions.hands availability
│   └── emotion_detection/                       # Emotion detection tests
│       └── test_emotion_debug.py               # Debug emotion detection logic
├── scripts/                    # 🚀 Shell scripts
│   └── run.sh                  # Execution helper (auto-resolves src/ paths)
├── LICENSE (optional)
└── README.old                  # Original documentation
```

`src/face_tracker.py` and `src/hand_tracker.py` are the single sources of truth for landmark indices
and drawing helpers used by both programs. `src/emotion_detection.py` consumes FaceMesh landmarks
and classifies 5 emotions via geometric features (EAR, MAR, mouth curve, brow slope, etc.) with softmax + EMA smoothing.

## 🤝 Contributing

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
pip install -r requirements/requirements-lock.txt
```

### `AttributeError: module 'mediapipe' has no attribute 'solutions'`
- **Cause**: You are using Python 3.14, or an incompatible mediapipe version. MediaPipe's `mp.solutions.pose` API is only available in version 0.10.14 and was removed in later 0.10.x patches (0.10.20+) and all 1.x versions.
- **Solution**: Recreate the venv with Python 3.8–3.13 and install the locked dependencies:
```bash
python3.12 -m venv .venv && .venv/bin/pip install -r requirements/requirements-lock.txt
```

### Avatar appears oversized/undersized
- **Solution**: Use `--scale` parameter to adjust avatar size

### Poor tracking performance
- **Solution**: Increase `--model-complexity` to 1 or 2 for more accurate detection, or use `--require-full-body` to ensure all joints are visible

### Hands not appearing
- **Cause**: `--enable-hands` was not passed, or the hand is barely visible / too close to the camera
- **Solution**: Enable the flag explicitly and step back so the whole hand is in frame (hands lock on at visibility ≥ 0.7). Hand tracking only engages when the corresponding arm is visible in the pose results.

### Face mesh not appearing
- **Cause**: Face is smaller than 25% of frame height, or `--enable-face` not passed
- **Solution**: Move closer to camera (face ≥25% of frame height triggers auto-activation) and ensure flag is passed

### Emotion HUD not appearing
- **Cause**: Face not detected/face too small, or `--detect-emotion` requires `--enable-face`
- **Solution**: Ensure both `--enable-face` and `--detect-emotion` are passed. HUD shows "Face too small" or "No face detected" when applicable.

## 📊 Performance Metrics

- **Camera FPS**: ~7.5 FPS
- **Render FPS**: 60 FPS (smooth interpolation)
- **Inference Time**: ~30-50ms (depending on CPU)
- **Memory Usage**: ~200-400 MB

## 🎯 Future Enhancements

Potential improvements for future versions:
- [ ] Full 3D avatar mesh (replace the capsule rig with a skinned model)
- [ ] Support for multi-person tracking
- [ ] Export pose data to 3D animation files
- [ ] VR/AR device support
- [ ] Custom avatar models and textures
- [ ] Machine learning-based motion retargeting

## 📜 Changelog

### v0.3.0 — Face Mesh + Emotion Detection
- Added MediaPipe Face Mesh integration via new `face_tracker.py` module
  (468 landmarks, color-coded face oval/eyes/brows/lips/nose, debug bounds)
- New `--enable-face` flag for both `pose_avatar.py` and `pose_webcam.py`
- Face mesh auto-activates at ≥25% frame height; Pose's 11 face landmarks
  auto-hidden while FaceMesh active
- Added `emotion_detection.py` — standalone geometric emotion classifier
  computing 12 features (EAR, MAR, mouth curve, brow-eye distance, etc.)
  with softmax + 1s EMA smoothing into 5 emotions (Happy/Sad/Angry/Surprised/Neutral)
- New `--detect-emotion` flag for `pose_avatar.py` — shows real-time probability
  bars in bottom-left 3D HUD with confidence threshold (60%)
- Removed deprecated `FaceAvatarTracker` and `MinimalFaceAnchor` yellow sphere

### v0.2.0 — Hand & Finger Tracking
- Added MediaPipe Hands integration via the new `hand_tracker.py` module
  (21 landmarks per hand, finger chains, reference geometry calibration)
- New `--enable-hands` flag for both `pose_avatar.py` and `pose_webcam.py`
- The 3D avatar now renders articulated hands: a symmetrical palm capsule
  (wrist → middle-finger base) plus 3 phalanx segments per finger, driven
  at the same 60 FPS interpolated rate as the body
- Hands activate per arm only while visible (visibility ≥ 0.7)
- New test tools: `test_hand_landmarks.py` and `test_hands_compatibility.py`

### v0.1.0 — Initial Release
- Real-time 3D avatar driven by MediaPipe Pose (capsule rig, HUD, camera preview)
- Standalone 2D skeleton viewer (`pose_webcam.py`)

## 📧 Support

For issues, questions, or suggestions, please submit an issue on the project repository.

---

**Enjoy creating your digital twin!** 🦾