#!/usr/bin/env python3
"""Drive a 3D humanoid avatar from webcam pose estimation (MediaPipe -> Ursina).

Scene units are METRES: one Ursina unit == one metre, so MediaPipe's metric
world landmarks can be used directly (the previous version scaled them by 6
while sizing the camera for a ~10 unit tall figure, which is why the avatar
rendered off the top of the screen).

Windows:
  * Ursina 3D scene: a simple humanoid avatar (head, torso, capsule limbs)
    that mirrors your pose, standing on a floor grid.
  * Optional cv2 window (--show-video) with the raw feed + 2D skeleton overlay.
    The two windows are placed next to each other, so the avatar and the live
    skeleton are visible side by side.

  NOTE: the webcam can only be opened by one process at a time, so the
  standalone pose_webcam.py cannot run at the same time as this program -
  use --show-video instead.

Controls (in the Ursina window):
  W / Up      move avatar away from camera (into the screen)
  S / Down    move avatar toward camera
  A / Left    move avatar left
  D / Right   move avatar right
  R / F       raise / lower avatar
  q / Esc     quit
"""
from __future__ import annotations

import argparse
import math
import queue
import sys
import threading
import time
from dataclasses import dataclass, field

import cv2
import mediapipe as mp
from ursina import (
    Ursina, Entity, Vec3, color, camera, window, Text, application,
    held_keys, time as ursina_time,
)

# ---------------------------------------------------------------------------
# MediaPipe pose landmark indices
# ---------------------------------------------------------------------------
NOSE = 0
L_EAR, R_EAR = 7, 8
L_SHOULDER, R_SHOULDER = 11, 12
L_ELBOW, R_ELBOW = 13, 14
L_WRIST, R_WRIST = 15, 16
L_HIP, R_HIP = 23, 24
L_KNEE, R_KNEE = 25, 26
L_ANKLE, R_ANKLE = 27, 28
L_FOOT, R_FOOT = 31, 32

# Derived (synthetic) point keys, used by the rig for mid-points.
P_MID_SHOULDER = -1
P_MID_HIP = -2
P_HEAD = -3

# Joints that must be visible for the pose to count as "full body".
FULL_BODY_JOINTS = (L_KNEE, R_KNEE, L_ANKLE, R_ANKLE)
LEG_JOINTS = (L_KNEE, R_KNEE, L_ANKLE, R_ANKLE, L_FOOT, R_FOOT)

# MediaPipe Pose exposes 33 landmarks (0..32).
POSE_LANDMARK_COUNT = 33

# ---------------------------------------------------------------------------
# Scene constants  (1 unit = 1 metre)
# ---------------------------------------------------------------------------
GROUND_Y = 0.0
HIP_HEIGHT = 0.95          # height of the avatar's hip centre above the floor
SCALE = 1.0                # MediaPipe world landmarks are already metric
MOVE_SPEED = 1.5           # metres per second
REFERENCE_TORSO = 0.50     # metres; used to normalise the avatar's proportions
REFERENCE_SHOULDER = 0.38  # metres; fallback normaliser when hips are hidden

CAMERA_POS = Vec3(1.45, 1.42, 3.95)     # fixed 3/4 view of the "room"
CAMERA_TARGET = Vec3(0.0, 0.95, 0.0)
# NOTE: Ursina's fov is the HORIZONTAL field of view (see ursina/camera.py),
# so a horizontal fov would crop the body as soon as the window aspect ratio
# changes.  We therefore fix the VERTICAL field of view and derive the
# horizontal one from the current aspect ratio: the avatar stays framed no
# matter how the window is resized.
# 28 deg vertical covers the whole silhouette of a 1.75 m person (which
# subtends ~23.2 deg from this camera) with a comfortable margin top and
# bottom, so the head is never clipped.
CAMERA_VERTICAL_FOV = 28.0      # degrees
DEFAULT_WINDOW_SIZE = (1280, 720)

GRID_EXTENT = 15           # grid lines run from -GRID_EXTENT .. +GRID_EXTENT
GRID_SPACING = 1.0

# ---------------------------------------------------------------------------
# Avatar body proportions (metres)
# ---------------------------------------------------------------------------
HEAD_RADIUS = 0.115
NECK_RADIUS = 0.055
SHOULDER_RADIUS = 0.085
UPPER_ARM_RADIUS = 0.085
FOREARM_RADIUS = 0.070
HAND_RADIUS = 0.055
HIP_RADIUS = 0.105
THIGH_RADIUS = 0.115
SHIN_RADIUS = 0.090
TORSO_WIDTH = 0.34         # left <-> right
TORSO_DEPTH = 0.20         # front <-> back
FOOT_WIDTH = 0.09
FOOT_DEPTH = 0.07

# NOTE: Ursina's color.rgb() takes 0..1 floats; color.rgb32() takes 0..255
# ints.  Using rgb() with 0..255 values silently clamps everything to white.
COLOR_SKIN = color.rgb32(232, 196, 168)
COLOR_SHIRT = color.rgb32(64, 126, 196)
COLOR_PANTS = color.rgb32(58, 68, 94)
COLOR_SHOE = color.rgb32(38, 38, 48)
COLOR_JOINT = color.azure

# ---------------------------------------------------------------------------
# cv2 overlay colors (BGR) for the optional --show-video window
# ---------------------------------------------------------------------------
LANDMARK_COLOR = (0, 255, 0)
BONE_COLOR = (0, 200, 255)
POINT_RADIUS = 4
BONE_THICKNESS = 2

# Name of the cv2 preview window.
PREVIEW_WINDOW = "pose"

# ---------------------------------------------------------------------------
# Ursina must be instantiated at module level so Ursina can discover the
# module-level update() and input() hooks.
# ---------------------------------------------------------------------------
app = Ursina(title="Pose Avatar", borderless=False, vsync=True,
             editor_ui_enabled=False, size=DEFAULT_WINDOW_SIZE)


# ---------------------------------------------------------------------------
# Small vector helpers
# ---------------------------------------------------------------------------
def _lerp(a: Vec3, b: Vec3, t: float) -> Vec3:
    return Vec3(a.x + (b.x - a.x) * t,
                a.y + (b.y - a.y) * t,
                a.z + (b.z - a.z) * t)


def _clamp(v: float, lo: float, hi: float) -> float:
    return lo if v < lo else (hi if v > hi else v)


def _safe_normalize(v: Vec3, fallback: Vec3 | None = None) -> Vec3 | None:
    length = v.length()
    if length < 1e-6:
        return fallback
    return Vec3(v.x / length, v.y / length, v.z / length)


def _horizontal_fov_for_vertical(vertical_deg: float, aspect: float) -> float:
    """Ursina's ``camera.fov`` is horizontal; convert a vertical fov to it."""
    v = math.radians(vertical_deg)
    return math.degrees(2.0 * math.atan(math.tan(v / 2.0) * max(aspect, 0.1)))


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
@dataclass
class Config:
    camera_index: int = 0
    width: int = 640
    height: int = 480
    model_complexity: int = 0
    min_visibility: float = 0.25
    show_video: bool = False
    flip: bool = True
    show_joints: bool = False
    require_full_body: bool = False
    exit_if_lost: float = 0.0
    scale: float = SCALE
    smooth_tau: float = 0.15      # EMA time constant (seconds) for landmarks
    interpolate: bool = True
    extrapolate: float = 0.15    # how far past the newest sample we may predict
    move_speed: float = MOVE_SPEED
    hip_height: float = HIP_HEIGHT

    # Window placement (so the 3D scene and the cv2 preview sit next to each
    # other on a wide desktop instead of on top of each other).
    window_width: int = 1280
    window_height: int = 720
    window_x: int = 0
    window_y: int = 0
    preview_width: int = 640
    preview_height: int = 480
    preview_x: int | None = None      # None -> right of the 3D window
    preview_y: int | None = None      # None -> same top as the 3D window


# ---------------------------------------------------------------------------
# Pose sample produced by the background worker
# ---------------------------------------------------------------------------
@dataclass
class PoseSample:
    t: float                                   # monotonic timestamp
    points: dict[int, Vec3] = field(default_factory=dict)   # raw world landmarks
    vis: dict[int, float] = field(default_factory=dict)      # raw visibility
    image: list[tuple[float, float, float]] = field(default_factory=list)
    frame: object | None = None


# ---------------------------------------------------------------------------
# Background capture + inference thread
# ---------------------------------------------------------------------------
class PoseWorker(threading.Thread):
    """Reads the camera and runs MediaPipe off the render thread.

    The webcam on this machine only delivers ~7.6 fps, so the render loop must
    not block on inference. Results are pushed into a 1-slot queue; the newest
    result always wins and stale ones are dropped.
    """

    def __init__(self, cfg: Config):
        super().__init__(daemon=True, name="pose-worker")
        self.cfg = cfg
        self.results: queue.Queue[PoseSample] = queue.Queue(maxsize=1)
        self.stop_event = threading.Event()
        self.error: str | None = None
        self.camera_fps = 0.0
        self.inference_ms = 0.0
        self.frames = 0

    # -- queue helpers ------------------------------------------------------
    def _publish(self, sample: PoseSample) -> None:
        try:
            self.results.get_nowait()
        except queue.Empty:
            pass
        try:
            self.results.put_nowait(sample)
        except queue.Full:
            pass

    # -- main loop ----------------------------------------------------------
    def run(self) -> None:
        cap = cv2.VideoCapture(self.cfg.camera_index)
        if not cap.isOpened():
            self.error = f"Could not open camera {self.cfg.camera_index}."
            self._publish(PoseSample(t=time.monotonic()))
            return

        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.cfg.width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.cfg.height)

        pose = mp.solutions.pose.Pose(
            static_image_mode=False,
            model_complexity=self.cfg.model_complexity,
            smooth_landmarks=True,
            enable_segmentation=False,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )

        fps_window_start = time.monotonic()
        fps_window_frames = 0
        try:
            while not self.stop_event.is_set():
                ok, frame = cap.read()
                if not ok:
                    time.sleep(0.005)
                    continue

                if self.cfg.flip:
                    frame = cv2.flip(frame, 1)

                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                rgb.flags.writeable = False

                t_infer = time.monotonic()
                result = pose.process(rgb)
                self.inference_ms = (time.monotonic() - t_infer) * 1000.0

                sample = PoseSample(t=time.monotonic())
                world = result.pose_world_landmarks
                if world is not None:
                    for i, lm in enumerate(world.landmark):
                        sample.points[i] = Vec3(lm.x, lm.y, lm.z)
                        sample.vis[i] = float(getattr(lm, "visibility", 0.0))
                image = result.pose_landmarks
                if image is not None:
                    for lm in image.landmark:
                        sample.image.append((lm.x, lm.y,
                                             float(getattr(lm, "visibility", 0.0))))
                if self.cfg.show_video:
                    sample.frame = frame

                self._publish(sample)

                # rolling camera fps
                self.frames += 1
                fps_window_frames += 1
                now = time.monotonic()
                elapsed = now - fps_window_start
                if elapsed >= 0.5:
                    self.camera_fps = fps_window_frames / elapsed
                    fps_window_frames = 0
                    fps_window_start = now
        finally:
            pose.close()
            cap.release()

    def stop(self) -> None:
        self.stop_event.set()


# ---------------------------------------------------------------------------
# Landmark smoothing, graceful degradation and pose interpolation
# ---------------------------------------------------------------------------
class PoseTracker:
    """Turns a stream of noisy MediaPipe samples into a smooth pose.

    Three things happen here:

    1. Per-landmark exponential smoothing (time aware).
    2. Graceful degradation: a landmark that drops below ``min_visibility``
       keeps its last good position instead of disappearing, so the skeleton
       does not flicker.
    3. Pose interpolation: the two most recent snapshots are interpolated at
       render time. The webcam only delivers ~7.6 fps, so this is what turns
       a choppy 8 Hz puppet into continuous motion.
    """

    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.pos: dict[int, Vec3] = {}          # smoothed raw landmarks
        self.vis: dict[int, float] = {}         # smoothed visibility
        self.have: set[int] = set()             # landmarks ever seen

        self.snap_prev: dict[int, Vec3] = {}
        self.snap_curr: dict[int, Vec3] = {}
        self.t_prev: float | None = None
        self.t_curr: float | None = None

        self.body_scale = 1.0
        self.last_detection: float | None = None
        self.last_sample_t: float | None = None

    # -- ingest -------------------------------------------------------------
    def ingest(self, sample: PoseSample) -> None:
        if not sample.points:
            # A frame with no person in it: keep the last good positions.
            self.snap_prev = dict(self.snap_curr)
            self.t_prev = self.t_curr
            self.snap_curr = dict(self.snap_curr)
            self.t_curr = sample.t
            self.last_sample_t = sample.t
            return

        self.last_detection = sample.t
        self.last_sample_t = sample.t

        now = sample.t
        prev_t = self.t_curr
        dt = (now - prev_t) if prev_t is not None else 0.0
        dt = _clamp(dt, 1.0 / 240.0, 1.0)

        # time-aware EMA alpha
        alpha = 1.0 - math.exp(-dt / max(self.cfg.smooth_tau, 1e-3))

        raw: dict[int, Vec3] = {}
        for idx, p in sample.points.items():
            v = sample.vis.get(idx, 0.0)
            # visibility is smoothed with a slower constant so a single bad
            # frame cannot make a limb vanish
            vis_alpha = 1.0 - math.exp(-dt / 0.35)
            prev_v = self.vis.get(idx, v)
            self.vis[idx] = prev_v + (v - prev_v) * vis_alpha

            if v >= self.cfg.min_visibility:
                prev_p = self.pos.get(idx)
                if prev_p is None:
                    self.pos[idx] = p
                else:
                    self.pos[idx] = _lerp(prev_p, p, alpha)
                self.have.add(idx)
                raw[idx] = self.pos[idx]
            elif idx in self.pos:
                # Graceful degradation: hold the last good position instead of
                # letting the limb disappear.
                raw[idx] = self.pos[idx]
            # else: never seen reliably -> omit entirely, so MediaPipe's
            # hallucinated landmarks never reach the rig.

        self._update_body_scale(raw, dt)

        self.snap_prev = dict(self.snap_curr)
        self.t_prev = self.t_curr
        self.snap_curr = raw
        self.t_curr = now

    # -- body normalisation -------------------------------------------------
    def _update_body_scale(self, pts: dict[int, Vec3], dt: float) -> None:
        """Normalise the avatar's proportions.

        MediaPipe's world landmarks are metric but noticeably noisy in the
        torso length. We slowly adapt a scale factor so the body always
        measures a canonical size, which keeps the avatar's limbs
        proportional and its feet near the floor.

        The torso length (mid-shoulder -> mid-hip) is the best measure but it
        needs the hips to be visible. When only the upper body is in frame we
        fall back to the shoulder width, which is nearly always detectable.
        """
        def ok(*idxs):
            return all(self.vis.get(i, 0.0) >= self.cfg.min_visibility
                       for i in idxs)

        target = None
        if ok(L_SHOULDER, R_SHOULDER, L_HIP, R_HIP):
            ms = (pts.get(L_SHOULDER) + pts.get(R_SHOULDER)) * 0.5
            mh = (pts.get(L_HIP) + pts.get(R_HIP)) * 0.5
            torso = (ms - mh).length()
            if 0.15 < torso < 1.2:
                target = torso / REFERENCE_TORSO
        if target is None and ok(L_SHOULDER, R_SHOULDER):
            width = (pts.get(L_SHOULDER) - pts.get(R_SHOULDER)).length()
            if 0.12 < width < 0.9:
                target = width / REFERENCE_SHOULDER

        if target is None:
            return
        target = _clamp(target, 0.55, 1.9)
        # slow adaptation (time constant ~2.5 s)
        beta = 1.0 - math.exp(-dt / 2.5)
        self.body_scale += (target - self.body_scale) * beta

    # -- render-time sampling ----------------------------------------------
    def _interpolated(self, now: float) -> dict[int, Vec3]:
        if not self.snap_curr:
            return {}
        if not self.cfg.interpolate or self.t_prev is None or self.t_curr is None:
            return dict(self.snap_curr)
        span = self.t_curr - self.t_prev
        if span <= 1e-4:
            return dict(self.snap_curr)
        alpha = (now - self.t_prev) / span
        alpha = _clamp(alpha, 0.0, 1.0 + max(self.cfg.extrapolate, 0.0))
        out: dict[int, Vec3] = {}
        for idx, curr in self.snap_curr.items():
            prev = self.snap_prev.get(idx)
            out[idx] = curr if prev is None else _lerp(prev, curr, alpha)
        return out

    def scene_points(self, now: float, offset: Vec3) -> dict[int, Vec3]:
        """Interpolated landmarks converted to Ursina scene coordinates."""
        raw = self._interpolated(now)
        if not raw:
            return {}

        hip = raw.get(L_HIP)
        hip2 = raw.get(R_HIP)
        if hip is not None and hip2 is not None:
            hip_mid = (hip + hip2) * 0.5
        else:
            hip_mid = Vec3(0, 0, 0)

        s = self.cfg.scale * self.body_scale
        out: dict[int, Vec3] = {}
        for idx, p in raw.items():
            out[idx] = Vec3(
                -(p.x - hip_mid.x) * s + offset.x,
                -(p.y - hip_mid.y) * s + offset.y,
                -(p.z - hip_mid.z) * s + offset.z,
            )

        # derived points
        ls, rs = out.get(L_SHOULDER), out.get(R_SHOULDER)
        lh, rh = out.get(L_HIP), out.get(R_HIP)
        if ls is not None and rs is not None:
            out[P_MID_SHOULDER] = (ls + rs) * 0.5
        if lh is not None and rh is not None:
            out[P_MID_HIP] = (lh + rh) * 0.5
        elif P_MID_SHOULDER in out:
            # MediaPipe's world landmarks are ALWAYS hip-centred by
            # construction, so the origin of the pose is a reliable stand-in
            # for the hip centre even when the hip landmarks themselves are
            # not detected. This keeps the torso drawn while the user only has
            # their upper body in frame.
            out[P_MID_HIP] = Vec3(offset.x, offset.y, offset.z)

        head = self._head_position(out)
        if head is not None:
            out[P_HEAD] = head
        return out

    def _head_position(self, pts: dict[int, Vec3]) -> Vec3 | None:
        le, re_ = pts.get(L_EAR), pts.get(R_EAR)
        if le is not None and re_ is not None:
            return (le + re_) * 0.5
        nose = pts.get(NOSE)
        if nose is not None:
            # MediaPipe's nose sits on the face; nudge it back towards the
            # body axis and up so the head sphere is roughly centred.
            mid = pts.get(P_MID_SHOULDER)
            head = Vec3(nose.x, nose.y + 0.04, nose.z)
            if mid is not None:
                head = _lerp(head, mid, 0.10)
            return head
        return None

    # -- status helpers -----------------------------------------------------
    def full_body(self) -> bool:
        return all(self.vis.get(i, 0.0) >= self.cfg.min_visibility
                   for i in FULL_BODY_JOINTS)

    def leg_quality(self) -> float:
        vals = [self.vis.get(i, 0.0) for i in LEG_JOINTS]
        if not vals:
            return 0.0
        return sum(vals) / len(vals)

    def tracking(self) -> bool:
        if self.last_detection is None:
            return False
        return (time.monotonic() - self.last_detection) < 0.6


# ---------------------------------------------------------------------------
# Avatar rig primitives
# ---------------------------------------------------------------------------
class Capsule:
    """An elongated sphere stretched between two landmarks."""

    def __init__(self, a: int, b: int, radius: float, col, parent=None):
        self.a, self.b = a, b
        self.radius = radius
        self.pivot = Entity(parent=parent)
        self.mesh = Entity(parent=self.pivot, model="sphere", color=col)

    def set(self, pts: dict[int, Vec3]) -> None:
        pa, pb = pts.get(self.a), pts.get(self.b)
        if pa is None or pb is None:
            self.pivot.enabled = False
            return
        length = (pb - pa).length()
        if length < 1e-4:
            self.pivot.enabled = False
            return
        self.pivot.enabled = True
        self.pivot.position = (pa + pb) * 0.5
        self.pivot.look_at(pb)
        # Ursina's sphere model has a radius of 0.5 at scale 1.
        self.mesh.scale = (self.radius * 2.0, self.radius * 2.0, length)


class Sphere:
    """A small sphere centred on a single landmark."""

    def __init__(self, a: int, radius: float, col, parent=None):
        self.a = a
        self.radius = radius
        self.entity = Entity(parent=parent, model="sphere", color=col)

    def set(self, pts: dict[int, Vec3]) -> None:
        p = pts.get(self.a)
        if p is None:
            self.entity.enabled = False
            return
        self.entity.enabled = True
        self.entity.position = p
        self.entity.scale = (self.radius * 2.0, self.radius * 2.0, self.radius * 2.0)


class Box:
    """A box stretched between two landmarks.

    The box is oriented so that its local +Z runs along the segment, its
    local +Y is the body's width axis and its local +X is the depth axis.
    ``size`` is therefore ``(depth, width)`` and the length comes from the
    distance between the two landmarks.
    """

    def __init__(self, a: int, b: int, size: tuple[float, float], col, parent=None):
        self.a, self.b = a, b
        self.size = size
        self.pivot = Entity(parent=parent)
        self.mesh = Entity(parent=self.pivot, model="cube", color=col)

    def set(self, pts: dict[int, Vec3]) -> None:
        pa, pb = pts.get(self.a), pts.get(self.b)
        if pa is None or pb is None:
            self.pivot.enabled = False
            return
        length = (pb - pa).length()
        if length < 1e-4:
            self.pivot.enabled = False
            return
        self.pivot.enabled = True
        self.pivot.position = (pa + pb) * 0.5

        forward = _safe_normalize(pb - pa)
        if forward is None:
            self.pivot.enabled = False
            return
        width_axis = body_width_axis(pts)
        if abs(forward.dot(width_axis)) > 0.95:
            width_axis = Vec3(0, 1, 0)
            if abs(forward.dot(width_axis)) > 0.95:
                width_axis = Vec3(1, 0, 0)
        self.pivot.look_at(pb, up=width_axis)
        self.mesh.scale = (self.size[0], self.size[1], length)


def body_width_axis(pts: dict[int, Vec3]) -> Vec3:
    """Left-to-right axis of the body, used to orient boxes."""
    ls, rs = pts.get(L_SHOULDER), pts.get(R_SHOULDER)
    if ls is not None and rs is not None:
        axis = _safe_normalize(rs - ls)
        if axis is not None:
            return axis
    lh, rh = pts.get(L_HIP), pts.get(R_HIP)
    if lh is not None and rh is not None:
        axis = _safe_normalize(rh - lh)
        if axis is not None:
            return axis
    return Vec3(1, 0, 0)


class Avatar:
    """The full humanoid rig."""

    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.root = Entity()
        self.parts: list = []
        self.joint_entities: dict[int, Entity] = {}

        # torso -------------------------------------------------------------
        self.parts.append(Box(P_MID_SHOULDER, P_MID_HIP,
                              (TORSO_DEPTH, TORSO_WIDTH), COLOR_SHIRT, self.root))
        self.parts.append(Capsule(L_SHOULDER, R_SHOULDER,
                                  SHOULDER_RADIUS, COLOR_SHIRT, self.root))
        self.parts.append(Capsule(L_HIP, R_HIP, HIP_RADIUS, COLOR_PANTS, self.root))
        self.parts.append(Capsule(P_MID_SHOULDER, P_HEAD,
                                  NECK_RADIUS, COLOR_SKIN, self.root))
        self.parts.append(Sphere(P_HEAD, HEAD_RADIUS, COLOR_SKIN, self.root))

        # arms --------------------------------------------------------------
        for shoulder, elbow, wrist in (
            (L_SHOULDER, L_ELBOW, L_WRIST),
            (R_SHOULDER, R_ELBOW, R_WRIST),
        ):
            self.parts.append(Capsule(shoulder, elbow, UPPER_ARM_RADIUS,
                                      COLOR_SHIRT, self.root))
            self.parts.append(Capsule(elbow, wrist, FOREARM_RADIUS,
                                      COLOR_SKIN, self.root))
            self.parts.append(Sphere(wrist, HAND_RADIUS, COLOR_SKIN, self.root))

        # legs --------------------------------------------------------------
        for hip, knee, ankle, foot in (
            (L_HIP, L_KNEE, L_ANKLE, L_FOOT),
            (R_HIP, R_KNEE, R_ANKLE, R_FOOT),
        ):
            self.parts.append(Capsule(hip, knee, THIGH_RADIUS, COLOR_PANTS, self.root))
            self.parts.append(Capsule(knee, ankle, SHIN_RADIUS, COLOR_PANTS, self.root))
            self.parts.append(Box(ankle, foot, (FOOT_DEPTH, FOOT_WIDTH),
                                  COLOR_SHOE, self.root))

        # optional debug joint markers --------------------------------------
        if cfg.show_joints:
            for idx in (NOSE, L_EAR, R_EAR, L_SHOULDER, R_SHOULDER, L_ELBOW,
                        R_ELBOW, L_WRIST, R_WRIST, L_HIP, R_HIP, L_KNEE, R_KNEE,
                        L_ANKLE, R_ANKLE, L_FOOT, R_FOOT):
                self.joint_entities[idx] = Entity(
                    parent=self.root, model="sphere", scale=0.05, color=COLOR_JOINT)

    def set(self, pts: dict[int, Vec3]) -> None:
        for part in self.parts:
            part.set(pts)
        for idx, ent in self.joint_entities.items():
            p = pts.get(idx)
            if p is None:
                ent.enabled = False
            else:
                ent.enabled = True
                ent.position = p

    def hide(self) -> None:
        for part in self.parts:
            if hasattr(part, "pivot"):
                part.pivot.enabled = False
            else:
                part.entity.enabled = False
        for ent in self.joint_entities.values():
            ent.enabled = False


# ---------------------------------------------------------------------------
# Scene
# ---------------------------------------------------------------------------
def build_scene() -> None:
    window.color = color.rgb32(18, 20, 28)
    # Ursina's development-mode widget (fps / close button) in the corner.
    try:
        window.editor_ui.enabled = False
    except AttributeError:
        pass

    camera.position = CAMERA_POS
    camera.look_at(CAMERA_TARGET)
    # Keep the vertical field of view fixed so the avatar stays framed even
    # when the window is resized.
    try:
        camera.perspective_lens.set_aspect_ratio(scene_aspect)
    except Exception:                                    # pragma: no cover
        pass
    camera.fov = _horizontal_fov_for_vertical(CAMERA_VERTICAL_FOV, scene_aspect)

    Entity(model="plane", scale=(40, 1, 40),
           color=color.rgb32(34, 36, 46), position=(0, GROUND_Y, 0))

    grid_color = color.rgb32(58, 62, 82)
    axis_color = color.rgb32(90, 100, 130)
    n = int(GRID_EXTENT / GRID_SPACING)
    for i in range(-n, n + 1):
        p = i * GRID_SPACING
        major = (i % 5 == 0)
        c = axis_color if major else grid_color
        Entity(model="cube", scale=(GRID_EXTENT * 2, 0.01, 0.02),
               color=c, position=(0, GROUND_Y + 0.01, p))
        Entity(model="cube", scale=(0.02, 0.01, GRID_EXTENT * 2),
               color=c, position=(p, GROUND_Y + 0.01, 0))


# ---------------------------------------------------------------------------
# Window placement
# ---------------------------------------------------------------------------
# Aspect ratio the scene is currently framed for.
scene_aspect = DEFAULT_WINDOW_SIZE[0] / DEFAULT_WINDOW_SIZE[1]


def apply_window_layout(cfg: Config) -> None:
    """Put the 3D scene and the cv2 preview next to each other.

    The 3D window is placed at ``--window-x/--window-y``; the cv2 preview is
    placed immediately to its right (or at ``--preview-x/--preview-y`` when
    those are given).  This makes the "avatar + live skeleton" view a genuine
    side-by-side layout instead of two windows stacked on top of each other.
    """
    global scene_aspect
    scene_aspect = cfg.window_width / max(1, cfg.window_height)

    try:
        window.size = (cfg.window_width, cfg.window_height)
        window.position = (cfg.window_x, cfg.window_y)
        camera.perspective_lens.set_aspect_ratio(scene_aspect)
    except Exception:                                    # pragma: no cover
        pass

    if not cfg.show_video:
        return

    px = cfg.preview_x
    if px is None:
        px = cfg.window_x + cfg.window_width
    py = cfg.preview_y
    if py is None:
        py = cfg.window_y

    try:
        cv2.namedWindow(PREVIEW_WINDOW, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(PREVIEW_WINDOW, cfg.preview_width, cfg.preview_height)
        cv2.moveWindow(PREVIEW_WINDOW, px, py)
    except cv2.error:                                    # pragma: no cover
        pass


# ---------------------------------------------------------------------------
# HUD
# ---------------------------------------------------------------------------
def _ui_pos(nx: float, ny: float) -> Vec3:
    """Normalised screen position (-1..1) -> camera.ui local coordinates."""
    return Vec3(nx * 0.5 * window.aspect_ratio, ny * 0.5, 0)


class Hud:
    def __init__(self) -> None:
        # Ursina's default Text origin (-.5, .5) anchors the text at its
        # top-left corner, which is what we want for a top-left HUD.
        self.status = Text(text="SEARCHING FOR POSE", scale=1.5, color=color.yellow)
        self.detail = Text(text="", scale=0.9, color=color.white)
        self.stats = Text(text="", scale=0.8, color=color.light_gray)
        self.hint = Text(text="WASD / arrows = move    R/F = up/down    Q = quit",
                         scale=0.8, color=color.light_gray)
        self.meter_bg = Entity(parent=camera.ui, model="quad", scale=(0.16, 0.016),
                               color=color.rgb32(60, 64, 84))
        self.meter_fill = Entity(parent=camera.ui, model="quad", scale=(0.001, 0.012),
                                 color=color.green)
        self.meter_label = Text(text="legs", scale=0.9, color=color.light_gray)
        self._aspect = None

    def layout(self) -> None:
        aspect = window.aspect_ratio
        if aspect == self._aspect:
            return
        self._aspect = aspect
        self.status.position = _ui_pos(-0.92, 0.88)
        self.detail.position = _ui_pos(-0.92, 0.80)
        self.stats.position = _ui_pos(-0.92, -0.80)
        self.hint.position = _ui_pos(-0.92, -0.90)
        self.meter_label.position = _ui_pos(-0.92, 0.70)
        self.meter_bg.position = _ui_pos(-0.78, 0.695)
        self.meter_fill.position = _ui_pos(-0.86, 0.695)

    def set_meter(self, value: float) -> None:
        value = _clamp(value, 0.0, 1.0)
        w = 0.001 + 0.158 * value
        self.meter_fill.scale_x = w
        self.meter_fill.x = _ui_pos(-0.86, 0.695).x + w * 0.5
        if value < 0.4:
            self.meter_fill.color = color.orange
        elif value < 0.7:
            self.meter_fill.color = color.yellow
        else:
            self.meter_fill.color = color.green


# ---------------------------------------------------------------------------
# Runtime state
# ---------------------------------------------------------------------------
cfg = Config()
worker: PoseWorker | None = None
tracker = PoseTracker(cfg)
avatar: Avatar | None = None
hud: Hud | None = None
avatar_offset = Vec3(0.0, HIP_HEIGHT, 0.0)
shadow: Entity | None = None
start_time = time.monotonic()
_full_body_seen = False
_preview_frame = None
_preview_placed = False


def _place_preview_window() -> None:
    """Move the cv2 preview window next to the 3D window (once)."""
    global _preview_placed
    _preview_placed = True
    px = cfg.preview_x
    if px is None:
        px = cfg.window_x + cfg.window_width
    py = cfg.preview_y
    if py is None:
        py = cfg.window_y
    try:
        cv2.moveWindow(PREVIEW_WINDOW, px, py)
    except cv2.error:
        pass


# ---------------------------------------------------------------------------
# Per-frame hook
# ---------------------------------------------------------------------------
def update() -> None:
    global _full_body_seen, _preview_frame, avatar_offset, _preview_placed

    if worker is None or avatar is None or hud is None:
        return

    now = time.monotonic()

    # --- pull the newest pose result ---------------------------------------
    try:
        sample = worker.results.get_nowait()
    except queue.Empty:
        sample = None
    if sample is not None:
        tracker.ingest(sample)

    # --- keyboard movement --------------------------------------------------
    dt = ursina_time.dt
    speed = cfg.move_speed * dt
    if held_keys["d"] or held_keys["right arrow"]:
        avatar_offset.x += speed
    if held_keys["a"] or held_keys["left arrow"]:
        avatar_offset.x -= speed
    if held_keys["w"] or held_keys["up arrow"]:
        avatar_offset.z -= speed          # -Z = into the screen
    if held_keys["s"] or held_keys["down arrow"]:
        avatar_offset.z += speed
    if held_keys["r"]:
        avatar_offset.y += speed
    if held_keys["f"]:
        avatar_offset.y -= speed

    # --- drive the rig -----------------------------------------------------
    tracking = tracker.tracking()
    full_body = tracker.full_body()
    if full_body:
        _full_body_seen = True

    pts = tracker.scene_points(now, avatar_offset) if tracking else {}

    ready = True
    if cfg.require_full_body and not _full_body_seen:
        ready = False

    if pts and ready:
        avatar.set(pts)
    else:
        avatar.hide()

    if shadow is not None:
        shadow.enabled = bool(pts) and ready
        shadow.position = (avatar_offset.x, GROUND_Y + 0.02, avatar_offset.z)

    # --- HUD ----------------------------------------------------------------
    hud.layout()
    leg_q = tracker.leg_quality()
    hud.set_meter(leg_q)

    if not tracking:
        hud.status.text = "SEARCHING FOR POSE"
        hud.status.color = color.yellow
        hud.detail.text = "step into frame so your upper body is visible"
    elif cfg.require_full_body and not _full_body_seen:
        hud.status.text = "WAITING FOR FULL BODY"
        hud.status.color = color.orange
        hud.detail.text = "step back until your knees and feet are in frame"
    elif not full_body:
        hud.status.text = "UPPER BODY ONLY"
        hud.status.color = color.orange
        hud.detail.text = "step back to include your legs"
    else:
        hud.status.text = "TRACKING"
        hud.status.color = color.green
        hud.detail.text = ""

    render_fps = 1.0 / max(dt, 1e-4)
    hud.stats.text = (
        f"render {render_fps:4.1f} fps   camera {worker.camera_fps:4.1f} fps   "
        f"inference {worker.inference_ms:4.0f} ms   scale {tracker.body_scale:0.2f}"
    )

    # --- optional 2D preview ------------------------------------------------
    if cfg.show_video:
        if sample is not None and sample.frame is not None:
            draw_pose_on_frame(sample.frame, sample.image, cfg.min_visibility)
            _preview_frame = sample.frame
        if _preview_frame is not None:
            if not _preview_placed:
                # The window manager sometimes ignores moveWindow() until the
                # window is actually mapped, so re-assert the position once.
                _place_preview_window()
            cv2.imshow(PREVIEW_WINDOW, _preview_frame)
            cv2.waitKey(1)

    # --- exit conditions ----------------------------------------------------
    if cfg.exit_if_lost > 0.0:
        lost_for = (now - (tracker.last_detection or start_time))
        if lost_for > cfg.exit_if_lost:
            print(f"Tracking lost for {lost_for:.1f}s - exiting.", file=sys.stderr)
            application.quit()


def input(key: str) -> None:  # noqa: A001
    if key in ("q", "escape"):
        application.quit()


# ---------------------------------------------------------------------------
# 2D overlay
# ---------------------------------------------------------------------------
def draw_pose_on_frame(frame, image_points, min_visibility: float) -> None:
    """Draw image-space landmarks/bones onto `frame` in place."""
    h, w = frame.shape[:2]
    pts: list[tuple[int, int] | None] = []
    for x, y, v in image_points:
        if v >= min_visibility:
            pts.append((int(x * w), int(y * h)))
        else:
            pts.append(None)

    # MediaPipe occasionally returns *no* image landmarks at all.  The
    # connection table still references landmark indices up to 32, so pad the
    # list out to the full landmark count with "not visible" placeholders
    # instead of blowing up with IndexError.
    while len(pts) < POSE_LANDMARK_COUNT:
        pts.append(None)

    for a, b in mp.solutions.pose.POSE_CONNECTIONS:
        pa, pb = pts[a], pts[b]
        if pa is None or pb is None:
            continue
        cv2.line(frame, pa, pb, BONE_COLOR, BONE_THICKNESS, cv2.LINE_AA)

    for p in pts:
        if p is None:
            continue
        cv2.circle(frame, p, POINT_RADIUS, LANDMARK_COLOR, -1, cv2.LINE_AA)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Drive a 3D humanoid avatar from webcam pose estimation."
    )
    p.add_argument("--camera", type=int, default=0,
                   help="Camera device index (default: 0).")
    p.add_argument("--width", type=int, default=640)
    p.add_argument("--height", type=int, default=480)
    p.add_argument("--model-complexity", type=int, choices=(0, 1, 2), default=0,
                   help="MediaPipe model size: 0 = fastest (default: 0).")
    p.add_argument("--min-visibility", type=float, default=0.25,
                   help="Landmarks below this confidence are held, not drawn "
                        "(default: 0.25).")
    p.add_argument("--show-video", action="store_true",
                   help="Also open a cv2 window with the camera feed + overlay.")
    p.add_argument("--no-flip", action="store_true",
                   help="Do not mirror the frame (selfie view is on by default).")
    p.add_argument("--show-joints", action="store_true",
                   help="Draw small markers on every tracked landmark.")
    p.add_argument("--require-full-body", action="store_true",
                   help="Wait until a full-body pose is detected before driving "
                        "the avatar.")
    p.add_argument("--exit-if-lost", type=float, default=0.0, metavar="SECONDS",
                   help="Quit if tracking is lost for this many seconds "
                        "(0 = never).")
    p.add_argument("--scale", type=float, default=SCALE,
                   help="Size multiplier for the avatar (default: 1.0).")
    p.add_argument("--smooth-tau", type=float, default=0.15, metavar="SECONDS",
                   help="Landmark smoothing time constant (default: 0.15).")
    p.add_argument("--no-interpolate", action="store_true",
                   help="Disable pose interpolation between camera frames.")
    p.add_argument("--move-speed", type=float, default=MOVE_SPEED,
                   help="Avatar movement speed in metres per second.")
    p.add_argument("--window-width", type=int, default=1280,
                   help="Width of the 3D window in pixels (default: 1280).")
    p.add_argument("--window-height", type=int, default=720,
                   help="Height of the 3D window in pixels (default: 720).")
    p.add_argument("--window-x", type=int, default=0,
                   help="X position of the 3D window (default: 0).")
    p.add_argument("--window-y", type=int, default=0,
                   help="Y position of the 3D window (default: 0).")
    p.add_argument("--preview-width", type=int, default=640,
                   help="Width of the --show-video window (default: 640).")
    p.add_argument("--preview-height", type=int, default=480,
                   help="Height of the --show-video window (default: 480).")
    p.add_argument("--preview-x", type=int, default=None,
                   help="X position of the --show-video window "
                        "(default: right of the 3D window).")
    p.add_argument("--preview-y", type=int, default=None,
                   help="Y position of the --show-video window "
                        "(default: same top as the 3D window).")
    return p.parse_args()


def main() -> None:
    global cfg, worker, tracker, avatar, hud, shadow, avatar_offset

    args = parse_args()
    cfg = Config(
        camera_index=args.camera,
        width=args.width,
        height=args.height,
        model_complexity=args.model_complexity,
        min_visibility=args.min_visibility,
        show_video=args.show_video,
        flip=not args.no_flip,
        show_joints=args.show_joints,
        require_full_body=args.require_full_body,
        exit_if_lost=args.exit_if_lost,
        scale=args.scale,
        smooth_tau=args.smooth_tau,
        interpolate=not args.no_interpolate,
        move_speed=args.move_speed,
        window_width=args.window_width,
        window_height=args.window_height,
        window_x=args.window_x,
        window_y=args.window_y,
        preview_width=args.preview_width,
        preview_height=args.preview_height,
        preview_x=args.preview_x,
        preview_y=args.preview_y,
    )
    tracker = PoseTracker(cfg)

    apply_window_layout(cfg)
    build_scene()
    avatar = Avatar(cfg)
    hud = Hud()
    shadow = Entity(model="circle", scale=0.7, color=color.rgb32(10, 10, 16),
                    position=(0, GROUND_Y + 0.02, 0))
    avatar_offset = Vec3(0.0, cfg.hip_height, 0.0)

    worker = PoseWorker(cfg)
    worker.start()
    # give the camera a moment to open so we can report failures early
    time.sleep(0.5)
    if worker.error:
        print(worker.error, file=sys.stderr)
        worker.stop()
        sys.exit(1)

    print("Pose avatar running.  q = quit, WASD = move.", flush=True)
    try:
        app.run()
    finally:
        if worker is not None:
            worker.stop()
            worker.join(timeout=1.0)
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
