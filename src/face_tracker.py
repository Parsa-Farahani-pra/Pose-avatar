#!/usr/bin/env python3
"""Modular face tracking module using MediaPipe Face Mesh.

Provides:
- FaceMesh landmark definitions (468 points)
- metrics world space vertices for precise shape alignment
- FaceBoundaryBounds, FaceDetectionGate to adapt camera appearances
"""

from __future__ import annotations

import argparse
import logging
import threading
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
from contextlib import contextmanager

import cv2
import mediapipe as mp
import numpy as np

logger = logging.getLogger(__name__)

# ============================================================================
# Face Mesh Landmark Indices (MediaPipe FaceMesh - 468 points)
# ============================================================================

NOSE_TIP = 1
NOSE_BRIDGE = 4


def _unique_indices(connections) -> List[int]:
    """Extract sorted unique landmark indices from a MediaPipe FACEMESH connection set.

    MediaPipe FaceMesh indices are NOT spatially ordered, so contiguous index
    ranges (e.g. range(0, 17)) do NOT correspond to facial regions. The only
    reliable way to obtain region landmarks is to derive them from the official
    FACEMESH_* connection topology.
    """
    idxs: set = set()
    for pair in connections:
        idxs.update(pair)
    return sorted(idxs)


_FM = mp.solutions.face_mesh

# Canonical region landmarks derived from MediaPipe's official topology:
FACE_MESH_FACE_OVAL = _unique_indices(_FM.FACEMESH_FACE_OVAL)   # jaw + forehead contour
FACE_MESH_JAWLINE = FACE_MESH_FACE_OVAL                         # alias: jaw is part of the oval
FACE_MESH_LEFT_EYE = _unique_indices(_FM.FACEMESH_LEFT_EYE)
FACE_MESH_RIGHT_EYE = _unique_indices(_FM.FACEMESH_RIGHT_EYE)
FACE_MESH_LEFT_EYEBROW = _unique_indices(_FM.FACEMESH_LEFT_EYEBROW)
FACE_MESH_RIGHT_EYEBROW = _unique_indices(_FM.FACEMESH_RIGHT_EYEBROW)
FACE_MESH_MOUTH = _unique_indices(_FM.FACEMESH_LIPS)            # outer + inner lips
FACE_MESH_NOSE = _unique_indices(_FM.FACEMESH_NOSE)

# Optional extras
FACE_MESH_EYES_WINK = (33, 133, 362, 263)  # inner/outer eye corners for wink detection

# Regions used by the lightweight ("subsample") debug overlay
DEBUG_FACE_REGIONS = list(FACE_MESH_JAWLINE + FACE_MESH_LEFT_EYEBROW +
                          FACE_MESH_RIGHT_EYEBROW + FACE_MESH_LEFT_EYE +
                          FACE_MESH_RIGHT_EYE + FACE_MESH_NOSE + FACE_MESH_MOUTH)
ALL_FACE_MESH_REGIONS = tuple(DEBUG_FACE_REGIONS)

# MediaPipe unified landmarks (shared across solutions)
TIP_NOSE = 1            # MediaPipe's anchor for nose
LEFT_EYE_2 = 33         # MediaPipe's critical eye index
RIGHT_EYE_2 = 362       # MediaPipe's critical eye index
L_ICE_INDEX = 33        # MediaPipe's pivot eye index
R_ICE_INDEX = 362

# World space relative references with metric growth
L_WRIST = 15            # MediaPipe's anchor wrist for shoulder matching
R_WRIST = 16            # MediaPipe's anchor wrist for shoulder matching
WRIST = 15              # MediaPipe's rmp anchor wrist for shoulder matching

# Face and hand global usages
PHALANX_RADIUS = 0.006
PALM_RADIUS = 0.030

# Bounding box thresholds
K_FACE_SIZE_THRESHOLD = 0.25  # 25% of frame height

# ============================================================================
# Bounding Box & Visibility Utilities
# ============================================================================

@dataclass
class FaceBoundaryBounds:
    """Quantifies the visual coverage of a face in the current frame.

    Attributes:
        min_x: Normalized left edge (0-1 relative to frame width)
        min_y: Normalized top edge (0-1 relative to frame height)
        max_x: Normalized right edge (0-1 relative to frame width)
        max_y: Normalized bottom edge (0-1 relative to frame height)
    """
    min_x: float
    min_y: float
    max_x: float
    max_y: float


def compute_face_bbox(landmarks_468: List[Tuple[int, Tuple[float, float, float]]],
                      width: int, height: int) -> FaceBoundaryBounds:
    """Compute normalized bounding box from FaceMesh landmarks.

    Args:
        landmarks_468: List of (index, (x, y, z)) normalized landmarks
        width: Frame width in pixels
        height: Frame height in pixels

    Returns:
        Normalized bounding box spanning all FaceMesh landmarks
    """
    xs = [lm[1][0] for lm in landmarks_468]
    ys = [lm[1][1] for lm in landmarks_468]
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)

    # Clip to [0, 1] in case of numerical errors
    normalized_min_x = max(0.0, min_x)
    normalized_max_x = min(1.0, max_x)
    normalized_min_y = max(0.0, min_y)
    normalized_max_y = min(1.0, max_y)

    return FaceBoundaryBounds(
        min_x=normalized_min_x,
        min_y=normalized_min_y,
        max_x=normalized_max_x,
        max_y=normalized_max_y
    )


def face_box_height(bounds: FaceBoundaryBounds) -> float:
    """Compute normalized face height in this frame.

    Args:
        bounds: Normalized bounding box

    Returns:
        Normalized height (max_y - min_y), ranging from 0 to 1
    """
    return bounds.max_y - bounds.min_y


def normalize_yz(frame_height: int, normalized_y: float) -> int:
    """Convert normalized Y to pixel coordinate for display.

    Args:
        frame_height: Frame height in pixels
        normalized_y: Normalized Y in [0, 1]

    Returns:
        Pixel Y coordinate (top‑down: frame_height - int(normalized_y * frame_height))
    """
    return frame_height - int(normalized_y * frame_height)


# ============================================================================
# Face Mesh Data Structure
# ============================================================================

@dataclass
class FaceLandmarks:
    """Compact data structure holding FaceMesh output.

    Attributes:
        landmarks_468: All 468 normalized FaceMesh landmarks (x, y, z in [0, 1])
        nose_tip_world: metric version of nose tip in MediaPipe world coordinates
        face_bbox: Normalized bounding box across all FaceMesh landmarks
        face_size_ratio: Normalized face height vs frame height
        is_detected: True if face detected and size threshold met
    """
    landmarks_468: List[Tuple[int, Tuple[float, float, float]]]
    nose_tip_world: np.ndarray  # metric version of nose tip
    face_bbox: FaceBoundaryBounds
    face_size_ratio: float
    is_detected: bool


# ============================================================================
# Face Mesh Drawing Utilities
# ============================================================================

# Region colors in BGR. Left/right regions share the same color so the
# visualization stays symmetric when the face is centered in frame.
NOSE_COLOR = (255, 255, 255)                  # White nose region
NOSE_TIP_COLOR = (0, 255, 255)                # Yellow anchor at the nose tip
JAW_COLOR = (0, 255, 0)                       # Green face-oval/jaw contour
EYEBROW_COLOR = (255, 255, 0)                 # Cyan eyebrows
EYE_COLOR = (255, 0, 255)                     # Magenta eyes
MOUTH_COLOR = (127, 0, 255)                   # Purple lips
MESH_COLOR = (170, 170, 170)                  # Soft gray for the rest of the mesh


# MediaPipe Pose face landmark indices (0-10): nose, eyes, ears, lips.
# These duplicate what FaceMesh renders, so they are hidden while
# FaceMesh is active and detecting a face, and shown again otherwise.
POSE_FACE_LANDMARKS = set(range(11))  # indices 0..10


def draw_face_mesh_subset(frame: np.ndarray,
                           face: FaceLandmarks,
                           mode: str = "full",
                           face_colors: Optional[Dict[int, np.ndarray]] = None) -> None:
    """Draw ALL 468 FaceMesh landmarks on the frame using region-based coloring.

    Args:
        frame: Input BGR image array compatible with cv2
        face: Full FaceMesh output with all 468 landmarks
        mode: "full" — render all 468 landmarks; "subsample" — schematic regions only
        face_colors: Optional dict mapping landmark indices to BGR colors

    Returns:
        None (modifies frame in place)
    """
    if not face.landmarks_468:
        return

    h, w = frame.shape[:2]

    # Default colors by region (BGR format). Assignment order matters only for
# overlapping indices; regions are largely disjoint in the official topology.
    if face_colors is None:
        face_colors = {}
        for idx in range(468):
            face_colors[idx] = MESH_COLOR            # rest of the mesh: soft gray
        for idx in FACE_MESH_FACE_OVAL:
            face_colors[idx] = JAW_COLOR             # green outer contour (jaw + forehead)
        for idx in FACE_MESH_LEFT_EYEBROW + FACE_MESH_RIGHT_EYEBROW:
            face_colors[idx] = EYEBROW_COLOR         # cyan eyebrows
        for idx in FACE_MESH_LEFT_EYE + FACE_MESH_RIGHT_EYE:
            face_colors[idx] = EYE_COLOR             # magenta eyes
        for idx in FACE_MESH_NOSE:
            face_colors[idx] = NOSE_COLOR            # white nose
        for idx in FACE_MESH_MOUTH:
            face_colors[idx] = MOUTH_COLOR           # purple lips
        face_colors[NOSE_TIP] = NOSE_TIP_COLOR       # yellow anchor at nose tip

    # Compute which landmarks to render
    indices_to_draw = []
    if mode == "full":
        # Render ALL 468 landmarks
        indices_to_draw = [lm[0] for lm in face.landmarks_468]
    else:  # "subsample"
        # Render key schematic regions only
        indices_to_draw = list({int(lm[0]) for lm in face.landmarks_468 if int(lm[0]) in ALL_FACE_MESH_REGIONS})

    # Scale normalized coordinates to frame
    scale_x = w
    scale_y = h

    # Draw all landmarks
    for idx in indices_to_draw:
        if idx not in face_colors:
            continue
        if idx >= len(face.landmarks_468):
            continue
        norm_x, norm_y, _ = face.landmarks_468[idx][1]
        x = int(np.clip(norm_x * scale_x, 0, w - 1))
        y = int(np.clip(norm_y * scale_y, 0, h - 1))
        cv2.circle(frame, (x, y), 2, face_colors[idx], -1)


def debug_face_bounds_display(frame: np.ndarray,
                              bounds: FaceBoundaryBounds,
                              face_size_ratio: float,
                              threshold: float = 0.25) -> None:
    """Overlay the computed face bounding box and ratio info on the frame.

    Args:
        frame: Input image (modified in place)
        bounds: Normalized face bounding box
        face_size_ratio: Normalized face height relative to frame
        threshold: Size threshold (e.g., 0.25 meaning 25% of frame height)

    Returns:
        None (modifies frame in place)
    """
    h, w = frame.shape[:2]

    # Convert normalized to pixel coordinates
    min_x = int(bounds.min_x * w)
    min_y = int(bounds.min_y * h)
    max_x = int(bounds.max_x * w)
    max_y = int(bounds.max_y * h)

    # Convert normalized ratio to pixels
    height_px = int(face_size_ratio * h)

    # Check if detected
    color = (0, 255, 0) if face_size_ratio >= threshold else (0, 0, 255)
    alpha = 0.6

    # Draw rectangle around face
    cv2.rectangle(frame, (min_x, min_y), (max_x, max_y), color, 1)

    # Draw text overlay
    status = "DETECTED" if face_size_ratio >= threshold else "NOT DETECTED"
    text = f"Face: {face_size_ratio:.2%} (Size: {status})"

    # Put text at top-left corner
    cv2.putText(frame, text, (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7, color, 2)


# ============================================================================
# Face Mesh Worker (Thread-Safe Processing)
# ============================================================================

@dataclass
class FaceWorkerConfig:
    """Configuration for FaceMesh processing.
    
    Attributes:
        enable_face: If False, return no-face payload; if True, process Face Mesh
        face_size_threshold: Minimum face size ratio (0-1) to activate face detection
    """
    enable_face: bool = False
    face_size_threshold: float = 0.25


class FaceWorker:
    """Thread-safe worker processing MediaPipe Face Mesh landmarks.
    
    Due to MediaPipe 0.10.14 limitations, we use a separate FaceMesh solution
    when enabled and gate processing by face bounding box size.
    """

    def __init__(self, cfg: FaceWorkerConfig | argparse.Namespace):
        """
        Initialize FaceMesh worker.
        
        Args:
            cfg: Either FaceWorkerConfig object or argparse.Namespace 
                (will be converted to FaceWorkerConfig)
        """
        # Convert Namespace to FaceWorkerConfig if needed
        if isinstance(cfg, argparse.Namespace):
            # Create FaceWorkerConfig from Namespace attributes
            face_size_threshold = getattr(cfg, 'face_size_threshold', 0.25)
            config = FaceWorkerConfig(
                enable_face=cfg.enable_face,
                face_size_threshold=face_size_threshold
            )
        else:
            config = cfg
        
        self.cfg = config
        self._lock = threading.RLock()
        self._face_mesh: Optional[mp.solutions.face_mesh.FaceMesh] = None
        self._initialized = False

        if self.cfg.enable_face:
            self._initialize_face_mesh()

        logger.info("FaceMesh worker initialized (enabled=%s)", self.cfg.enable_face)

    def _initialize_face_mesh(self) -> None:
        """Initialize MediaPipe Face Mesh solution."""
        try:
            self._face_mesh = mp.solutions.face_mesh.FaceMesh(
                static_image_mode=False,
                refine_landmarks=True,
                max_num_faces=1,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5
            )
            self._initialized = True
            logger.debug("FaceMesh MediaPipe solution successfully initialized")
        except Exception as e:
            logger.error("Failed to initialize FaceMesh: %s", e)
            self._initialized = False

    def close(self) -> None:
        """Clean up resources."""
        if self._face_mesh is not None:
            self._face_mesh.close()
            self._face_mesh = None
            logger.debug("FaceMesh MediaPipe solution released")

    def process_frame(self, frame: np.ndarray) -> Optional[FaceLandmarks]:
        """
        Process a single frame and return FaceMesh output.
        
        Args:
            frame: Input BGR image
            
        Returns:
            FaceLandmarks object with full 468 landmarks if face detected and meets size threshold,
            None otherwise
        """
        with self._lock:
            if not self.cfg.enable_face or not self._initialized or self._face_mesh is None:
                return None

            try:
                mp_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = self._face_mesh.process(mp_frame)

                if results.multi_face_landmarks is None or len(results.multi_face_landmarks) == 0:
                    return None

                # Use the first detected face (single person assumption)
                face_landmarks = results.multi_face_landmarks[0]

                # Build landmarks_468 list
                landmarks_468 = []
                for idx in range(468):
                    x = face_landmarks.landmark[idx].x
                    y = face_landmarks.landmark[idx].y
                    z = face_landmarks.landmark[idx].z
                    landmarks_468.append((idx, (x, y, z)))

                # Extract nose tip world coordinates
                nose_x = face_landmarks.landmark[NOSE_TIP].x
                nose_y = face_landmarks.landmark[NOSE_TIP].y
                nose_z = face_landmarks.landmark[NOSE_TIP].z

                # Handle invalid coordinates (shouldn't happen with proper model)
                if nose_x <= 0 or nose_y <= 0 or nose_z <= 0:
                    nose_x, nose_y, nose_z = 0.5, 0.5, 0.5

                nose_world = np.array([nose_x, nose_y, nose_z], dtype=np.float32)

                # Compute bounding box and size ratio
                face_bounds = compute_face_bbox(landmarks_468, frame.shape[1], frame.shape[0])
                face_size_ratio = face_box_height(face_bounds)
                is_detected = face_size_ratio >= self.cfg.face_size_threshold

                logger.debug("Face: %s (size=%.2f)", "detected" if is_detected else "not detected", face_size_ratio)

                # Return full FaceLandmarks object
                return FaceLandmarks(
                    landmarks_468=landmarks_468,
                    nose_tip_world=nose_world,
                    face_bbox=face_bounds,
                    face_size_ratio=face_size_ratio,
                    is_detected=is_detected
                )

            except Exception as e:
                logger.error("Error processing FaceMesh: %s", e)
                return None


# ============================================================================
# Module Exports
# ============================================================================

__all__ = [
    "FaceWorker",
    "FaceWorkerConfig",
    "FaceLandmarks",
    "FaceBoundaryBounds",
    "compute_face_bbox",
    "face_box_height",
    "draw_face_mesh_subset",
    "debug_face_bounds_display",
    "NOSE_TIP",
    "NOSE_BRIDGE",
    "ALL_FACE_MESH_REGIONS",
    "FACE_MESH_JAWLINE",
    "FACE_MESH_LEFT_EYEBROW",
    "FACE_MESH_RIGHT_EYEBROW",
    "FACE_MESH_LEFT_EYE",
    "FACE_MESH_RIGHT_EYE",
    "FACE_MESH_EYES_WINK",
    "FACE_MESH_BROWS_COMPLETE",
    "FACE_MESH_MOUTH",
    "FACE_MESH_FACE_OVAL",
    "FACE_MESH_UPPER_FACE",
    "FACE_MESH_CHIN",
    "K_FACE_SIZE_THRESHOLD",
    "PHALANX_RADIUS",
    "PALM_RADIUS",
]

