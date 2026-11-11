#!/usr/bin/env python3
"""Emotion detection from MediaPipe Face Mesh landmarks.

Computes geometric features from the 468 normalized landmarks and classifies
into 5 core emotions (Happy, Sad, Angry, Surprised, Neutral) with certainty
percentages. Includes temporal smoothing for stable output.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np


# ============================================================================
# Emotion result dataclass
# ============================================================================

@dataclass
class EmotionResult:
    """Result of one emotion detection update.

    Attributes:
        percentages: Dict mapping emotion name -> certainty [0, 100].
        top_emotion: The emotion with highest certainty.
        top_certainty: Certainty of top_emotion in [0, 100].
        is_confident: True if top_certainty >= 60% (tunable threshold).
    """
    percentages: Dict[str, float]
    top_emotion: str
    top_certainty: float
    is_confident: bool

    def __str__(self) -> str:
        bars = []
        for emo, pct in sorted(self.percentages.items(), key=lambda x: -x[1]):
            bar = "█" * int(pct / 5) + "░" * (20 - int(pct / 5))
            marker = " ←" if emo == self.top_emotion else ""
            bars.append(f"  {emo:9s} {bar} {pct:5.1f}%{marker}")
        return "\n".join(bars)


# ============================================================================
# MediaPipe FaceMesh topology (canonical indices from FACEMESH_* sets)
# ============================================================================

# Left eye (subject's anatomical left = image right): 16 indices
# Ordered for EAR: [outer, inner, top1, top2, bottom1, bottom2, ...]
EYE_LEFT_IDXS = [33, 133, 159, 158, 145, 144]  # outer, inner, top, top, bottom, bottom

# Right eye (subject's anatomical right = image left): 16 indices
EYE_RIGHT_IDXS = [362, 263, 386, 385, 374, 373]  # outer, inner, top, top, bottom, bottom

# Lips: 40 indices (outer + inner contour)
# For MAR: corners, top center, bottom center
MOUTH_CORNER_LEFT = 61
MOUTH_CORNER_RIGHT = 291
MOUTH_TOP_CENTER = 13
MOUTH_BOTTOM_CENTER = 14
MOUTH_INNER_TOP = 82
MOUTH_INNER_BOTTOM = 87

# Eyebrows
EYEBROW_LEFT_IDXS = [70, 63, 105, 66, 107]  # medial to lateral
EYEBROW_RIGHT_IDXS = [300, 293, 334, 296, 336]  # medial to lateral

# Face oval for width
FACE_OVAL_IDXS = [10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136, 172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109, 10]

# Cheeks (for smile cheek raise)
CHEEK_LEFT_IDXS = [116, 117, 118, 119, 120, 121, 126, 142, 36, 205, 206, 207, 213, 192, 147, 187, 205, 206, 207]
CHEEK_RIGHT_IDXS = [345, 346, 347, 348, 349, 350, 355, 371, 266, 425, 426, 427, 433, 412, 376, 416, 425, 426, 427]


def _norm(landmarks_468: List[Tuple[int, Tuple[float, float, float]]], idx: int) -> np.ndarray:
    """Get normalized (x, y) for a landmark index."""
    x, y, _ = landmarks_468[idx][1]
    return np.array([x, y], dtype=np.float32)


def _eye_aspect_ratio(landmarks_468, idxs):
    """Eye aspect ratio (EAR).
    
    idxs: [outer_corner, inner_corner, top1, top2, bottom1, bottom2]
    EAR = (||top1-bottom1|| + ||top2-bottom2||) / (2 * ||outer-inner||)
    """
    outer = _norm(landmarks_468, idxs[0])
    inner = _norm(landmarks_468, idxs[1])
    top1 = _norm(landmarks_468, idxs[2])
    top2 = _norm(landmarks_468, idxs[3])
    bot1 = _norm(landmarks_468, idxs[4])
    bot2 = _norm(landmarks_468, idxs[5])
    
    h = np.linalg.norm(outer - inner)
    if h < 1e-6:
        return 0.0
    v1 = np.linalg.norm(top1 - bot1)
    v2 = np.linalg.norm(top2 - bot2)
    return (v1 + v2) / (2.0 * h)


def _mouth_aspect_ratio(landmarks_468):
    """Mouth aspect ratio (MAR) - height / width."""
    top = _norm(landmarks_468, MOUTH_TOP_CENTER)
    bottom = _norm(landmarks_468, MOUTH_BOTTOM_CENTER)
    left = _norm(landmarks_468, MOUTH_CORNER_LEFT)
    right = _norm(landmarks_468, MOUTH_CORNER_RIGHT)
    
    h = np.linalg.norm(top - bottom)
    w = np.linalg.norm(left - right)
    if w < 1e-6:
        return 0.0
    return h / w


def _mouth_corner_curvature(landmarks_468):
    """Corners relative to lip center. >0 = corners up (smile), <0 = down (frown)."""
    left_corner = _norm(landmarks_468, MOUTH_CORNER_LEFT)
    right_corner = _norm(landmarks_468, MOUTH_CORNER_RIGHT)
    center_top = _norm(landmarks_468, MOUTH_TOP_CENTER)
    center_bottom = _norm(landmarks_468, MOUTH_BOTTOM_CENTER)
    lip_center_y = (center_top[1] + center_bottom[1]) / 2.0
    avg_corner_y = (left_corner[1] + right_corner[1]) / 2.0
    # In image coords, y increases downward, so "up" is negative
    return lip_center_y - avg_corner_y


def _mouth_width_ratio(landmarks_468):
    """Mouth width / face width."""
    left = _norm(landmarks_468, MOUTH_CORNER_LEFT)
    right = _norm(landmarks_468, MOUTH_CORNER_RIGHT)
    mouth_w = np.linalg.norm(left - right)
    
    face_left = _norm(landmarks_468, FACE_OVAL_IDXS[0])
    face_right = _norm(landmarks_468, FACE_OVAL_IDXS[len(FACE_OVAL_IDXS)//2])
    face_w = np.linalg.norm(face_left - face_right)
    
    if face_w < 1e-6:
        return 0.0
    return mouth_w / face_w


def _eyebrow_eye_distance(landmarks_468, brow_idxs, eye_idxs):
    """Average vertical distance from eyebrow to eye. Smaller = lowered brows."""
    brow_y = np.mean([_norm(landmarks_468, i)[1] for i in brow_idxs])
    eye_y = np.mean([_norm(landmarks_468, i)[1] for i in eye_idxs])
    return eye_y - brow_y  # positive = brow above eye


def _eyebrow_slope(landmarks_468, brow_idxs):
    """Inner vs outer eyebrow height. <0 = inner up (sad), >0 = inner down (angry)."""
    inner = _norm(landmarks_468, brow_idxs[0])  # medial end
    outer = _norm(landmarks_468, brow_idxs[-1]) # lateral end
    return inner[1] - outer[1]  # negative = inner higher (sad)


def _cheek_raise(landmarks_468):
    """Average cheek y relative to nose tip. Lower (more negative) = raised (smile)."""
    nose_tip = _norm(landmarks_468, 1)
    left_cheek = np.mean([_norm(landmarks_468, i) for i in CHEEK_LEFT_IDXS], axis=0)
    right_cheek = np.mean([_norm(landmarks_468, i) for i in CHEEK_RIGHT_IDXS], axis=0)
    avg_cheek_y = (left_cheek[1] + right_cheek[1]) / 2.0
    return nose_tip[1] - avg_cheek_y  # positive = cheeks raised


def extract_features(landmarks_468: List[Tuple[int, Tuple[float, float, float]]]) -> Dict[str, float]:
    """Compute all geometric features from normalized landmarks."""
    feats = {}

    # Eyes - EAR
    left_ear = _eye_aspect_ratio(landmarks_468, EYE_LEFT_IDXS)
    right_ear = _eye_aspect_ratio(landmarks_468, EYE_RIGHT_IDXS)
    feats["ear_avg"] = (left_ear + right_ear) / 2.0
    feats["ear_left"] = left_ear
    feats["ear_right"] = right_ear
    feats["ear_asym"] = abs(left_ear - right_ear)

    # Mouth
    feats["mar"] = _mouth_aspect_ratio(landmarks_468)
    feats["mouth_curve"] = _mouth_corner_curvature(landmarks_468)
    feats["mouth_width_ratio"] = _mouth_width_ratio(landmarks_468)

    # Eyebrows
    left_brow_dist = _eyebrow_eye_distance(landmarks_468, EYEBROW_LEFT_IDXS, EYE_LEFT_IDXS)
    right_brow_dist = _eyebrow_eye_distance(landmarks_468, EYEBROW_RIGHT_IDXS, EYE_RIGHT_IDXS)
    feats["brow_eye_dist_avg"] = (left_brow_dist + right_brow_dist) / 2.0
    left_brow_slope = _eyebrow_slope(landmarks_468, EYEBROW_LEFT_IDXS)
    right_brow_slope = _eyebrow_slope(landmarks_468, EYEBROW_RIGHT_IDXS)
    feats["brow_slope_avg"] = (left_brow_slope + right_brow_slope) / 2.0

    # Cheeks
    feats["cheek_raise"] = _cheek_raise(landmarks_468)

    return feats


# ============================================================================
# EmotionDetector — stateful, smoothed per-frame
# ============================================================================

# Resting/neutral feature values (from neutral synthetic face baseline)
REST = {
    "ear_avg": 0.40,           # eyes open
    "mar": 0.25,               # mouth closed
    "mouth_curve": 0.0,        # neutral corners
    "mouth_width_ratio": 0.34, # relaxed mouth width
    "brow_eye_dist_avg": 0.074,# brows at rest
    "brow_slope_avg": 0.0,     # brows level
    "cheek_raise": 0.013,      # cheeks neutral
}


def _score_happy(feats):
    # Smile: corners up, mouth wide, cheeks raised, slight squint (EAR < rest)
    s = 0.0
    s += max(0, feats["mouth_curve"] * 40)           # corners up
    s += max(0, feats["cheek_raise"] * 20)           # cheeks raised
    s += max(0, (feats["mouth_width_ratio"] - REST["mouth_width_ratio"]) * 10)
    s += max(0, (REST["ear_avg"] - feats["ear_avg"]) * 10)  # slight squint
    return s


def _score_sad(feats):
    # Frown: corners down AND inner brows up (negative slope)
    # Gate with mouth_curve: only score if mouth is frowning
    frown_gate = max(0, -feats["mouth_curve"])
    if frown_gate == 0:
        return 0.0
    s = 0.0
    s += frown_gate * 35                              # corners down (gated)
    s += max(0, -feats["brow_slope_avg"] * 25)       # inner brows up
    s += max(0, (feats["brow_eye_dist_avg"] - REST["brow_eye_dist_avg"]) * 10)
    return s


def _score_angry(feats):
    # Brows down and together (small brow_eye_dist), inner brows down (positive slope), eyes narrowed
    s = 0.0
    s += max(0, (REST["brow_eye_dist_avg"] - feats["brow_eye_dist_avg"]) * 40)  # brows lowered
    s += max(0, feats["brow_slope_avg"] * 20)        # inner brows down
    s += max(0, (REST["ear_avg"] - feats["ear_avg"]) * 25)  # eyes narrowed
    s += max(0, -feats["mouth_curve"] * 5)           # slight frown
    return s


def _score_surprised(feats):
    # Requires COMBINATION: wide eyes AND raised brows AND open mouth
    # Use multiplicative gating so all three must be present
    eye_open = max(0, feats["ear_avg"] - REST["ear_avg"])       # >0 = wider than rest
    brows_up = max(0, feats["brow_eye_dist_avg"] - REST["brow_eye_dist_avg"])  # >0 = raised
    mouth_open = max(0, feats["mar"] - REST["mar"])             # >0 = open mouth
    
    # Multiplicative - all three must be positive for high score
    s = eye_open * brows_up * mouth_open * 1500
    
    # Small additive bonus for each individually
    s += eye_open * 5
    s += brows_up * 8
    s += mouth_open * 3
    return s


def _score_neutral(feats):
    # Baseline: penalize deviation from rest on all features
    s = 0.0
    s -= abs(feats["ear_avg"] - REST["ear_avg"]) * 15
    s -= abs(feats["mar"] - REST["mar"]) * 15
    s -= abs(feats["mouth_curve"]) * 15
    s -= abs(feats["brow_eye_dist_avg"] - REST["brow_eye_dist_avg"]) * 15
    s -= abs(feats["brow_slope_avg"]) * 15
    s -= abs(feats["cheek_raise"] - REST["cheek_raise"]) * 10
    s -= abs(feats["mouth_width_ratio"] - REST["mouth_width_ratio"]) * 10
    return s


_SCORERS = {
    "Happy": _score_happy,
    "Sad": _score_sad,
    "Angry": _score_angry,
    "Surprised": _score_surprised,
    "Neutral": _score_neutral,
}

_EMOTIONS = list(_SCORERS.keys())


class EmotionDetector:
    """Detects emotion from FaceMesh landmarks with temporal smoothing.

    Args:
        smoothing_tau: Time constant for exponential moving average (seconds).
                       0.8-1.2 gives ~1s response, stable output.
        min_face_size: Minimum face_size_ratio from FaceLandmarks to trust.
    """
    def __init__(self, smoothing_tau: float = 1.0, min_face_size: float = 0.25):
        self.smoothing_tau = smoothing_tau
        self.min_face_size = min_face_size
        self._smoothed_scores: Dict[str, float] = {e: 0.0 for e in _EMOTIONS}
        self._last_result: Optional[EmotionResult] = None
        self._last_face_size: float = 0.0
        self._initialized = False

    def update(self, face, dt: float) -> Optional[EmotionResult]:
        """Process one frame's FaceLandmarks.

        Args:
            face: FaceLandmarks object from face_tracker (468 points, bbox, is_detected).
            dt: Time since last update (seconds).

        Returns:
            EmotionResult if face.is_detected and face_size >= min_face_size,
            None otherwise (face too small or not detected).
        """
        if face is None or not face.is_detected:
            self._last_result = None
            self._initialized = False
            return None

        if face.face_size_ratio < self.min_face_size:
            self._last_result = None
            self._initialized = False
            return None

        feats = extract_features(face.landmarks_468)

        # Raw scores for each emotion
        raw = {e: fn(feats) for e, fn in _SCORERS.items()}

        # Softmax normalization → percentages sum to 100
        max_raw = max(raw.values())
        exp = {e: math.exp(v - max_raw) for e, v in raw.items()}
        total = sum(exp.values())
        percentages = {e: 100.0 * v / total for e, v in exp.items()}

        # Temporal smoothing (EMA)
        alpha = 1.0 - math.exp(-dt / max(self.smoothing_tau, 1e-3))
        for e in _EMOTIONS:
            if not self._initialized:
                self._smoothed_scores[e] = percentages[e]
            else:
                self._smoothed_scores[e] += alpha * (percentages[e] - self._smoothed_scores[e])

        # Normalize smoothed values to sum to 100
        smoothed_sum = sum(self._smoothed_scores.values())
        smoothed_pct = {e: 100.0 * v / smoothed_sum for e, v in self._smoothed_scores.items()}

        top = max(smoothed_pct, key=smoothed_pct.get)
        top_pct = smoothed_pct[top]

        result = EmotionResult(
            percentages=smoothed_pct,
            top_emotion=top,
            top_certainty=top_pct,
            is_confident=top_pct >= 60.0,
        )
        self._last_result = result
        self._initialized = True
        self._last_face_size = face.face_size_ratio
        return result

    def get_last_result(self) -> Optional[EmotionResult]:
        """Return last result even if current frame had no face."""
        return self._last_result

    def reset(self) -> None:
        self._smoothed_scores = {e: 0.0 for e in _EMOTIONS}
        self._last_result = None
        self._initialized = False


# ============================================================================
# Standalone test with synthetic landmarks
# ============================================================================

def _make_synthetic_face(shape: str) -> List[Tuple[int, Tuple[float, float, float]]]:
    """Create a 468-point face with realistic base shape + expression."""
    # Start with a realistic face template (normalized coordinates)
    # Key landmarks positioned approximately correctly
    base_coords = {}
    
    # Face oval (jaw + forehead)
    oval_pts = FACE_OVAL_IDXS
    for i, idx in enumerate(oval_pts):
        angle = 2 * math.pi * i / len(oval_pts)
        # Ellipse: wider than tall
        base_coords[idx] = (0.5 + 0.35 * math.cos(angle), 0.5 + 0.45 * math.sin(angle))
    
    # Eyes
    base_coords[33] = (0.35, 0.42)    # left outer
    base_coords[133] = (0.45, 0.42)   # left inner
    base_coords[159] = (0.40, 0.40)   # left top
    base_coords[158] = (0.40, 0.40)   # left top
    base_coords[145] = (0.40, 0.44)   # left bottom
    base_coords[144] = (0.40, 0.44)   # left bottom
    
    base_coords[362] = (0.65, 0.42)   # right outer
    base_coords[263] = (0.55, 0.42)   # right inner
    base_coords[386] = (0.60, 0.40)   # right top
    base_coords[385] = (0.60, 0.40)   # right top
    base_coords[374] = (0.60, 0.44)   # right bottom
    base_coords[373] = (0.60, 0.44)   # right bottom
    
    # Mouth
    base_coords[61] = (0.38, 0.65)    # left corner
    base_coords[291] = (0.62, 0.65)   # right corner
    base_coords[13] = (0.50, 0.62)    # top center
    base_coords[14] = (0.50, 0.68)    # bottom center
    base_coords[82] = (0.50, 0.63)    # inner top
    base_coords[87] = (0.50, 0.67)    # inner bottom
    
    # Eyebrows
    base_coords[70] = (0.37, 0.35)    # left medial
    base_coords[63] = (0.40, 0.34)
    base_coords[105] = (0.43, 0.34)
    base_coords[66] = (0.45, 0.35)
    base_coords[107] = (0.48, 0.35)   # left lateral
    
    base_coords[300] = (0.52, 0.35)   # right medial
    base_coords[293] = (0.55, 0.34)
    base_coords[334] = (0.57, 0.34)
    base_coords[296] = (0.59, 0.35)
    base_coords[336] = (0.63, 0.35)   # right lateral
    
    # Nose tip
    base_coords[1] = (0.50, 0.55)
    
    # Cheeks
    base_coords[116] = (0.32, 0.55)
    base_coords[117] = (0.33, 0.57)
    base_coords[345] = (0.68, 0.55)
    base_coords[346] = (0.67, 0.57)

    # Fill remaining with interpolation
    for i in range(468):
        if i not in base_coords:
            # Find nearest known landmark
            nearest = min(base_coords.keys(), key=lambda k: abs(k - i))
            base_coords[i] = base_coords[nearest]

    # Apply expression perturbations
    def perturb(idx, dx, dy):
        if idx in base_coords:
            x, y = base_coords[idx]
            base_coords[idx] = (x + dx, y + dy)

    if shape == "smile":
        # Corners up, mouth wider, cheeks raised, slight squint
        perturb(61, 0.015, -0.025)
        perturb(291, -0.015, -0.025)
        perturb(13, 0, -0.005)
        perturb(14, 0, 0.005)
        perturb(82, 0, -0.003)
        perturb(87, 0, 0.003)
        perturb(116, 0, -0.015)
        perturb(117, 0, -0.015)
        perturb(345, 0, -0.015)
        perturb(346, 0, -0.015)
        perturb(159, 0, -0.005)
        perturb(145, 0, 0.005)
        perturb(386, 0, -0.005)
        perturb(374, 0, 0.005)
    elif shape == "frown":
        # Corners down, inner brows up
        perturb(61, 0, 0.02)
        perturb(291, 0, 0.02)
        perturb(70, 0, -0.02)   # left medial up
        perturb(300, 0, -0.02)  # right medial up
        perturb(63, 0, -0.015)
        perturb(293, 0, -0.015)
    elif shape == "wide_eyes":
        # Eyes wide open, brows raised
        perturb(159, 0, -0.015)
        perturb(145, 0, 0.015)
        perturb(386, 0, -0.015)
        perturb(374, 0, 0.015)
        perturb(70, 0, -0.015)
        perturb(300, 0, -0.015)
        perturb(63, 0, -0.01)
        perturb(293, 0, -0.01)
    elif shape == "brows_up":
        perturb(70, 0, -0.02)
        perturb(300, 0, -0.02)
        perturb(63, 0, -0.015)
        perturb(293, 0, -0.015)
    elif shape == "brows_down":
        perturb(70, 0, 0.02)
        perturb(300, 0, 0.02)
        perturb(63, 0, 0.015)
        perturb(293, 0, 0.015)

    # Build final list
    return [(i, (*base_coords[i], 0.0)) for i in range(468)]


def test_emotion_detector():
    """Run synthetic face through detector and print results."""
    print("Testing EmotionDetector with synthetic faces...\n")
    for shape in ["smile", "frown", "wide_eyes", "brows_up", "brows_down"]:
        face_lms = _make_synthetic_face(shape)
        face = type('Face', (), {
            'landmarks_468': face_lms,
            'is_detected': True,
            'face_size_ratio': 0.6,
        })()
        detector = EmotionDetector(smoothing_tau=0.0)  # no smoothing for test
        result = detector.update(face, 1/30)
        print(f"=== {shape.upper()} ===")
        print(result)
        print(f"Top: {result.top_emotion} ({result.top_certainty:.1f}%)\n")


if __name__ == "__main__":
    test_emotion_detector()