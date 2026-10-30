#!/usr/bin/env python3
"""Modular hand and finger tracking module using MediaPipe Hands.

Provides:
- HandLandmarks: Dataclass storing 21 landmarks + 3D relative positions
- HandReference: Pinned reference geometry measured on first detection
- HandWorker: Thread-safe worker that captures or processes frames
- Helper methods for palm calculation and finger angles
"""

from __future__ import annotations

import math
import queue
import threading
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import cv2
import mediapipe as mp

# MediaPipe Hand landmark indices
WRIST = 0
THUMB_CMC, THUMB_MCP, THUMB_IP, THUMB_TIP = 1, 2, 3, 4
INDEX_MCP, INDEX_PIP, INDEX_DIP, INDEX_TIP = 5, 6, 7, 8
MIDDLE_MCP, MIDDLE_PIP, MIDDLE_DIP, MIDDLE_TIP = 9, 10, 11, 12
RING_MCP, RING_PIP, RING_DIP, RING_TIP = 13, 14, 15, 16
PINKY_MCP, PINKY_PIP, PINKY_DIP, PINKY_TIP = 17, 18, 19, 20

# Finger hierarchies for simple phalanx / armature:
# (Base/MCP -> PIP -> DIP -> TIP)
FINGER_CHAINS = {
    "thumb": [THUMB_CMC, THUMB_MCP, THUMB_IP, THUMB_TIP],
    "index": [INDEX_MCP, INDEX_PIP, INDEX_DIP, INDEX_TIP],
    "middle": [MIDDLE_MCP, MIDDLE_PIP, MIDDLE_DIP, MIDDLE_TIP],
    "ring": [RING_MCP, RING_PIP, RING_DIP, RING_TIP],
    "pinky": [PINKY_MCP, PINKY_PIP, PINKY_DIP, PINKY_TIP],
}


@dataclass
class HandConfig:
    min_detection_confidence: float = 0.7
    min_tracking_confidence: float = 0.7
    max_num_hands: int = 2
    smooth_tau: float = 0.12  # Smoothing time constant for hands


@dataclass
class HandLandmarks:
    handedness: str  # "Left" or "Right"
    landmarks_2d: Dict[int, Tuple[float, float]] = field(default_factory=dict)
    landmarks_3d: Dict[int, Tuple[float, float, float]] = field(default_factory=dict)
    visibility: Dict[int, float] = field(default_factory=dict)
    timestamp: float = 0.0


@dataclass
class HandReference:
    """Pinned reference measurements taken once on first clean detection."""
    palm_size: float = 0.08  # Distance between wrist and middle MCP in metres
    palm_width: float = 0.075  # Distance between index MCP and pinky MCP
    bone_lengths: Dict[str, List[float]] = field(default_factory=dict)
    is_set: bool = False

    def calibrate(self, hand_lms: HandLandmarks) -> None:
        """Measure bone segments from the first solid hand sample."""
        pts = hand_lms.landmarks_3d
        if not pts or len(pts) < 21:
            return

        def dist(i1, i2):
            p1, p2 = pts[i1], pts[i2]
            return math.sqrt(
                (p1[0] - p2[0]) ** 2 +
                (p1[1] - p2[1]) ** 2 +
                (p1[2] - p2[2]) ** 2
            )

        # Palm measurements
        self.palm_size = dist(WRIST, MIDDLE_MCP)
        self.palm_width = dist(INDEX_MCP, PINKY_MCP)

        # Bone segments along each digit
        for finger_name, chain in FINGER_CHAINS.items():
            lengths = []
            for i in range(len(chain) - 1):
                lengths.append(dist(chain[i], chain[i + 1]))
            self.bone_lengths[finger_name] = lengths

        self.is_set = True


class HandTracker:
    """Processes images with MediaPipe Hands and formats outputs."""

    def __init__(self, config: Optional[HandConfig] = None):
        self.cfg = config or HandConfig()
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=self.cfg.max_num_hands,
            min_detection_confidence=self.cfg.min_detection_confidence,
            min_tracking_confidence=self.cfg.min_tracking_confidence,
        )
        self.references: Dict[str, HandReference] = {
            "Left": HandReference(),
            "Right": HandReference(),
        }

    def process_frame(
        self, rgb_frame: cv2.Mat, timestamp: Optional[float] = None
    ) -> Dict[str, HandLandmarks]:
        """Process RGB frame and extract normalized and metric coordinates."""
        now = timestamp if timestamp is not None else time.monotonic()
        results = self.hands.process(rgb_frame)

        detected_hands: Dict[str, HandLandmarks] = {}
        if not results.multi_hand_landmarks or not results.multi_handedness:
            return detected_hands

        # multi_hand_world_landmarks provides metric coordinates if available
        world_lms = getattr(results, "multi_hand_world_landmarks", None)

        for idx, (landmarks, handedness_info) in enumerate(
            zip(results.multi_hand_landmarks, results.multi_handedness)
        ):
            # MediaPipe reports handedness for selfie view by default
            label = handedness_info.classification[0].label  # "Left" or "Right"

            hand_data = HandLandmarks(handedness=label, timestamp=now)

            # 2D screen coordinates (0.0 to 1.0)
            for i, lm in enumerate(landmarks.landmark):
                hand_data.landmarks_2d[i] = (lm.x, lm.y)
                hand_data.visibility[i] = getattr(lm, "visibility", 1.0)

            # 3D world or pseudo-world coordinates
            if world_lms and idx < len(world_lms):
                for i, lm in enumerate(world_lms[idx].landmark):
                    hand_data.landmarks_3d[i] = (lm.x, lm.y, lm.z)
            else:
                for i, lm in enumerate(landmarks.landmark):
                    hand_data.landmarks_3d[i] = (lm.x, lm.y, lm.z)

            # Auto-calibrate reference pose once per hand side
            ref = self.references[label]
            if not ref.is_set:
                ref.calibrate(hand_data)

            detected_hands[label] = hand_data

        return detected_hands

    def close(self):
        self.hands.close()


def draw_hand_on_frame(
    frame: cv2.Mat,
    hand_lms: HandLandmarks,
    color_bone=(0, 255, 255),
    color_joint=(255, 0, 128),
) -> None:
    """Draw 2D skeleton overlay on a video frame."""
    h, w = frame.shape[:2]

    # Draw finger chain lines
    for chain in FINGER_CHAINS.values():
        for i in range(len(chain) - 1):
            idx1, idx2 = chain[i], chain[i + 1]
            if idx1 in hand_lms.landmarks_2d and idx2 in hand_lms.landmarks_2d:
                p1 = (
                    int(hand_lms.landmarks_2d[idx1][0] * w),
                    int(hand_lms.landmarks_2d[idx1][1] * h),
                )
                p2 = (
                    int(hand_lms.landmarks_2d[idx2][0] * w),
                    int(hand_lms.landmarks_2d[idx2][1] * h),
                )
                cv2.line(frame, p1, p2, color_bone, 2, cv2.LINE_AA)

    # Connect palm base (wrist to each MCP)
    wrist_pt = hand_lms.landmarks_2d.get(WRIST)
    if wrist_pt:
        w_px = (int(wrist_pt[0] * w), int(wrist_pt[1] * h))
        for mcp in (THUMB_CMC, INDEX_MCP, MIDDLE_MCP, RING_MCP, PINKY_MCP):
            if mcp in hand_lms.landmarks_2d:
                m_px = (
                    int(hand_lms.landmarks_2d[mcp][0] * w),
                    int(hand_lms.landmarks_2d[mcp][1] * h),
                )
                cv2.line(frame, w_px, m_px, color_bone, 1, cv2.LINE_AA)

    # Draw joint markers
    for idx, (nx, ny) in hand_lms.landmarks_2d.items():
        px = (int(nx * w), int(ny * h))
        radius = 4 if idx in (THUMB_TIP, INDEX_TIP, MIDDLE_TIP, RING_TIP, PINKY_TIP) else 2
        cv2.circle(frame, px, radius, color_joint, -1, cv2.LINE_AA)
