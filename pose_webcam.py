#!/usr/bin/env python3
"""Live single-person pose estimation from a webcam (MediaPipe Pose, CPU)."""

from __future__ import annotations

import argparse
import sys
import time

import cv2
import mediapipe as mp

from hand_tracker import HandTracker, draw_hand_on_frame
from face_tracker import (
    FaceWorker, draw_face_mesh_subset, debug_face_bounds_display,
    NOSE_TIP, ALL_FACE_MESH_REGIONS, POSE_FACE_LANDMARKS
)

LANDMARK_COLOR = (0, 255, 0)
BONE_COLOR = (0, 200, 255)
FPS_COLOR = (255, 255, 255)
POINT_RADIUS = 4
BONE_THICKNESS = 2


def draw_pose(
    frame,
    landmarks,
    connections,
    min_visibility: float,
    show_pose_face: bool = True,
) -> None:
    """Draw pose skeleton, optionally hiding Pose's own face landmarks.

    Args:
        frame: Image frame to draw on (modified in place).
        landmarks: MediaPipe Pose landmarks.
        connections: Pose connections (bone pairs) to draw.
        min_visibility: Draw a landmark only if visibility >= this value.
        show_pose_face: If False, skip Pose face landmarks (indices 0-10)
            so FaceMesh is the only face visualization on screen.
    """
    height, width = frame.shape[:2]
    # Keyed by ORIGINAL landmark index -> pixel position.
    # Using a dict avoids index-shift bugs when landmarks are skipped.
    landmark_positions: dict[int, tuple[int, int]] = {}

    for idx, landmark in enumerate(landmarks.landmark):
        if not show_pose_face and idx in POSE_FACE_LANDMARKS:
            continue
        if landmark.visibility >= min_visibility:
            x = int(landmark.x * width)
            y = int(landmark.y * height)
            landmark_positions[idx] = (x, y)

    # Bones are drawn only when BOTH endpoints are visible landmarks;
    # connections touching hidden face landmarks are skipped automatically.
    for start_idx, end_idx in connections:
        start = landmark_positions.get(start_idx)
        end = landmark_positions.get(end_idx)
        if start is None or end is None:
            continue
        cv2.line(frame, start, end, BONE_COLOR, BONE_THICKNESS, cv2.LINE_AA)

    for point in landmark_positions.values():
        cv2.circle(frame, point, POINT_RADIUS, LANDMARK_COLOR, -1, cv2.LINE_AA)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Show live webcam pose landmarks (single person, CPU)."
    )
    parser.add_argument(
        "-c",
        "--camera",
        type=int,
        default=0,
        help="Camera device index (default: 0).",
    )
    parser.add_argument(
        "--min-visibility",
        type=float,
        default=0.5,
        help="Draw a landmark only if visibility is at least this value (default: 0.5).",
    )
    parser.add_argument(
        "--no-flip",
        action="store_true",
        help="Do not mirror the frame (selfie view is on by default).",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=640,
        help="Requested capture width (default: 640).",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=480,
        help="Requested capture height (default: 480).",
    )
    parser.add_argument(
        "--enable-hands",
        action="store_true",
        help="Enable simultaneous hand and finger tracking.",
    )
    parser.add_argument(
        "--enable-face",
        action="store_true",
        help="Enable MediaPipe Face Mesh tracking (lightweight 30-40 landmark subset for performance).",
    )
    return parser.parse_args()


def open_camera(index: int, width: int, height: int) -> cv2.VideoCapture:
    cap = cv2.VideoCapture(index)
    if not cap.isOpened():
        print(
            f"Could not open camera {index}. "
            "Try another device with --camera 1 (or a higher index), "
            "and make sure no other app is using the webcam.",
            file=sys.stderr,
        )
        sys.exit(1)

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

    ok, _frame = cap.read()
    if not ok:
        cap.release()
        print(
            f"Camera {index} opened but no frame could be read. "
            "Try --camera with a different index, or check that the device is connected.",
            file=sys.stderr,
        )
        sys.exit(1)

    return cap


def main() -> None:
    args = parse_args()
    cap = open_camera(args.camera, args.width, args.height)
    flip = not args.no_flip

    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(
        static_image_mode=False,
        model_complexity=1,
        smooth_landmarks=True,
        enable_segmentation=False,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    hand_tracker = HandTracker() if args.enable_hands else None
    face_worker = FaceWorker(args) if args.enable_face else None

    prev_time = time.perf_counter()
    window_name = "Pose (q or Esc to quit)"

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("Lost camera feed; exiting.", file=sys.stderr)
                break

            if flip:
                frame = cv2.flip(frame, 1)

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            rgb.flags.writeable = False
            result = pose.process(rgb)

            # Process face mesh once per frame; the result drives both the
            # skeleton face-landmark toggle and the face mesh overlay.
            face_landmarks = None
            if face_worker is not None:
                face_landmarks = face_worker.process_frame(frame)
            face_detected = face_landmarks is not None and face_landmarks.is_detected

            if result.pose_landmarks:
                # Hide Pose's 11 face landmarks while FaceMesh is actively
                # showing the full face mesh; restore them otherwise.
                draw_pose(
                    frame,
                    result.pose_landmarks,
                    mp_pose.POSE_CONNECTIONS,
                    args.min_visibility,
                    show_pose_face=not face_detected,
                )

            if hand_tracker is not None:
                hands_dict = hand_tracker.process_frame(rgb)
                for label, hand_lms in hands_dict.items():
                    draw_hand_on_frame(frame, hand_lms)

            # Render the full 468-landmark face mesh when a face is detected
            if face_detected:
                draw_face_mesh_subset(frame, face_landmarks, mode="full")
                debug_face_bounds_display(frame, face_landmarks.face_bbox, face_landmarks.face_size_ratio)

            now = time.perf_counter()
            elapsed = now - prev_time
            prev_time = now
            fps = 1.0 / elapsed if elapsed > 0 else 0.0
            cv2.putText(
                frame,
                f"{fps:.1f} FPS",
                (10, 24),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                FPS_COLOR,
                2,
                cv2.LINE_AA,
            )

            cv2.imshow(window_name, frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("f"):
                print("Face mesh enabled" if face_worker.cfg.enable_face else "Face mesh disabled")
            if key in (ord("q"), 27):
                break
    finally:
        if hand_tracker is not None:
            hand_tracker.close()
        if face_worker is not None:
            # FaceWorker doesn't need explicit close() but we call it for consistency
            face_worker.close()
        pose.close()
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
