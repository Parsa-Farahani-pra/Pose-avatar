#!/usr/bin/env python3
"""Live single-person pose estimation from a webcam (MediaPipe Pose, CPU)."""

from __future__ import annotations

import argparse
import sys
import time

import cv2
import mediapipe as mp

from hand_tracker import HandTracker, draw_hand_on_frame

LANDMARK_COLOR = (0, 255, 0)
BONE_COLOR = (0, 200, 255)
FPS_COLOR = (255, 255, 255)
POINT_RADIUS = 4
BONE_THICKNESS = 2


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


def draw_pose(
    frame,
    landmarks,
    connections,
    min_visibility: float,
) -> None:
    height, width = frame.shape[:2]
    points: list[tuple[int, int] | None] = []

    for landmark in landmarks.landmark:
        if landmark.visibility >= min_visibility:
            x = int(landmark.x * width)
            y = int(landmark.y * height)
            points.append((x, y))
        else:
            points.append(None)

    for start_idx, end_idx in connections:
        start = points[start_idx]
        end = points[end_idx]
        if start is None or end is None:
            continue
        cv2.line(frame, start, end, BONE_COLOR, BONE_THICKNESS, cv2.LINE_AA)

    for point in points:
        if point is None:
            continue
        cv2.circle(frame, point, POINT_RADIUS, LANDMARK_COLOR, -1, cv2.LINE_AA)


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

            if result.pose_landmarks:
                draw_pose(
                    frame,
                    result.pose_landmarks,
                    mp_pose.POSE_CONNECTIONS,
                    args.min_visibility,
                )

            if hand_tracker is not None:
                hands_dict = hand_tracker.process_frame(rgb)
                for label, hand_lms in hands_dict.items():
                    draw_hand_on_frame(frame, hand_lms)

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
            if key in (ord("q"), 27):
                break
    finally:
        if hand_tracker is not None:
            hand_tracker.close()
        pose.close()
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
