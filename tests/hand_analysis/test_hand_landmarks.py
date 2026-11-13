#!/usr/bin/env python3
"""Test MediaPipe hand landmarks detection and visualization."""

from __future__ import annotations

import argparse
import cv2
import mediapipe as mp
import time
from pathlib import Path

# Add tests/ to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.hand_tracker import (
    WRIST, THUMB_TIP, INDEX_TIP, MIDDLE_TIP, RING_TIP, PINKY_TIP,
    draw_hand_on_frame
)

def draw_hand_landmarks(frame, hand_landmarks, mp_hands, frame_idx=0):
    """Draw hand landmarks and anatomical connections on frame."""
    h, w = frame.shape[:2]

    # Color coding by finger
    colors = {
        0: (255, 0, 255),           # Thumb base (magenta)
        1: (255, 0, 255),           # Thumb MCP
        4: (255, 0, 255),           # Thumb tip
        5: (0, 255, 255),           # Index MCP
        8: (0, 255, 255),           # Index tip
        9: (0, 255, 0),             # Middle MCP
        12: (0, 255, 0),            # Middle tip
        13: (0, 255, 255),          # Ring MCP
        16: (0, 255, 255),          # Ring tip
        17: (0, 255, 255),          # Pinky MCP
        20: (0, 255, 255),          # Pinky tip
    }

    # Draw connections (MediaPipe predefined)
    connections = mp_hands.HAND_CONNECTIONS

    # Draw bone connections
    for start, end in connections:
        start_pt = None
        end_pt = None

        for lm_idx, landmark in enumerate(hand_landmarks.landmark):
            if lm_idx == start:
                start_pt = (int(landmark.x * w), int(landmark.y * h))
            if lm_idx == end:
                end_pt = (int(landmark.x * w), int(landmark.y * h))

        if start_pt and end_pt:
            # Color based on finger
            mid_idx = (start + end) // 2
            color = colors.get(mid_idx, (255, 255, 255))
            cv2.line(frame, start_pt, end_pt, color, 3, cv2.LINE_AA)

    # Draw landmarks on top
    for idx, landmark in enumerate(hand_landmarks.landmark):
        x, y = int(landmark.x * w), int(landmark.y * h)
        vis = landmark.visibility

        # Select color based on finger index
        color = colors.get(idx, (255, 255, 255))

        # Size based on visibility
        radius = 4 if vis > 0.5 else 2
        alpha = int(vis * 255) if idx not in (0, 4, 8, 12, 16, 20) else 255

        # Create colored circle
        overlay = frame.copy()
        cv2.circle(overlay, (x, y), radius, color, -1, cv2.LINE_AA)
        cv2.addWeighted(overlay, alpha / 255, frame, 1 - alpha / 255, 0, frame)

    # Label finger tips
    finger_labels = {
        4: "Thumb",
        8: "Index",
        12: "Middle",
        16: "Ring",
        20: "Pinky"
    }

    for tip_idx, label in finger_labels.items():
        tip = hand_landmarks.landmark[tip_idx]
        label_pt = (int(tip.x * w), int(tip.y * h - 15))
        cv2.putText(frame, label, label_pt, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    return frame

def analyze_hand_data(hand_landmarks, handedness):
    """Analyze hand landmark data and return quality metrics."""
    landmarks = hand_landmarks.landmark

    # Absolute distances between two landmark points
    def landmark_distance(idx1, idx2):
        p1 = landmarks[idx1]
        p2 = landmarks[idx2]
        dx = p1.x - p2.x
        dy = p1.y - p2.y
        dz = p1.z - p2.z
        return (dx*dx + dy*dy + dz*dz) ** 0.5

    # Count visible landmarks
    visible_count = sum(1 for lm in landmarks if lm.visibility > 0.2)
    all_visible = all(lm.visibility > 0.5 for lm in landmarks)

    # Calculate palm width (thumb base to pinky base distance)
    palm_size = landmark_distance(1, 17)

    # Calculate wrist to finger 8 (palm depth estimate)
    wrist_to_index = landmark_distance(0, 8)

    # Finger lengths
    finger_lengths = {
        "thumb": landmark_distance(1, 4),
        "index": landmark_distance(5, 8),
        "middle": landmark_distance(9, 12),
        "ring": landmark_distance(13, 16),
        "pinky": landmark_distance(17, 20),
    }

    return {
        'handedness': handedness,
        'visible_count': visible_count,
        'all_visible': all_visible,
        'palm_size': palm_size,
        'finger_lengths': finger_lengths,
    }

def print_hand_stats(stats, frame_idx=0):
    """Print formatted statistics for current hand detection."""
    print(f"\n{'='*60}")
    print(f"Frame {frame_idx}")
    print(f"{'='*60}")
    print(f"Hand: {stats['handedness'].upper()}")
    print(f"Visible landmarks: {stats['visible_count']}/21 ({stats['visible_count']/21*100:.1f}%)")
    print(f"Palm size: {stats['palm_size']:.3f}m (palm extent)")
    print(f"Palm width: {stats['palm_size']:.3f}m (thumb to pinky)")

    if stats.get('finger_lengths'):
        print("\nFinger lengths (relative):")
        for finger, length in sorted(stats['finger_lengths'].items()):
            print(f"  - {finger}: {length:.3f}m")

    # Detection quality assessment
    if stats['all_visible']:
        quality = "EXCELLENT"
        color = (0, 255, 0)
    elif stats['visible_count'] >= 14:
        quality = "GOOD"
        color = (255, 165, 0)
    else:
        quality = "POOR"
        color = (255, 0, 0)

    print(f"\nDetection quality: {quality}")

    quality

    return quality

def main():
    parser = argparse.ArgumentParser(description="Test MediaPipe hand landmarks")
    parser.add_argument("--camera", type=int, default=0, help="Camera device index")
    parser.add_argument("--confidence", type=float, default=0.7,
                       help="Minimum detection confidence (0.7-0.9)")
    parser.add_argument("--min-visibility", type=float, default=0.2,
                       help="Minimum landmark visibility threshold")
    parser.add_argument("--min-hand-count", type=int, default=1,
                       help="Minimum number of hands to detect before showing")
    args = parser.parse_args()

    print("=" * 60)
    print("MediaPipe Hand Landmarks Test")
    print("=" * 60)
    print(f"\nConfiguration:")
    print(f"  • Camera: {args.camera}")
    print(f"  • Detection confidence: {args.confidence}")
    print(f"  • Landmark visibility: {args.min_visibility}")
    print(f"  • Minimum hands: {args.min_hand_count}")

    # Open camera
    print(f"\nOpening camera {args.camera}...")
    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        print(f"\n✗ Could not open camera {args.camera}")
        print("   Tip: Try --camera 1 or --camera 2")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    # Initialize MediaPipe Hands
    print("\nInitializing MediaPipe Hands...")
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=2,
        min_detection_confidence=args.confidence,
        min_tracking_confidence=args.min_visibility + 0.2
    )

    print("✓ MediaPipe Hands initialized")

    # Test image for compatibility check
    test_img = cv2.imread('/tmp/test_hand.jpg', cv2.IMREAD_COLOR)

    # Stats tracking
    frames = 0
    hand_detections = []
    detection_times = []
    avg_fps = 0.0

    # Window setup
    cv2.namedWindow("Hand Landmarks Test")

    prev_time = time.perf_counter()
    hands_detected_count = 0

    print("\n" + "=" * 60)
    print("Controls:")
    print("  • Hold up hands to test detection")
    print("  • Spread fingers to test articulation")
    print("  • Close fingers to test curl detection")
    print("  • Press 'q' to quit")
    print("=" * 60)

    running = True
    while running and frames < 600:  # Run for 10 seconds (600 frames @ 60fps)
        ret, frame = cap.read()
        if not ret:
            print("\n✗ Could not read camera frame")
            break

        # Flip for selfie view
        frame = cv2.flip(frame, 1)

        # Process hands
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rgb.flags.writeable = False

        start_time = time.perf_counter()
        results = hands.process(rgb)
        process_time = time.perf_counter() - start_time

        # Echo inference timing
        fps_now = 1.0 / (time.perf_counter() - prev_time) if time.perf_counter() - prev_time > 0 else 0.0
        prev_time = time.perf_counter()
        avg_fps = 0.95 * avg_fps + 0.05 * fps_now

        # Process detections
        if results.multi_hand_landmarks:
            current_hand_count = len(results.multi_hand_landmarks)
            hands_detected_count += current_hand_count

            # Check if we have minimum hands before processing
            if current_hand_count >= args.min_hand_count:
                for idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
                    handedness = results.multi_handedness[idx].classification[0].label

                    # Draw landmarks
                    frame = draw_hand_landmarks(frame, hand_landmarks, mp_hands, frames)

                    # Analyze data
                    stats = analyze_hand_data(hand_landmarks, handedness)
                    quality = print_hand_stats(stats, frames)

                    hand_detections.append({
                        'frame': frames,
                        'hand_idx': idx,
                        'handedness': handedness,
                        'stats': stats,
                        'quality': quality
                    })

                    # Store detection time and landmark data
                    detection_times.append({
                        'frame': frames,
                        'hand_idx': idx,
                        'timestamp': time.perf_counter(),
                        'landmarks': [dict(pose=(lm.x, lm.y, lm.z), vis=lm.visibility)
                                     for lm in hand_landmarks.landmark],
                        'handedness': handedness
                    })
        else:
            # No hands detected - occasionally show message
            if frames % 10 == 0:
                cv2.putText(frame, "No hands detected - Hold up both hands",
                           (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

                # Draw a simple hand outline
                skeleton_img = cv2.imread('/tmp/skeleton_hint.png', cv2.IMREAD_COLOR)
                if skeleton_img is not None:
                    frame[100:200, 200:300, :] = skeleton_img

        # Display camera stats and landmarks info
        cv2.putText(frame, f"FPS: {avg_fps:.1f}", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(frame, f"Hands detected: {hands_detected_count}", (10, 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)
        cv2.putText(frame, f"Frames: {frames}", (10, 90),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(frame, f"Process time: {process_time*1000:.1f}ms", (10, 120),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 2)

        # Try signature hints from library markdown
        try:
            import mediapipe.solutions.hands
            sig_path = getattr(mediapipe.solutions.hands, '__file__', None)
            if sig_path:
                import datetime
                # Show when module was loaded/frozen
                load_time = datetime.datetime.fromtimestamp(
                    getattr(mediapipe, '_loading_fresh', __file__))
        except Exception:
            pass

        # Display frame
        cv2.imshow("Hand Landmarks Test", frame)

        # Check quit
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or key == 27:
            running = False

        frames += 1

    # Cleanup
    hands.close()
    cap.release()
    cv2.destroyAllWindows()

    # Print summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    print(f"Total frames processed: {frames}")
    print(f"Hands detected: {hands_detected_count}")

    if detection_times:
        print(f"\nDetection Statistics:")
        print(f"  • Total detections: {len(detection_times)}")

        # Count left/right
        left_count = sum(1 for d in detection_times if d['handedness'] == 'Left')
        right_count = sum(1 for d in detection_times if d['handedness'] == 'Right')
        print(f"  • Left hand: {left_count}")
        print(f"  • Right hand: {right_count}")

        # Quality distribution
        excellent = sum(1 for d in hand_detections if d['quality'] == 'EXCELLENT')
        good = sum(1 for d in hand_detections if d['quality'] == 'GOOD')
        poor = sum(1 for d in hand_detections if d['quality'] == 'POOR')

        print(f"\nDetection Quality:")
        print(f"  • Excellent (all landmarks ≥ 0.5): {excellent}")
        print(f"  • Good (≥ 14 landmarks): {good}")
        print(f"  • Poor (< 14 landmarks): {poor}")

        if excellent > 0:
            print("\n✓ SUCCESS: Hands are being detected and visualized correctly!")
            print("   You can proceed to integration into your pose_avatar.py")
        else:
            print("\n✗ QUALITY ISSUE: Most hands have poor visibility")

    else:
        print("\n✗ No hands detected")
        print("   Try: move closer, raise hands above shoulder, spread fingers")
        print("   Check: mediaPipe version compatibility, lighting, frame exposure")

if __name__ == "__main__":
    main()