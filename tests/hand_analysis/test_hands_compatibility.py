#!/usr/bin/env python3
"""Test MediaPipe hand landmarks compatibility."""

from __future__ import annotations

from pathlib import Path
import sys

# Add tests/ sibling (src/) to path if not already there
current_dir = Path(__file__).parent.parent
if str(current_dir) not in sys.path:
    sys.path.insert(0, str(current_dir))

import mediapipe as mp
import cv2

def main():
    print("=" * 60)
    print("MediaPipe Hands Compatibility Test")
    print("=" * 60)

    print(f"\nMediapipe version: {mp.__version__}")

    # Check if Hands solution exists
    print("\n1. Checking for mp.solutions.hands module...")
    try:
        mp_hands = mp.solutions.hands
        print("   ✓ mp.solutions.hands found!")

        # Test creating a Hands object
        print("\n2. Creating mp.solutions.hands.Hands object...")
        with mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        ) as hands:
            print("   ✓ Successfully created mp.hands.Hands object")
            print(f"      - max_num_hands: {hands.max_num_hands}")

        # Test processing a test image
        print("\n3. Testing with a sample image...")
        # Create a small test image
        test_img = cv2.imread('/tmp/test_hand.jpg', cv2.IMREAD_COLOR)
        if test_img is None:
            # Create a blank white image
            test_img = cv2.imread('/dev/stdin', cv2.IMREAD_COLOR)
            if test_img is None:
                print("   * No image file found, skipping image test")
        else:
            print("   ✓ Image loaded")
            rgb = cv2.cvtColor(test_img, cv2.COLOR_BGR2RGB)
            rgb.flags.writeable = False
            results = hands.process(rgb)

            if results.multi_hand_landmarks:
                print(f"   ✓ Hand detection working! Found {len(results.multi_hand_landmarks)} hand(s)")
                for i, hand_landmarks in enumerate(results.multi_hand_landmarks):
                    print(f"      Hand {i}: {len(hand_landmarks.landmark)} landmarks")
            else:
                print("   ✗ No hands detected in test image (expected for blank image)")

    except AttributeError as e:
        print(f"\n   ✗ AttributeError: {e}")
        print("\n   → MediaPipe version does NOT include mp.solutions.hands")
        print("   → You will need to upgrade MediaPipe")
    except Exception as e:
        print(f"\n   ! Unexpected error: {type(e).__name__}: {e}")

    # Check available solutions
    print("\n" + "=" * 60)
    print("Available MediaPipe Solutions:")
    print("=" * 60)
    print(f"\nmp.solutions has the following modules:")
    try:
        import importlib
        solutions = [name for name in dir(mp.solutions) 
                    if not name.startswith('_') and name[0].islower()]
        for sol in sorted(solutions):
            module = getattr(mp.solutions, sol)
            print(f"  - {sol}")

        # Specific checks
        if hasattr(mp.solutions, 'pose'):
            print("\n  ✓ mp.solutions.pose: AVAILABLE")
        else:
            print("\n  ✗ mp.solutions.pose: MISSING")

        if hasattr(mp.solutions, 'hands'):
            print("  ✓ mp.solutions.hands: AVAILABLE")
        else:
            print("  ✗ mp.solutions.hands: MISSING")

        if hasattr(mp.solutions, 'face_mesh'):
            print("  ✓ mp.solutions.face_mesh: AVAILABLE")
        else:
            print("  ✗ mp.solutions.face_mesh: MISSING")
    except Exception as e:
        print(f"   Error listing solutions: {e}")

    print("\n" + "=" * 60)
    print("Test Complete!")
    print("=" * 60)

if __name__ == "__main__":
    main()