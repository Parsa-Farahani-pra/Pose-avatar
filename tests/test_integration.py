#!/usr/bin/env python3
"""Test pose_avatar.py emotion integration without GUI."""

import sys
import time
# No need for sys.path here - using src imports

from src.emotion_detection import EmotionDetector, _make_synthetic_face
from src.pose_avatar import Config, PoseWorker, PoseTracker, HandAvatarTracker

def test():
    print("Testing emotion HUD integration logic...")
    print("=" * 70)

    # Create a mock worker with synthetic faces
    class MockSample:
        def __init__(self, face_shape):
            face_lms = _make_synthetic_face(face_shape)
            self.face = type('Face', (), {
                'landmarks_468': face_lms,
                'is_detected': True,
                'face_size_ratio': 0.6,
            })()
            self.image = None
            self.frame = None

    # Create detector
    detector = EmotionDetector(smoothing_tau=1.0)

    # Simulate 10 frames
    expressions = ["neutral", "smile", "smile", "smile", "neutral",
                   "frown", "frown", "neutral", "happy", "neutral"]

    for i, expr in enumerate(expressions, 1):
        # Create mock sample
        sample = MockSample(expr)

        # Call the integration logic (similar to pose_avatar's update)
        cfg_detect = True
        emotion_det = detector

        if cfg_detect and emotion_det is not None:
            if sample is not None and sample.face is not None:
                if sample.face.is_detected:
                    # Face detected - get emotion result
                    result = emotion_det.update(sample.face, 0.033)
                    if result is not None:
                        print(f"Frame {i:2d} {expr:6s}: {result.top_emotion}")
                        for emo, pct in sorted(result.percentages.items(), key=lambda x: -x[1]):
                            marker = " ←" if emo == result.top_emotion else ""
                            print(f"         {emo:9s}: {int(pct):3d}%{marker}")
                    else:
                        print(f"Frame {i:2d} {expr:6s}: Too small")

    print("\n" + "=" * 70)
    print("Now run: ./run.sh pose_avatar.py --enable-hands --enable-face --detect-emotion")
    print("If you see [EMOTION] debug messages, the detector is being called.")
    print("If you DON'T see debug messages, check if --detect-emotion flag is active.")

if __name__ == "__main__":
    test()