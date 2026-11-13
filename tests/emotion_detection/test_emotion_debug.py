#!/usr/bin/env python3
"""Test emotion detection with synthetic faces."""

from pathlib import Path
import sys

# Add src/ to path
src_dir = Path(__file__).parent.parent
sys.path.insert(0, str(src_dir))

from src.emotion_detection import EmotionDetector, EmotionResult, _make_synthetic_face
import time

def main():
    detector = EmotionDetector(smoothing_tau=1.0)
    
    print("Testing EmotionDetector with synthetic faces...")
    print("=" * 60)
    
    # Create a sequence of faces
    faces = []
    for shape in ["smile", "frown", "wide_eyes", "neutral", "neutral", "neutral"]:
        face_lms = _make_synthetic_face(shape)
        face = type('Face', (), {
            'landmarks_468': face_lms,
            'is_detected': True,
            'face_size_ratio': 0.6,
        })()
        faces.append(face)
    
    # Process them
    print("\nProcessing sequence...")
    for i, face in enumerate(faces, 1):
        result = detector.update(face, 1/30)
        if result:
            print(f"  Frame {i}: {result.top_emotion}")
            for emo, pct in sorted(result.percentages.items(), key=lambda x: -x[1]):
                print(f"    {emo}: {pct:5.1f}%")
        else:
            print(f"  Frame {i}: TRACKER: Face too small (<0.25)")
    
    print("\n" + "=" * 60)

if __name__ == "__main__":
    main()