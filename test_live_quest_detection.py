#!/usr/bin/env python3
"""Test quest marker detection in live game."""

import time
from pathlib import Path
from src.diabot.core.implementations import WindowsScreenCapture
from src.diabot.vision.yolo_detector import YOLODetector

def main():
    # Initialize
    print("🤖 Testing quest marker detection in live game...")
    
    screen_capture = WindowsScreenCapture(window_title="Diablo II: Resurrected")
    detector = YOLODetector(
        "runs/detect/runs/train/diablo-yolo3/weights/best.pt",
        confidence_threshold=0.35,
        debug=True
    )
    
    # Capture multiple frames
    for i in range(5):
        print(f"\n[Frame {i}]")
        frame = screen_capture.get_frame()
        
        if frame is None:
            print("  ❌ Failed to capture frame")
            continue
        
        print(f"  📸 Captured frame: {frame.shape}")
        
        # Detect
        detections = detector.detect(frame)
        print(f"  Found {len(detections)} detections:")
        
        quest_found = False
        for det in detections:
            print(f"    - {det.class_name}: {det.confidence:.2f} at {det.center}")
            if det.class_name == "quest":
                quest_found = True
        
        if not quest_found:
            print("  ⚠️  No quest marker found")
        else:
            print("  ✓ Quest marker detected!")
        
        time.sleep(0.5)

if __name__ == "__main__":
    main()
