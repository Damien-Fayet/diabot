#!/usr/bin/env python3
"""Test YOLO detection on live game capture.

Captures frames from Diablo II window and runs YOLO inference in real-time.
Press 'q' to quit, 's' to save current frame.

Usage:
    python test_yolo_live.py --model runs/detect/runs/train/diablo-yolo3/weights/best.pt --conf 0.35
"""

import argparse
import time
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

try:
    from src.diabot.core.implementations import WindowsScreenCapture
except ImportError:
    print("Error: Could not import WindowsScreenCapture")
    print("Make sure you're running on Windows with src.diabot installed")
    exit(1)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Test YOLO detection on live game")
    parser.add_argument("--model", required=True, type=str, help="Path to YOLO model (.pt)")
    parser.add_argument("--conf", type=float, default=0.35, help="Confidence threshold")
    parser.add_argument("--window-name", type=str, default="Diablo II: Resurrected", 
                        help="Game window title")
    parser.add_argument("--fps-limit", type=int, default=5, 
                        help="Max FPS for inference (to avoid overload)")
    parser.add_argument("--save-dir", type=Path, default=Path("data/screenshots/outputs/live"),
                        help="Directory to save captured frames (when pressing 's')")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    
    model_path = Path(args.model)
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")
    
    # Load YOLO model
    print(f"🚀 Loading model: {model_path}")
    model = YOLO(str(model_path))
    print(f"✓ Model loaded")
    
    # Initialize screen capture
    print(f"🎮 Looking for window: {args.window_name}")
    capture = WindowsScreenCapture(window_title=args.window_name)
    
    # Test capture
    test_frame = capture.get_frame()
    if test_frame is None:
        raise RuntimeError(f"Could not capture window '{args.window_name}'")
    print(f"✓ Window found: {test_frame.shape[1]}x{test_frame.shape[0]}")
    
    # Prepare save directory
    args.save_dir.mkdir(parents=True, exist_ok=True)
    
    # FPS limiting
    frame_time = 1.0 / args.fps_limit
    last_inference_time = 0
    frame_count = 0
    
    print("\n" + "="*60)
    print("LIVE YOLO DETECTION")
    print("="*60)
    print(f"Confidence threshold: {args.conf}")
    print(f"FPS limit: {args.fps_limit}")
    print("\nControls:")
    print("  'q' - Quit")
    print("  's' - Save current annotated frame")
    print("="*60 + "\n")
    
    try:
        while True:
            current_time = time.time()
            
            # Rate limiting
            if current_time - last_inference_time < frame_time:
                time.sleep(0.01)
                continue
            
            # Capture frame
            frame = capture.get_frame()
            if frame is None:
                print("⚠️  Failed to capture frame")
                time.sleep(0.1)
                continue
            
            # Run inference
            results = model.predict(source=frame, conf=args.conf, verbose=False)
            
            # Annotate frame
            annotated = results[0].plot()
            
            # Draw stats overlay
            detections = results[0].boxes
            fps = 1.0 / (current_time - last_inference_time) if last_inference_time > 0 else 0
            
            info_text = f"FPS: {fps:.1f} | Detections: {len(detections)} | Frame: {frame_count}"
            cv2.putText(annotated, info_text, (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # Display class names for detected objects
            y_offset = 60
            for box in detections:
                cls_id = int(box.cls.item())
                conf = float(box.conf.item())
                cls_name = model.names[cls_id]
                text = f"{cls_name}: {conf:.2f}"
                cv2.putText(annotated, text, (10, y_offset),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                y_offset += 30
            
            # Show frame
            cv2.imshow("YOLO Live Detection", annotated)
            
            # Handle keyboard
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                print("\n👋 Quitting...")
                break
            elif key == ord('s'):
                timestamp = time.strftime("%Y%m%d_%H%M%S")
                save_path = args.save_dir / f"live_{timestamp}.jpg"
                cv2.imwrite(str(save_path), annotated)
                print(f"💾 Saved frame: {save_path}")
            
            last_inference_time = current_time
            frame_count += 1
            
    except KeyboardInterrupt:
        print("\n👋 Interrupted by user")
    finally:
        cv2.destroyAllWindows()
        print(f"\n📊 Stats:")
        print(f"  Total frames processed: {frame_count}")
        print(f"  Average FPS: {frame_count / (time.time() - last_inference_time):.1f}")


if __name__ == "__main__":
    main()
