"""
Visualize ROI candidates extracted from environment vision.

Shows background removal and foreground object detection.

Usage:
    python show_roi_candidates.py
    python show_roi_candidates.py --input path/to/screenshot.jpg
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np

from src.diabot.vision.environment_vision import EnvironmentVisionModule
from src.diabot.vision.screen_regions import ENVIRONMENT_REGIONS


def draw_roi_overlay(frame: np.ndarray, roi_candidates) -> np.ndarray:
    """Draw ROI candidates on frame with color-coded bounding boxes."""
    overlay = frame.copy()
    
    # Extract playfield region to match ROI coordinates
    playfield_region = ENVIRONMENT_REGIONS['playfield']
    frame_h, frame_w = frame.shape[:2]
    
    # Get playfield offset in full frame
    offset_x, offset_y, _, _ = playfield_region.get_bounds(frame_h, frame_w)
    
    for roi in roi_candidates:
        x, y, w, h = roi.bbox
        
        # Adjust coordinates to full frame
        x += offset_x
        y += offset_y
        
        # Color based on type
        if roi.roi_type == "static":
            color = (0, 255, 0)  # Green for static (NPCs)
            label = "STATIC"
        else:
            color = (0, 165, 255)  # Orange for moving (enemies)
            label = "MOVING"
        
        # Draw bounding box
        cv2.rectangle(overlay, (x, y), (x + w, y + h), color, 2)
        
        # Draw label with confidence
        text = f"{label} {roi.confidence:.2f}"
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(overlay, (x, y - th - 4), (x + tw, y), color, -1)
        cv2.putText(overlay, text, (x, y - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        
        # Draw features
        feature_text = f"A:{int(roi.features['area'])} S:{roi.features['solidity']:.2f}"
        cv2.putText(overlay, feature_text, (x, y + h + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1, cv2.LINE_AA)
    
    return overlay


def main() -> int:
    parser = argparse.ArgumentParser(description="Visualize ROI candidates from environment vision")
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data/screenshots/outputs/live_capture/live_capture_raw.jpg"),
        help="Path to input screenshot",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/screenshots/outputs/diagnostic/roi_candidates.png"),
        help="Where to save overlay image",
    )
    parser.add_argument("--debug", action="store_true", help="Enable debug output")
    parser.add_argument("--show", action="store_true", help="Display result with matplotlib")
    args = parser.parse_args()
    
    if not args.input.exists():
        print(f"ERROR: Input not found: {args.input}")
        return 1
    
    frame = cv2.imread(str(args.input))
    if frame is None:
        print(f"ERROR: Failed to read: {args.input}")
        return 1
    
    print("\n" + "=" * 60)
    print("ROI CANDIDATE EXTRACTION")
    print("=" * 60)
    
    env_vision = EnvironmentVisionModule(debug=args.debug)
    
    # Enable debug output for intermediate images
    debug_dir = "data/screenshots/outputs/diagnostic/roi_steps" if args.debug else None
    
    state = env_vision.analyze(frame, extract_rois=True, detect_templates=False, 
                              detect_enemies=False, debug_output_dir=debug_dir)
    
    if debug_dir:
        print(f"\nIntermediate processing images saved to: {debug_dir}/")
    
    print(f"\nExtracted {len(state.roi_candidates)} ROI candidates:")
    for i, roi in enumerate(state.roi_candidates[:10], 1):
        x, y, w, h = roi.bbox
        print(f"  {i}. {roi.roi_type:7s} @ ({x:4d}, {y:4d}, {w:3d}x{h:3d}) "
              f"conf={roi.confidence:.2f} area={int(roi.features['area']):5d}")
    
    if len(state.roi_candidates) > 10:
        print(f"  ... and {len(state.roi_candidates) - 10} more")
    
    # Count by type
    static_count = sum(1 for r in state.roi_candidates if r.roi_type == "static")
    moving_count = sum(1 for r in state.roi_candidates if r.roi_type == "moving")
    
    print(f"\nBreakdown:")
    print(f"  - Static (NPC/object): {static_count}")
    print(f"  - Moving (enemy): {moving_count}")
    
    # Draw overlay
    overlay = draw_roi_overlay(frame, state.roi_candidates)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(args.output), overlay)
    print(f"\nSaved overlay: {args.output}")
    
    if args.show:
        import matplotlib.pyplot as plt
        rgb = cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)
        plt.figure(figsize=(14, 8))
        plt.imshow(rgb)
        plt.title(f"ROI Candidates: {len(state.roi_candidates)} total ({static_count} static, {moving_count} moving)")
        plt.axis("off")
        plt.tight_layout()
        plt.show()
    
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
