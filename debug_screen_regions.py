#!/usr/bin/env python3
"""
Visualize screen regions on a game screenshot.
Helps understand how regions map to the actual image.
"""

import cv2
import numpy as np
from pathlib import Path
from src.diabot.vision.screen_regions import ScreenRegion, UI_REGIONS, ENVIRONMENT_REGIONS

def draw_region(frame, region: ScreenRegion, name: str, color: tuple, thickness: int = 2):
    """Draw a region rectangle on frame with label."""
    h, w = frame.shape[:2]
    x, y, region_w, region_h = region.get_bounds(h, w)
    
    # Draw rectangle
    cv2.rectangle(frame, (x, y), (x + region_w, y + region_h), color, thickness)
    
    # Draw label with background
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.6
    font_color = (0, 0, 0)  # Black text
    
    text = f"{name} ({region_w}x{region_h}px)"
    (text_w, text_h), baseline = cv2.getTextSize(text, font, font_scale, 1)
    
    # Background for text
    cv2.rectangle(frame,
                  (x, y - text_h - baseline - 5),
                  (x + text_w + 5, y - baseline),
                  color, -1)
    
    # Text
    cv2.putText(frame, text, (x + 2, y - baseline - 2),
                font, font_scale, font_color, 1)

def main():
    # Load image
    image_path = Path("data/screenshots/inputs/game.jpg")
    if not image_path.exists():
        print(f"❌ Image not found: {image_path}")
        return
    
    frame = cv2.imread(str(image_path))
    if frame is None:
        print(f"❌ Failed to load image: {image_path}")
        return
    
    h, w = frame.shape[:2]
    print(f"📸 Image loaded: {image_path}")
    print(f"   Dimensions: {w}x{h}px")
    print()
    
    # Create a copy for drawing
    frame_with_regions = frame.copy()
    
    # Draw UI regions
    print("🎨 Drawing UI regions...")
    ui_color = (0, 255, 255)  # Cyan for all UI regions
    
    for region_name, region in UI_REGIONS.items():
        draw_region(frame_with_regions, region, region_name, ui_color, 3)
        x, y, rw, rh = region.get_bounds(h, w)
        print(f"   {region_name:15} → x={x:4}, y={y:4}, w={rw:4}, h={rh:4}")
    
    print()
    
    # Draw environment regions
    print("🎨 Drawing environment regions...")
    env_color = (0, 255, 0)  # Green for all environment regions
    
    for region_name, region in ENVIRONMENT_REGIONS.items():
        draw_region(frame_with_regions, region, region_name, env_color, 2)
        x, y, rw, rh = region.get_bounds(h, w)
        print(f"   {region_name:15} → x={x:4}, y={y:4}, w={rw:4}, h={rh:4}")
    
    print()
    
    # Show summary
    print("📊 Region summary:")
    print(f"   🎯 UI regions: {len(UI_REGIONS)} (cyan/blue/orange boxes)")
    print(f"   🎯 Environment regions: {len(ENVIRONMENT_REGIONS)} (green/red boxes)")
    print()
    
    # Save output
    output_path = Path("data/screenshots/outputs/game_with_regions.jpg")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), frame_with_regions)
    print(f"✅ Saved visualization to: {output_path}")
    
    # Also display
    print()
    print("📺 Displaying image (press any key to close)...")
    
    # Resize for display if too large
    display_frame = frame_with_regions
    if w > 1200 or h > 900:
        scale = min(1200 / w, 900 / h)
        display_frame = cv2.resize(frame_with_regions, 
                                  (int(w * scale), int(h * scale)))
        print(f"   (Resized to {display_frame.shape[1]}x{display_frame.shape[0]}px for display)")
    
    cv2.imshow("Screen Regions Visualization", display_frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
