#!/usr/bin/env python3
"""
Script to visualize all defined screen regions on a screenshot.
Useful for debugging and verifying region positions.
"""

import cv2
import numpy as np
from pathlib import Path
from src.diabot.vision.screen_regions import ALL_REGIONS

def draw_regions(image_path: str, output_path: str = None):
    """Draw all regions on the image with labels."""
    img = cv2.imread(image_path)
    if img is None:
        print(f"❌ Cannot load image: {image_path}")
        return
    
    height, width = img.shape[:2]
    print(f"Image size: {width}x{height}\n")
    
    # Create overlay for semi-transparent rectangles
    overlay = img.copy()
    
    # Colors for different region types (BGR format)
    colors = {
        'lifebar': (0, 0, 255),      # Red
        'manabar': (255, 0, 0),      # Blue
        'minimap': (0, 255, 0),      # Green
        'inventory': (0, 255, 255),  # Yellow
        'stamina': (255, 255, 0),    # Cyan
        'exp': (255, 0, 255),        # Magenta
        'belt': (128, 128, 255),     # Light red
        'default': (200, 200, 200)   # Gray
    }
    
    print("=" * 80)
    print("REGIONS VISUALIZATION")
    print("=" * 80)
    
    for region_name, region_def in ALL_REGIONS.items():
        # Calculate pixel coordinates
        x = int(region_def.x_ratio * width)
        y = int(region_def.y_ratio * height)
        w = int(region_def.w_ratio * width)
        h = int(region_def.h_ratio * height)
        
        # Choose color based on region name
        color = colors['default']
        for key, col in colors.items():
            if key in region_name.lower():
                color = col
                break
        
        # Draw semi-transparent rectangle
        cv2.rectangle(overlay, (x, y), (x + w, y + h), color, -1)
        
        # Draw border
        cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
        
        # Draw label with background
        label = region_name
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5
        thickness = 1
        (label_w, label_h), baseline = cv2.getTextSize(label, font, font_scale, thickness)
        
        # Position label above region
        label_x = x
        label_y = y - 5
        if label_y < label_h + 5:
            label_y = y + h + label_h + 5
        
        # Draw label background
        cv2.rectangle(img, 
                     (label_x, label_y - label_h - 3),
                     (label_x + label_w + 4, label_y + 3),
                     color, -1)
        
        # Draw label text
        cv2.putText(img, label, (label_x + 2, label_y),
                   font, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)
        
        print(f"{region_name:20s}: x={x:4d} y={y:4d} w={w:4d} h={h:4d} "
              f"(ratios: {region_def.x_ratio:.3f}, {region_def.y_ratio:.3f}, "
              f"{region_def.w_ratio:.3f}, {region_def.h_ratio:.3f})")
    
    # Blend overlay with original image
    alpha = 0.3
    result = cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0)
    
    # Add legend
    legend_y = 30
    legend_x = 10
    cv2.putText(result, "Screen Regions Visualization", (legend_x, legend_y),
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
    
    # Save output
    if output_path is None:
        output_path = "data/screenshots/outputs/regions_visual.png"
    
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(output_path, result)
    
    print("\n" + "=" * 80)
    print(f"✓ Saved visualization to: {output_path}")
    print("=" * 80)
    
    return result


if __name__ == "__main__":
    import sys
    
    # Default to game_a1.png
    image_path = "data/screenshots/inputs/game_a1.png"
    
    if len(sys.argv) > 1:
        image_path = sys.argv[1]
    
    print(f"\nVisualizing regions on: {image_path}\n")
    draw_regions(image_path)
