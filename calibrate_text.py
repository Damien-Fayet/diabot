#!/usr/bin/env python3
"""
Calibrate text regions above HP/Mana orbs.
The text displays numeric values like "120/150" above the orbs.
"""

import cv2
import numpy as np
from pathlib import Path

def find_text_region(image_path: str):
    """Find text regions above HP and Mana orbs."""
    img = cv2.imread(image_path)
    if img is None:
        print(f"❌ Cannot load: {image_path}")
        return
    
    height, width = img.shape[:2]
    print(f"Image size: {width}x{height}\n")
    
    # Current orb regions (from screen_regions.py)
    hp_orb = {'x': 0.217, 'y': 0.872, 'w': 0.070, 'h': 0.127}
    mana_orb = {'x': 0.731, 'y': 0.870, 'w': 0.070, 'h': 0.120}
    
    # Convert to pixels
    hp_orb_px = {
        'x': int(hp_orb['x'] * width),
        'y': int(hp_orb['y'] * height),
        'w': int(hp_orb['w'] * width),
        'h': int(hp_orb['h'] * height)
    }
    mana_orb_px = {
        'x': int(mana_orb['x'] * width),
        'y': int(mana_orb['y'] * height),
        'w': int(mana_orb['w'] * width),
        'h': int(mana_orb['h'] * height)
    }
    
    print("=" * 80)
    print("SEARCHING FOR TEXT REGIONS ABOVE ORBS")
    print("=" * 80)
    
    # Search area: above the orbs
    # Text is typically 40-60 pixels above the orb top
    search_height = 80  # pixels to search above orb
    text_height_estimate = 25  # typical text height
    
    def find_text_above_orb(orb_px, orb_name, color_name):
        """Find text region above an orb by looking for bright pixels."""
        # Define search area above orb
        search_x = orb_px['x'] - 20  # slightly wider
        search_y = orb_px['y'] - search_height
        search_w = orb_px['w'] + 40
        search_h = search_height
        
        # Clamp to image bounds
        search_x = max(0, search_x)
        search_y = max(0, search_y)
        search_w = min(search_w, width - search_x)
        search_h = min(search_h, height - search_y)
        
        # Extract search region
        search_roi = img[search_y:search_y+search_h, search_x:search_x+search_w]
        
        # Convert to grayscale
        gray = cv2.cvtColor(search_roi, cv2.COLOR_BGR2GRAY)
        
        # Text is typically bright (white or light color)
        # Threshold to find bright regions
        _, thresh = cv2.threshold(gray, 180, 255, cv2.THRESH_BINARY)
        
        # Find contours
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            print(f"⚠️  No text contours found for {orb_name}")
            # Use default position
            text_x = orb_px['x']
            text_y = orb_px['y'] - 45
            text_w = orb_px['w']
            text_h = 25
        else:
            # Find the largest contour (likely the text)
            largest = max(contours, key=cv2.contourArea)
            x, y, w, h = cv2.boundingRect(largest)
            
            # Convert back to full image coordinates
            text_x = search_x + x
            text_y = search_y + y
            text_w = w
            text_h = h
            
            # Expand slightly to ensure we capture all text
            margin = 5
            text_x = max(0, text_x - margin)
            text_y = max(0, text_y - margin)
            text_w = min(text_w + 2*margin, width - text_x)
            text_h = min(text_h + 2*margin, height - text_y)
        
        # Convert to ratios
        x_ratio = text_x / width
        y_ratio = text_y / height
        w_ratio = text_w / width
        h_ratio = text_h / height
        
        print(f"\n{orb_name} Text Region ({color_name}):")
        print(f"  Pixels: x={text_x}, y={text_y}, w={text_w}, h={text_h}")
        print(f"  Ratios: x={x_ratio:.3f}, y={y_ratio:.3f}, w={w_ratio:.3f}, h={h_ratio:.3f}")
        
        # Draw on image for visualization
        vis = img.copy()
        # Draw orb region in one color
        cv2.rectangle(vis, 
                     (orb_px['x'], orb_px['y']),
                     (orb_px['x'] + orb_px['w'], orb_px['y'] + orb_px['h']),
                     (0, 255, 255), 2)  # Yellow for orb
        
        # Draw text region
        cv2.rectangle(vis, 
                     (text_x, text_y),
                     (text_x + text_w, text_y + text_h),
                     (0, 255, 0), 2)  # Green for text
        
        # Draw search area
        cv2.rectangle(vis,
                     (search_x, search_y),
                     (search_x + search_w, search_y + search_h),
                     (255, 0, 0), 1)  # Blue for search area
        
        # Labels
        cv2.putText(vis, f"{orb_name} Orb", 
                   (orb_px['x'], orb_px['y'] - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
        cv2.putText(vis, f"{orb_name} Text", 
                   (text_x, text_y - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        # Save debug visualization
        output_path = f"data/screenshots/outputs/calibration_text_{orb_name.lower()}.png"
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(output_path, vis)
        print(f"  ✓ Saved: {output_path}")
        
        # Also save just the text region crop
        text_crop = img[text_y:text_y+text_h, text_x:text_x+text_w]
        crop_path = f"data/screenshots/outputs/text_{orb_name.lower()}_crop.png"
        cv2.imwrite(crop_path, text_crop)
        print(f"  ✓ Saved crop: {crop_path}")
        
        # Save thresholded search region for debugging
        thresh_path = f"data/screenshots/outputs/text_{orb_name.lower()}_thresh.png"
        cv2.imwrite(thresh_path, thresh)
        print(f"  ✓ Saved threshold: {thresh_path}")
        
        return {
            'x_ratio': x_ratio,
            'y_ratio': y_ratio,
            'w_ratio': w_ratio,
            'h_ratio': h_ratio,
            'pixels': {'x': text_x, 'y': text_y, 'w': text_w, 'h': text_h}
        }
    
    # Find text regions
    hp_text = find_text_above_orb(hp_orb_px, "HP", "red")
    mana_text = find_text_above_orb(mana_orb_px, "Mana", "blue")
    
    print("\n" + "=" * 80)
    print("SUGGESTED UPDATES FOR screen_regions.py")
    print("=" * 80)
    print(f"""
    'lifebar_ui': ScreenRegion(
        name='lifebar_ui',
        x_ratio={hp_text['x_ratio']:.3f},
        y_ratio={hp_text['y_ratio']:.3f},
        w_ratio={hp_text['w_ratio']:.3f},
        h_ratio={hp_text['h_ratio']:.3f},
    ),
    'manabar_ui': ScreenRegion(
        name='manabar_ui',
        x_ratio={mana_text['x_ratio']:.3f},
        y_ratio={mana_text['y_ratio']:.3f},
        w_ratio={mana_text['w_ratio']:.3f},
        h_ratio={mana_text['h_ratio']:.3f},
    ),
""")
    
    print("=" * 80)
    print("✓ Analysis complete. Check outputs/ for visualizations.")
    print("=" * 80)


if __name__ == "__main__":
    import sys
    
    image_path = "data/screenshots/inputs/game_a1.png"
    if len(sys.argv) > 1:
        image_path = sys.argv[1]
    
    print(f"\nCalibrating text regions from: {image_path}\n")
    find_text_region(image_path)
