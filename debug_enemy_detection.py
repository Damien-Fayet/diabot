#!/usr/bin/env python3
"""
Debug vision: show detection process step by step.
Shows masks, contours, final detections.
"""

import cv2
import numpy as np
from pathlib import Path
from src.diabot.vision.screen_regions import ENVIRONMENT_REGIONS

def main():
    # Load image
    image_path = Path("data/screenshots/inputs/game.jpg")
    frame = cv2.imread(str(image_path))
    
    if frame is None:
        print(f"❌ Failed to load {image_path}")
        return
    
    print("=" * 60)
    print("🔍 ENEMY DETECTION - STEP BY STEP")
    print("=" * 60)
    print()
    
    # Convert to HSV
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
    # Extract playfield
    playfield_hsv = ENVIRONMENT_REGIONS['playfield'].extract_from_frame(hsv)
    h, w = playfield_hsv.shape[:2]
    
    print(f"📍 Playfield region: {w}x{h}px")
    print(f"   Area: {w*h} pixels")
    print()
    
    # Step 1: Red mask
    print("Step 1️⃣ : RED MASK (H=0-10, S=100-255, V=100-255)")
    red_mask = cv2.inRange(playfield_hsv,
                           np.array([0, 100, 100]),
                           np.array([10, 255, 255]))
    red_pixels = cv2.countNonZero(red_mask)
    print(f"   Red pixels found: {red_pixels}")
    print()
    
    # Step 2: Orange mask
    print("Step 2️⃣ : ORANGE MASK (H=10-25, S=100-255, V=100-255)")
    orange_mask = cv2.inRange(playfield_hsv,
                              np.array([10, 100, 100]),
                              np.array([25, 255, 255]))
    orange_pixels = cv2.countNonZero(orange_mask)
    print(f"   Orange pixels found: {orange_pixels}")
    print()
    
    # Step 3: Combined mask
    print("Step 3️⃣ : COMBINED MASK (Red + Orange)")
    threat_mask = cv2.bitwise_or(red_mask, orange_mask)
    threat_pixels = cv2.countNonZero(threat_mask)
    print(f"   Total threat pixels: {threat_pixels}")
    print(f"   Percentage of playfield: {100*threat_pixels/(w*h):.2f}%")
    print()
    
    # Step 4: Find contours
    print("Step 4️⃣ : FIND CONTOURS")
    contours, _ = cv2.findContours(threat_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    print(f"   Contours found: {len(contours)}")
    print()
    
    # Step 5: Filter by size
    print("Step 5️⃣ : FILTER BY SIZE")
    playfield_area = w * h
    min_size = 50
    max_size = playfield_area * 0.05
    
    print(f"   Min size: {min_size} pixels")
    print(f"   Max size: {max_size:.0f} pixels (5% of {playfield_area})")
    print()
    
    valid_enemies = []
    for i, contour in enumerate(contours):
        area = cv2.contourArea(contour)
        x, y, w_box, h_box = cv2.boundingRect(contour)
        
        if min_size < area < max_size:
            enemy_type = "LARGE" if area > 500 else "small"
            confidence = min(1.0, area / 500)
            valid_enemies.append({
                'idx': i,
                'area': area,
                'type': enemy_type,
                'bbox': (x, y, w_box, h_box),
                'confidence': confidence,
                'center': (x + w_box//2, y + h_box//2)
            })
            print(f"   ✓ #{i:2d} | Area={area:5.0f}px | {enemy_type:5s} | Conf={confidence:.1%} | Center=({x+w_box//2:3d}, {y+h_box//2:3d})")
        else:
            status = "TOO_SMALL" if area < min_size else "TOO_LARGE"
            if i < 5 or i > len(contours) - 5:  # Show first and last few
                print(f"   ✗ #{i:2d} | Area={area:5.0f}px | {status:9s} (filtered out)")
            elif i == 5:
                print(f"   ... ({len(contours) - 10} contours filtered) ...")
    
    print()
    print(f"🎯 FINAL RESULT")
    print(f"   Valid enemies: {len(valid_enemies)}")
    print()
    
    if len(valid_enemies) > 0:
        large = len([e for e in valid_enemies if e['type'] == 'LARGE'])
        small = len([e for e in valid_enemies if e['type'] == 'small'])
        print(f"   - {large} LARGE enemies")
        print(f"   - {small} small enemies")
        print()
        print("   Sizes:")
        sizes = [e['area'] for e in valid_enemies]
        print(f"   - Min: {min(sizes):.0f}px")
        print(f"   - Max: {max(sizes):.0f}px")
        print(f"   - Avg: {np.mean(sizes):.0f}px")
    
    print()
    print("=" * 60)
    print()

if __name__ == "__main__":
    main()
