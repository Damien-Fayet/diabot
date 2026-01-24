#!/usr/bin/env python3
"""Analyze screenshot images to understand game structure."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import cv2
import numpy as np


def analyze_image(image_path: str):
    """Analyze a single image."""
    print(f"\n📸 Analyzing: {image_path}")
    print("=" * 60)
    
    img = cv2.imread(image_path)
    if img is None:
        print(f"❌ Failed to load {image_path}")
        return
    
    h, w, c = img.shape
    print(f"  Shape: {h}x{w}x{c}")
    
    # Analyze color distribution
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    
    # Detect common game UI colors
    colors = {
        "Red (enemies/threat)": (cv2.inRange(hsv, np.array([0, 100, 100]), np.array([10, 255, 255])),),
        "Green (health)": (cv2.inRange(hsv, np.array([40, 100, 100]), np.array([80, 255, 255])),),
        "Blue (mana)": (cv2.inRange(hsv, np.array([100, 100, 100]), np.array([140, 255, 255])),),
        "Yellow (items)": (cv2.inRange(hsv, np.array([20, 100, 100]), np.array([40, 255, 255])),),
        "Black (UI bg)": (cv2.inRange(hsv, np.array([0, 0, 0]), np.array([180, 255, 50])),),
    }
    
    for color_name, (mask,) in colors.items():
        count = cv2.countNonZero(mask)
        pct = (count / (h * w)) * 100
        print(f"  {color_name}: {pct:.2f}%")
    
    # Detect contours/objects
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    print(f"  Contours found: {len(contours)}")
    
    # Analyze center region (player location hypothesis)
    cy, cx = h // 2, w // 2
    roi = img[max(0, cy-50):min(h, cy+50), max(0, cx-50):min(w, cx+50)]
    roi_gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    roi_mean = np.mean(roi_gray)
    print(f"  Center brightness: {roi_mean:.1f}")


def main():
    """Analyze all screenshots."""
    screenshot_dir = Path(__file__).parent.parent / "data" / "screenshots" / "inputs"
    
    images = list(screenshot_dir.glob("*.jpg")) + list(screenshot_dir.glob("*.png"))
    
    print("🎮 Screenshot Analysis Tool")
    print("=" * 60)
    
    for img_path in sorted(images):
        analyze_image(str(img_path))
    
    print("\n✅ Analysis complete!")


if __name__ == "__main__":
    main()
