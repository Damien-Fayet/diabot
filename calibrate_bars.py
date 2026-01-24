#!/usr/bin/env python3
"""
Calibration script for HP/Mana bar regions on real Diablo 2 screenshots.

This script:
1. Loads the screenshot
2. Displays the current regions
3. Analyzes color patterns to find HP/Mana bars
4. Extracts percentage values
5. Suggests updated region definitions
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

import cv2
import numpy as np

from diabot.vision.screen_regions import UI_REGIONS


def analyze_image(image_path: str):
    """Analyze screenshot to find HP/Mana bar locations."""
    print(f"\n{'='*80}")
    print(f"ANALYZING: {Path(image_path).name}")
    print(f"{'='*80}\n")
    
    # Load image
    frame = cv2.imread(image_path)
    if frame is None:
        print(f"❌ Failed to load image: {image_path}")
        return
    
    h, w = frame.shape[:2]
    print(f"Image size: {w}x{h}")
    print()
    
    # Current regions
    print("Current LifeBar Region:")
    lifebar_region = UI_REGIONS['lifebar_ui']
    print(f"  x_ratio={lifebar_region.x_ratio}, y_ratio={lifebar_region.y_ratio}")
    print(f"  w_ratio={lifebar_region.w_ratio}, h_ratio={lifebar_region.h_ratio}")
    x, y, region_w, region_h = lifebar_region.get_bounds(h, w)
    print(f"  Pixels: x={x}, y={y}, w={region_w}, h={region_h}")
    print()
    
    print("Current ManaBar Region:")
    manabar_region = UI_REGIONS['manabar_ui']
    print(f"  x_ratio={manabar_region.x_ratio}, y_ratio={manabar_region.y_ratio}")
    print(f"  w_ratio={manabar_region.w_ratio}, h_ratio={manabar_region.h_ratio}")
    x, y, region_w, region_h = manabar_region.get_bounds(h, w)
    print(f"  Pixels: x={x}, y={y}, w={region_w}, h={region_h}")
    print()
    
    # Extract regions
    lifebar_crop = lifebar_region.extract_from_frame(frame)
    manabar_crop = manabar_region.extract_from_frame(frame)
    
    # Analyze HP bar (red channel)
    print("=" * 80)
    print("HP BAR ANALYSIS")
    print("=" * 80)
    hp_ratio, hp_details = analyze_health_bar(lifebar_crop)
    print(f"HP Ratio: {hp_ratio:.1%}")
    print(f"Details: {hp_details}")
    print()
    
    # Analyze Mana bar (blue channel)
    print("=" * 80)
    print("MANA BAR ANALYSIS")
    print("=" * 80)
    mana_ratio, mana_details = analyze_mana_bar(manabar_crop)
    print(f"Mana Ratio: {mana_ratio:.1%}")
    print(f"Details: {mana_details}")
    print()
    
    # Save debug images
    output_dir = Path("data/screenshots/outputs")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    cv2.imwrite(str(output_dir / "calibration_hp.png"), lifebar_crop)
    cv2.imwrite(str(output_dir / "calibration_mana.png"), manabar_crop)
    print(f"✓ Saved debug crops to {output_dir}")
    
    # Create visualization
    vis = frame.copy()
    
    # Draw HP region
    x, y, region_w, region_h = lifebar_region.get_bounds(h, w)
    cv2.rectangle(vis, (x, y), (x + region_w, y + region_h), (0, 0, 255), 2)
    cv2.putText(vis, f"HP: {hp_ratio:.0%}", (x, y - 10), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
    
    # Draw Mana region
    x, y, region_w, region_h = manabar_region.get_bounds(h, w)
    cv2.rectangle(vis, (x, y), (x + region_w, y + region_h), (255, 0, 0), 2)
    cv2.putText(vis, f"Mana: {mana_ratio:.0%}", (x, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
    
    cv2.imwrite(str(output_dir / "calibration_visual.png"), vis)
    print(f"✓ Saved visualization to {output_dir / 'calibration_visual.png'}")
    print()
    
    # Scan for better regions
    print("=" * 80)
    print("REGION OPTIMIZATION")
    print("=" * 80)
    
    # Scan bottom portion of screen for red (HP) and blue (Mana) concentrations
    bottom_region = frame[int(h * 0.7):, :]  # Bottom 30%
    
    # Find HP bar (red)
    hp_suggestions = find_red_region(bottom_region, h, w)
    if hp_suggestions:
        print("\nSuggested HP Region:")
        for suggestion in hp_suggestions[:3]:  # Top 3
            print(f"  x_ratio={suggestion['x_ratio']:.3f}, y_ratio={suggestion['y_ratio']:.3f}")
            print(f"  w_ratio={suggestion['w_ratio']:.3f}, h_ratio={suggestion['h_ratio']:.3f}")
            print(f"  (score={suggestion['score']:.0f})")
    
    # Find Mana bar (blue)
    mana_suggestions = find_blue_region(bottom_region, h, w)
    if mana_suggestions:
        print("\nSuggested Mana Region:")
        for suggestion in mana_suggestions[:3]:  # Top 3
            print(f"  x_ratio={suggestion['x_ratio']:.3f}, y_ratio={suggestion['y_ratio']:.3f}")
            print(f"  w_ratio={suggestion['w_ratio']:.3f}, h_ratio={suggestion['h_ratio']:.3f}")
            print(f"  (score={suggestion['score']:.0f})")
    
    print()


def analyze_health_bar(crop: np.ndarray) -> tuple[float, dict]:
    """
    Analyze HP bar crop to extract percentage.
    
    HP bar is typically red/dark red gradient.
    """
    if crop.size == 0:
        return 0.0, {"error": "empty crop"}
    
    # Convert to HSV for better color detection
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    
    # Red hue is at 0 and 180 in HSV
    # Define two ranges for red
    lower_red1 = np.array([0, 100, 50])
    upper_red1 = np.array([10, 255, 255])
    
    lower_red2 = np.array([170, 100, 50])
    upper_red2 = np.array([180, 255, 255])
    
    # Create masks
    mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
    red_mask = mask1 | mask2
    
    # Count red pixels
    red_pixels = np.sum(red_mask > 0)
    total_pixels = crop.shape[0] * crop.shape[1]
    
    # HP ratio is based on red pixel density
    hp_ratio = red_pixels / total_pixels if total_pixels > 0 else 0.0
    
    # Also check horizontal extent (bar fills left to right)
    if red_mask.any():
        # Find rightmost red pixel in each row
        red_extents = []
        for row in red_mask:
            if row.any():
                rightmost = np.where(row > 0)[0][-1]
                red_extents.append(rightmost)
        
        if red_extents:
            avg_extent = np.mean(red_extents)
            hp_ratio_extent = avg_extent / crop.shape[1]
            
            # Use extent as primary metric
            hp_ratio = hp_ratio_extent
    
    details = {
        "red_pixels": int(red_pixels),
        "total_pixels": int(total_pixels),
        "density": hp_ratio,
        "crop_shape": crop.shape,
    }
    
    return hp_ratio, details


def analyze_mana_bar(crop: np.ndarray) -> tuple[float, dict]:
    """
    Analyze Mana bar crop to extract percentage.
    
    Mana bar is typically blue gradient.
    """
    if crop.size == 0:
        return 0.0, {"error": "empty crop"}
    
    # Convert to HSV
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    
    # Blue hue range
    lower_blue = np.array([100, 100, 50])
    upper_blue = np.array([140, 255, 255])
    
    blue_mask = cv2.inRange(hsv, lower_blue, upper_blue)
    
    # Count blue pixels
    blue_pixels = np.sum(blue_mask > 0)
    total_pixels = crop.shape[0] * crop.shape[1]
    
    mana_ratio = blue_pixels / total_pixels if total_pixels > 0 else 0.0
    
    # Check horizontal extent
    if blue_mask.any():
        blue_extents = []
        for row in blue_mask:
            if row.any():
                rightmost = np.where(row > 0)[0][-1]
                blue_extents.append(rightmost)
        
        if blue_extents:
            avg_extent = np.mean(blue_extents)
            mana_ratio_extent = avg_extent / crop.shape[1]
            mana_ratio = mana_ratio_extent
    
    details = {
        "blue_pixels": int(blue_pixels),
        "total_pixels": int(total_pixels),
        "density": mana_ratio,
        "crop_shape": crop.shape,
    }
    
    return mana_ratio, details


def find_red_region(region: np.ndarray, full_h: int, full_w: int) -> list[dict]:
    """Find regions with high red concentration."""
    hsv = cv2.cvtColor(region, cv2.COLOR_BGR2HSV)
    
    # Red mask
    lower_red1 = np.array([0, 100, 50])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([170, 100, 50])
    upper_red2 = np.array([180, 255, 255])
    
    mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
    red_mask = mask1 | mask2
    
    # Find contours
    contours, _ = cv2.findContours(red_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    suggestions = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        area = w * h
        
        if area > 500:  # Minimum area
            # Convert to full image coordinates
            y_full = y + int(full_h * 0.7)
            
            suggestions.append({
                "x_ratio": x / full_w,
                "y_ratio": y_full / full_h,
                "w_ratio": w / full_w,
                "h_ratio": h / full_h,
                "score": area,
            })
    
    suggestions.sort(key=lambda s: s['score'], reverse=True)
    return suggestions


def find_blue_region(region: np.ndarray, full_h: int, full_w: int) -> list[dict]:
    """Find regions with high blue concentration."""
    hsv = cv2.cvtColor(region, cv2.COLOR_BGR2HSV)
    
    # Blue mask
    lower_blue = np.array([100, 100, 50])
    upper_blue = np.array([140, 255, 255])
    
    blue_mask = cv2.inRange(hsv, lower_blue, upper_blue)
    
    # Find contours
    contours, _ = cv2.findContours(blue_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    suggestions = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        area = w * h
        
        if area > 500:  # Minimum area
            # Convert to full image coordinates
            y_full = y + int(full_h * 0.7)
            
            suggestions.append({
                "x_ratio": x / full_w,
                "y_ratio": y_full / full_h,
                "w_ratio": w / full_w,
                "h_ratio": h / full_h,
                "score": area,
            })
    
    suggestions.sort(key=lambda s: s['score'], reverse=True)
    return suggestions


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Calibrate HP/Mana bar regions")
    parser.add_argument(
        "image",
        nargs="?",
        default="data/screenshots/inputs/game_a1.png",
        help="Path to screenshot"
    )
    
    args = parser.parse_args()
    
    analyze_image(args.image)
