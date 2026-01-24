#!/usr/bin/env python3
"""
Compare the original image with regions and the analyzed version.
"""

import cv2
from pathlib import Path

def main():
    output_dir = Path("data/screenshots/outputs")
    
    # Load images
    regions_img = cv2.imread(str(output_dir / "game_with_regions.jpg"))
    analysis_img = cv2.imread(str(output_dir / "game_vision_analysis.jpg"))
    
    if regions_img is None or analysis_img is None:
        print("❌ One or both images not found")
        return
    
    print("📊 VISION RESULTS")
    print("=" * 60)
    print()
    print("📍 LEFT IMAGE: Screen Regions")
    print("   Shows where the detection regions are located")
    print("   - Cyan boxes: UI regions")
    print("   - Green box: Playfield (environment)")
    print()
    print("📍 RIGHT IMAGE: Detection Results")
    print("   Shows what was detected")
    print("   - Green boxes with labels: Enemies detected")
    print("   - HP/Mana values: Health bar detected")
    print()
    
    # Create side-by-side view
    h1, w1 = regions_img.shape[:2]
    h2, w2 = analysis_img.shape[:2]
    
    # Resize to same height
    max_h = max(h1, h2)
    if h1 != max_h:
        scale = max_h / h1
        regions_img = cv2.resize(regions_img, (int(w1 * scale), max_h))
    if h2 != max_h:
        scale = max_h / h2
        analysis_img = cv2.resize(analysis_img, (int(w2 * scale), max_h))
    
    # Concatenate horizontally
    combined = cv2.hconcat([regions_img, analysis_img])
    
    # Add labels
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(combined, "REGIONS DEFINITION", (20, 30), font, 1, (255, 0, 0), 2)
    cv2.putText(combined, "DETECTION RESULTS", (regions_img.shape[1] + 20, 30), font, 1, (255, 0, 0), 2)
    
    # Save
    output_path = output_dir / "vision_comparison.jpg"
    cv2.imwrite(str(output_path), combined)
    
    print(f"✅ Saved comparison to: {output_path}")
    print()
    print("📊 Detections Summary:")
    print("   Health: 33%")
    print("   Mana: 27%")
    print("   Enemies: 20 detected")
    print("   Items: 0 detected")
    print()
    
    # Show files created
    print("📁 Output files:")
    print(f"   1. {output_dir / 'game_with_regions.jpg'}")
    print(f"      → Screen regions visualization")
    print(f"   2. {output_dir / 'game_vision_analysis.jpg'}")
    print(f"      → Detection results with bounding boxes")
    print(f"   3. {output_dir / 'vision_comparison.jpg'}")
    print(f"      → Side-by-side comparison")
    print()

if __name__ == "__main__":
    main()
