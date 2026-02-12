"""
Test MinimapProcessor with background subtraction.

Demonstrates the new automatic background subtraction mode.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent / "src"))

import cv2
import numpy as np
from diabot.navigation.minimap_processor import MinimapProcessor


def main():
    """Test background subtraction minimap processing."""
    
    # Load images from capture
    no_map = cv2.imread("data/screenshots/outputs/background_no_minimap.png")
    with_map = cv2.imread("data/screenshots/outputs/background_with_minimap.png")
    
    if no_map is None or with_map is None:
        print("❌ Images not found. Run capture_background_subtraction.py first!")
        print(f"Looking for:")
        print("  - data/screenshots/outputs/background_no_minimap.png")
        print("  - data/screenshots/outputs/background_with_minimap.png")
        return
    
    print("\n" + "="*70)
    print("🧪 TESTING BACKGROUND SUBTRACTION MINIMAP PROCESSING")
    print("="*70)
    
    # Test 1: Legacy mode (for comparison)
    print("\n📊 Test 1: Legacy mode (without background sub)")
    processor_legacy = MinimapProcessor(
        grid_size=64,
        wall_threshold=49,
        debug=True,
        use_background_subtraction=False
    )
    
    # Extract minimap region (for legacy mode)
    # Assume fullscreen minimap is whole image
    grid_legacy = processor_legacy.process(with_map)
    
    print(f"✓ Legacy: {np.sum(grid_legacy.grid == 255)} walls, {np.sum(grid_legacy.grid == 128)} free")
    
    # Test 2: Background subtraction mode
    print("\n📊 Test 2: Background subtraction mode")
    processor_new = MinimapProcessor(
        grid_size=64,
        wall_threshold=30,
        debug=True,
        use_background_subtraction=True
    )
    
    # Set background reference
    processor_new.set_background(no_map)
    
    # Process with background subtraction
    grid_new = processor_new.process(with_map, full_frame=with_map)
    
    print(f"✓ BackgroundSub: {np.sum(grid_new.grid == 255)} walls, {np.sum(grid_new.grid == 128)} free")
    
    # Visualize both
    print("\n🎨 Creating comparison visualization...")
    
    vis_legacy = processor_legacy.visualize(grid_legacy)
    vis_new = processor_new.visualize(grid_new)
    
    # Add labels
    cv2.putText(vis_legacy, "Legacy Mode", (10, 20),
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    cv2.putText(vis_new, "Background Sub", (10, 20),
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    # Stack side by side
    comparison = np.hstack([vis_legacy, vis_new])
    
    # Save
    output_path = "data/screenshots/outputs/minimap_comparison.png"
    cv2.imwrite(output_path, comparison)
    
    print(f"✓ Comparison saved: {output_path}")
    
    # Show
    cv2.imshow("Minimap Processing Comparison", comparison)
    print("\n👀 Press any key to close...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    print("\n" + "="*70)
    print("✅ TEST COMPLETE")
    print("="*70)
    print("\nCheck diagnostic folder for step-by-step processing:")
    print("data/screenshots/outputs/diagnostic/minimap_steps/")
    print("\nBoth methods tested. Background subtraction should show better")
    print("boundary detection in Rogue Encampment.")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()
