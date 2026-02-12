"""
Capture background subtraction to isolate minimap.

Captures:
1. Frame WITHOUT minimap (Tab off)
2. Frame WITH minimap (Tab on)
3. Computes difference to isolate pure minimap

This removes background pollution and reveals true map structure.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent / "src"))

import cv2
import numpy as np
import time

from diabot.core.implementations import WindowsScreenCapture


def capture_with_background_subtraction():
    """Capture and subtract background to isolate minimap."""
    
    print("\n" + "="*70)
    print("🎬 BACKGROUND SUBTRACTION FOR MINIMAP")
    print("="*70 + "\n")
    
    try:
        capturer = WindowsScreenCapture()
        print("✓ Screen capture initialized\n")
    except Exception as e:
        print(f"❌ Failed to initialize capture: {e}")
        return
    
    print("📋 Instructions:")
    print("   1. Make sure Diablo 2 window is visible")
    print("   2. Stand still in game")
    print("   3. Press ENTER when ready...")
    input()
    
    # Step 1: Capture WITHOUT minimap
    print("\n[1/3] Capturing frame WITHOUT minimap...")
    print("      → Make sure minimap is CLOSED (press Tab if needed)")
    print("      → Press ENTER when minimap is OFF...")
    input()
    
    time.sleep(0.5)  # Small delay
    frame_no_map = capturer.get_frame()
    
    if frame_no_map is None:
        print("❌ Failed to capture frame without minimap")
        return
    
    print(f"      ✓ Captured: {frame_no_map.shape}")
    
    # Save
    output_dir = Path("data/screenshots/outputs")
    cv2.imwrite(str(output_dir / "background_no_minimap.png"), frame_no_map)
    print(f"      ✓ Saved: background_no_minimap.png\n")
    
    # Step 2: Capture WITH minimap
    print("[2/3] Capturing frame WITH minimap...")
    print("      → Press TAB to open fullscreen minimap")
    print("      → Press ENTER when minimap is VISIBLE...")
    input()
    
    time.sleep(0.5)  # Small delay
    frame_with_map = capturer.get_frame()
    
    if frame_with_map is None:
        print("❌ Failed to capture frame with minimap")
        return
    
    print(f"      ✓ Captured: {frame_with_map.shape}")
    
    # Save
    cv2.imwrite(str(output_dir / "background_with_minimap.png"), frame_with_map)
    print(f"      ✓ Saved: background_with_minimap.png\n")
    
    # Step 3: Compute difference
    print("[3/3] Computing difference...")
    
    # Convert to grayscale for better comparison
    gray_no_map = cv2.cvtColor(frame_no_map, cv2.COLOR_BGR2GRAY)
    gray_with_map = cv2.cvtColor(frame_with_map, cv2.COLOR_BGR2GRAY)
    
    # Absolute difference
    diff = cv2.absdiff(gray_with_map, gray_no_map)
    
    print(f"      ✓ Difference computed")
    
    # Threshold to remove noise
    _, diff_binary = cv2.threshold(diff, 10, 255, cv2.THRESH_BINARY)
    
    # Save difference
    cv2.imwrite(str(output_dir / "background_diff_raw.png"), diff)
    cv2.imwrite(str(output_dir / "background_diff_binary.png"), diff_binary)
    print(f"      ✓ Saved: background_diff_raw.png")
    print(f"      ✓ Saved: background_diff_binary.png\n")
    
    # Also show color difference
    diff_color = cv2.absdiff(frame_with_map, frame_no_map)
    cv2.imwrite(str(output_dir / "background_diff_color.png"), diff_color)
    print(f"      ✓ Saved: background_diff_color.png\n")
    
    # Extract minimap region from difference
    from diabot.vision.screen_regions import UI_REGIONS
    minimap_region = UI_REGIONS.get("minimap_ui")
    
    if minimap_region:
        h, w = diff.shape
        x, y, mw, mh = minimap_region.get_bounds(h, w)
        
        # Crop minimap from difference
        minimap_diff = diff[y:y+mh, x:x+mw]
        minimap_diff_binary = diff_binary[y:y+mh, x:x+mw]
        
        cv2.imwrite(str(output_dir / "minimap_diff_isolated.png"), minimap_diff)
        cv2.imwrite(str(output_dir / "minimap_diff_binary.png"), minimap_diff_binary)
        print(f"      ✓ Saved: minimap_diff_isolated.png")
        print(f"      ✓ Saved: minimap_diff_binary.png\n")
    
    # Statistics
    print("="*70)
    print("📊 ANALYSIS")
    print("="*70)
    print(f"Difference range: {diff.min()} - {diff.max()}")
    print(f"Mean difference: {diff.mean():.1f}")
    print(f"Non-zero pixels: {np.count_nonzero(diff_binary)} ({np.count_nonzero(diff_binary)/diff_binary.size*100:.1f}%)")
    print("="*70 + "\n")
    
    # Create comparison visualization
    print("🎨 Creating comparison visualization...")
    
    # Resize for display
    scale = 0.5
    h_display = int(frame_no_map.shape[0] * scale)
    w_display = int(frame_no_map.shape[1] * scale)
    
    no_map_small = cv2.resize(frame_no_map, (w_display, h_display))
    with_map_small = cv2.resize(frame_with_map, (w_display, h_display))
    diff_color_small = cv2.resize(diff_color, (w_display, h_display))
    
    # Stack horizontally
    comparison = np.hstack([no_map_small, with_map_small, diff_color_small])
    
    # Add labels
    cv2.putText(comparison, "WITHOUT MINIMAP", (10, 30),
               cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(comparison, "WITH MINIMAP", (w_display + 10, 30),
               cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(comparison, "DIFFERENCE (Isolated Minimap)", (w_display*2 + 10, 30),
               cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    
    cv2.imwrite(str(output_dir / "background_subtraction_comparison.png"), comparison)
    print(f"   ✓ Saved: background_subtraction_comparison.png\n")
    
    print("="*70)
    print("✅ COMPLETE")
    print("="*70)
    print("\nFiles saved in data/screenshots/outputs/:")
    print("  • background_no_minimap.png - Original without map")
    print("  • background_with_minimap.png - Original with map")
    print("  • background_diff_raw.png - Raw difference (grayscale)")
    print("  • background_diff_binary.png - Binary threshold")
    print("  • background_diff_color.png - Color difference")
    print("  • minimap_diff_isolated.png - Cropped minimap difference")
    print("  • background_subtraction_comparison.png - Side-by-side comparison")
    print("="*70 + "\n")
    
    # Display
    cv2.imshow("Background Subtraction Result", comparison)
    print("Press any key to exit...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    capture_with_background_subtraction()
