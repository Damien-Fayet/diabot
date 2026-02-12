"""
Debug minimap extraction to understand coordinate conversion.
Shows the actual minimap region and its relationship to the game screen.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent / "src"))

import cv2
import numpy as np

from diabot.vision.screen_regions import UI_REGIONS


def debug_minimap():
    """Visualize minimap extraction and sizes."""
    
    print("\n" + "="*70)
    print("🔍 MINIMAP SIZE DEBUG")
    print("="*70 + "\n")
    
    # Load game screenshot
    game_path = Path("data/screenshots/inputs/game_screenshot.png")
    if not game_path.exists():
        print(f"❌ Game screenshot not found: {game_path}")
        return
    
    frame = cv2.imread(str(game_path))
    frame_h, frame_w = frame.shape[:2]
    print(f"📸 Game frame: {frame_w}x{frame_h}\n")
    
    # Get minimap region
    minimap_region = UI_REGIONS.get("minimap_ui")
    if not minimap_region:
        print("❌ Minimap region not configured")
        return
    
    print(f"🗺️  Minimap region config:")
    print(f"   x_ratio: {minimap_region.x_ratio} ({minimap_region.x_ratio*100:.0f}%)")
    print(f"   y_ratio: {minimap_region.y_ratio} ({minimap_region.y_ratio*100:.0f}%)")
    print(f"   w_ratio: {minimap_region.w_ratio} ({minimap_region.w_ratio*100:.0f}%)")
    print(f"   h_ratio: {minimap_region.h_ratio} ({minimap_region.h_ratio*100:.0f}%)\n")
    
    # Extract minimap
    x, y, w, h = minimap_region.get_bounds(frame_h, frame_w)
    print(f"📐 Extracted bounds:")
    print(f"   x={x}, y={y}, w={w}, h={h}")
    print(f"   Minimap size: {w}x{h}\n")
    
    x_end = min(x + w, frame_w)
    y_end = min(y + h, frame_h)
    minimap = frame[y:y_end, x:x_end].copy()
    
    # Draw rectangle on frame
    frame_vis = frame.copy()
    cv2.rectangle(frame_vis, (x, y), (x_end, y_end), (0, 255, 0), 3)
    cv2.putText(frame_vis, f"Minimap: {w}x{h}", (x, y-10),
               cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    
    # Problem analysis
    print("⚠️  PROBLEM ANALYSIS:")
    print(f"   Current minimap extraction: {w}x{h} pixels")
    print(f"   Expected minimap size: ~120x120 pixels")
    print(f"   Ratio: {w/120:.1f}x too large!\n")
    
    # Calculate correct ratios
    # D2R minimap is approximately 120x120 in a 1920x1080 screen
    # Top-right corner, with some margin
    target_size = 120
    correct_w_ratio = target_size / frame_w
    correct_h_ratio = target_size / frame_h
    correct_x_ratio = 1.0 - correct_w_ratio - 0.02  # 2% margin from right
    correct_y_ratio = 0.02  # 2% margin from top
    
    print("✅ SUGGESTED CORRECTION:")
    print(f"   minimap_ui: ScreenRegion(")
    print(f"       x_ratio={correct_x_ratio:.4f},")
    print(f"       y_ratio={correct_y_ratio:.4f},")
    print(f"       w_ratio={correct_w_ratio:.4f},")
    print(f"       h_ratio={correct_h_ratio:.4f},")
    print(f"   )\n")
    
    # Test with corrected values
    x_new = int(correct_x_ratio * frame_w)
    y_new = int(correct_y_ratio * frame_h)
    w_new = int(correct_w_ratio * frame_w)
    h_new = int(correct_h_ratio * frame_h)
    
    minimap_corrected = frame[y_new:y_new+h_new, x_new:x_new+w_new].copy()
    
    cv2.rectangle(frame_vis, (x_new, y_new), (x_new+w_new, y_new+h_new), (0, 0, 255), 3)
    cv2.putText(frame_vis, f"Corrected: {w_new}x{h_new}", (x_new, y_new-10),
               cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    
    print(f"   New extraction size: {w_new}x{h_new} pixels\n")
    
    # Save visualizations
    output_dir = Path("data/screenshots/outputs")
    
    # Scale down frame for display
    scale = 0.5
    frame_display = cv2.resize(frame_vis, None, fx=scale, fy=scale)
    minimap_large = cv2.resize(minimap, (400, 400), interpolation=cv2.INTER_LINEAR)
    minimap_corrected_large = cv2.resize(minimap_corrected, (400, 400), interpolation=cv2.INTER_LINEAR)
    
    cv2.imwrite(str(output_dir / "debug_minimap_frame.png"), frame_display)
    cv2.imwrite(str(output_dir / "debug_minimap_current.png"), minimap_large)
    cv2.imwrite(str(output_dir / "debug_minimap_corrected.png"), minimap_corrected_large)
    
    print("💾 Saved:")
    print(f"   {output_dir / 'debug_minimap_frame.png'}")
    print(f"   {output_dir / 'debug_minimap_current.png'}")
    print(f"   {output_dir / 'debug_minimap_corrected.png'}\n")
    
    # Display comparison
    comparison = np.hstack([minimap_large, minimap_corrected_large])
    
    # Add labels
    cv2.putText(comparison, "Current (WRONG)", (10, 30),
               cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(comparison, "Corrected", (410, 30),
               cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    
    print("="*70)
    print("📊 IMPACT ON POI POSITIONING:")
    print("="*70)
    print(f"Current scale: {w}px minimap / 64 cells = {w/64:.1f} px/cell")
    print(f"Correct scale: {w_new}px minimap / 64 cells = {w_new/64:.1f} px/cell")
    print(f"\nScale ratio difference: {(w/64) / (w_new/64):.1f}x")
    print(f"POIs are currently displaced by ~{(w/64) / (w_new/64):.1f}x their real distance!\n")
    print("="*70 + "\n")
    
    cv2.imshow("Minimap Comparison", comparison)
    print("Press any key to exit...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    debug_minimap()
