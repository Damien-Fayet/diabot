"""
Debug static map localization with detailed step-by-step visualization.

This script shows:
1. Original minimap from game
2. Processed minimap grid
3. Static reference map
4. Edge detection on both
5. Template matching result
6. Final localization
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent / "src"))

import cv2
import numpy as np
import json

from diabot.navigation.static_map_localizer import StaticMapLocalizer, load_zone_static_map
from diabot.navigation.minimap_processor import MinimapProcessor


def debug_localization():
    """Debug localization with step-by-step visualization."""
    
    print("\n" + "="*70)
    print("🔍 STATIC MAP LOCALIZATION DEBUG")
    print("="*70 + "\n")
    
    # 1. Load static map
    print("[1] Loading static map...")
    zone_name = "ROGUE ENCAMPMENT"
    static_map_path = load_zone_static_map(zone_name)
    
    if static_map_path is None:
        print("❌ Static map not found")
        return
    
    static_map = cv2.imread(str(static_map_path))
    static_gray = cv2.cvtColor(static_map, cv2.COLOR_BGR2GRAY)
    print(f"✓ Static map: {static_map.shape[:2][::-1]} (WxH)")
    
    # Load annotations
    annotations_path = static_map_path.parent / f"{static_map_path.stem}_annotations.json"
    if annotations_path.exists():
        with open(annotations_path, 'r') as f:
            annotations = json.load(f)
        print(f"✓ Annotations: {len(annotations.get('pois', []))} POIs")
    
    # 2. Load game minimap
    print("\n[2] Loading captured minimap...")
    minimap_path = "data/screenshots/outputs/minimap_fullscreen_capture.png"
    
    if not Path(minimap_path).exists():
        print(f"❌ Minimap not found: {minimap_path}")
        print("Run the bot first to capture a minimap")
        return
    
    minimap_img = cv2.imread(minimap_path)
    print(f"✓ Game minimap: {minimap_img.shape[:2][::-1]} (WxH)")
    
    # 3. Process minimap to grid
    print("\n[3] Processing minimap to grid...")
    processor = MinimapProcessor(grid_size=64, wall_threshold=49, debug=False)
    minimap_grid = processor.process(minimap_img)
    
    # Get grid as image
    grid_img = minimap_grid.grid.astype(np.uint8)
    print(f"✓ Grid: {grid_img.shape[:2][::-1]} (WxH)")
    print(f"  Walls: {np.sum(grid_img == 255)}")
    print(f"  Free: {np.sum(grid_img == 128)}")
    
    # 4. Edge detection
    print("\n[4] Applying edge detection...")
    
    # Static map edges
    static_edges = cv2.Canny(static_gray, 50, 150)
    kernel = np.ones((3, 3), np.uint8)
    static_edges = cv2.dilate(static_edges, kernel, iterations=1)
    
    # Minimap edges
    minimap_edges = cv2.Canny(grid_img, 50, 150)
    minimap_edges = cv2.dilate(minimap_edges, kernel, iterations=1)
    
    print(f"✓ Static edges: {np.sum(static_edges > 0)} pixels")
    print(f"✓ Minimap edges: {np.sum(minimap_edges > 0)} pixels")
    
    # 5. Try multiple scales
    print("\n[5] Testing template matching at different scales...")
    
    best_results = []
    
    for scale in [0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0]:
        # Resize minimap edges
        new_w = int(minimap_edges.shape[1] * scale)
        new_h = int(minimap_edges.shape[0] * scale)
        
        if new_w > static_edges.shape[1] or new_h > static_edges.shape[0]:
            continue
        
        if new_w < 10 or new_h < 10:
            continue
        
        scaled_template = cv2.resize(minimap_edges, (new_w, new_h))
        
        # Match
        result = cv2.matchTemplate(static_edges, scaled_template, cv2.TM_CCOEFF_NORMED)
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
        
        # Store result
        best_results.append({
            'scale': scale,
            'confidence': max_val,
            'location': max_loc,
            'size': (new_w, new_h)
        })
        
        print(f"  Scale {scale:.2f}x ({new_w}x{new_h}): confidence={max_val:.3f}")
    
    # Find best match
    best = max(best_results, key=lambda x: x['confidence'])
    print(f"\n✓ Best match: scale={best['scale']:.2f}x, confidence={best['confidence']:.3f}")
    
    # 6. Visualize best match
    print("\n[6] Creating visualization...")
    
    # Prepare images for display
    vis_static = static_map.copy()
    
    # Draw match on static map
    x, y = best['location']
    w, h = best['size']
    cx = x + w // 2
    cy = y + h // 2
    
    # Draw rectangle
    cv2.rectangle(vis_static, (x, y), (x + w, y + h), (0, 255, 0), 2)
    
    # Draw crosshair at center
    cv2.drawMarker(vis_static, (cx, cy), (0, 0, 255),
                  markerType=cv2.MARKER_CROSS, markerSize=20, thickness=2)
    
    # Add text
    cv2.putText(vis_static, f"Match: {best['confidence']:.3f}", (cx + 25, cy - 20),
               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    cv2.putText(vis_static, f"Pos: ({cx}, {cy})", (cx + 25, cy),
               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    cv2.putText(vis_static, f"Scale: {best['scale']:.2f}x", (cx + 25, cy + 20),
               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    # Mark exit POI if available
    if annotations_path.exists():
        exits = [p for p in annotations.get('pois', []) if p['type'] == 'exit']
        for exit_poi in exits:
            ex, ey = exit_poi['position']
            cv2.circle(vis_static, (ex, ey), 8, (0, 0, 255), -1)
            cv2.putText(vis_static, exit_poi['name'], (ex + 15, ey),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)
    
    # Create detailed panel
    # Resize for display
    grid_vis = cv2.cvtColor(grid_img, cv2.COLOR_GRAY2BGR)
    grid_vis = cv2.resize(grid_vis, (227, 168))
    cv2.putText(grid_vis, "Minimap Grid", (5, 15),
               cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
    
    minimap_edges_vis = cv2.cvtColor(minimap_edges, cv2.COLOR_GRAY2BGR)
    minimap_edges_vis = cv2.resize(minimap_edges_vis, (227, 168))
    cv2.putText(minimap_edges_vis, "Minimap Edges", (5, 15),
               cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)
    
    static_edges_vis = cv2.cvtColor(static_edges, cv2.COLOR_GRAY2BGR)
    cv2.putText(static_edges_vis, "Static Map Edges", (5, 15),
               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
    
    # Stack minimap views vertically
    minimap_panel = np.vstack([grid_vis, minimap_edges_vis])
    
    # Stack horizontally
    top_row = np.hstack([minimap_panel, static_edges_vis])
    
    # Bottom row: matched static map (full size)
    bottom_row = vis_static
    
    # Add info text to bottom
    info_text = [
        f"Confidence: {best['confidence']:.3f} ({'GOOD' if best['confidence'] >= 0.35 else 'POOR'})",
        f"Threshold: 0.35 for good, 0.25 for poor",
        f"Current: {'PASSED' if best['confidence'] >= 0.35 else 'FAILED'} (too low)",
    ]
    
    y_offset = vis_static.shape[0] - 60
    for i, line in enumerate(info_text):
        color = (0, 255, 0) if best['confidence'] >= 0.35 else (0, 0, 255)
        cv2.putText(bottom_row, line, (10, y_offset + i * 20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    
    # Stack vertically
    # Make widths match
    if top_row.shape[1] < bottom_row.shape[1]:
        padding_w = bottom_row.shape[1] - top_row.shape[1]
        padding = np.zeros((top_row.shape[0], padding_w, 3), dtype=np.uint8)
        top_row = np.hstack([top_row, padding])
    elif top_row.shape[1] > bottom_row.shape[1]:
        padding_w = top_row.shape[1] - bottom_row.shape[1]
        padding = np.zeros((bottom_row.shape[0], padding_w, 3), dtype=np.uint8)
        bottom_row = np.hstack([bottom_row, padding])
    
    final_vis = np.vstack([top_row, bottom_row])
    
    # Save
    output_path = "data/screenshots/outputs/localization_debug_full.png"
    cv2.imwrite(output_path, final_vis)
    print(f"✓ Saved: {output_path}")
    
    # Show
    cv2.imshow("Static Map Localization Debug", final_vis)
    print("\n👀 Press any key to close...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    # 7. Recommendations
    print("\n" + "="*70)
    print("📋 DIAGNOSIS & RECOMMENDATIONS")
    print("="*70)
    
    if best['confidence'] >= 0.50:
        print("\n✅ EXCELLENT match quality")
        print("   → No changes needed")
    elif best['confidence'] >= 0.35:
        print("\n✓ GOOD match quality")
        print("   → Localization should work")
    elif best['confidence'] >= 0.25:
        print("\n⚠️  POOR match quality")
        print("   → Localization unreliable, will fallback to standard navigation")
    else:
        print("\n❌ FAILED match")
        print("   → Static map doesn't match game minimap")
    
    print("\nPossible improvements:")
    print("  1. Lower thresholds (current: 0.35 good, 0.25 poor)")
    print("  2. Use different preprocessing (blur, contrast)")
    print("  3. Try different edge detection parameters")
    print("  4. Use feature matching instead of template matching")
    print("  5. Redraw static map to match game minimap style")
    
    print("\n" + "="*70 + "\n")


if __name__ == "__main__":
    debug_localization()
