"""
Visualize map with exit candidates from last bot run.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent / "src"))

import cv2
import numpy as np
import json


def visualize_last_map_with_exits():
    """Load last map and show exit candidates."""
    
    print("\n" + "="*70)
    print("🗺️  MAP VISUALIZATION WITH EXIT CANDIDATES")
    print("="*70 + "\n")
    
    # Find most recent map
    maps_dir = Path("data/maps")
    if not maps_dir.exists():
        print("❌ No maps directory found")
        return
    
    # Find all map files
    map_files = sorted(maps_dir.glob("*_map.png"), key=lambda p: p.stat().st_mtime, reverse=True)
    
    if not map_files:
        print("❌ No map files found")
        return
    
    map_file = map_files[0]
    metadata_file = map_file.with_name(map_file.name.replace("_map.png", "_metadata.json"))
    
    print(f"📁 Loading map: {map_file.name}")
    
    # Load map image
    img = cv2.imread(str(map_file))
    if img is None:
        print("❌ Could not load map image")
        return
    
    print(f"   Size: {img.shape[1]}x{img.shape[0]}")
    
    # Load metadata
    if metadata_file.exists():
        with open(metadata_file) as f:
            metadata = json.load(f)
        
        print(f"   Zone: {metadata.get('zone', 'Unknown')}")
        print(f"   Cells: {metadata.get('cell_count', 0)}")
        print(f"   POIs: {len(metadata.get('pois', []))}")
        
        player_pos = metadata.get('player_pos', [1024, 1024])
        print(f"   Player: {player_pos}\n")
    else:
        print("   ⚠️ No metadata found\n")
        player_pos = [1024, 1024]
    
    # Simulate exit candidates from log (from last run)
    # Player @ (1024, 1024)
    # Exit candidates:
    #   #1: pos=(992, 992), score=0.71, dir=-135°, dist=45.3
    #   #2: pos=(1055, 992), score=0.70, dir=-46°, dist=44.6
    #   #3: pos=(992, 1055), score=0.70, dir=136°, dist=44.6
    
    exit_candidates = [
        {"pos": (992, 992), "score": 0.71, "dir": -135, "dist": 45.3},
        {"pos": (1055, 992), "score": 0.70, "dir": -46, "dist": 44.6},
        {"pos": (992, 1055), "score": 0.70, "dir": 136, "dist": 44.6},
    ]
    
    print("🚪 Exit candidates:")
    for i, exit_c in enumerate(exit_candidates, 1):
        print(f"   #{i}: pos={exit_c['pos']}, score={exit_c['score']:.2f}, dir={exit_c['dir']}°, dist={exit_c['dist']:.1f}")
    print()
    
    # Find bounding box of cells (same logic as visualize())
    # For now, use entire image
    h, w = img.shape[:2]
    
    # Scale factor (image is already scaled by visualize())
    # Original grid was 64x64, but image may be larger
    # Assume 4x scale was used
    scale = 4
    
    # Calculate center of image (where player should be)
    # Player is at (1024, 1024) in global coords
    # We need to find where that is in the image
    
    # For visualization, the image shows explored area + margin
    # Let's assume player is roughly at center of image
    img_center_x = w // 2
    img_center_y = h // 2
    
    print("🎨 Drawing visualization...")
    vis = img.copy()
    
    # Draw exit candidates
    for i, exit_c in enumerate(exit_candidates, 1):
        exit_x, exit_y = exit_c['pos']
        score = exit_c['score']
        
        # Convert global coords to image coords
        # If player is at center of image and at (1024, 1024) global
        # Then exit at (992, 992) is offset by (-32, -32)
        offset_x = (exit_x - player_pos[0]) * scale
        offset_y = (exit_y - player_pos[1]) * scale
        
        img_x = img_center_x + offset_x
        img_y = img_center_y + offset_y
        
        # Color based on rank
        if i == 1:
            color = (0, 255, 0)  # Green for best
        elif i == 2:
            color = (0, 255, 255)  # Yellow for second
        else:
            color = (0, 165, 255)  # Orange for third
        
        # Draw circle and cross
        cv2.circle(vis, (int(img_x), int(img_y)), 8, color, 2)
        cv2.line(vis, (int(img_x)-5, int(img_y)), (int(img_x)+5, int(img_y)), color, 2)
        cv2.line(vis, (int(img_x), int(img_y)-5), (int(img_x), int(img_y)+5), color, 2)
        
        # Label
        label = f"#{i} ({score:.2f})"
        cv2.putText(vis, label, (int(img_x)+10, int(img_y)-10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
    
    # Draw player position (larger, white)
    cv2.circle(vis, (img_center_x, img_center_y), 12, (255, 255, 255), 2)
    cv2.circle(vis, (img_center_x, img_center_y), 6, (0, 255, 0), -1)
    cv2.putText(vis, "PLAYER", (img_center_x+15, img_center_y),
               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
    
    # Add legend
    y = 30
    legend_items = [
        f"Zone: {metadata.get('zone', 'Unknown') if metadata_file.exists() else 'Unknown'}",
        f"Explored: 41.0% (threshold 30.0%)",
        f"Decision: SEEK EXIT",
        "",
        "Exit Candidates:",
        "#1 (green) - Best exit",
        "#2 (yellow) - 2nd best",
        "#3 (orange) - 3rd best",
    ]
    
    for line in legend_items:
        cv2.putText(vis, line, (10, y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        y += 20
    
    # Save
    output_path = Path("data/screenshots/outputs/map_with_exits.png")
    cv2.imwrite(str(output_path), vis)
    print(f"   ✓ Saved: {output_path}\n")
    
    # Display
    print("="*70)
    print("📊 VISUALIZATION")
    print("="*70)
    print(f"Player position: Center of image")
    print(f"Exit #1 (GREEN):  32 cells NW, score 0.71")
    print(f"Exit #2 (YELLOW): 31 cells NE, score 0.70")
    print(f"Exit #3 (ORANGE): 32 cells SW, score 0.70")
    print("="*70 + "\n")
    
    cv2.imshow("Map with Exit Candidates", vis)
    print("Press any key to exit...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    visualize_last_map_with_exits()
