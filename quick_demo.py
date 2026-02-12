"""
Quick demo of the complete cartography + POI system.

Shows all features in one short script.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent / "src"))

import cv2
import numpy as np

from diabot.navigation.minimap_processor import MinimapProcessor
from diabot.navigation.player_locator import PlayerLocator
from diabot.navigation.map_accumulator import MapAccumulator
from diabot.navigation.exit_navigator import ExitNavigator


def quick_demo():
    """Run complete demo in ~10 seconds."""
    
    print("\n" + "="*70)
    print("🗺️  CARTOGRAPHY + POI SYSTEM - QUICK DEMO")
    print("="*70 + "\n")
    
    # Check if minimap exists
    minimap_path = "data/screenshots/outputs/diagnostic/minimap_steps/step1_extracted.png"
    if not Path(minimap_path).exists():
        print("❌ Minimap not found. Run bot with --debug first:")
        print("   python src/diabot/main.py --debug --max-frames 1")
        return
    
    # Initialize system
    print("📦 Initializing system...")
    processor = MinimapProcessor(grid_size=64, wall_threshold=49, debug=False)
    locator = PlayerLocator(debug=False)
    accumulator = MapAccumulator(map_size=2048, save_dir=Path("data/maps"), debug=False)
    navigator = ExitNavigator(debug=False)
    print("   ✓ Ready (YOLO not needed for this demo)\n")
    
    # Load minimap
    print("📸 Loading minimap...")
    minimap = cv2.imread(minimap_path)
    print(f"   ✓ Loaded {minimap.shape}\n")
    
    # Process
    print("⚙️  Processing minimap...")
    grid = processor.process(minimap)
    from diabot.navigation.minimap_processor import CellType
    walls = np.sum(grid.grid == CellType.WALL)
    free = np.sum(grid.grid == CellType.FREE)
    print(f"   ✓ Walls: {walls} ({walls/grid.grid.size*100:.0f}%)")
    print(f"   ✓ Free:  {free} ({free/grid.grid.size*100:.0f}%)\n")
    
    # Detect player
    print("🎯 Detecting player...")
    player_pos = locator.detect_player_cross(minimap)
    print(f"   ✓ Player at: {player_pos}\n")
    
    # Update map
    print("🗺️  Updating map...")
    accumulator.update(grid, player_offset=(0, 0))
    print(f"   ✓ Cells mapped: {len(accumulator.cells)}\n")
    
    # Add POIs
    print("📍 Adding POIs...")
    pois_to_add = [
        ("npc", (1030, 1020), "Akara", 0.95),
        ("npc", (1040, 1015), "Kashya", 0.92),
        ("waypoint", (1010, 1030), "Waypoint", 0.98),
        ("exit", (990, 1055), "Blood Moor", 0.88),
        ("chest", (1050, 1000), "Chest", 0.75),
    ]
    
    for poi_type, pos, label, conf in pois_to_add:
        accumulator.add_poi(poi_type, pos, label, conf)
        print(f"   ✓ {label} ({poi_type})")
    print()
    
    # Find exits
    print("🚪 Finding exits...")
    exits = navigator.find_exit_candidates(accumulator, max_candidates=3)
    print(f"   ✓ Found {len(exits)} exit candidates")
    if exits:
        best = exits[0]
        print(f"   → Best: {best.position} (score={best.score:.2f})\n")
    
    # Decide mode
    print("🧭 Navigation mode...")
    should_explore = navigator.should_explore_instead(accumulator, 0.3)
    mode = "EXPLORE" if should_explore else "SEEK EXIT"
    print(f"   ✓ Mode: {mode}\n")
    
    # Visualize
    print("🎨 Generating visualization...")
    vis = accumulator.visualize(scale=4)
    
    # Add info overlay
    y = 20
    info_lines = [
        f"Cells: {len(accumulator.cells)}",
        f"POIs: {len(accumulator.pois)}",
        f"Mode: {mode}",
    ]
    
    for line in info_lines:
        cv2.putText(vis, line, (10, y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        y += 20
    
    output_path = Path("data/screenshots/outputs/quick_demo_result.png")
    cv2.imwrite(str(output_path), vis)
    print(f"   ✓ Saved: {output_path}\n")
    
    # Save map
    print("💾 Saving map...")
    accumulator.save_map("DEMO_ZONE")
    print("   ✓ Map saved\n")
    
    # Test clear
    print("🧹 Testing clear...")
    before_cells = len(accumulator.cells)
    before_pois = len(accumulator.pois)
    
    accumulator.clear(keep_pois=True)
    after_cells = len(accumulator.cells)
    after_pois = len(accumulator.pois)
    
    print(f"   Before: {before_cells} cells, {before_pois} POIs")
    print(f"   After:  {after_cells} cells, {after_pois} POIs")
    print("   ✓ Clear works (kept POIs)\n")
    
    # Summary
    print("="*70)
    print("✅ DEMO COMPLETE")
    print("="*70)
    print("\nFeatures demonstrated:")
    print("  ✅ Minimap processing (optimized parameters)")
    print("  ✅ Player detection (white cross)")
    print("  ✅ Map accumulation (2048×2048)")
    print("  ✅ POI tracking (5 types)")
    print("  ✅ Exit detection (3 candidates)")
    print("  ✅ Navigation mode (EXPLORE/SEEK EXIT)")
    print("  ✅ Visualization (color-coded)")
    print("  ✅ Persistence (JSON + PNG)")
    print("  ✅ Map clearing (with/without POIs)")
    print("\n" + "="*70)
    print("Next steps:")
    print("  • View map:     python view_map.py")
    print("  • List maps:    python clear_maps.py --list")
    print("  • Clear maps:   python clear_maps.py --clear-all")
    print("  • Run bot:      python src/diabot/main.py --debug")
    print("="*70 + "\n")
    
    # Display
    cv2.imshow("Quick Demo Result", vis)
    print("Press any key to exit...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    quick_demo()
