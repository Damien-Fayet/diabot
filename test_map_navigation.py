"""
Test the integrated map-aware navigation system.

This script demonstrates:
1. Optimized minimap processing with tuned parameters
2. Player position detection (white cross)
3. Map accumulation in memory
4. Exit detection and navigation
5. Visualization of accumulated map
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


def test_map_navigation():
    """Test complete navigation pipeline on a single minimap image."""
    
    # Load test minimap image
    minimap_path = "data/screenshots/outputs/diagnostic/minimap_steps/step1_extracted.png"
    
    if not Path(minimap_path).exists():
        print(f"❌ Minimap image not found: {minimap_path}")
        print("Run the bot with --debug first to generate diagnostic images")
        return
    
    print("="*70)
    print("MAP-AWARE NAVIGATION SYSTEM TEST")
    print("="*70)
    
    # 1. Initialize components with optimized parameters
    print("\n[1] Initializing components...")
    processor = MinimapProcessor(
        grid_size=64,
        wall_threshold=49,  # From minimap_tuned_params.txt
        debug=True
    )
    locator = PlayerLocator(debug=True)
    accumulator = MapAccumulator(
        map_size=2048,
        cell_size=1.0,
        save_dir=Path("data/maps"),
        debug=True
    )
    navigator = ExitNavigator(debug=True)
    
    # 2. Load and process minimap
    print("\n[2] Loading minimap...")
    minimap = cv2.imread(minimap_path)
    if minimap is None:
        print(f"❌ Failed to load: {minimap_path}")
        return
    
    print(f"✓ Loaded minimap: {minimap.shape}")
    
    # 3. Process minimap to grid
    print("\n[3] Processing minimap to occupancy grid...")
    minimap_grid = processor.process(minimap)
    print(f"✓ Grid shape: {minimap_grid.shape}")
    print(f"✓ Grid center (player): {minimap_grid.center}")
    
    # Count wall/free cells
    from diabot.navigation.minimap_processor import CellType
    wall_count = np.sum(minimap_grid.grid == CellType.WALL)
    free_count = np.sum(minimap_grid.grid == CellType.FREE)
    total = minimap_grid.grid.size
    print(f"✓ Walls: {wall_count} ({wall_count/total*100:.1f}%)")
    print(f"✓ Free:  {free_count} ({free_count/total*100:.1f}%)")
    
    # 4. Detect player position
    print("\n[4] Detecting player position (white cross)...")
    player_pos = locator.detect_player_cross(minimap)
    if player_pos:
        print(f"✓ Player at: {player_pos}")
    else:
        print("⚠ Player position not detected, using center")
    
    # 5. Update accumulated map
    print("\n[5] Updating accumulated map...")
    accumulator.update(minimap_grid, player_offset=(0, 0))
    print(f"✓ Map cells: {len(accumulator.cells)}")
    print(f"✓ Player world pos: {accumulator.player_world_pos}")
    
    # 6. Find exit candidates
    print("\n[6] Finding exit candidates...")
    exit_candidates = navigator.find_exit_candidates(accumulator, max_candidates=5)
    
    if exit_candidates:
        print(f"✓ Found {len(exit_candidates)} exit candidates:")
        for i, candidate in enumerate(exit_candidates):
            print(f"  #{i+1}: pos={candidate.position}, score={candidate.score:.2f}, "
                  f"dir={candidate.direction:.0f}°, dist={candidate.distance:.1f}")
        
        # 7. Get navigation target for best exit
        print("\n[7] Computing navigation target for best exit...")
        best_exit = exit_candidates[0]
        target = navigator.get_navigation_target(best_exit, accumulator, minimap_grid)
        
        if target:
            rel_x, rel_y = target
            print(f"✓ Click target: ({rel_x:.2f}, {rel_y:.2f}) relative")
            print(f"  (Would click at {rel_x*100:.0f}%, {rel_y*100:.0f}% of screen)")
    else:
        print("⚠ No exit candidates found")
    
    # 8. Decide exploration vs exit-seeking
    print("\n[8] Deciding navigation strategy...")
    should_explore = navigator.should_explore_instead(accumulator, exploration_threshold=0.3)
    mode = "EXPLORE" if should_explore else "SEEK EXIT"
    print(f"✓ Navigation mode: {mode}")
    
    # 9. Visualize accumulated map
    print("\n[9] Generating visualizations...")
    
    # Minimap with player detection
    player_vis = locator.visualize_detection(minimap)
    cv2.imwrite("data/screenshots/outputs/test_player_detection.png", player_vis)
    print("✓ Saved: test_player_detection.png")
    
    # Occupancy grid
    grid_vis = processor.visualize(minimap_grid)
    cv2.imwrite("data/screenshots/outputs/test_occupancy_grid.png", grid_vis)
    print("✓ Saved: test_occupancy_grid.png")
    
    # Accumulated map
    map_vis = accumulator.visualize(scale=4)
    cv2.imwrite("data/screenshots/outputs/test_accumulated_map.png", map_vis)
    print("✓ Saved: test_accumulated_map.png")
    
    # Combined visualization
    print("\n[10] Creating combined visualization...")
    
    # Resize for side-by-side display
    h1, w1 = player_vis.shape[:2]
    h2, w2 = grid_vis.shape[:2]
    h3, w3 = map_vis.shape[:2]
    
    target_h = 400
    player_scaled = cv2.resize(player_vis, (int(w1*target_h/h1), target_h))
    grid_scaled = cv2.resize(grid_vis, (int(w2*target_h/h2), target_h))
    map_scaled = cv2.resize(map_vis, (int(w3*target_h/h3), target_h))
    
    # Stack horizontally
    combined = np.hstack([player_scaled, grid_scaled, map_scaled])
    
    # Add titles
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(combined, "1. Player Detection", (10, 30), font, 0.7, (0, 255, 255), 2)
    cv2.putText(combined, "2. Occupancy Grid", (player_scaled.shape[1] + 10, 30), 
               font, 0.7, (0, 255, 255), 2)
    cv2.putText(combined, "3. Accumulated Map", 
               (player_scaled.shape[1] + grid_scaled.shape[1] + 10, 30), 
               font, 0.7, (0, 255, 255), 2)
    
    cv2.imwrite("data/screenshots/outputs/test_navigation_pipeline.png", combined)
    print("✓ Saved: test_navigation_pipeline.png")
    
    # Display
    cv2.imshow("Navigation Pipeline Test", combined)
    print("\n" + "="*70)
    print("✓ TEST COMPLETE")
    print("="*70)
    print("\nPress any key in the image window to exit...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    # 11. Save accumulated map
    print("\n[11] Saving accumulated map...")
    accumulator.save_map("TEST_ZONE")
    print("✓ Map saved to data/maps/")
    
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print(f"Minimap processing: {wall_count/total*100:.1f}% walls, {free_count/total*100:.1f}% free")
    print(f"Player position: {player_pos}")
    print(f"Map cells accumulated: {len(accumulator.cells)}")
    print(f"Exit candidates found: {len(exit_candidates)}")
    print(f"Navigation mode: {mode}")
    print("\nAll visualizations saved to: data/screenshots/outputs/")
    print("="*70)


if __name__ == "__main__":
    test_map_navigation()
