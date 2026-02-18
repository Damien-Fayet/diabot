"""
Demo script for MinimapSLAM system.

Demonstrates:
1. Static minimap processing (developer mode)
2. Motion estimation between frames
3. Global map building
4. POI tracking
5. Loop closure detection
6. Visualization dashboard

This script runs in developer mode using static images.
No Windows-specific dependencies required.
"""

import cv2
import numpy as np
from pathlib import Path
import time

from src.diabot.navigation.minimap_slam import MinimapSLAM
from src.diabot.navigation.slam_visualizer import SLAMVisualizer


def create_synthetic_minimap(size: int = 200, walls_pattern: str = "corridor") -> np.ndarray:
    """
    Create synthetic minimap for testing.
    
    Args:
        size: Image size
        walls_pattern: Pattern type (corridor, room, intersection, loop)
        
    Returns:
        Synthetic minimap image
    """
    minimap = np.zeros((size, size, 3), dtype=np.uint8)
    
    # Background (dark)
    minimap[:] = (20, 20, 20)
    
    if walls_pattern == "corridor":
        # Vertical corridor
        cv2.rectangle(minimap, (80, 0), (120, size), (200, 200, 200), -1)
        # Walls
        cv2.rectangle(minimap, (70, 0), (75, size), (255, 255, 255), -1)
        cv2.rectangle(minimap, (125, 0), (130, size), (255, 255, 255), -1)
    
    elif walls_pattern == "room":
        # Room with walls
        cv2.rectangle(minimap, (40, 40), (160, 160), (200, 200, 200), -1)
        cv2.rectangle(minimap, (40, 40), (160, 160), (255, 255, 255), 3)
        # Door opening
        cv2.rectangle(minimap, (95, 40), (105, 45), (200, 200, 200), -1)
    
    elif walls_pattern == "intersection":
        # Cross intersection
        # Horizontal corridor
        cv2.rectangle(minimap, (0, 90), (size, 110), (200, 200, 200), -1)
        # Vertical corridor
        cv2.rectangle(minimap, (90, 0), (110, size), (200, 200, 200), -1)
        # Walls
        cv2.rectangle(minimap, (0, 85), (size, 90), (255, 255, 255), -1)
        cv2.rectangle(minimap, (0, 110), (size, 115), (255, 255, 255), -1)
        cv2.rectangle(minimap, (85, 0), (90, size), (255, 255, 255), -1)
        cv2.rectangle(minimap, (110, 0), (115, size), (255, 255, 255), -1)
    
    elif walls_pattern == "loop":
        # Circular corridor for loop closure testing
        center = (size // 2, size // 2)
        radius_outer = 80
        radius_inner = 60
        cv2.circle(minimap, center, radius_outer, (255, 255, 255), -1)
        cv2.circle(minimap, center, radius_inner, (20, 20, 20), -1)
    
    # Add player marker (blue cross)
    cx, cy = size // 2, size // 2
    cv2.line(minimap, (cx - 5, cy), (cx + 5, cy), (255, 200, 0), 2)
    cv2.line(minimap, (cx, cy - 5), (cx, cy + 5), (255, 200, 0), 2)
    
    return minimap


def simulate_movement_sequence() -> list:
    """
    Generate a sequence of minimaps simulating player movement.
    
    Returns:
        List of (minimap, movement_description) tuples
    """
    sequences = []
    
    # 1. Start in corridor
    base = create_synthetic_minimap(200, "corridor")
    sequences.append((base.copy(), "Start - Vertical corridor"))
    
    # 2. Move down (shift image up)
    for i in range(3):
        shifted = np.roll(base, -10, axis=0)
        sequences.append((shifted.copy(), f"Moving down ({i+1})"))
    
    # 3. Enter room
    room = create_synthetic_minimap(200, "room")
    sequences.append((room.copy(), "Entered room"))
    
    # 4. Stay in room (minimal movement)
    for i in range(2):
        shifted = np.roll(room, np.random.randint(-3, 3), axis=0)
        shifted = np.roll(shifted, np.random.randint(-3, 3), axis=1)
        sequences.append((shifted.copy(), f"In room ({i+1})"))
    
    # 5. Move to intersection
    intersection = create_synthetic_minimap(200, "intersection")
    sequences.append((intersection.copy(), "Reached intersection"))
    
    # 6. Move right (shift image left)
    for i in range(2):
        shifted = np.roll(intersection, -10, axis=1)
        sequences.append((shifted.copy(), f"Moving right ({i+1})"))
    
    # 7. Return to intersection (loop closure)
    sequences.append((intersection.copy(), "Back to intersection (LOOP CLOSURE)"))
    
    return sequences


def demo_static_minimap():
    """Demo 1: Process a single static minimap."""
    print("\n" + "="*70)
    print("DEMO 1: Static Minimap Processing")
    print("="*70)
    
    # Check for real minimap
    minimap_path = Path("data/screenshots/outputs/diagnostic/minimap_steps/step1_extracted.png")
    
    if minimap_path.exists():
        print(f"Loading real minimap from: {minimap_path}")
        minimap = cv2.imread(str(minimap_path))
    else:
        print("No real minimap found, using synthetic data")
        minimap = create_synthetic_minimap(200, "intersection")
    
    print(f"Minimap size: {minimap.shape[1]}x{minimap.shape[0]}")
    
    # Initialize SLAM
    slam = MinimapSLAM(
        map_size=2048,
        movement_threshold=2.0,
        debug=True
    )
    
    # Process minimap
    print("\nProcessing minimap...")
    skeleton = slam.preprocess_minimap(minimap)
    
    print(f"Skeleton extracted: {skeleton.shape}")
    print(f"Wall pixels: {np.sum(skeleton > 127)}")
    
    # Update SLAM (first frame)
    slam.update(minimap)
    
    # Get stats
    stats = slam.get_stats()
    print("\nSLAM Statistics:")
    for key, value in stats.items():
        print(f"  {key}: {value}")
    
    # Visualize
    visualizer = SLAMVisualizer()
    processing_vis = visualizer.visualize_minimap_processing(minimap, skeleton)
    local_map_vis = visualizer.visualize_local_map(slam, radius=150)
    
    # Show results
    cv2.imshow("Processing", processing_vis)
    cv2.imshow("Local Map", local_map_vis)
    
    print("\nPress any key to continue...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    return slam


def demo_movement_sequence():
    """Demo 2: Simulate player movement through multiple areas."""
    print("\n" + "="*70)
    print("DEMO 2: Movement Sequence with Loop Closure")
    print("="*70)
    
    # Initialize SLAM
    slam = MinimapSLAM(
        map_size=2048,
        movement_threshold=2.0,
        loop_closure_threshold=0.75,
        signature_interval=3,
        debug=True
    )
    
    # Initialize visualizer
    visualizer = SLAMVisualizer(window_name="SLAM Demo", map_view_size=400)
    
    # Generate movement sequence
    sequences = simulate_movement_sequence()
    print(f"\nGenerated {len(sequences)} frames")
    
    # Process each frame
    fps_history = []
    for i, (minimap, description) in enumerate(sequences):
        print(f"\n--- Frame {i+1}/{len(sequences)}: {description} ---")
        
        # Timing
        start_time = time.time()
        
        # Preprocess
        skeleton = slam.preprocess_minimap(minimap)
        
        # Estimate motion (if not first frame)
        dx, dy = 0.0, 0.0
        if slam.prev_skeleton is not None:
            dx, dy, confidence = slam.estimate_motion(skeleton, slam.prev_skeleton)
            print(f"Motion: dx={dx:.2f}, dy={dy:.2f}, confidence={confidence:.3f}")
            visualizer.add_motion_to_history(dx, dy)
        
        # Update SLAM
        slam.update(minimap)
        
        # Add synthetic POIs (simulate detection)
        if "intersection" in description.lower():
            slam.add_poi("npc", (100, 100), confidence=0.9, metadata={"name": "Vendor"})
        if "room" in description.lower():
            slam.add_poi("chest", (120, 80), confidence=0.8)
        
        # Calculate FPS
        elapsed = time.time() - start_time
        fps = 1.0 / elapsed if elapsed > 0 else 0
        fps_history.append(fps)
        avg_fps = np.mean(fps_history[-10:])
        
        # Get stats
        stats = slam.get_stats()
        print(f"World offset: {stats['world_offset']}")
        print(f"Known cells: {stats['known_cells']}")
        print(f"Loop closures: {stats['loop_closures']}")
        
        # Create dashboard
        dashboard = visualizer.create_dashboard(
            slam=slam,
            current_minimap=minimap,
            skeleton=skeleton,
            dx=dx,
            dy=dy,
            fps=avg_fps
        )
        
        # Show
        key = visualizer.show(dashboard, wait_key=500)  # 500ms per frame
        
        if key == ord('q'):
            print("\nQuitting...")
            break
        elif key == ord('s'):
            # Save map
            slam.save_map()
            print("\nMap saved!")
    
    # Final stats
    print("\n" + "="*70)
    print("DEMO COMPLETE")
    print("="*70)
    final_stats = slam.get_stats()
    print("\nFinal Statistics:")
    for key, value in final_stats.items():
        print(f"  {key}: {value}")
    
    print(f"\nAverage FPS: {np.mean(fps_history):.1f}")
    
    # Show final global map
    print("\nShowing final global map...")
    global_map_vis = visualizer.visualize_global_map(slam, view_size=800)
    cv2.imshow("Final Global Map", global_map_vis)
    
    print("\nPress any key to exit...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    return slam


def demo_poi_tracking():
    """Demo 3: POI tracking and persistence."""
    print("\n" + "="*70)
    print("DEMO 3: POI Tracking")
    print("="*70)
    
    # Initialize SLAM
    slam = MinimapSLAM(map_size=2048, debug=True)
    
    # Add test minimap
    minimap = create_synthetic_minimap(200, "intersection")
    slam.update(minimap)
    
    # Add various POIs
    print("\nAdding POIs...")
    slam.add_poi("npc", (100, 100), confidence=0.9, metadata={"name": "Akara"})
    slam.add_poi("waypoint", (100, 120), confidence=1.0, metadata={"zone": "Rogue Encampment"})
    slam.add_poi("exit", (150, 100), confidence=0.8, metadata={"destination": "Blood Moor"})
    slam.add_poi("chest", (80, 80), confidence=0.7)
    slam.add_poi("shrine", (120, 120), confidence=0.75)
    
    # List POIs
    print(f"\nTotal POIs: {len(slam.current_level.pois)}")
    for poi in slam.current_level.pois:
        print(f"  {poi.poi_type:10s} @ {poi.pos} (confidence: {poi.confidence:.2f})")
    
    # Visualize
    visualizer = SLAMVisualizer()
    local_map = visualizer.visualize_local_map(slam, radius=150)
    
    cv2.imshow("POI Map", local_map)
    
    # Save and load test
    print("\nSaving map...")
    slam.save_map("test_poi_map.npz")
    
    print("Loading map...")
    slam_loaded = MinimapSLAM(map_size=2048, debug=True)
    slam_loaded.load_map("test_poi_map.npz")
    
    print(f"Loaded POIs: {len(slam_loaded.current_level.pois)}")
    
    print("\nPress any key to continue...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def demo_multi_level():
    """Demo 4: Multi-level map support."""
    print("\n" + "="*70)
    print("DEMO 4: Multi-Level Support")
    print("="*70)
    
    # Initialize SLAM
    slam = MinimapSLAM(map_size=2048, debug=True)
    
    # Level 0: Rogue Encampment
    print("\n--- Level 0: Rogue Encampment ---")
    minimap0 = create_synthetic_minimap(200, "intersection")
    slam.update(minimap0)
    slam.add_poi("stairs", (150, 100), metadata={"destination": "Blood Moor"})
    
    print(f"Current level: {slam.current_level_id}")
    print(f"Levels: {list(slam.levels.keys())}")
    
    # Switch to Level 1: Blood Moor
    print("\n--- Switching to Level 1 ---")
    slam.switch_level("blood_moor", transition_type="stairs")
    
    minimap1 = create_synthetic_minimap(200, "corridor")
    slam.update(minimap1)
    slam.add_poi("npc", (100, 80), metadata={"name": "Flavie"})
    
    print(f"Current level: {slam.current_level_id}")
    print(f"Levels: {list(slam.levels.keys())}")
    
    # Switch to Level 2: Cold Plains
    print("\n--- Switching to Level 2 ---")
    slam.switch_level("cold_plains", transition_type="exit")
    
    minimap2 = create_synthetic_minimap(200, "room")
    slam.update(minimap2)
    slam.add_poi("waypoint", (100, 100))
    
    print(f"Current level: {slam.current_level_id}")
    print(f"Levels: {list(slam.levels.keys())}")
    
    # Show level transitions
    print("\n--- Level Transitions ---")
    for level_id, level in slam.levels.items():
        print(f"{level_id}:")
        for target, pos in level.transitions.items():
            print(f"  -> {target} @ {pos}")
    
    # Visualize each level
    visualizer = SLAMVisualizer()
    
    for level_id in slam.levels.keys():
        slam.current_level_id = level_id
        local_map = visualizer.visualize_local_map(slam, radius=150)
        cv2.imshow(f"Level: {level_id}", local_map)
    
    print("\nPress any key to continue...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def main():
    """Run all demos."""
    print("\n" + "="*70)
    print("MINIMAP SLAM DEMONSTRATION")
    print("="*70)
    print("\nThis demo shows a 2D SLAM system for Diablo II using ONLY minimap.")
    print("No game coordinates, memory access, or physics engine.")
    print("\nKey concepts:")
    print("  - Player position is FIXED")
    print("  - World moves around player")
    print("  - Motion estimated from visual changes")
    print("  - Loop closure corrects drift")
    print("  - Multi-level support for stairs/portals")
    
    print("\nAvailable demos:")
    print("  1. Static minimap processing")
    print("  2. Movement sequence with loop closure")
    print("  3. POI tracking and persistence")
    print("  4. Multi-level support")
    print("  5. Run all demos")
    print("  q. Quit")
    
    choice = input("\nSelect demo (1-5, q): ").strip()
    
    if choice == '1':
        demo_static_minimap()
    elif choice == '2':
        demo_movement_sequence()
    elif choice == '3':
        demo_poi_tracking()
    elif choice == '4':
        demo_multi_level()
    elif choice == '5':
        demo_static_minimap()
        demo_movement_sequence()
        demo_poi_tracking()
        demo_multi_level()
    elif choice.lower() == 'q':
        print("Goodbye!")
        return
    else:
        print("Invalid choice")
        return
    
    print("\n" + "="*70)
    print("Demo complete!")
    print("="*70)


if __name__ == "__main__":
    main()
