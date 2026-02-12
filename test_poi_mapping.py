"""
Test POI detection and mapping from YOLO.

Demonstrates how YOLO detections are added to the accumulated map.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent / "src"))

import cv2
import numpy as np

from diabot.navigation.minimap_processor import MinimapProcessor
from diabot.navigation.player_locator import PlayerLocator
from diabot.navigation.map_accumulator import MapAccumulator, MapPOI
from diabot.vision.yolo_detector import YOLODetector


def test_poi_mapping():
    """Test POI detection and map integration."""
    
    print("="*70)
    print("POI DETECTION AND MAPPING TEST")
    print("="*70)
    
    # 1. Initialize components
    print("\n[1] Initializing components...")
    
    # Use trained YOLO model (fallback to generic if not found)
    yolo_model = Path("runs/detect/runs/train/diablo-yolo3/weights/best.pt")
    if not yolo_model.exists():
        print(f"⚠ Trained model not found: {yolo_model}")
        print("  Trying generic YOLO model as fallback...")
        yolo_model = Path("yolo11n.pt")
        if not yolo_model.exists():
            print(f"⚠ Generic model not found either: {yolo_model}")
            print("This test will demonstrate POI system without real detections")
            yolo_detector = None
        else:
            yolo_detector = YOLODetector(model_path=yolo_model, confidence_threshold=0.35, debug=True)
    else:
        print(f"✓ Using trained model: {yolo_model}")
        yolo_detector = YOLODetector(model_path=yolo_model, confidence_threshold=0.35, debug=True)
    
    processor = MinimapProcessor(grid_size=64, wall_threshold=49, debug=True)
    locator = PlayerLocator(debug=True)
    accumulator = MapAccumulator(map_size=2048, save_dir=Path("data/maps"), debug=True)
    
    # 2. Load test images
    print("\n[2] Loading test images...")
    
    minimap_path = "data/screenshots/outputs/diagnostic/minimap_steps/step1_extracted.png"
    game_frame_path = "data/screenshots/inputs/game_screenshot.png"
    
    if not Path(minimap_path).exists():
        print(f"❌ Minimap not found: {minimap_path}")
        return
    
    minimap = cv2.imread(minimap_path)
    print(f"✓ Loaded minimap: {minimap.shape}")
    
    # Try to load game frame for YOLO
    if Path(game_frame_path).exists():
        game_frame = cv2.imread(game_frame_path)
        print(f"✓ Loaded game frame: {game_frame.shape}")
    else:
        game_frame = None
        print(f"⚠ Game frame not found: {game_frame_path}")
    
    # 3. Process minimap
    print("\n[3] Processing minimap...")
    minimap_grid = processor.process(minimap)
    player_pos = locator.detect_player_cross(minimap)
    accumulator.update(minimap_grid, player_offset=(0, 0))
    print(f"✓ Map updated: {len(accumulator.cells)} cells")
    
    # 4. Detect POIs with YOLO
    print("\n[4] Detecting POIs with YOLO...")
    
    if yolo_detector and game_frame is not None:
        detections = yolo_detector.detect(game_frame)
        print(f"✓ YOLO detected {len(detections)} objects")
        
        # Screen-to-map calibration
        # Game viewport shows ~64 cells (same as minimap grid)
        # Screen 1922x1114 / 64 cells = ~24 px/cell
        SCREEN_PIXELS_PER_CELL = 24.0  # Calibrated for minimap grid
        
        for det in detections:
            print(f"  - {det.class_name} ({det.confidence:.2f}) @ {det.center}")
            
            # Map to POI type
            poi_type_map = {
                # Generic
                "person": "npc",
                "npc": "npc",
                # D2 NPCs
                "akara": "npc",
                "kashya": "npc",
                "warriv": "npc",
                # Objects
                "waypoint": "waypoint",
                "stash": "chest",
                "chest": "chest",
                "shrine": "shrine",
                "quest": "quest",
                # Exits
                "door": "exit",
                "portal": "exit",
            }
            
            poi_type = poi_type_map.get(det.class_name.lower(), "unknown")
            
            if poi_type != "unknown":
                # Convert screen coords to map coords
                cx, cy = det.center
                
                # Player is at center of screen
                player_screen_x = game_frame.shape[1] // 2
                player_screen_y = game_frame.shape[0] // 2
                
                # Offset in pixels from player
                screen_dx = cx - player_screen_x
                screen_dy = cy - player_screen_y
                
                # Convert to map cells
                offset_x = int(screen_dx / SCREEN_PIXELS_PER_CELL)
                offset_y = int(screen_dy / SCREEN_PIXELS_PER_CELL)
                
                # Global position
                global_x = accumulator.player_world_pos[0] + offset_x
                global_y = accumulator.player_world_pos[1] + offset_y
                
                accumulator.add_poi(
                    poi_type=poi_type,
                    position=(global_x, global_y),
                    label=det.class_name,
                    confidence=det.confidence
                )
    else:
        print("⚠ No YOLO detector or game frame, adding simulated POIs...")
        
        # Add simulated POIs for demonstration
        simulated_pois = [
            ("npc", (1030, 1020), "Akara", 0.95),
            ("npc", (1040, 1015), "Kashya", 0.92),
            ("waypoint", (1010, 1030), "Waypoint", 0.98),
            ("exit", (990, 1055), "Blood Moor Exit", 0.88),
            ("chest", (1050, 1000), "Chest", 0.75),
        ]
        
        for poi_type, pos, label, conf in simulated_pois:
            accumulator.add_poi(
                poi_type=poi_type,
                position=pos,
                label=label,
                confidence=conf
            )
        
        print(f"✓ Added {len(simulated_pois)} simulated POIs")
    
    # 5. Display POI summary
    print("\n[5] POI Summary:")
    print(f"Total POIs: {len(accumulator.pois)}")
    
    poi_types = {}
    for poi in accumulator.pois:
        poi_types[poi.poi_type] = poi_types.get(poi.poi_type, 0) + 1
    
    for poi_type, count in sorted(poi_types.items()):
        print(f"  {poi_type}: {count}")
    
    # List all POIs
    print("\nDetailed POI list:")
    for i, poi in enumerate(accumulator.pois, 1):
        print(f"  #{i}: {poi.label} ({poi.poi_type}) @ {poi.position} - conf={poi.confidence:.2f}")
    
    # 6. Visualize map with POIs
    print("\n[6] Generating visualization...")
    map_vis = accumulator.visualize(scale=4)
    
    # Add legend
    legend_y = 20
    legend_items = [
        ("NPC", (255, 255, 0)),
        ("Exit", (0, 165, 255)),
        ("Waypoint", (255, 0, 255)),
        ("Chest", (0, 215, 255)),
        ("Shrine", (203, 192, 255)),
    ]
    
    for label, color in legend_items:
        cv2.circle(map_vis, (20, legend_y), 5, color, -1)
        cv2.putText(map_vis, label, (30, legend_y + 5),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        legend_y += 25
    
    output_path = Path("data/screenshots/outputs/test_poi_mapping.png")
    cv2.imwrite(str(output_path), map_vis)
    print(f"✓ Saved visualization: {output_path}")
    
    # 7. Save map with POIs
    print("\n[7] Saving map...")
    accumulator.save_map("TEST_POI_ZONE")
    
    # 8. Test map clearing
    print("\n[8] Testing map clear functionality...")
    print(f"Before clear: {len(accumulator.cells)} cells, {len(accumulator.pois)} POIs")
    
    # Clear map but keep POIs
    accumulator.clear(keep_pois=True)
    print(f"After clear (keep_pois=True): {len(accumulator.cells)} cells, {len(accumulator.pois)} POIs")
    
    # Re-update map
    accumulator.update(minimap_grid, player_offset=(0, 0))
    print(f"After re-update: {len(accumulator.cells)} cells, {len(accumulator.pois)} POIs")
    
    # Clear everything
    accumulator.clear(keep_pois=False)
    print(f"After clear (keep_pois=False): {len(accumulator.cells)} cells, {len(accumulator.pois)} POIs")
    
    print("\n" + "="*70)
    print("✓ TEST COMPLETE")
    print("="*70)
    print("\nFeatures demonstrated:")
    print("  ✓ POI detection from YOLO")
    print("  ✓ POI addition to accumulated map")
    print("  ✓ POI visualization (color-coded markers)")
    print("  ✓ POI persistence in JSON metadata")
    print("  ✓ Map clearing with/without POI retention")
    print("="*70)
    
    # Display
    cv2.imshow("Map with POIs", map_vis)
    print("\nPress any key to exit...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    test_poi_mapping()
