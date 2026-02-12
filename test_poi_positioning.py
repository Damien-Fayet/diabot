"""
Test POI positioning accuracy on real game frame.
Verify that POIs are placed correctly relative to detected objects.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent / "src"))

import cv2
import numpy as np

from diabot.vision.yolo_detector import YOLODetector
from diabot.navigation.minimap_processor import MinimapProcessor
from diabot.navigation.player_locator import PlayerLocator
from diabot.navigation.map_accumulator import MapAccumulator


def test_poi_positioning():
    """Test POI positioning with a real game frame."""
    
    print("\n" + "="*70)
    print("🎯 POI POSITIONING TEST")
    print("="*70 + "\n")
    
    # Look for game screenshot
    screenshot_path = Path("data/screenshots/inputs/game_001.png")
    if not screenshot_path.exists():
        print("❌ No game screenshot found at:", screenshot_path)
        print("   Please run bot with --debug first to capture a frame.")
        return
    
    print(f"📸 Loading screenshot: {screenshot_path}")
    frame = cv2.imread(str(screenshot_path))
    print(f"   ✓ Frame size: {frame.shape}\n")
    
    # Initialize systems
    print("📦 Initializing systems...")
    
    # Use trained YOLO model
    yolo_model = Path("runs/detect/runs/train/diablo-yolo3/weights/best.pt")
    if not yolo_model.exists():
        print(f"   ❌ Trained model not found: {yolo_model}")
        print("   Run training first: python scripts/train_yolo.py")
        return
    
    yolo = YOLODetector(model_path=yolo_model, confidence_threshold=0.3, debug=False)
    processor = MinimapProcessor(grid_size=64, wall_threshold=49, debug=False)
    locator = PlayerLocator(debug=False)
    accumulator = MapAccumulator(map_size=2048, save_dir=Path("data/maps"), debug=False)
    print("   ✓ Ready\n")
    
    # Detect objects with YOLO
    print("🔍 Running YOLO detection...")
    detections = yolo.detect(frame)
    print(f"   ✓ Found {len(detections)} objects\n")
    
    # Display detections
    if detections:
        print("   Detected objects:")
        for det in detections:
            x, y, w, h = det.bbox
            cx, cy = x + w//2, y + h//2
            print(f"     • {det.class_name} @ ({cx}, {cy}) - {det.confidence:.0%}")
    
    # Extract and process minimap
    print("\n📍 Processing minimap...")
    from diabot.vision.screen_regions import UI_REGIONS
    minimap_region = UI_REGIONS.get("minimap_ui")
    
    if not minimap_region:
        print("   ❌ Minimap region not configured")
        return
    
    frame_h, frame_w = frame.shape[:2]
    x, y, w, h = minimap_region.get_bounds(frame_h, frame_w)
    minimap = frame[y:y+h, x:x+w].copy()
    
    # Process minimap
    grid = processor.process(minimap)
    player_pos = locator.detect_player_cross(minimap)
    
    if player_pos is None:
        print("   ❌ Could not detect player on minimap")
        return
    
    print(f"   ✓ Player at: {player_pos} (minimap coords)")
    
    # Update map
    accumulator.update(grid, player_offset=(0, 0))
    print(f"   ✓ Map updated: {len(accumulator.cells)} cells\n")
    
    # Convert detections to POIs
    print("🗺️  Converting detections to POIs...")
    
    poi_type_map = {
        # Generic
        "npc": "npc",
        "person": "npc",
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
        "exit": "exit",
        "portal": "exit",
        "door": "exit",
    }
    
    poi_count = 0
    for det in detections:
        x, y, w, h = det.bbox
        cx, cy = x + w//2, y + h//2
        
        # Determine POI type
        poi_type = None
        for key, value in poi_type_map.items():
            if key.lower() in det.class_name.lower():
                poi_type = value
                break
        
        if poi_type:
            # Convert screen coords to map coords
            # Use improved conversion: 2 pixels per map cell
            screen_scale = 2.0
            offset_x = int((cx - frame_w//2) / screen_scale)
            offset_y = int((cy - frame_h//2) / screen_scale)
            
            global_x = accumulator.player_world_pos[0] + offset_x
            global_y = accumulator.player_world_pos[1] + offset_y
            
            accumulator.add_poi(
                poi_type=poi_type,
                position=(global_x, global_y),
                label=det.class_name,
                confidence=det.confidence
            )
            
            print(f"   ✓ {det.class_name}: screen({cx},{cy}) → offset({offset_x},{offset_y}) → global({global_x},{global_y})")
            poi_count += 1
    
    print(f"\n   Total POIs added: {poi_count}\n")
    
    # Visualize
    print("🎨 Creating visualizations...\n")
    
    # 1. Frame with YOLO boxes
    frame_vis = frame.copy()
    for det in detections:
        x, y, w, h = det.bbox
        cv2.rectangle(frame_vis, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(frame_vis, det.class_name, (x, y-5),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    # 2. Map with POIs
    map_vis = accumulator.visualize(scale=4)
    
    # Add comparison info
    info_y = 30
    info_lines = [
        f"Detections: {len(detections)}",
        f"POIs: {len(accumulator.pois)}",
        f"Player: {accumulator.player_world_pos}",
        f"Screen scale: 2.0 px/cell",
    ]
    
    for line in info_lines:
        cv2.putText(map_vis, line, (10, info_y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        info_y += 25
    
    # Save outputs
    output_dir = Path("data/screenshots/outputs")
    frame_path = output_dir / "poi_positioning_frame.png"
    map_path = output_dir / "poi_positioning_map.png"
    
    cv2.imwrite(str(frame_path), frame_vis)
    cv2.imwrite(str(map_path), map_vis)
    
    print(f"   ✓ Saved frame: {frame_path}")
    print(f"   ✓ Saved map: {map_path}\n")
    
    # Display side by side
    # Resize frame to match map height for comparison
    map_h = map_vis.shape[0]
    scale_factor = map_h / frame_vis.shape[0]
    frame_resized = cv2.resize(frame_vis, None, fx=scale_factor, fy=scale_factor)
    
    # Combine horizontally
    combined = np.hstack([frame_resized, map_vis])
    
    combined_path = output_dir / "poi_positioning_comparison.png"
    cv2.imwrite(str(combined_path), combined)
    print(f"   ✓ Saved comparison: {combined_path}\n")
    
    print("="*70)
    print("✅ TEST COMPLETE")
    print("="*70)
    print("\nCheck the output images to verify POI positioning:")
    print("  1. Frame with YOLO detections (green boxes)")
    print("  2. Map with POIs (color-coded dots)")
    print("  3. Side-by-side comparison")
    print("\nPOI positions should align with detected objects.")
    print("="*70 + "\n")
    
    # Display
    cv2.imshow("POI Positioning Test", combined)
    print("Press any key to exit...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    test_poi_positioning()
