"""
Example: Extract minimap POIs and build navigation graph from live game.

This script demonstrates:
1. Extracting minimap region from captured frame
2. Detecting POIs on minimap (waypoints, exits, monsters)
3. Registering them in the world map
4. Building navigation graph
"""

import cv2
import numpy as np
from pathlib import Path
from src.diabot.vision.screen_regions import UI_REGIONS
from src.diabot.navigation import Navigator, MinimapPOIDetector


def extract_minimap(frame: np.ndarray) -> np.ndarray:
    """Extract minimap region from frame."""
    minimap_region = UI_REGIONS.get('minimap_ui')
    if minimap_region:
        minimap = minimap_region.extract_from_frame(frame)
        return minimap
    return None


def demo_extract_and_analyze_minimap():
    """Demo extracting minimap and analyzing POIs."""
    
    print("="*80)
    print("MINIMAP POI EXTRACTION DEMO")
    print("="*80)
    
    # Load live capture
    live_capture_path = Path("data/screenshots/outputs/live_capture/live_capture_raw.jpg")
    if not live_capture_path.exists():
        print(f"❌ No live capture found at {live_capture_path}")
        print("   Run: python test_live_capture.py (single capture mode)")
        return
    
    frame = cv2.imread(str(live_capture_path))
    if frame is None:
        print(f"❌ Failed to load frame")
        return
    
    print(f"\n✓ Loaded frame: {frame.shape}")
    
    # Extract minimap
    minimap = extract_minimap(frame)
    if minimap is None or minimap.size == 0:
        print("❌ Failed to extract minimap region")
        return
    
    print(f"✓ Extracted minimap: {minimap.shape}")
    
    # Save minimap for inspection
    minimap_output = Path("data/screenshots/outputs/live_capture/minimap_extracted.png")
    minimap_output.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(minimap_output), minimap)
    print(f"✓ Saved minimap: {minimap_output}")
    
    # Detect POIs on minimap
    detector = MinimapPOIDetector(debug=True)
    pois = detector.detect(minimap)
    
    print(f"\n✓ Detected {len(pois)} POIs on minimap:")
    for poi in pois:
        print(f"  - {poi.poi_type}: {poi.position} (confidence: {poi.confidence:.2f})")
    
    # Initialize navigator and register the zone
    nav = Navigator(debug=True)
    
    # Register zone (assuming ROGUE ENCAMPMENT from UI analysis)
    zone_name = "ROGUE ENCAMPMENT"
    act = "a1"
    
    nav.visit_zone(zone_name, act)
    
    # Register detected POIs
    for poi in pois:
        if poi.poi_type == 'waypoint':
            nav.add_waypoint(zone_name, poi.position)
            print(f"  ✓ Registered waypoint at {poi.position}")
        elif poi.poi_type == 'exit':
            # Note: would need to know target zone
            nav.add_exit(zone_name, "UNKNOWN", poi.position)
            print(f"  ✓ Registered exit at {poi.position}")
    
    # Save maps
    nav.save_maps()
    print(f"\n✓ Saved navigation data")
    
    # Print status
    nav.print_status()


if __name__ == '__main__':
    demo_extract_and_analyze_minimap()
