"""
Visualize map with POI markers in real-time.

Shows accumulated map with detected POIs color-coded by type.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent / "src"))

import cv2
import json


def visualize_saved_map(map_json_path: str):
    """
    Load and visualize a saved map with POIs.
    
    Args:
        map_json_path: Path to map metadata JSON file
    """
    json_path = Path(map_json_path)
    
    if not json_path.exists():
        print(f"❌ Map not found: {json_path}")
        return
    
    # Load metadata
    with open(json_path, 'r') as f:
        metadata = json.load(f)
    
    zone = metadata.get("zone", "UNKNOWN")
    cells = metadata.get("cell_count", 0)
    pois = metadata.get("pois", [])
    
    print("="*70)
    print(f"MAP VISUALIZATION: {zone}")
    print("="*70)
    print(f"Cells: {cells}")
    print(f"POIs: {len(pois)}")
    
    # Load PNG
    png_path = json_path.with_suffix(".png").name.replace("_metadata", "_map")
    png_path = json_path.parent / png_path
    
    if not png_path.exists():
        print(f"❌ Map image not found: {png_path}")
        return
    
    img = cv2.imread(str(png_path))
    print(f"Image: {img.shape}")
    
    # Add POI info overlay
    y_offset = 30
    for poi in pois:
        poi_type = poi.get("type", "unknown")
        label = poi.get("label", "")
        confidence = poi.get("confidence", 0.0)
        
        text = f"{label} ({poi_type}) - {confidence:.0%}"
        
        # Color by type
        colors = {
            "npc": (255, 255, 0),
            "exit": (0, 165, 255),
            "waypoint": (255, 0, 255),
            "chest": (0, 215, 255),
            "shrine": (203, 192, 255),
            "quest": (0, 0, 255),
        }
        color = colors.get(poi_type, (255, 255, 255))
        
        cv2.putText(img, text, (10, y_offset),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        y_offset += 25
    
    # Add title
    cv2.putText(img, f"{zone} - {len(pois)} POIs", (10, img.shape[0] - 10),
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    # Display
    print("\n" + "="*70)
    print("POI List:")
    for i, poi in enumerate(pois, 1):
        print(f"  {i}. {poi['label']} ({poi['type']}) @ {poi['position']} - {poi['confidence']:.0%}")
    print("="*70)
    
    cv2.imshow("Map Visualization", img)
    print("\nPress any key to exit...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def main():
    """Find and display most recent map."""
    maps_dir = Path("data/maps")
    
    if not maps_dir.exists():
        print("No maps directory found")
        return
    
    # Find most recent JSON
    json_files = sorted(maps_dir.glob("*_metadata.json"), key=lambda p: p.stat().st_mtime, reverse=True)
    
    if not json_files:
        print("No saved maps found")
        print("\nRun one of these first:")
        print("  python test_map_navigation.py")
        print("  python test_poi_mapping.py")
        return
    
    # Show most recent
    most_recent = json_files[0]
    print(f"Loading most recent map: {most_recent.name}")
    visualize_saved_map(str(most_recent))


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        # Specific map provided
        visualize_saved_map(sys.argv[1])
    else:
        # Auto-find most recent
        main()
