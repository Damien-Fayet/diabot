"""
Interactive tool to annotate POIs on static map.

Click on the map to mark POI locations:
- Left click: Add POI
- Right click: Remove nearest POI
- Keys 1-5: Change POI type
  1 = Exit (red)
  2 = Waypoint (green)
  3 = Stash (orange)
  4 = NPC (blue)
  5 = Quest (yellow)
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent / "src"))

import cv2
import numpy as np
import json


class MapAnnotator:
    """Interactive map annotation tool."""
    
    def __init__(self, map_path: Path, zone_name: str):
        """Initialize annotator."""
        self.map_path = Path(map_path)
        self.zone_name = zone_name
        self.annotations_file = self.map_path.parent / f"{self.map_path.stem}_annotations.json"
        
        # Load map
        self.original_map = cv2.imread(str(map_path))
        if self.original_map is None:
            raise ValueError(f"Could not load map: {map_path}")
        
        self.display_map = self.original_map.copy()
        self.window_name = f"Annotate: {zone_name}"
        
        # POI types and colors (BGR)
        self.poi_types = {
            1: ("exit", (0, 0, 255)),      # Red
            2: ("waypoint", (0, 255, 0)),  # Green
            3: ("stash", (0, 165, 255)),   # Orange
            4: ("npc", (255, 0, 0)),       # Blue
            5: ("quest", (0, 255, 255)),   # Yellow
        }
        
        self.current_type = 1  # Default to exit
        
        # Annotations: list of {type, position, name}
        self.annotations = []
        
        # Load existing annotations if any
        self._load_annotations()
        
        # Mouse callback
        cv2.namedWindow(self.window_name)
        cv2.setMouseCallback(self.window_name, self._mouse_callback)
        
        print("\n" + "="*70)
        print(f"🎨 MAP ANNOTATION TOOL: {zone_name}")
        print("="*70)
        print("\nControls:")
        print("  Left Click  : Add POI")
        print("  Right Click : Remove nearest POI")
        print("  Keys 1-5    : Change POI type")
        print("    1 = Exit (red)")
        print("    2 = Waypoint (green)")
        print("    3 = Stash (orange)")
        print("    4 = NPC (blue)")
        print("    5 = Quest (yellow)")
        print("  's' : Save annotations")
        print("  'c' : Clear all")
        print("  'q' : Quit")
        print("="*70 + "\n")
        
        self._update_display()
    
    def _load_annotations(self):
        """Load existing annotations from JSON."""
        if self.annotations_file.exists():
            try:
                with open(self.annotations_file, 'r') as f:
                    data = json.load(f)
                    self.annotations = data.get('pois', [])
                print(f"✓ Loaded {len(self.annotations)} existing annotations")
            except Exception as e:
                print(f"⚠️  Could not load annotations: {e}")
    
    def _save_annotations(self):
        """Save annotations to JSON."""
        data = {
            'zone': self.zone_name,
            'map_file': self.map_path.name,
            'map_size': {
                'width': self.original_map.shape[1],
                'height': self.original_map.shape[0]
            },
            'pois': self.annotations
        }
        
        with open(self.annotations_file, 'w') as f:
            json.dump(data, f, indent=2)
        
        print(f"✓ Saved {len(self.annotations)} annotations to {self.annotations_file.name}")
    
    def _mouse_callback(self, event, x, y, flags, param):
        """Handle mouse events."""
        if event == cv2.EVENT_LBUTTONDOWN:
            # Add POI
            poi_type_name, _ = self.poi_types[self.current_type]
            
            # Generate default name
            count = sum(1 for p in self.annotations if p['type'] == poi_type_name)
            default_name = f"{poi_type_name}_{count + 1}"
            
            poi = {
                'type': poi_type_name,
                'position': [x, y],
                'name': default_name
            }
            
            self.annotations.append(poi)
            print(f"➕ Added {poi_type_name} at ({x}, {y})")
            self._update_display()
        
        elif event == cv2.EVENT_RBUTTONDOWN:
            # Remove nearest POI
            if self.annotations:
                # Find nearest
                distances = [np.sqrt((p['position'][0] - x)**2 + (p['position'][1] - y)**2)
                           for p in self.annotations]
                nearest_idx = np.argmin(distances)
                
                if distances[nearest_idx] < 20:  # Within 20 pixels
                    removed = self.annotations.pop(nearest_idx)
                    print(f"➖ Removed {removed['type']} at {removed['position']}")
                    self._update_display()
    
    def _update_display(self):
        """Redraw map with annotations."""
        self.display_map = self.original_map.copy()
        
        # Draw all POIs
        for poi in self.annotations:
            poi_type = poi['type']
            x, y = poi['position']
            
            # Find color for this type
            color = None
            for key, (type_name, type_color) in self.poi_types.items():
                if type_name == poi_type:
                    color = type_color
                    break
            
            if color is None:
                color = (128, 128, 128)  # Gray for unknown
            
            # Draw marker
            cv2.circle(self.display_map, (x, y), 8, color, -1)
            cv2.circle(self.display_map, (x, y), 8, (255, 255, 255), 2)
            
            # Draw name
            cv2.putText(self.display_map, poi['name'], (x + 12, y - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
        
        # Draw current mode indicator
        type_name, type_color = self.poi_types[self.current_type]
        cv2.rectangle(self.display_map, (10, 10), (200, 40), (0, 0, 0), -1)
        cv2.putText(self.display_map, f"Mode: {type_name.upper()}", (15, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, type_color, 2)
        
        # Show count
        cv2.putText(self.display_map, f"POIs: {len(self.annotations)}", 
                   (self.display_map.shape[1] - 120, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        cv2.imshow(self.window_name, self.display_map)
    
    def run(self):
        """Run annotation loop."""
        while True:
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q'):
                # Quit
                break
            
            elif key == ord('s'):
                # Save
                self._save_annotations()
            
            elif key == ord('c'):
                # Clear all
                if self.annotations:
                    print(f"🗑️  Cleared {len(self.annotations)} annotations")
                    self.annotations = []
                    self._update_display()
            
            elif ord('1') <= key <= ord('5'):
                # Change POI type
                self.current_type = key - ord('0')
                type_name, _ = self.poi_types[self.current_type]
                print(f"🎨 Mode: {type_name.upper()}")
                self._update_display()
        
        cv2.destroyAllWindows()
        
        # Auto-save on exit
        if self.annotations:
            self._save_annotations()


def auto_preannotate(map_path: Path, zone_name: str):
    """
    Automatically detect colored markers on the map and create pre-annotations.
    
    Looks for:
    - Red points (exits)
    - Green points (waypoint)
    - Orange points (stash)
    - Blue points (NPCs)
    """
    map_img = cv2.imread(str(map_path))
    if map_img is None:
        print(f"❌ Could not load map: {map_path}")
        return None
    
    print("\n" + "="*70)
    print(f"🔍 AUTO-DETECTING POIs ON MAP")
    print("="*70 + "\n")
    
    hsv = cv2.cvtColor(map_img, cv2.COLOR_BGR2HSV)
    
    annotations = []
    
    # Define color ranges in HSV
    color_ranges = {
        'exit': {
            'name': 'exit',
            'lower': np.array([0, 150, 150]),    # Red
            'upper': np.array([10, 255, 255]),
        },
        'waypoint': {
            'name': 'waypoint',
            'lower': np.array([50, 150, 150]),   # Green
            'upper': np.array([70, 255, 255]),
        },
        'stash': {
            'name': 'stash',
            'lower': np.array([10, 150, 150]),   # Orange
            'upper': np.array([25, 255, 255]),
        },
        'npc': {
            'name': 'npc',
            'lower': np.array([100, 150, 150]),  # Blue
            'upper': np.array([130, 255, 255]),
        },
    }
    
    for poi_type, ranges in color_ranges.items():
        # Create mask for this color
        mask = cv2.inRange(hsv, ranges['lower'], ranges['upper'])
        
        # Find contours
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        count = 0
        for contour in contours:
            area = cv2.contourArea(contour)
            
            # Filter by area (looking for small markers)
            if 10 < area < 500:
                # Get center
                M = cv2.moments(contour)
                if M['m00'] > 0:
                    cx = int(M['m10'] / M['m00'])
                    cy = int(M['m01'] / M['m00'])
                    
                    annotations.append({
                        'type': ranges['name'],
                        'position': [cx, cy],
                        'name': f"{ranges['name']}_{count + 1}"
                    })
                    
                    count += 1
        
        print(f"  ✓ Found {count} {poi_type} marker(s)")
    
    print(f"\n✓ Total: {len(annotations)} POIs detected\n")
    
    return annotations


def main():
    """Main entry point."""
    # Path to static map
    map_path = Path("data/maps/minimap_images/A1-ROGUE ENCAMPMENT.png")
    zone_name = "ROGUE ENCAMPMENT"
    
    if not map_path.exists():
        print(f"❌ Map not found: {map_path}")
        print("Place your static map at: data/maps/minimap_images/A1-ROGUE ENCAMPMENT.png")
        return
    
    # Try auto-detection first
    print("Attempting automatic POI detection...")
    auto_annotations = auto_preannotate(map_path, zone_name)
    
    if auto_annotations:
        # Save pre-annotations
        annotations_file = map_path.parent / f"{map_path.stem}_annotations.json"
        data = {
            'zone': zone_name,
            'map_file': map_path.name,
            'map_size': {
                'width': cv2.imread(str(map_path)).shape[1],
                'height': cv2.imread(str(map_path)).shape[0]
            },
            'pois': auto_annotations
        }
        
        with open(annotations_file, 'w') as f:
            json.dump(data, f, indent=2)
        
        print(f"💾 Pre-annotations saved to: {annotations_file.name}")
    
    # Launch interactive annotator
    print("\nLaunching interactive annotator...")
    print("(You can refine the auto-detected POIs or add new ones)")
    
    annotator = MapAnnotator(map_path, zone_name)
    annotator.run()
    
    print("\n" + "="*70)
    print("✅ ANNOTATION COMPLETE")
    print("="*70)
    print(f"\nAnnotations saved to: {annotator.annotations_file}")
    print(f"Use these annotations for navigation in the bot.")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()
