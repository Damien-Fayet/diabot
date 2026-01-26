"""
Navigation & World Map System for Diablo 2 Bot

This system manages:
1. World map data (zones, POIs, connections)
2. Minimap POI detection (waypoints, exits, monsters)
3. Pathfinding and route planning
4. Persistent storage of offline maps

STRUCTURE:
----------

### world_map.py
- POI: Point Of Interest data class
- ZoneMap: Represents a zone with its POIs and connections
- WorldMapManager: Manages all zones, loads/saves from JSON

### minimap_detector.py
- MinimapPOI: Detected POI on minimap
- MinimapPOIDetector: Detects POIs using color-based analysis
  - Waypoints (blue)
  - Player position (yellow)
  - Monsters (red)
  - Exits (bright red/orange)

### navigator.py
- NavigationPath: Represents a path through zones
- Navigator: Main navigation system
  - Visit/explore zones
  - Register waypoints and exits
  - Find paths using BFS
  - Plan optimal routes

DATA STORAGE:
-------------

Maps are stored in: data/maps/

- zones_maps.json: Main map database
  {
    "version": "1.0",
    "last_updated": "2026-01-24T...",
    "zones": [
      {
        "zone_name": "ROGUE ENCAMPMENT",
        "act": "a1",
        "pois": [
          {
            "name": "Waypoint",
            "poi_type": "waypoint",
            "position": [50, 100],
            "zone": "ROGUE ENCAMPMENT"
          }
        ],
        "connections": {
          "Exit to Blood Moor": "BLOOD MOOR"
        },
        "discovered_at": "2026-01-24T..."
      }
    ]
  }

- minimap_images/: Cached minimap PNG files for each zone

USAGE EXAMPLE:
--------------

from src.diabot.navigation import Navigator, MinimapPOIDetector

# Initialize navigator
nav = Navigator(debug=True)

# Explore a zone
nav.visit_zone("ROGUE ENCAMPMENT", "a1")

# Extract minimap and detect POIs
detector = MinimapPOIDetector(debug=True)
minimap = ...  # Your minimap image
pois = detector.detect(minimap)

# Register POIs
for poi in pois:
    if poi.poi_type == 'waypoint':
        nav.add_waypoint("ROGUE ENCAMPMENT", poi.position)

# Register connections
nav.add_exit("ROGUE ENCAMPMENT", "BLOOD MOOR", exit_position)

# Plan routes
path = nav.plan_route("ROGUE ENCAMPMENT", "COLD PLAINS")
print(f"Route: {path.zones}")  # ['ROGUE ENCAMPMENT', 'BLOOD MOOR', 'COLD PLAINS']

# Save
nav.save_maps()

WORKFLOW FOR NEW ZONES:
-----------------------

1. Enter zone (first time)
2. Extract minimap from screen_regions.py UI_REGIONS['minimap_ui']
3. Detect POIs using MinimapPOIDetector
4. Register zone in Navigator
5. Register all detected POIs
6. Save maps

NEXT IMPROVEMENTS:
------------------

- [ ] Cluster nearby waypoints (same actual waypoint)
- [ ] Detect room/level exits automatically
- [ ] Build graph visualization
- [ ] A* pathfinding instead of BFS
- [ ] Cache frequently used paths
- [ ] Detect level transitions and populate automatically
"""

print(__doc__)
