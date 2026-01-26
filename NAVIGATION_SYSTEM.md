# Vision-Based Navigation System

## Overview

This is a robotics-inspired navigation system for Diablo 2 that operates **purely on visual observations** from the minimap overlay. It does not use memory reading, game APIs, or injected map data.

## Architecture

### Core Principles

1. **Vision-Only**: All navigation decisions are based on image processing
2. **Cross-Platform**: Developer mode works on macOS/Linux/Windows
3. **Incremental Mapping**: Builds a local map as the player explores
4. **Frontier-Based**: Explores by targeting the boundary of known/unknown space
5. **Dead Reckoning**: Tracks player movement between observations

### Components

```
navigation/
├── minimap_extractor.py      # Extract minimap from game frame
├── minimap_processor.py       # Convert to occupancy grid (walls/free)
├── local_map.py               # Incremental 2D map building
├── pose_estimator.py          # Dead reckoning for player pose
├── frontier_navigator.py      # Main navigation controller
└── nav_visualization.py       # Debug overlays
```

## How It Works

### 1. Minimap Extraction

**MinimapExtractor** crops the minimap region from the full game frame using predefined screen regions.

```python
from diabot.navigation import MinimapExtractor

extractor = MinimapExtractor(debug=True)
minimap = extractor.extract(game_frame)
```

### 2. Minimap Processing

**MinimapProcessor** converts the minimap image to a binary occupancy grid:
- **WALL**: Dark areas (obstacles)
- **FREE**: Bright areas (walkable paths)
- **UNKNOWN**: Areas outside visible range

```python
from diabot.navigation import MinimapProcessor

processor = MinimapProcessor(grid_size=64, wall_threshold=80)
minimap_grid = processor.process(minimap)

# minimap_grid.grid is a 64x64 numpy array
# minimap_grid.center is the player position (always center)
```

**Algorithm:**
1. Convert to grayscale
2. Apply bilateral filter (noise reduction)
3. Threshold: dark → walls, bright → free
4. Morphological operations (remove noise)
5. Resize to fixed grid size

### 3. Local Map Building

**LocalMap** maintains an incremental 2D occupancy grid:
- Fixed size (e.g., 200×200 cells)
- Centered on player
- Updates from minimap observations
- Tracks visited cells
- Automatically recenters when player moves far from center

```python
from diabot.navigation import LocalMap

local_map = LocalMap(map_size=200, recenter_threshold=40)

# Update with new minimap observation
local_map.update_from_minimap(minimap_grid.grid, player_position)

# Mark current position as visited
local_map.mark_visited(player_x, player_y)

# Get frontier cells for exploration
frontiers = local_map.get_frontiers(max_frontiers=20)

# Find path between two points
path = local_map.find_path(start=(x1, y1), goal=(x2, y2))
```

**Map Recentering:**
When the player moves more than `recenter_threshold` cells from center, the map shifts to keep the player near the center. This allows infinite exploration within a fixed-size map.

### 4. Pose Estimation

**PoseEstimator** tracks player position and orientation using dead reckoning:

```python
from diabot.navigation import PoseEstimator

pose_estimator = PoseEstimator(
    initial_x=100,
    initial_y=100,
    movement_speed=2.0  # cells per second
)

# Update from movement command
pose_estimator.update_from_movement("forward", duration=0.5)

# Update from rotation
pose_estimator.update_rotation(angle_delta=30)  # degrees

# Get current pose
pose = pose_estimator.get_pose()
print(f"Position: ({pose.x}, {pose.y}), Angle: {pose.angle}°")

# Apply correction from observations
pose_estimator.correct_position(new_x, new_y, confidence=0.8)
```

**Dead Reckoning Errors:**
Dead reckoning accumulates drift over time. Future enhancements could use:
- Minimap alignment (template matching)
- Visual odometry
- Landmark recognition

### 5. Frontier-Based Navigation

**FrontierNavigator** is the main navigation controller that integrates all components:

```python
from diabot.navigation import FrontierNavigator, NavigationAction

navigator = FrontierNavigator(
    minimap_grid_size=64,
    local_map_size=200,
    movement_speed=2.0,
    debug=True
)

# Update navigation (call every frame or periodically)
nav_state = navigator.update(game_frame)

# Check action
if nav_state.action == NavigationAction.MOVE_FORWARD:
    # Execute forward movement
    execute_move_forward()
    navigator.report_movement("forward", 0.5)
    
elif nav_state.action == NavigationAction.TURN_LEFT:
    # Execute left turn
    execute_turn_left()
    navigator.report_rotation(-30)
```

**Navigation State:**
```python
nav_state.action              # NavigationAction enum
nav_state.current_position    # (x, y) in map coordinates
nav_state.current_angle       # Orientation in degrees
nav_state.target_position     # Target frontier or None
nav_state.path                # Planned path or None
nav_state.frontiers_available # Number of frontiers found
nav_state.exploration_progress # 0.0 to 1.0
```

**Frontier Detection Algorithm:**
1. Find all FREE cells adjacent to UNKNOWN cells
2. Calculate distance from player to each frontier
3. Sort by distance (closest first)
4. Return top N frontiers

**Path Planning:**
- Uses breadth-first search (BFS)
- Only considers FREE cells as walkable
- 4-connected grid (up, down, left, right)
- Returns list of waypoints from start to goal

**Action Generation:**
1. Select closest frontier as target
2. Plan path to frontier using BFS
3. Get next waypoint on path
4. Calculate angle to waypoint
5. If angle difference > 30°: turn
6. Otherwise: move forward

### 6. Debug Visualization

**NavigationOverlay** provides visual debugging:

```python
from diabot.navigation import NavigationOverlay

overlay = NavigationOverlay(
    show_local_map=True,
    show_path=True,
    show_frontiers=True,
    show_minimap_grid=True
)

# Draw overlay on frame
vis_frame = overlay.draw(
    frame=game_frame,
    nav_state=nav_state,
    local_map=navigator.get_local_map(),
    minimap_grid=minimap_grid
)

# Display or save
cv2.imshow("Navigation Debug", vis_frame)
```

**Visualization Elements:**
- **Text Info**: Action, position, angle, target, frontiers, progress
- **Local Map**: Top-down view with explored areas, walls, frontiers
- **Minimap Grid**: Processed occupancy grid from current observation
- **Player**: Green dot at current position
- **Path**: Yellow line showing planned route
- **Target**: Red dot at navigation goal
- **Frontiers**: Cyan dots at exploration boundaries

## Usage Examples

### Example 1: Basic Navigation Loop

```python
from diabot.navigation import FrontierNavigator, NavigationAction

# Initialize
navigator = FrontierNavigator(debug=True)

# Main loop
while exploring:
    # Get game frame
    frame = capture_screen()
    
    # Update navigation
    nav_state = navigator.update(frame)
    
    # Execute action
    if nav_state.action == NavigationAction.MOVE_FORWARD:
        move_forward(duration=0.5)
        navigator.report_movement("forward", 0.5)
        
    elif nav_state.action == NavigationAction.TURN_LEFT:
        turn_left(angle=30)
        navigator.report_rotation(-30)
        
    elif nav_state.action == NavigationAction.TURN_RIGHT:
        turn_right(angle=30)
        navigator.report_rotation(30)
        
    elif nav_state.action == NavigationAction.STOP:
        # No more frontiers or stuck
        break
```

### Example 2: Developer Mode (Static Screenshot)

```python
import cv2
from diabot.navigation import FrontierNavigator, NavigationOverlay

# Load screenshot
frame = cv2.imread("data/screenshots/game.png")

# Initialize
navigator = FrontierNavigator(debug=True)
overlay = NavigationOverlay()

# Process
nav_state = navigator.update(frame)

# Visualize
vis = overlay.draw(frame, nav_state, navigator.get_local_map())
cv2.imwrite("output.png", vis)
```

### Example 3: Custom Map Exploration

```python
from diabot.navigation import (
    MinimapExtractor,
    MinimapProcessor,
    LocalMap
)

# Manual component usage
extractor = MinimapExtractor()
processor = MinimapProcessor(grid_size=64)
local_map = LocalMap(map_size=300)

# Process frame
minimap = extractor.extract(frame)
minimap_grid = processor.process(minimap)

# Update map
player_pos = (150, 150)  # center of 300x300 map
local_map.update_from_minimap(minimap_grid.grid, player_pos)

# Get frontiers
frontiers = local_map.get_frontiers()
print(f"Found {len(frontiers)} frontiers")

# Visualize
map_vis = local_map.visualize(show_frontiers=True)
cv2.imshow("Local Map", map_vis)
```

## Configuration

### Minimap Processing

```python
processor = MinimapProcessor(
    grid_size=64,          # Output grid size (64x64)
    wall_threshold=80,     # Brightness threshold (0-255)
    debug=False            # Enable debug output
)
```

**Tuning wall_threshold:**
- Lower values (60-70): More aggressive wall detection
- Higher values (90-100): More conservative, less noise
- Default (80): Good balance for most cases

### Local Map

```python
local_map = LocalMap(
    map_size=200,          # Grid size (200x200)
    recenter_threshold=40, # Distance before recentering
    debug=False
)
```

**Tuning map_size:**
- Smaller (100-150): Less memory, faster, limited exploration
- Larger (250-500): More memory, slower, extensive exploration
- Default (200): Good balance for local exploration

### Pose Estimator

```python
pose_estimator = PoseEstimator(
    movement_speed=2.0,    # Grid cells per second
    debug=False
)
```

**Tuning movement_speed:**
- Depends on character speed and grid scale
- Adjust based on observed movement vs estimated position drift
- Higher values = faster movement estimation

## Limitations and Future Work

### Current Limitations

1. **Dead Reckoning Drift**: Pose estimation accumulates error over time
2. **No Visual Odometry**: No correction from visual observations yet
3. **Simple Path Planning**: BFS is not A* (not optimal for long paths)
4. **No Dynamic Obstacles**: Assumes static environment
5. **Fixed Grid Scale**: Grid doesn't adapt to minimap zoom/scale

### Future Enhancements

1. **Minimap Alignment**: Use template matching to correct pose drift
2. **Visual Landmarks**: Detect waypoints, portals, distinctive features
3. **A* Path Planning**: More efficient long-distance pathfinding
4. **Zone Transition**: Detect and handle zone boundaries
5. **Obstacle Avoidance**: React to dynamic obstacles (monsters, NPCs)
6. **Multi-Scale Mapping**: Coarse global map + fine local map
7. **Loop Closure**: Detect when returning to previously visited areas

## Technical Details

### Coordinate Systems

**Frame Coordinates:**
- Origin: Top-left corner
- X-axis: Right
- Y-axis: Down
- Units: Pixels

**Map Coordinates:**
- Origin: Top-left of map grid
- X-axis: Right
- Y-axis: Down
- Units: Grid cells

**Minimap Coordinates:**
- Origin: Top-left of minimap
- Player: Always at center
- Units: Grid cells

### Angle Convention

- 0° = Right (positive X)
- 90° = Down (positive Y)
- 180° = Left (negative X)
- 270° = Up (negative Y)

Rotation: Clockwise is positive

### Cell Types

```python
class CellType(IntEnum):
    UNKNOWN = 0      # Not yet observed
    FREE = 128       # Walkable space
    WALL = 255       # Obstacle/wall
```

## Testing

Run the demo script to test the navigation system:

```bash
python demo_frontier_navigation.py \
    --image data/screenshots/inputs/game_frame.png \
    --iterations 10 \
    --output data/screenshots/outputs/nav_demo.png
```

## Integration with Existing Bot

The navigation system integrates with the existing bot architecture:

```python
# In decision engine or FSM
from diabot.navigation import FrontierNavigator, NavigationAction

class ExploreState:
    def __init__(self):
        self.navigator = FrontierNavigator(debug=True)
    
    def update(self, frame):
        # Get navigation decision
        nav_state = self.navigator.update(frame)
        
        # Translate to bot actions
        if nav_state.action == NavigationAction.MOVE_FORWARD:
            return Action(type="move", direction="forward")
        elif nav_state.action == NavigationAction.TURN_LEFT:
            return Action(type="rotate", angle=-30)
        # ... etc
```

## Performance

**Computational Cost:**
- Minimap extraction: ~1ms
- Minimap processing: ~5-10ms
- Local map update: ~5ms
- Frontier detection: ~10-20ms
- Path planning: ~5-50ms (depends on distance)

**Total per frame:** 25-100ms (10-40 FPS)

**Optimization opportunities:**
- Process minimap every N frames, not every frame
- Limit frontier search radius
- Use A* instead of BFS for long paths
- Parallelize minimap processing

## Troubleshooting

### No frontiers detected
- Check minimap extraction (verify region coordinates)
- Adjust wall_threshold in MinimapProcessor
- Verify player is in explorable area

### Pose drift
- Report movement/rotation accurately
- Implement visual correction
- Reset pose when entering new zone

### Stuck in loops
- Increase stuck_counter threshold
- Implement backtracking
- Add visited cell penalties

### Poor path planning
- Increase local_map size
- Use A* instead of BFS
- Add path smoothing

## References

This navigation system is inspired by:
- **Frontier-Based Exploration**: Yamauchi (1997)
- **Occupancy Grid Mapping**: Moravec & Elfes (1985)
- **SLAM**: Simultaneous Localization and Mapping
- **Dead Reckoning**: Classic navigation technique

The implementation is simplified for game navigation but follows established robotics principles.
