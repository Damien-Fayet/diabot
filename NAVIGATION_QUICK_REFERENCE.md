# Navigation System Quick Reference

## Installation

No additional dependencies needed. Uses existing:
- numpy
- opencv-python (cv2)

## Quick Start

### 1. Basic Usage

```python
from diabot.navigation import FrontierNavigator

# Initialize
nav = FrontierNavigator(debug=True)

# Update with frame
nav_state = nav.update(game_frame)

# Check action
print(nav_state.action)  # NavigationAction enum
```

### 2. With Visualization

```python
from diabot.navigation import FrontierNavigator, NavigationOverlay

nav = FrontierNavigator(debug=True)
overlay = NavigationOverlay()

nav_state = nav.update(frame)
vis = overlay.draw(frame, nav_state, nav.get_local_map())

cv2.imshow("Nav", vis)
```

### 3. Full Loop

```python
nav = FrontierNavigator()

while exploring:
    state = nav.update(frame)
    
    if state.action == NavigationAction.MOVE_FORWARD:
        move_forward(0.5)  # 0.5 seconds
        nav.report_movement("forward", 0.5)
    
    elif state.action == NavigationAction.TURN_LEFT:
        turn_left(30)  # 30 degrees
        nav.report_rotation(-30)
    
    elif state.action == NavigationAction.STOP:
        break  # Done exploring
```

## API Reference

### FrontierNavigator

**Constructor:**
```python
FrontierNavigator(
    minimap_grid_size=64,    # Minimap processing resolution
    local_map_size=200,      # Local map size (cells)
    movement_speed=2.0,      # Movement speed (cells/sec)
    debug=False              # Enable debug output
)
```

**Methods:**
- `update(frame) -> NavigationState`: Main update, returns navigation decision
- `report_movement(direction, duration)`: Report executed movement
- `report_rotation(angle_delta)`: Report executed rotation
- `reset()`: Reset navigation state
- `get_local_map() -> LocalMap`: Get local map for visualization
- `get_pose() -> Pose`: Get current pose estimate

**NavigationState Fields:**
- `action`: NavigationAction enum (STOP, MOVE_FORWARD, TURN_LEFT, TURN_RIGHT)
- `current_position`: (x, y) tuple
- `current_angle`: float (degrees)
- `target_position`: (x, y) or None
- `path`: list of (x, y) waypoints or None
- `frontiers_available`: int
- `exploration_progress`: float (0.0 to 1.0)

### NavigationAction Enum

```python
NavigationAction.STOP           # No action
NavigationAction.MOVE_FORWARD   # Move forward
NavigationAction.TURN_LEFT      # Turn left
NavigationAction.TURN_RIGHT     # Turn right
```

### NavigationOverlay

**Constructor:**
```python
NavigationOverlay(
    show_local_map=True,     # Show map in corner
    show_path=True,          # Show planned path
    show_frontiers=True,     # Show frontier cells
    show_minimap_grid=False  # Show processed minimap
)
```

**Methods:**
- `draw(frame, nav_state, local_map, minimap_grid) -> frame`: Draw overlay

### LocalMap

**Methods:**
- `update_from_minimap(grid, player_pos)`: Update with observation
- `mark_visited(x, y)`: Mark cell as visited
- `get_frontiers(max_frontiers) -> list`: Find frontier cells
- `find_path(start, goal) -> list`: Plan path (BFS)
- `visualize() -> frame`: Create visualization

### PoseEstimator

**Methods:**
- `update_from_movement(direction, duration)`: Update from movement
- `update_rotation(angle_delta)`: Update from rotation
- `correct_position(x, y, confidence)`: Apply correction
- `get_pose() -> Pose`: Get current pose
- `reset(x, y, angle)`: Reset pose

## Configuration Tips

### Tuning Wall Threshold

```python
processor = MinimapProcessor(
    wall_threshold=80  # Default
    # Lower (60-70): More aggressive wall detection
    # Higher (90-100): More conservative
)
```

### Adjusting Map Size

```python
nav = FrontierNavigator(
    local_map_size=200  # Default
    # Smaller (100-150): Faster, less memory
    # Larger (250-500): More exploration range
)
```

### Movement Speed Calibration

```python
nav = FrontierNavigator(
    movement_speed=2.0  # Default: 2 cells/second
    # Adjust based on character and grid scale
)
```

## Common Patterns

### Zone Reset

```python
# Reset when entering new zone
if zone_changed():
    nav.reset()
```

### Conditional Navigation

```python
# Only navigate in certain states
if current_state == State.EXPLORE:
    nav_state = nav.update(frame)
    # Execute action
else:
    # Use different controller
    pass
```

### Error Handling

```python
try:
    nav_state = nav.update(frame)
except Exception as e:
    print(f"Navigation error: {e}")
    # Fallback behavior
```

## Debugging

### Enable Debug Output

```python
nav = FrontierNavigator(debug=True)
# Prints internal state to console
```

### Visualize Local Map

```python
local_map = nav.get_local_map()
vis = local_map.visualize(show_frontiers=True)
cv2.imshow("Map", vis)
```

### Visualize Minimap Processing

```python
from diabot.navigation import MinimapExtractor, MinimapProcessor

extractor = MinimapExtractor()
processor = MinimapProcessor(debug=True)

minimap = extractor.extract(frame)
grid = processor.process(minimap)
vis = processor.visualize(grid)
cv2.imshow("Grid", vis)
```

### Save Debug Frames

```python
vis = overlay.draw(frame, nav_state, nav.get_local_map())
cv2.imwrite(f"debug/nav_{frame_num}.png", vis)
```

## Troubleshooting

### No frontiers detected
- Check minimap extraction region
- Adjust `wall_threshold`
- Verify player is in explorable area

### Pose drifts over time
- Report movements accurately
- Implement visual correction
- Reset on zone transitions

### Navigator stuck
- Increase `stuck_counter` threshold in code
- Add obstacle avoidance
- Implement backtracking

### Poor path planning
- Increase `local_map_size`
- Consider implementing A* pathfinding
- Add path smoothing

## Performance

Typical timing per frame:
- Minimap extraction: ~1ms
- Minimap processing: ~5-10ms
- Local map update: ~5ms
- Frontier detection: ~10-20ms
- Path planning: ~5-50ms

**Total: 25-100ms (10-40 FPS)**

## Examples

See:
- `demo_frontier_navigation.py` - Basic demo
- `example_navigation_integration.py` - Bot integration
- `test_navigation_system.py` - Unit tests

## Documentation

Full docs:
- `NAVIGATION_SYSTEM.md` - Complete technical documentation
- `NAVIGATION_IMPLEMENTATION.md` - Implementation summary

## Support

For issues or questions:
1. Check debug output with `debug=True`
2. Verify minimap extraction with visualization
3. Check test suite: `python test_navigation_system.py`
