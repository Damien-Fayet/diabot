# Vision-Based Navigation Module

A robotics-inspired navigation system for Diablo 2 that operates purely on visual observations from the minimap overlay.

## Features

- **Vision-Only**: No memory reading or game API access
- **Cross-Platform**: Works on Windows, macOS, and Linux in developer mode
- **Frontier-Based**: Systematic exploration using frontier detection
- **Incremental Mapping**: Builds local occupancy grid as player explores
- **Dead Reckoning**: Tracks player pose between observations
- **Path Planning**: BFS pathfinding on occupancy grid
- **Debug Visualization**: Comprehensive overlay system

## Quick Start

```python
from diabot.navigation import FrontierNavigator, NavigationAction

# Initialize
navigator = FrontierNavigator(debug=True)

# Main loop
while exploring:
    # Update with game frame
    nav_state = navigator.update(frame)
    
    # Execute action
    if nav_state.action == NavigationAction.MOVE_FORWARD:
        move_forward(duration=0.5)
        navigator.report_movement("forward", 0.5)
    
    elif nav_state.action == NavigationAction.TURN_LEFT:
        turn_left(angle=30)
        navigator.report_rotation(-30)
    
    elif nav_state.action == NavigationAction.STOP:
        break  # No more frontiers
```

## Components

### Core Modules

- **MinimapExtractor**: Extracts minimap region from game frame
- **MinimapProcessor**: Converts minimap to binary occupancy grid
- **LocalMap**: Maintains incremental 2D map with frontier detection
- **PoseEstimator**: Tracks player position/orientation via dead reckoning
- **FrontierNavigator**: Main controller integrating all components
- **NavigationOverlay**: Debug visualization system

### Architecture

```
Frame → MinimapExtractor → MinimapProcessor → MinimapGrid
                                                    ↓
LocalMap ← update ← MinimapGrid + PlayerPose
    ↓
Frontiers → PathPlanning → NavigationAction
                              ↓
                        ActionExecutor
                              ↓
                        PoseEstimator ← MovementFeedback
```

## Navigation Pipeline

1. **Extract** minimap from full frame
2. **Process** minimap to occupancy grid (walls/free/unknown)
3. **Update** local map with new observations
4. **Track** player pose using dead reckoning
5. **Detect** frontier cells (boundary of explored/unexplored)
6. **Plan** path to nearest frontier
7. **Generate** navigation action (move/turn/stop)

## Configuration

```python
navigator = FrontierNavigator(
    minimap_grid_size=64,      # Minimap processing resolution
    local_map_size=200,        # Local map size (200×200 cells)
    movement_speed=2.0,        # Movement speed (cells/second)
    debug=False                # Enable debug output
)
```

## Visualization

```python
from diabot.navigation import NavigationOverlay

overlay = NavigationOverlay(
    show_local_map=True,       # Show explored map
    show_path=True,            # Show planned path
    show_frontiers=True,       # Show exploration targets
    show_minimap_grid=False    # Show processed minimap
)

vis_frame = overlay.draw(frame, nav_state, navigator.get_local_map())
cv2.imshow("Navigation", vis_frame)
```

## Testing

Run unit tests:
```bash
python test_navigation_system.py
```

Run demo:
```bash
python demo_frontier_navigation.py --image data/screenshots/inputs/game_frame.png
```

## Documentation

- **NAVIGATION_SYSTEM.md** - Complete technical documentation
- **NAVIGATION_IMPLEMENTATION.md** - Implementation summary
- **NAVIGATION_QUICK_REFERENCE.md** - Quick API reference

## Integration

See `example_navigation_integration.py` for integration with existing bot FSM.

```python
from diabot.navigation import FrontierNavigator

class ExploreState:
    def __init__(self):
        self.nav = FrontierNavigator()
    
    def update(self, frame):
        nav_state = self.nav.update(frame)
        return self.translate_action(nav_state.action)
```

## Performance

- **25-100ms per frame** (10-40 FPS)
- **1-5MB memory** for local map
- Suitable for real-time navigation

## Coordinate Systems

- **Frame**: Pixels, origin at top-left
- **Map**: Grid cells, origin at top-left
- **Angles**: Degrees, 0°=right, 90°=down (clockwise)

## Limitations

1. Dead reckoning drift (future: visual correction)
2. No dynamic obstacles (future: obstacle avoidance)
3. BFS not optimal for long paths (future: A*)
4. Fixed grid scale (future: adaptive scaling)

## Future Enhancements

### High Priority
- Visual correction via template matching
- Landmark detection (waypoints, portals)
- Zone transition handling

### Medium Priority
- A* pathfinding
- Dynamic obstacle avoidance
- Loop closure detection

### Low Priority
- Multi-scale mapping
- Visual odometry
- Learned features via ML

## Troubleshooting

### No frontiers detected
- Verify minimap extraction region
- Adjust `wall_threshold` in MinimapProcessor
- Check player is in explorable area

### Pose drifts
- Report movements accurately
- Implement visual correction
- Reset on zone changes

### Navigator stuck
- Increase stuck threshold
- Add backtracking logic
- Implement obstacle avoidance

## References

Inspired by:
- Frontier-Based Exploration (Yamauchi, 1997)
- Occupancy Grid Mapping (Moravec & Elfes, 1985)
- SLAM (Simultaneous Localization and Mapping)

## License

Part of the diabot project. See main LICENSE file.

## Status

✅ **Production Ready**
- All tests passing
- Fully documented
- Cross-platform compatible
- Integrated with existing bot architecture
