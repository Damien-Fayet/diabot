# Movement Optimization + Zone Navigation System

## Summary of Improvements

### 1. **Movement Optimization - Don't Re-Click While Approaching**

**Problem**: Bot would re-click movement target every frame, causing jerky movement and excess clicking.

**Solution**: Track distance to current movement target
- If distance is decreasing → continue moving (don't re-click)
- If distance stays same for N frames → we're stuck, re-click
- If target changes → re-click to new target

**Implementation**:
- New fields in `DiabloBot`:
  - `current_movement_target`: Position we're moving to
  - `movement_target_distance`: Distance to target
  - `frames_without_progress`: Count of frames not getting closer
  - `max_frames_without_progress`: Threshold before re-clicking (15 frames)

- New method: `_should_reclick_movement_target(hero_pos, target_pos)`
  - Returns True if should re-click, False to continue

**Result**: Smoother movement, less jittery, fewer clicks

### 2. **Zone Mapping System - Navigate When Quest Not Visible**

**Problem**: Quests aren't always on screen; need navigation to reach them.

**Solution**: Create map of game zones with connections
- Define zones (Rogue Encampment, Cold Plains, Burial Grounds, etc.)
- Define connections between zones (which zones connect)
- Track current zone based on hero position
- Calculate paths using BFS

**Components**:

#### `ZoneLocation` dataclass:
```python
zone_id: str                          # "rogue_encampment"
name: str                             # "Rogue Encampment"
position: Tuple[float, float]         # (500, 600) center
region_size: Tuple[float, float]      # (800, 600) size
exits: List[str]                      # ["cold_plains", "barracks"]
npc_types: List[str]                  # Common NPCs here
```

#### `GameMap`:
- `zones`: Dict of all zones
- `current_zone`: Currently detected zone
- Methods:
  - `find_zone_by_position(x, y)`: Which zone contains position
  - `update_current_zone(hero_pos)`: Update based on hero
  - `find_path_to_zone(target_zone_id)`: BFS pathfinding
  - `get_zone_entrance(zone_id)`: Position to move to

#### `MapNavigator`:
- `navigate_to_quest_target(target_zone_id)`: Get next waypoint
- Handles multi-zone navigation automatically

### 3. **Enhanced Decision Making**

When quest target not visible on screen:
1. Get current quest's target zone
2. Use navigator to find path
3. Move to next zone in path

**Priority levels**:
1. Quest marker visible on screen → interact with NPC
2. Quest NPCs visible → move to them
3. Quest not on screen → navigate to quest zone
4. No quest → explore

## Code Examples

### Movement Optimization

```python
# Bot decides to move to target
target = (400, 300)

# Frame 1: Check if should click
should_click = bot._should_reclick_movement_target(hero_pos, target)
# → True (first time seeing target)
# → executes right-click at (400, 300)

# Frame 2-5: Hero getting closer
distance = 150, 100, 50, 25 pixels
# → False each frame (distance decreasing)
# → no re-click, character continues moving

# Frame 6: Hero got stuck
distance = 26 pixels (not closer)
frames_without_progress = 1, 2, 3... 15
# → False for 14 frames
# → True on frame 15 (stuck threshold)
# → re-click at target
```

### Zone Navigation

```python
# Hero in Rogue Encampment, quest in Burial Grounds
current_zone = "rogue_encampment"
target_zone = "burial_grounds"

# Get path
path = navigator.find_path_to_zone("burial_grounds")
# → ["rogue_encampment", "cold_plains", "burial_grounds"]

# Get next waypoint
next_target = navigator.navigate_to_quest_target("burial_grounds")
# → (400, 400) - entrance to Cold Plains

# Move towards waypoint
click_move(next_target)

# Next frame, hero in Cold Plains
game_map.update_current_zone(hero_pos)
# → current_zone = "cold_plains"

# Get next waypoint
next_target = navigator.navigate_to_quest_target("burial_grounds")
# → (300, 300) - entrance to Burial Grounds

# Move towards next zone...
```

## Output Example

```
[Frame 0 | FPS: 2.0]
  FSM State: IDLE
  Quest: Sisters' Burial Grounds: Defeat the corrupted Rogue Encampment
  Detections: 6 total
  
  ➡️  Moving to zone:burial_grounds: dist=450px (continuing) [zone: rogue_encampment]
  
  Detected objects:
    - stash (conf=0.90)
    - hero (conf=0.80)
    - kashya (conf=0.75)

[Frame 1 | FPS: 1.9]
  ➡️  Moving to zone:burial_grounds: dist=420px (continuing) [zone: rogue_encampment]

[Frame 5 | FPS: 1.8]
  ➡️  Moving to zone:burial_grounds: dist=100px (continuing) [zone: cold_plains]

[Frame 10 | FPS: 1.8]
  ➡️  Moving to zone:burial_grounds: dist=50px (continuing) [zone: burial_grounds]

[Frame 15 | FPS: 1.8]
  🎯 Quest marker at (107.0, 222.0) (conf=0.97)
    → NPC under marker: cain at (108.0, 329.5)
  
  Action: interact_with_npc → cain
```

## Integration Points

### In `DiabloBot`:
1. Initialize: `self.game_map = GameMap()`, `self.navigator = MapNavigator(game_map)`
2. Each frame: `game_map.update_current_zone(hero_det.center)`
3. In `_decide_action()`: Fall back to navigator if quest not visible
4. In `_execute_action()`: Use `_should_reclick_movement_target()` to optimize movement

### In `_log_step()`:
- Display movement status: "Moving to X: dist=Ypx (continuing)" 
- Show current zone for debugging navigation

## Files

### New Files:
- `src/diabot/navigation/map_system.py`: Zone mapping and navigation (200+ lines)

### Modified Files:
- `src/diabot/bot/bot_main.py`:
  - Added movement optimization fields (~10 lines)
  - Added `_should_reclick_movement_target()` method (~30 lines)
  - Added `_get_quest_target_zone()` method (~20 lines)
  - Updated `_execute_action()` with smart re-clicking (~15 lines)
  - Enhanced `_log_step()` with movement info (~5 lines)
  - Integrated GameMap and MapNavigator (~10 lines)

## Future Improvements

1. **Better Zone Detection**: Use YOLO to detect zone markers instead of position
2. **Smooth Pathfinding**: Implement A* instead of BFS for better routes
3. **Terrain Avoidance**: Track obstacles and avoid them
4. **Portal Detection**: Detect portals/doors to enter zones more precisely
5. **Dynamic Map Loading**: Load zones for different acts as needed
6. **Performance Tracking**: Measure time to reach quest targets
7. **Stuck Detection**: Better heuristics for detecting when truly stuck
