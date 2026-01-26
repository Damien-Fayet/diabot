# NPC Detection Under Quest Markers - Improvements

## What was improved

Instead of simply clicking below the quest marker with a fixed offset, the bot now:

1. **Finds the exact NPC under the marker** using geometric scoring
   - Detects all NPCs (akara, kashya, warriv, etc.)
   - Scores each based on vertical position below marker + horizontal alignment
   - Selects the best match
   - Result: Clicks on actual NPC bounding box, not approximate offset

2. **Tracks distance from hero to NPC** for debugging and future features
   - Calculates Euclidean distance each frame
   - Tracks if distance is decreasing (bot approaching NPC)
   - Counts consecutive frames where distance decreased
   - Result: Can detect when bot successfully navigates to NPC

## Implementation Details

### 1. Improved NPC Detection Under Marker

**Method**: `DiabloBot._find_npc_under_marker(quest_marker, detections)`

```python
# Scoring formula:
vertical_score = max(0, gap_from_marker_bottom)  # Prefer NPCs directly below
horizontal_score = horizontal_distance * 0.5      # Secondary weight

score = vertical_score + horizontal_score
```

**Key parameters**:
- NPCs must be below marker (vertical gap >= -20px allows slight overlap)
- NPCs too far horizontally are penalized but not excluded
- Returns highest-confidence match

### 2. Click on NPC Bbox, Not Marker

**Old behavior**: 
```python
click_y = marker_y + 35  # Fixed 35px offset down
click_at_screen(marker_x, click_y)
```

**New behavior**:
```python
# Use actual NPC position from detection
npc_x, npc_y = npc.center  # Center of NPC bounding box
interact_with_object(screen_x, screen_y)  # Click on NPC directly
```

### 3. Distance Tracking

**Method**: `DiabloBot._update_distance_tracking(hero_det, npc_det)`

Tracks:
- `last_distance`: Distance from hero to NPC (pixels)
- `distance_decreasing_frames`: Counter of frames where distance decreased
- `last_hero_pos`, `last_npc_pos`: Last known positions

Formula:
```python
distance = sqrt((hero_x - npc_x)² + (hero_y - npc_y)²)
```

## Output Example

```
[Frame 0 | FPS: 2.0]
  FSM State: IDLE
  Quest: Sisters' Burial Grounds: Defeat the corrupted Rogue Encampment
  Detections: 6 total
  
  🎯 Quest marker at (107.0, 222.0) (conf=0.97)
    → NPC under marker: warriv at (108.0, 329.5)
  
  📍 Distance: 863px ↓ (Δ+0px, 0 frames closer)
    Hero at (958.0, 476.5), NPC at (108.0, 329.5)
  
  Detected objects:
    - quest (conf=0.97)
    - hero (conf=0.97)
    - waypoint (conf=0.96)
    - warriv (conf=0.77)
    - kashya (conf=0.73)
  
  Action: interact_with_npc → warriv
```

## Files Modified

### `src/diabot/bot/bot_main.py`

1. Added initialization fields:
   - `last_hero_pos`, `last_npc_pos`, `last_distance`, `distance_decreasing_frames`

2. Added new methods:
   - `_calculate_distance()`: Euclidean distance between points
   - `_update_distance_tracking()`: Track hero/NPC distance each frame
   - `_find_hero()`: Locate hero detection
   - `_find_npc_under_marker()`: Improved geometric NPC matching

3. Modified methods:
   - `run_step()`: Find hero, track distance, pass all info to decision engine
   - `_decide_action()`: Accept detections and pre-found NPC
   - `_execute_action()`: Click on NPC center instead of marker offset
   - `_log_step()`: Display distance info with arrows and frame counters

### `src/diabot/core/action_executor.py`

- Added `interact_with_object()`: Generic click on object center
- Updated `click_on_npc()`: Uses NPC position directly (kept for backward compat)

## Benefits

1. **More accurate interaction**: Clicks on actual NPC bbox, not estimated offset
2. **Better debugging**: See distance changes each frame
3. **Foundation for future features**:
   - Can detect if bot is stuck (distance not decreasing for N frames)
   - Can optimize pathfinding to NPCs
   - Can detect successful interaction (NPC despawns or quest marker disappears)

## Testing

Run demos:
```bash
# Live detection with distance tracking
python demo_npc_detection.py

# Full bot with all improvements
python run_bot_live.py --model runs/detect/runs/train/diablo-yolo3/weights/best.pt --fps 2 --max-frames 20
```

## Next Steps

1. **Quest completion detection**: Track when quest marker disappears after NPC interaction
2. **Combat system**: Test bot response when enemies appear
3. **HP extraction**: Replace dummy HP values with real screen values
4. **Movement pathfinding**: Improve distance-decreasing rate
