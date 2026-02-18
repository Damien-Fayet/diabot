# Minimap SLAM System Documentation

## Overview

This document describes the **Minimap SLAM** (Simultaneous Localization and Mapping) system for Diablo II Resurrected. This system builds a persistent map and localizes the player using **ONLY** visual information from the minimap—no game coordinates, memory access, or physics engine.

---

## Core Philosophy

### The Fixed-Player Paradigm

**Traditional approach** (coordinate-based):
- Player has position (x, y) in world
- World is static
- Movement updates player position

**Our approach** (vision-based):
- **Player position is FIXED** in our internal map
- **World moves around the player**
- Movement is inferred by comparing consecutive minimap images

### Why This Matters

In Diablo II, we have no access to:
- Player coordinates
- Game memory
- Physics engine
- Movement vectors

We can only observe:
- Minimap image changes
- Visual wall patterns
- POI appearances

Therefore, we don't ask: *"Did the player move?"*  
We ask: *"Did the minimap content shift between frames?"*

---

## System Architecture

### Data Structures

#### 1. Global Occupancy Grid
```python
# Large 2D array representing the known world
occupancy_grid: np.ndarray[H, W]

# Cell types:
UNKNOWN = -1  # Unexplored
FREE    = 0   # Walkable space
WALL    = 1   # Obstacle
```

#### 2. Player State
```python
player_center: (x, y)  # FIXED position in global map (typically center)
```

#### 3. World Offset
```python
world_offset_x: int  # Cumulative X displacement
world_offset_y: int  # Cumulative Y displacement
```

When the minimap shifts right (+dx), the world moved left, so:
```python
world_offset_x -= dx  # Opposite direction!
```

#### 4. Points of Interest (POI)
```python
@dataclass
class POI:
    poi_type: str          # 'npc', 'exit', 'stairs', 'waypoint', etc.
    pos: (gx, gy)          # GLOBAL coordinates
    confidence: float      # Detection reliability
    metadata: dict         # Additional info (e.g., NPC name)
```

#### 5. Levels (Multi-Level Support)
```python
@dataclass
class Level:
    level_id: str
    occupancy_grid: np.ndarray
    pois: List[POI]
    transitions: Dict[str, Tuple[int, int]]  # {target_level: (gx, gy)}
```

---

## Pipeline

### Step 1: Minimap Preprocessing

**Input:** Raw minimap image (BGR)  
**Output:** Binary skeleton (walls = 255, free = 0)

```python
def preprocess_minimap(minimap: np.ndarray) -> np.ndarray:
    1. Convert to grayscale
    2. Enhance contrast (CLAHE)
    3. Threshold (Otsu adaptive)
    4. Morphological cleanup (open/close)
    5. Skeletonize walls to 1-pixel width
    return skeleton
```

**Why skeletonize?**  
- Reduces noise
- Makes motion estimation more robust
- Standardizes wall representation

---

### Step 2: Motion Estimation

**Input:** Current skeleton, previous skeleton  
**Output:** (dx, dy, confidence)

```python
def estimate_motion(current, previous) -> (dx, dy, confidence):
    # Use phase correlation (sub-pixel accuracy)
    (dx, dy), conf = cv2.phaseCorrelate(previous, current)
    return dx, dy, conf
```

**Phase Correlation:**
- Robust to noise
- Sub-pixel accuracy
- Handles translations (not rotations)
- Works well for structured environments (walls)

**Alternative:** Optical flow (Farneback)
```python
flow = cv2.calcOpticalFlowFarneback(prev, curr, ...)
dx = np.median(flow[:, :, 0])
dy = np.median(flow[:, :, 1])
```

---

### Step 3: Update World Offset

```python
def update_world_offset(dx: float, dy: float):
    # IMPORTANT: World moves OPPOSITE to minimap shift
    if abs(dx) > movement_threshold or abs(dy) > movement_threshold:
        world_offset_x -= int(dx)  # Opposite!
        world_offset_y -= int(dy)  # Opposite!
```

**Intuition:**
- If minimap shifted **right** (+dx), player moved **left**
- But player is **fixed** in our frame
- So world moved **right** in player's frame
- Therefore, world_offset decreases (moves left in absolute terms)

---

### Step 4: Global Map Fusion

**For each pixel in minimap skeleton:**

```python
def fuse_minimap_to_global(skeleton: np.ndarray):
    h, w = skeleton.shape
    cx, cy = w // 2, h // 2  # Minimap center
    
    for y in range(h):
        for x in range(w):
            # Local offset from minimap center
            local_x = x - cx
            local_y = y - cy
            
            # Convert to GLOBAL coordinates
            gx = player_center_x + world_offset_x + local_x
            gy = player_center_y + world_offset_y + local_y
            
            # Update global map
            if skeleton[y, x] > 127:
                global_map[gy, gx] = WALL
            else:
                if global_map[gy, gx] == UNKNOWN:
                    global_map[gy, gx] = FREE
```

**Fusion Rules:**
- `WALL` is sticky (never overwritten)
- `FREE` overwrites `UNKNOWN`
- Multiple observations increase confidence

---

### Step 5: Loop Closure Detection

**Purpose:** Correct accumulated drift when returning to known locations.

```python
def capture_signature(skeleton) -> MapSignature:
    # Compact representation of local area
    return MapSignature(
        wall_hash=md5(skeleton),
        orientation_hist=compute_wall_orientations(skeleton),
        intersection_count=count_intersections(skeleton),
        world_offset=(world_offset_x, world_offset_y)
    )
```

**Detection:**
```python
def detect_loop_closure(current_sig):
    for prev_sig in signature_history:
        similarity = current_sig.similarity(prev_sig)
        if similarity > threshold:
            return prev_sig  # Loop detected!
    return None
```

**Drift Correction:**
```python
def correct_drift(current_sig, matched_sig):
    error_x = current_sig.world_offset[0] - matched_sig.world_offset[0]
    error_y = current_sig.world_offset[1] - matched_sig.world_offset[1]
    
    # Apply partial correction (avoid jarring jumps)
    world_offset_x -= error_x * 0.5
    world_offset_y -= error_y * 0.5
```

---

### Step 6: Blocked Movement Detection

**Use case:** Collision detection without physics.

```python
def detect_blocked_movement(expected_dx, expected_dy, actual_dx, actual_dy):
    expected_mag = sqrt(expected_dx² + expected_dy²)
    actual_mag = sqrt(actual_dx² + actual_dy²)
    
    # Movement blocked if actual << expected
    if expected_mag > threshold and actual_mag < threshold:
        return True  # Collision!
    return False
```

**Application:**
- Pathfinding validation
- Obstacle detection
- Door/gate status inference

---

### Step 7: POI Tracking

**Adding POIs:**
```python
def add_poi(poi_type, local_pos, confidence):
    lx, ly = local_pos  # Local minimap coordinates
    
    # Convert to global
    gx = player_center_x + world_offset_x + (lx - minimap_cx)
    gy = player_center_y + world_offset_y + (ly - minimap_cy)
    
    # Store in global coordinates
    pois.append(POI(type=poi_type, pos=(gx, gy), confidence=confidence))
```

**POI Merging:**
```python
# If POI detected near existing POI
for existing_poi in pois:
    if distance(new_poi, existing_poi) < threshold:
        existing_poi.confidence += 0.1  # Increase confidence
        return
```

---

### Step 8: Level Changes

**Detection:**
```python
def detect_level_change(current_minimap, prev_minimap):
    diff = cv2.absdiff(current_minimap, prev_minimap)
    change_ratio = np.sum(diff > threshold) / diff.size
    
    # Drastic change = level change
    return change_ratio > 0.7
```

**Switching Levels:**
```python
def switch_level(new_level_id):
    # Store transition point
    current_level.transitions[new_level_id] = current_global_pos
    
    # Switch to new level
    current_level_id = new_level_id
    
    # Reset world offset for new level
    world_offset_x = 0
    world_offset_y = 0
```

---

## Usage Examples

### Basic Usage

```python
from diabot.navigation import MinimapSLAM, SLAMVisualizer

# Initialize
slam = MinimapSLAM(
    map_size=4096,
    movement_threshold=2.0,
    loop_closure_threshold=0.85,
    debug=True
)

# Main loop
while game_running:
    # Capture minimap
    minimap = screen_capture.get_minimap()
    
    # Update SLAM
    slam.update(minimap)
    
    # Add detected POIs
    if npc_detected:
        slam.add_poi("npc", npc_local_pos, confidence=0.9)
    
    # Get stats
    stats = slam.get_stats()
    print(f"Known area: {stats['known_cells']} cells")
```

### With Visualization

```python
visualizer = SLAMVisualizer()

while game_running:
    minimap = screen_capture.get_minimap()
    skeleton = slam.preprocess_minimap(minimap)
    
    slam.update(minimap)
    
    # Create dashboard
    dashboard = visualizer.create_dashboard(
        slam=slam,
        current_minimap=minimap,
        skeleton=skeleton,
        dx=slam.world_offset_x,
        dy=slam.world_offset_y
    )
    
    visualizer.show(dashboard)
```

### Save/Load Maps

```python
# Save
slam.save_map("rogue_encampment.npz")

# Load
slam_loaded = MinimapSLAM()
slam_loaded.load_map("rogue_encampment.npz")
```

---

## Navigation Integration

### A* Pathfinding

```python
def astar(start, goal, global_map):
    # Use occupancy grid for pathfinding
    # WALL = infinite cost
    # UNKNOWN = high cost
    # FREE = low cost
    pass
```

### Return to POI

```python
def navigate_to_poi(slam, poi_type):
    # Find POI in global coordinates
    target_poi = find_poi(slam.current_level.pois, poi_type)
    
    if target_poi:
        # Convert global POI position to movement commands
        path = plan_path(slam.player_center, target_poi.pos, slam.global_map)
        execute_path(path)
```

### Frontier Exploration

```python
def find_frontiers(global_map):
    # Find boundaries between FREE and UNKNOWN
    frontiers = []
    for y in range(map_size):
        for x in range(map_size):
            if global_map[y, x] == FREE:
                # Check neighbors for UNKNOWN
                if has_unknown_neighbor(global_map, x, y):
                    frontiers.append((x, y))
    return frontiers
```

---

## Performance Characteristics

### Computational Complexity

| Operation | Complexity | Notes |
|-----------|-----------|-------|
| Preprocessing | O(N) | N = minimap pixels (~40k) |
| Phase Correlation | O(N log N) | FFT-based |
| Map Fusion | O(N) | Per-pixel update |
| Loop Closure | O(S) | S = signature count |
| Total per frame | **~10-20ms** | On modern CPU |

### Memory Usage

| Component | Size | Notes |
|-----------|------|-------|
| Global map (4096²) | ~16 MB | int8 array |
| Signatures (1000) | ~1 MB | Compressed |
| POIs (100) | ~10 KB | Small |
| **Total** | **~20 MB** | Per level |

### Accuracy

| Metric | Value | Improvement |
|--------|-------|------------|
| Position drift | ±5-10 px per 100 frames | Loop closure: ±2 px |
| POI localization | ±3 px | Re-observation: ±1 px |
| Wall alignment | 95% correct | After fusion |

---

## Debugging

### Visualization Modes

1. **Minimap Processing**: See preprocessing steps
2. **Local Map**: View area around player
3. **Global Map**: See entire explored area
4. **Dashboard**: All visualizations combined

### Debug Flags

```python
slam = MinimapSLAM(debug=True)  # Enable verbose output
```

Output:
```
[MinimapSLAM] Motion detected: dx=5.2, dy=-2.1
[MinimapSLAM] World offset now: (-52, 21)
[MinimapSLAM] Loop closure detected! Similarity=0.89
[MinimapSLAM] Drift corrected: (-3, 2)
```

### Common Issues

**Issue:** High drift accumulation  
**Solution:** Increase signature capture frequency

**Issue:** False loop closures  
**Solution:** Increase similarity threshold

**Issue:** Motion jitter  
**Solution:** Increase movement threshold

**Issue:** Wall gaps in map  
**Solution:** Adjust preprocessing parameters

---

## Advanced Topics

### Multi-Scale SLAM

Combine minimap SLAM with full-screen structural features:
```python
# Coarse: minimap (global consistency)
# Fine: screen corners/doors (local precision)
```

### Semantic Mapping

Classify areas by type:
```python
area_types = {
    'town': high_npc_density,
    'dungeon': high_wall_density,
    'outdoor': low_wall_density
}
```

### Collaborative SLAM

Merge maps from multiple bot instances:
```python
def merge_maps(map1, map2):
    # Align using POI correspondences
    # Fuse occupancy grids
    # Combine POI lists
    pass
```

---

## References

### Key Algorithms

- **Phase Correlation**: [Reddy & Chatterji, 1996]
- **Loop Closure**: [Cummins & Newman, 2008]
- **Occupancy Grids**: [Moravec & Elfes, 1985]

### Related Work

- **ORB-SLAM2**: [Mur-Artal et al., 2017]
- **Cartographer**: [Hess et al., 2016]
- **RTAB-Map**: [Labbé & Michaud, 2013]

---

## License

See LICENSE file in repository root.

## Contributing

See CONTRIBUTING.md for guidelines.

---

## Appendix: Coordinate Systems

### Minimap Coordinates
- Origin: Top-left corner
- X: Right
- Y: Down
- Size: Typically 200x200 pixels

### Global Coordinates
- Origin: Arbitrary (map center)
- X: Right
- Y: Down
- Size: 4096x4096 cells (configurable)

### Conversion Formula
```python
global_x = player_center_x + world_offset_x + (local_x - minimap_center_x)
global_y = player_center_y + world_offset_y + (local_y - minimap_center_y)
```

---

**End of Documentation**
