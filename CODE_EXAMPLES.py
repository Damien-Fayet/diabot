"""
Code Examples: NPC Detection & Distance Tracking

This file shows the key code snippets that implement the improvements.
"""

# ============================================================================
# 1. DISTANCE CALCULATION
# ============================================================================

def _calculate_distance(pos1, pos2):
    """Calculate straight-line distance between two points in pixels.
    
    Args:
        pos1: (x, y) tuple
        pos2: (x, y) tuple
    
    Returns:
        Distance in pixels
    """
    x1, y1 = pos1
    x2, y2 = pos2
    return ((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5

# Usage:
# distance = _calculate_distance((100, 200), (150, 300))
# >>> 111.8 pixels


# ============================================================================
# 2. NPC DETECTION UNDER MARKER
# ============================================================================

def _find_npc_under_marker(quest_marker, detections):
    """Find NPC geometrically positioned under quest marker.
    
    Algorithm:
    1. Get marker position and bottom edge
    2. Filter detections to only NPC classes
    3. For each NPC, score based on:
       - Vertical gap from marker bottom (primary)
       - Horizontal alignment (secondary)
    4. Return lowest-scoring NPC (best match)
    """
    marker_x, marker_y = quest_marker.center
    marker_bottom = quest_marker.bbox[3]  # y2
    
    # NPC class whitelist
    npc_classes = ["akara", "kashya", "warriv", "cain", "charsi", 
                   "stash", "waypoint", "hero"]
    
    npcs = [d for d in detections if d.class_name in npc_classes]
    
    best_npc = None
    best_score = float('inf')
    
    for npc in npcs:
        npc_x, npc_y = npc.center
        npc_top = npc.bbox[1]  # y1
        
        # NPC should be below marker (allow slight overlap)
        vertical_gap = npc_top - marker_bottom
        if vertical_gap < -20:
            continue
        
        # Horizontal alignment preference
        horizontal_distance = abs(npc_x - marker_x)
        
        # Combined score (lower is better)
        vertical_score = max(0, vertical_gap)
        horizontal_score = horizontal_distance * 0.5
        score = vertical_score + horizontal_score
        
        if score < best_score:
            best_score = score
            best_npc = npc
    
    return best_npc

# Usage:
# if quest_marker:
#     npc = _find_npc_under_marker(quest_marker, detections)
#     if npc:
#         print(f"Found {npc.class_name} under marker")
#         # Click on NPC instead of marker


# ============================================================================
# 3. DISTANCE TRACKING
# ============================================================================

class DistanceTracker:
    """Track hero-to-NPC distance and movement trends."""
    
    def __init__(self):
        self.last_distance = 0.0
        self.last_hero_pos = None
        self.last_npc_pos = None
        self.distance_decreasing_frames = 0
    
    def update(self, hero_det, npc_det):
        """Update distance tracking with new detections.
        
        Args:
            hero_det: Hero detection or None
            npc_det: NPC detection or None
        """
        if not hero_det or not npc_det:
            return
        
        # Calculate new distance
        current_distance = self._distance_between(
            hero_det.center, npc_det.center
        )
        
        # Track trend
        if self.last_distance > 0 and current_distance < self.last_distance:
            self.distance_decreasing_frames += 1
        else:
            self.distance_decreasing_frames = 0
        
        # Store for next frame
        self.last_distance = current_distance
        self.last_hero_pos = hero_det.center
        self.last_npc_pos = npc_det.center
    
    def get_status(self):
        """Get human-readable distance status.
        
        Returns:
            Dictionary with distance info
        """
        delta = 0  # Would compute delta from stored history
        trend = "APPROACHING" if self.distance_decreasing_frames > 0 else "STABLE"
        
        return {
            "distance": self.last_distance,
            "trend": trend,
            "approaching_frames": self.distance_decreasing_frames,
            "hero_pos": self.last_hero_pos,
            "npc_pos": self.last_npc_pos,
        }
    
    @staticmethod
    def _distance_between(pos1, pos2):
        x1, y1 = pos1
        x2, y2 = pos2
        return ((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5

# Usage:
# tracker = DistanceTracker()
# for frame in frames:
#     detections = model.detect(frame)
#     hero = find_by_class(detections, "hero")
#     npc = find_by_class(detections, "kashya")
#     
#     tracker.update(hero, npc)
#     status = tracker.get_status()
#     print(f"Distance: {status['distance']:.0f}px, "
#           f"Trend: {status['trend']}")


# ============================================================================
# 4. CLICK ON NPC BBOX (NOT MARKER)
# ============================================================================

def execute_npc_interaction(npc_detection, window_rect, executor):
    """Click on NPC bounding box center.
    
    Args:
        npc_detection: Detection object with bbox and center
        window_rect: (x, y, right, bottom) of game window in screen coords
        executor: ActionExecutor instance
    """
    # Get NPC center in window-relative coordinates
    npc_x, npc_y = npc_detection.center
    
    # Convert to screen-absolute coordinates
    window_x, window_y = window_rect[:2]
    screen_x = window_x + int(npc_x)
    screen_y = window_y + int(npc_y)
    
    # Click on NPC (not marker)
    executor.interact_with_object(screen_x, screen_y)
    
    return screen_x, screen_y

# Usage:
# rect = screen_capture.get_window_rect()
# screen_coords = execute_npc_interaction(npc, rect, executor)
# print(f"Clicked at {screen_coords}")


# ============================================================================
# 5. FULL INTEGRATION IN BOT STEP
# ============================================================================

def bot_step_example(screen_capture, detector, executor, tracker):
    """Example bot step with all features integrated.
    
    Flow:
    1. Capture frame
    2. Run YOLO detection
    3. Find quest marker and NPC under it
    4. Find hero
    5. Track distance to NPC
    6. Decide and execute interaction
    """
    # Step 1: Capture
    frame = screen_capture.get_frame()
    if not frame:
        return False
    
    # Step 2: Detect
    detections = detector.detect(frame)
    
    # Step 3: Find quest targets
    quest_markers = [d for d in detections if d.class_name == "quest"]
    if not quest_markers:
        print("No quest marker -> exploring")
        return True
    
    quest_marker = quest_markers[0]
    
    # Step 4: Find NPC under marker
    npc = find_npc_under_marker(quest_marker, detections)
    if not npc:
        print(f"Quest marker at {quest_marker.center}, no NPC found")
        return True
    
    # Step 5: Find hero
    heroes = [d for d in detections if d.class_name == "hero"]
    hero = heroes[0] if heroes else None
    
    # Step 6: Track distance
    tracker.update(hero, npc)
    status = tracker.get_status()
    
    print(f"Quest: {quest_marker.class_name}")
    print(f"NPC:   {npc.class_name} at {npc.center}")
    if hero:
        print(f"Hero:  at {hero.center}")
        print(f"Distance: {status['distance']:.0f}px "
              f"({status['trend']}, {status['approaching_frames']} frames)")
    
    # Step 7: Interact
    rect = screen_capture.get_window_rect()
    execute_npc_interaction(npc, rect, executor)
    
    return True

# Usage:
# while True:
#     if not bot_step_example(capture, detector, executor, tracker):
#         break
