"""
Distance and NPC tracking utilities for quest progression.
"""

from typing import Optional, List, Tuple

class DistanceTracker:
    """Track hero-to-NPC distance and movement trends."""
    def __init__(self):
        self.last_distance = 0.0
        self.last_hero_pos = None
        self.last_npc_pos = None
        self.distance_decreasing_frames = 0

    def update(self, hero_det, npc_det):
        if not hero_det or not npc_det:
            return
        current_distance = self._distance_between(hero_det.center, npc_det.center)
        if self.last_distance > 0 and current_distance < self.last_distance:
            self.distance_decreasing_frames += 1
        else:
            self.distance_decreasing_frames = 0
        self.last_distance = current_distance
        self.last_hero_pos = hero_det.center
        self.last_npc_pos = npc_det.center

    def get_status(self):
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

def find_npc_under_marker(quest_marker, detections):
    marker_x, marker_y = quest_marker.center
    marker_bottom = quest_marker.bbox[3]
    npc_classes = ["akara", "kashya", "warriv", "cain", "charsi", "stash", "waypoint", "hero"]
    npcs = [d for d in detections if d.class_name in npc_classes]
    best_npc = None
    best_score = float('inf')
    for npc in npcs:
        npc_x, npc_y = npc.center
        npc_top = npc.bbox[1]
        vertical_gap = npc_top - marker_bottom
        if vertical_gap < -20:
            continue
        horizontal_distance = abs(npc_x - marker_x)
        vertical_score = max(0, vertical_gap)
        horizontal_score = horizontal_distance * 0.5
        score = vertical_score + horizontal_score
        if score < best_score:
            best_score = score
            best_npc = npc
    return best_npc
