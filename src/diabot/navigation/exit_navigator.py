"""
Exit navigator - finds and navigates toward likely exits.

Uses accumulated map data to identify unexplored corridors and edges
that are likely to contain zone exits.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np

from .map_accumulator import MapAccumulator
from .minimap_processor import MinimapGrid


@dataclass
class ExitCandidate:
    """
    A potential exit location.
    
    Attributes:
        position: (x, y) in global map coordinates
        score: Exit likelihood score (higher = more likely)
        direction: Direction to exit from player (in degrees)
        distance: Distance from player in cells
    """
    position: Tuple[int, int]
    score: float
    direction: float
    distance: float


class ExitNavigator:
    """
    Finds and navigates toward likely zone exits.
    
    Strategy:
    1. Analyze accumulated map for exit candidates
    2. Score candidates by:
       - Edge proximity (unexplored neighbors)
       - Corridor shape (long narrow passages)
       - Distance from starting position
    3. Convert best candidate to minimap click target
    """
    
    def __init__(self, debug: bool = False):
        """
        Initialize exit navigator.
        
        Args:
            debug: Enable debug output
        """
        self.debug = debug
    
    def find_exit_candidates(
        self,
        map_accumulator: MapAccumulator,
        max_candidates: int = 5
    ) -> List[ExitCandidate]:
        """
        Find likely exit positions in accumulated map.
        
        Args:
            map_accumulator: The accumulated world map
            max_candidates: Maximum number of candidates to return
            
        Returns:
            List of exit candidates sorted by score (best first)
        """
        # Get likely exits from map accumulator
        exit_positions = map_accumulator.find_likely_exits(search_radius=40)
        
        if not exit_positions:
            if self.debug:
                print("[ExitNavigator] No exit candidates found")
            return []
        
        # Score each candidate
        candidates = []
        px, py = map_accumulator.player_world_pos
        
        for ex, ey in exit_positions:
            # Calculate distance
            dx = ex - px
            dy = ey - py
            distance = np.sqrt(dx*dx + dy*dy)
            
            # Calculate direction (angle in degrees)
            direction = np.degrees(np.arctan2(dy, dx))
            
            # Score based on:
            # 1. Distance (prefer farther = more explored)
            # 2. Edge proximity (already in find_likely_exits)
            distance_score = min(50, distance) / 50  # Normalize to 0-1
            
            # Get unknown neighbor count
            unknown_count = 0
            for ndy in [-1, 0, 1]:
                for ndx in [-1, 0, 1]:
                    nx, ny = ex + ndx, ey + ndy
                    if not map_accumulator.is_explored(nx, ny):
                        unknown_count += 1
            
            edge_score = unknown_count / 8.0  # Normalize to 0-1
            
            # Combined score
            score = edge_score * 0.7 + distance_score * 0.3
            
            candidates.append(ExitCandidate(
                position=(ex, ey),
                score=score,
                direction=direction,
                distance=distance
            ))
        
        # Sort by score descending
        candidates.sort(key=lambda c: c.score, reverse=True)
        
        if self.debug:
            print(f"[ExitNavigator] Found {len(candidates)} exit candidates")
            for i, cand in enumerate(candidates[:3]):
                print(f"  #{i+1}: pos={cand.position}, score={cand.score:.2f}, "
                      f"dir={cand.direction:.0f}°, dist={cand.distance:.1f}")
        
        return candidates[:max_candidates]
    
    def get_navigation_target(
        self,
        exit_candidate: ExitCandidate,
        map_accumulator: MapAccumulator,
        minimap_grid: MinimapGrid
    ) -> Optional[Tuple[float, float]]:
        """
        Convert exit candidate to minimap click target.
        
        Args:
            exit_candidate: The exit to navigate toward
            map_accumulator: Accumulated world map
            minimap_grid: Current minimap grid
            
        Returns:
            (rel_x, rel_y) click position on minimap (0.0-1.0 range), or None
        """
        # Get exit position in global coords
        ex, ey = exit_candidate.position
        
        # Get player position in global coords
        px, py = map_accumulator.player_world_pos
        
        # Calculate offset from player
        dx = ex - px
        dy = ey - py
        
        # Get minimap dimensions
        h, w = minimap_grid.shape
        minimap_cx, minimap_cy = minimap_grid.center
        
        # Convert global offset to minimap coordinates
        # (assuming 1:1 scale between grid and minimap)
        target_x = minimap_cx + dx
        target_y = minimap_cy + dy
        
        # Check if target is on minimap
        if not (0 <= target_x < w and 0 <= target_y < h):
            # Target is off minimap - navigate in direction
            # Clamp to minimap edge
            if self.debug:
                print(f"[ExitNavigator] Target off minimap, using direction {exit_candidate.direction:.0f}°")
            
            # Use direction to find edge point
            angle_rad = np.radians(exit_candidate.direction)
            
            # Project to minimap edge
            edge_dist = min(w, h) // 2 - 5  # Stay inside minimap
            target_x = minimap_cx + int(edge_dist * np.cos(angle_rad))
            target_y = minimap_cy + int(edge_dist * np.sin(angle_rad))
            
            # Clamp to minimap bounds
            target_x = max(0, min(w - 1, target_x))
            target_y = max(0, min(h - 1, target_y))
        
        # Check if target is walkable
        if not minimap_grid.is_free(target_x, target_y):
            # Find nearest free cell
            if self.debug:
                print(f"[ExitNavigator] Target blocked, searching for free path")
            
            # Search in expanding radius
            for radius in range(1, 10):
                for dy in range(-radius, radius + 1):
                    for dx in range(-radius, radius + 1):
                        nx = target_x + dx
                        ny = target_y + dy
                        if minimap_grid.is_free(nx, ny):
                            target_x = nx
                            target_y = ny
                            break
                    else:
                        continue
                    break
                else:
                    continue
                break
        
        # Convert to relative coordinates (0.0-1.0)
        rel_x = target_x / w
        rel_y = target_y / h
        
        # Clamp to safe range (avoid UI edges)
        rel_x = max(0.1, min(0.9, rel_x))
        rel_y = max(0.1, min(0.9, rel_y))
        
        if self.debug:
            print(f"[ExitNavigator] Click target: ({rel_x:.2f}, {rel_y:.2f}) "
                  f"grid=({target_x}, {target_y})")
        
        return (rel_x, rel_y)
    
    def should_explore_instead(
        self,
        map_accumulator: MapAccumulator,
        exploration_threshold: float = 0.3
    ) -> bool:
        """
        Decide if bot should explore instead of seeking exit.
        
        Args:
            map_accumulator: Accumulated map
            exploration_threshold: Min fraction of map to explore before seeking exit
            
        Returns:
            True if should explore more, False if should seek exit
        """
        # Count explored cells near player
        px, py = map_accumulator.player_world_pos
        radius = 50
        
        total_checked = 0
        explored_count = 0
        
        for y in range(py - radius, py + radius):
            for x in range(px - radius, px + radius):
                if not (0 <= x < map_accumulator.map_size and 
                       0 <= y < map_accumulator.map_size):
                    continue
                total_checked += 1
                if map_accumulator.is_explored(x, y):
                    explored_count += 1
        
        if total_checked == 0:
            return True
        
        explored_ratio = explored_count / total_checked
        
        should_explore = explored_ratio < exploration_threshold
        
        if self.debug:
            print(f"[ExitNavigator] Explored {explored_ratio:.1%} of nearby area "
                  f"(threshold {exploration_threshold:.1%})")
            print(f"[ExitNavigator] Decision: {'EXPLORE' if should_explore else 'SEEK EXIT'}")
        
        return should_explore
