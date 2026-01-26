"""
Local map building module.

Maintains an incremental 2D occupancy grid of the explored area.
Updates based on minimap observations as the player moves.
Provides frontier detection for exploration.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple, Optional
from collections import deque

import cv2
import numpy as np

from .minimap_processor import CellType


@dataclass
class Frontier:
    """
    A frontier cell for exploration.
    
    Frontiers are FREE cells adjacent to UNKNOWN cells.
    They represent the boundary of explored vs unexplored space.
    """
    position: Tuple[int, int]  # (x, y) in map coordinates
    distance: float = 0.0      # Distance from player (for prioritization)
    neighbors_unknown: int = 0  # Number of unknown neighbors


class LocalMap:
    """
    Incremental 2D occupancy grid map.
    
    Maintains a local map centered on the player's estimated position.
    Updates from minimap observations as the player explores.
    Tracks visited cells and provides frontier detection.
    
    The map uses a fixed-size grid that moves with the player.
    When the player moves far from center, the map shifts to recenter.
    
    Coordinate system:
    - Map coordinates are in grid cells
    - Origin (0, 0) is top-left of map
    - Player position is tracked in map coordinates
    """
    
    def __init__(
        self,
        map_size: int = 200,
        recenter_threshold: int = 40,
        debug: bool = False
    ):
        """
        Initialize local map.
        
        Args:
            map_size: Size of map grid (map_size x map_size)
            recenter_threshold: Distance from center before recentering
            debug: Enable debug output
        """
        self.map_size = map_size
        self.recenter_threshold = recenter_threshold
        self.debug = debug
        
        # Initialize empty map (all unknown)
        self.grid = np.full((map_size, map_size), CellType.UNKNOWN, dtype=np.uint8)
        
        # Player starts at center
        self.player_x = map_size // 2
        self.player_y = map_size // 2
        
        # Track visited cells
        self.visited = np.zeros((map_size, map_size), dtype=bool)
        
        # Offset for map coordinate system (for recentering)
        self.offset_x = 0
        self.offset_y = 0
        
        if self.debug:
            print(f"[LocalMap] Initialized {map_size}x{map_size} map")
    
    def update_from_minimap(
        self,
        minimap_grid: np.ndarray,
        player_map_pos: Tuple[int, int]
    ):
        """
        Update local map with new minimap observation.
        
        The minimap grid is centered on the player's current position.
        We overlay it onto the local map at the player's coordinates.
        
        Args:
            minimap_grid: Binary occupancy grid from minimap
            player_map_pos: Current player position in map coordinates (x, y)
        """
        if minimap_grid is None or minimap_grid.size == 0:
            return
        
        self.player_x, self.player_y = player_map_pos
        
        # Mark current position as visited
        if self._in_bounds(self.player_x, self.player_y):
            self.visited[self.player_y, self.player_x] = True
        
        # Get minimap dimensions
        mm_h, mm_w = minimap_grid.shape
        mm_cx = mm_w // 2
        mm_cy = mm_h // 2
        
        # Calculate overlay region in map coordinates
        # Minimap center aligns with player position
        map_x_start = self.player_x - mm_cx
        map_y_start = self.player_y - mm_cy
        
        # Overlay minimap onto local map
        for mm_y in range(mm_h):
            for mm_x in range(mm_w):
                map_x = map_x_start + mm_x
                map_y = map_y_start + mm_y
                
                if self._in_bounds(map_x, map_y):
                    # Only update if minimap has information (not UNKNOWN)
                    mm_value = minimap_grid[mm_y, mm_x]
                    if mm_value != CellType.UNKNOWN:
                        self.grid[map_y, map_x] = mm_value
        
        # Check if recentering is needed
        self._check_recenter()
        
        if self.debug:
            print(f"[LocalMap] Updated at ({self.player_x}, {self.player_y})")
    
    def mark_visited(self, x: int, y: int):
        """
        Mark a cell as visited.
        
        Args:
            x: X coordinate in map
            y: Y coordinate in map
        """
        if self._in_bounds(x, y):
            self.visited[y, x] = True
    
    def get_frontiers(self, max_frontiers: int = 20) -> List[Frontier]:
        """
        Find frontier cells for exploration.
        
        Frontiers are FREE cells adjacent to UNKNOWN cells.
        Returns closest frontiers to player for exploration targeting.
        
        Args:
            max_frontiers: Maximum number of frontiers to return
            
        Returns:
            List of Frontier objects, sorted by distance from player
        """
        frontiers = []
        
        # Search in a reasonable radius around player
        search_radius = min(50, self.map_size // 4)
        
        x_min = max(0, self.player_x - search_radius)
        x_max = min(self.map_size, self.player_x + search_radius)
        y_min = max(0, self.player_y - search_radius)
        y_max = min(self.map_size, self.player_y + search_radius)
        
        for y in range(y_min, y_max):
            for x in range(x_min, x_max):
                # Must be free space
                if self.grid[y, x] != CellType.FREE:
                    continue
                
                # Check if adjacent to unknown
                unknown_neighbors = self._count_unknown_neighbors(x, y)
                if unknown_neighbors > 0:
                    # Calculate distance from player
                    dx = x - self.player_x
                    dy = y - self.player_y
                    distance = np.sqrt(dx * dx + dy * dy)
                    
                    frontiers.append(Frontier(
                        position=(x, y),
                        distance=distance,
                        neighbors_unknown=unknown_neighbors
                    ))
        
        # Sort by distance (closest first)
        frontiers.sort(key=lambda f: f.distance)
        
        if self.debug and frontiers:
            print(f"[LocalMap] Found {len(frontiers)} frontiers")
        
        return frontiers[:max_frontiers]
    
    def find_path(
        self,
        start: Tuple[int, int],
        goal: Tuple[int, int]
    ) -> Optional[List[Tuple[int, int]]]:
        """
        Find path from start to goal using BFS.
        
        Simple breadth-first search for path finding.
        Only considers FREE cells as walkable.
        
        Args:
            start: Start position (x, y)
            goal: Goal position (x, y)
            
        Returns:
            List of (x, y) waypoints from start to goal, or None if no path
        """
        if not self._in_bounds(*start) or not self._in_bounds(*goal):
            return None
        
        # BFS setup
        queue = deque([start])
        came_from = {start: None}
        
        directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]  # 4-connected
        
        while queue:
            current = queue.popleft()
            
            if current == goal:
                # Reconstruct path
                path = []
                while current is not None:
                    path.append(current)
                    current = came_from[current]
                path.reverse()
                return path
            
            cx, cy = current
            
            for dx, dy in directions:
                next_x = cx + dx
                next_y = cy + dy
                next_pos = (next_x, next_y)
                
                # Check if valid and not visited
                if next_pos in came_from:
                    continue
                
                if not self._in_bounds(next_x, next_y):
                    continue
                
                # Must be free space
                if self.grid[next_y, next_x] != CellType.FREE:
                    continue
                
                queue.append(next_pos)
                came_from[next_pos] = current
        
        # No path found
        return None
    
    def get_cell(self, x: int, y: int) -> int:
        """Get cell value at position."""
        if self._in_bounds(x, y):
            return self.grid[y, x]
        return CellType.UNKNOWN
    
    def visualize(self, show_frontiers: bool = True) -> np.ndarray:
        """
        Create visualization of local map.
        
        Args:
            show_frontiers: Whether to highlight frontier cells
            
        Returns:
            BGR image for display
        """
        # Create color visualization
        vis = np.zeros((self.map_size, self.map_size, 3), dtype=np.uint8)
        
        # Color coding
        vis[self.grid == CellType.FREE] = [200, 200, 200]      # Light gray
        vis[self.grid == CellType.WALL] = [50, 50, 50]         # Dark gray
        vis[self.grid == CellType.UNKNOWN] = [0, 0, 0]         # Black
        
        # Highlight visited cells with green tint
        visited_mask = self.visited
        if np.any(visited_mask):
            # Blend with green for visited cells
            vis[visited_mask] = (vis[visited_mask] * 0.7 + np.array([100, 150, 100], dtype=np.uint8) * 0.3).astype(np.uint8)
        
        # Show frontiers
        if show_frontiers:
            frontiers = self.get_frontiers()
            for frontier in frontiers:
                fx, fy = frontier.position
                if self._in_bounds(fx, fy):
                    vis[fy, fx] = [0, 255, 255]  # Cyan for frontiers
        
        # Draw player position
        if self._in_bounds(self.player_x, self.player_y):
            cv2.circle(vis, (self.player_x, self.player_y), 2, (0, 255, 0), -1)
        
        return vis
    
    def _in_bounds(self, x: int, y: int) -> bool:
        """Check if coordinates are within map bounds."""
        return 0 <= x < self.map_size and 0 <= y < self.map_size
    
    def _count_unknown_neighbors(self, x: int, y: int) -> int:
        """Count unknown neighbors (4-connected)."""
        count = 0
        for dx, dy in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
            nx, ny = x + dx, y + dy
            if self._in_bounds(nx, ny):
                if self.grid[ny, nx] == CellType.UNKNOWN:
                    count += 1
        return count
    
    def _check_recenter(self):
        """
        Check if map needs recentering.
        
        If player moves too far from center, shift the map
        to keep player near center.
        """
        center_x = self.map_size // 2
        center_y = self.map_size // 2
        
        dx = self.player_x - center_x
        dy = self.player_y - center_y
        distance_from_center = np.sqrt(dx * dx + dy * dy)
        
        if distance_from_center > self.recenter_threshold:
            # Shift map to recenter player
            shift_x = dx
            shift_y = dy
            
            new_grid = np.full((self.map_size, self.map_size), CellType.UNKNOWN, dtype=np.uint8)
            new_visited = np.zeros((self.map_size, self.map_size), dtype=bool)
            
            # Copy existing data with offset
            for y in range(self.map_size):
                for x in range(self.map_size):
                    old_x = x + shift_x
                    old_y = y + shift_y
                    
                    if self._in_bounds(old_x, old_y):
                        new_grid[y, x] = self.grid[old_y, old_x]
                        new_visited[y, x] = self.visited[old_y, old_x]
            
            self.grid = new_grid
            self.visited = new_visited
            self.player_x = center_x
            self.player_y = center_y
            self.offset_x += shift_x
            self.offset_y += shift_y
            
            if self.debug:
                print(f"[LocalMap] Recentered map (offset: {self.offset_x}, {self.offset_y})")
