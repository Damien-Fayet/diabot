"""Navigation grid and pathfinding utilities."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import heapq
import numpy as np


@dataclass
class PathResult:
    path: List[Tuple[int, int]]  # list of (x, y) grid coords
    cost: float
    success: bool


class OccupancyGrid:
    """Simple occupancy grid: 0 free, >0 blocked/unknown."""

    def __init__(self, grid: np.ndarray):
        if grid.ndim != 2:
            raise ValueError("Grid must be 2D")
        self.grid = grid.astype(np.uint8)
        self.w = grid.shape[1]
        self.h = grid.shape[0]

    def is_free(self, x: int, y: int) -> bool:
        if x < 0 or y < 0 or x >= self.w or y >= self.h:
            return False
        return self.grid[y, x] == 0 or self.grid[y, x] == 1  # 0 unknown->treated as free, 1 walkable

    def neighbors(self, x: int, y: int) -> List[Tuple[int, int]]:
        res = []
        for dx, dy in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
            nx, ny = x + dx, y + dy
            if self.is_free(nx, ny):
                res.append((nx, ny))
        return res

    def frontier_cells(self) -> List[Tuple[int, int]]:
        """Cells that are free and adjacent to unknown (>1) are frontier targets."""
        frontiers: List[Tuple[int, int]] = []
        for y in range(self.h):
            for x in range(self.w):
                if self.is_free(x, y):
                    for dx, dy in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
                        nx, ny = x + dx, y + dy
                        if 0 <= nx < self.w and 0 <= ny < self.h and self.grid[ny, nx] > 1:
                            frontiers.append((x, y))
                            break
        return frontiers


class AStarPlanner:
    """Basic A* on occupancy grid."""

    def __init__(self, grid: OccupancyGrid):
        self.grid = grid

    def heuristic(self, a: Tuple[int, int], b: Tuple[int, int]) -> float:
        return abs(a[0] - b[0]) + abs(a[1] - b[1])

    def plan(self, start: Tuple[int, int], goal: Tuple[int, int]) -> PathResult:
        if not self.grid.is_free(*start):
            return PathResult([], float('inf'), False)
        if not self.grid.is_free(*goal):
            return PathResult([], float('inf'), False)

        open_set: List[Tuple[float, Tuple[int, int]]] = []
        heapq.heappush(open_set, (0, start))

        came_from: Dict[Tuple[int, int], Optional[Tuple[int, int]]] = {start: None}
        g_score: Dict[Tuple[int, int], float] = {start: 0}

        while open_set:
            _, current = heapq.heappop(open_set)
            if current == goal:
                path = self._reconstruct(came_from, current)
                return PathResult(path=path, cost=g_score[current], success=True)

            for neighbor in self.grid.neighbors(*current):
                tentative_g = g_score[current] + 1
                if tentative_g < g_score.get(neighbor, float('inf')):
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g
                    f = tentative_g + self.heuristic(neighbor, goal)
                    heapq.heappush(open_set, (f, neighbor))

        return PathResult([], float('inf'), False)

    @staticmethod
    def _reconstruct(came_from: Dict[Tuple[int, int], Optional[Tuple[int, int]]],
                      current: Tuple[int, int]) -> List[Tuple[int, int]]:
        path = [current]
        while came_from[current] is not None:
            current = came_from[current]
            path.append(current)
        path.reverse()
        return path
