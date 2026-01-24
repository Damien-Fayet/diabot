"""High-level navigation utilities: goal selection and planning."""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np

from .grid import AStarPlanner, OccupancyGrid, PathResult


@dataclass
class NavigationGoal:
    kind: str  # "frontier", "landmark", "target"
    position: Tuple[int, int]
    priority: int = 1


class Navigator:
    """Combines occupancy grid, goal selection, and path planning."""

    def __init__(self, grid: np.ndarray):
        self.occ = OccupancyGrid(grid)
        self.astar = AStarPlanner(self.occ)

    def plan_to(self, start: Tuple[int, int], goal: NavigationGoal) -> PathResult:
        return self.astar.plan(start, goal.position)

    def pick_frontier(self, start: Tuple[int, int]) -> Optional[NavigationGoal]:
        frontiers = self.occ.frontier_cells()
        if not frontiers:
            return None
        # Pick closest frontier (Manhattan distance)
        fx, fy = min(frontiers, key=lambda p: abs(p[0]-start[0]) + abs(p[1]-start[1]))
        return NavigationGoal(kind="frontier", position=(fx, fy), priority=1)


class GoalManager:
    """Simple goal selector between landmark goals and exploration frontier."""

    def __init__(self):
        self.current: Optional[NavigationGoal] = None

    def choose_goal(self,
                    start: Tuple[int, int],
                    landmarks: List[Tuple[str, Tuple[int, int]]],
                    navigator: Navigator) -> Optional[NavigationGoal]:
        # Prefer landmarks if any
        if landmarks:
            kind, pos = landmarks[0]
            return NavigationGoal(kind=kind, position=pos, priority=0)
        # Else frontier exploration
        return navigator.pick_frontier(start)
