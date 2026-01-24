"""Minimap parsing utilities.

Goal: convert minimap into a coarse occupancy grid and extract anchors
(player marker, points of interest) for navigation.

This is a vision-only stub that can be refined once we have more samples.
"""
from dataclasses import dataclass, field
from typing import Dict, List, Tuple

import cv2
import numpy as np

from .screen_regions import UI_REGIONS


@dataclass
class Landmark:
    kind: str  # "portal", "waypoint", "stairs", "door", etc.
    position: Tuple[int, int]  # grid coordinates (cx, cy)
    score: float = 1.0


@dataclass
class MinimapParseResult:
    grid: np.ndarray  # uint8 grid: 0 = unknown, 1 = free, 255 = blocked
    player_pos: Tuple[int, int]
    landmarks: List[Landmark] = field(default_factory=list)


class MinimapParser:
    """Parse minimap region into an occupancy grid.

    This implementation is intentionally simple and deterministic:
    - Extract minimap UI region (fallback to full frame if missing)
    - Convert to grayscale; bright pixels assumed walkable; dark = walls
    - Detect player marker as brightest blob; project to grid
    - Landmarks detection left as placeholders (color masks to be calibrated)
    """

    def __init__(self, grid_size: int = 96):
        self.grid_size = grid_size  # square grid resolution

    def parse(self, frame: np.ndarray) -> MinimapParseResult:
        if frame is None or frame.size == 0:
            raise ValueError("Frame is empty")

        minimap = self._extract_minimap(frame)
        if minimap.size == 0:
            # return empty grid to avoid crashes
            return MinimapParseResult(
                grid=np.zeros((self.grid_size, self.grid_size), dtype=np.uint8),
                player_pos=(self.grid_size // 2, self.grid_size // 2),
                landmarks=[],
            )

        # Convert to grayscale for simple thresholding
        gray = cv2.cvtColor(minimap, cv2.COLOR_BGR2GRAY) if minimap.shape[2] == 3 else minimap

        # Normalize and threshold: bright = walkable, dark = wall/unknown
        norm = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX)
        _, walkable = cv2.threshold(norm, 180, 255, cv2.THRESH_BINARY)

        # Resize to fixed grid
        grid = cv2.resize(walkable, (self.grid_size, self.grid_size), interpolation=cv2.INTER_AREA)
        grid = grid.astype(np.uint8)

        # Player marker: brightest spot
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(norm)
        player_x = int(max_loc[0] * self.grid_size / norm.shape[1])
        player_y = int(max_loc[1] * self.grid_size / norm.shape[0])
        player_pos = (player_x, player_y)

        # Placeholder landmarks (to be calibrated later)
        landmarks: List[Landmark] = []

        return MinimapParseResult(grid=grid, player_pos=player_pos, landmarks=landmarks)

    def _extract_minimap(self, frame: np.ndarray) -> np.ndarray:
        # Try UI minimap region; fall back to full frame
        if 'minimap_ui' in UI_REGIONS:
            region = UI_REGIONS['minimap_ui']
            x, y, w, h = region.get_bounds(frame.shape[0], frame.shape[1])
            return frame[y:y + h, x:x + w]
        return frame
