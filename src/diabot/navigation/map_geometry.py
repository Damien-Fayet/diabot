"""
Minimap geometry extraction: build a local occupancy grid (walls/limits) relative to player.

Input: minimap image (BGR), player marker is the blue cross on minimap.
Output: occupancy grid (0=free, 1=wall), player position in pixels and grid cells, debug mask.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np


@dataclass
class GeometryResult:
    """Result of minimap geometry extraction."""
    player_px: Optional[Tuple[int, int]]
    player_cell: Optional[Tuple[int, int]]
    occupancy: np.ndarray  # uint8 grid, 0 = free, 1 = wall
    cell_size: int
    wall_mask: np.ndarray  # binary mask same size as minimap


class MinimapGeometryExtractor:
    """Extract walls/limits from minimap and align them to the player position."""

    def __init__(self, cell_size: int = 4, wall_threshold: float = 0.25, debug: bool = False):
        """
        Args:
            cell_size: Size (in pixels) of one grid cell when downsampling the minimap.
            wall_threshold: Fraction of dark pixels to consider a cell as wall.
            debug: Enable verbose logging.
        """
        self.cell_size = cell_size
        self.wall_threshold = wall_threshold
        self.debug = debug

        # HSV ranges to find the blue player cross
        self.player_hsv_ranges = [
            (np.array([105, 120, 120]), np.array([130, 255, 255])),
            (np.array([90, 100, 120]), np.array([110, 255, 255])),
        ]

    def _detect_player(self, minimap: np.ndarray) -> Optional[Tuple[int, int]]:
        """Detect the blue player cross on the minimap.

        Returns:
            (x, y) in pixel coordinates, or None if not found.
        """
        hsv = cv2.cvtColor(minimap, cv2.COLOR_BGR2HSV)
        combined = np.zeros(hsv.shape[:2], dtype=np.uint8)
        for lower, upper in self.player_hsv_ranges:
            mask = cv2.inRange(hsv, lower, upper)
            combined = cv2.bitwise_or(combined, mask)

        combined = cv2.morphologyEx(combined, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        contours, _ = cv2.findContours(combined, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None

        # Pick the largest contour as player marker
        contour = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(contour)
        if area < 5:  # too small
            return None

        M = cv2.moments(contour)
        if M["m00"] == 0:
            return None
        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])
        return (cx, cy)

    def _build_wall_mask(self, minimap: np.ndarray) -> np.ndarray:
        """Convert minimap to a binary wall mask (1=wall, 0=free)."""
        gray = cv2.cvtColor(minimap, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (3, 3), 0)
        # Otsu threshold: bright lines/shapes (walls/blocked) -> 255, dark background -> 0
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        wall_mask = cv2.morphologyEx(binary, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        wall_mask = cv2.morphologyEx(wall_mask, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
        wall_mask = (wall_mask > 0).astype(np.uint8)
        return wall_mask

    def _downsample_to_grid(self, wall_mask: np.ndarray) -> np.ndarray:
        """Downsample wall mask to occupancy grid."""
        h, w = wall_mask.shape
        cell = self.cell_size
        grid_h = (h + cell - 1) // cell
        grid_w = (w + cell - 1) // cell
        grid = np.zeros((grid_h, grid_w), dtype=np.uint8)

        for gy in range(grid_h):
            for gx in range(grid_w):
                y1 = gy * cell
                x1 = gx * cell
                y2 = min(y1 + cell, h)
                x2 = min(x1 + cell, w)
                patch = wall_mask[y1:y2, x1:x2]
                wall_fraction = patch.mean()
                # wall_fraction is 0..1 since mask is 0/1
                if wall_fraction >= self.wall_threshold:
                    grid[gy, gx] = 1
        return grid

    def extract(self, minimap: np.ndarray, player_px: Optional[Tuple[int, int]] = None) -> GeometryResult:
        """
        Extract occupancy grid and player position from minimap.

        Args:
            minimap: Minimap image (BGR)
            player_px: Optional pre-detected player position in pixels

        Returns:
            GeometryResult with occupancy grid and player position
        """
        if minimap is None or minimap.size == 0:
            raise ValueError("Minimap is empty")

        if player_px is None:
            player_px = self._detect_player(minimap)
            if self.debug:
                print(f"[GEOM] Player detected: {player_px}")

        wall_mask = self._build_wall_mask(minimap)
        occupancy = self._downsample_to_grid(wall_mask)

        player_cell = None
        if player_px is not None:
            px, py = player_px
            player_cell = (px // self.cell_size, py // self.cell_size)

        return GeometryResult(
            player_px=player_px,
            player_cell=player_cell,
            occupancy=occupancy,
            cell_size=self.cell_size,
            wall_mask=wall_mask,
        )

    def overlay(self, minimap: np.ndarray, geom: GeometryResult) -> np.ndarray:
        """Create a debug overlay showing walls and player position."""
        vis = minimap.copy()

        # Draw walls in red on minimap (mask)
        red = np.zeros_like(vis)
        red[:, :, 2] = 255
        mask = geom.wall_mask > 0
        vis[mask] = cv2.addWeighted(vis, 0.6, red, 0.4, 0)[mask]

        # Draw player
        if geom.player_px:
            cv2.drawMarker(vis, geom.player_px, (255, 255, 0), markerType=cv2.MARKER_CROSS, markerSize=12, thickness=2)

        return vis


if __name__ == "__main__":
    # Quick self-test on a saved minimap
    minimap_path = Path("data/screenshots/outputs/live_capture/minimap_extracted.png")
    if minimap_path.exists():
        img = cv2.imread(str(minimap_path))
        extractor = MinimapGeometryExtractor(debug=True)
        geom = extractor.extract(img)
        overlay = extractor.overlay(img, geom)
        cv2.imwrite("data/maps/minimap_geometry_overlay.png", overlay)
        print("Saved overlay to data/maps/minimap_geometry_overlay.png")
    else:
        print("No minimap found for test")
