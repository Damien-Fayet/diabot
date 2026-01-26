"""
Minimap processing module.

Converts minimap image to binary occupancy grid (walls vs free space).
Uses computer vision to detect walkable paths and obstacles.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
from typing import Tuple

import cv2
import numpy as np


class CellType(IntEnum):
    """Cell types in occupancy grid."""
    UNKNOWN = 0      # Not yet observed
    FREE = 128       # Walkable space
    WALL = 255       # Obstacle/wall


@dataclass
class MinimapGrid:
    """
    Binary occupancy grid representation of minimap.
    
    Attributes:
        grid: 2D array where each cell is WALL, FREE, or UNKNOWN
        center: (x, y) coordinates of player position in grid (always center)
        cell_size: Size of each grid cell in pixels (from original minimap)
    """
    grid: np.ndarray
    center: Tuple[int, int]
    cell_size: float
    
    @property
    def shape(self) -> Tuple[int, int]:
        """Get grid shape (height, width)."""
        return self.grid.shape
    
    def is_free(self, x: int, y: int) -> bool:
        """Check if cell is walkable."""
        if 0 <= y < self.grid.shape[0] and 0 <= x < self.grid.shape[1]:
            return self.grid[y, x] == CellType.FREE
        return False
    
    def is_wall(self, x: int, y: int) -> bool:
        """Check if cell is a wall."""
        if 0 <= y < self.grid.shape[0] and 0 <= x < self.grid.shape[1]:
            return self.grid[y, x] == CellType.WALL
        return False


class MinimapProcessor:
    """
    Process minimap images into occupancy grids.
    
    Converts RGB minimap to binary grid representation:
    - Detects walls using brightness and color thresholds
    - Identifies free/walkable space
    - Handles noise with morphological operations
    
    Assumptions:
    - Minimap shows walls as dark/black
    - Walkable paths are lighter (brown, tan, or visible floor)
    - Player is always at center of minimap
    """
    
    def __init__(
        self,
        grid_size: int = 64,
        wall_threshold: int = 49,
        debug: bool = False
    ):
        """
        Initialize minimap processor with optimized parameters.
        
        Args:
            grid_size: Size of output grid (grid_size x grid_size)
            wall_threshold: Brightness threshold for wall detection (0-255)
                          D2R walls are BRIGHT (white/gray), free space is DARK
                          Optimized: 49 from parameter tuning
            debug: Enable debug output
        """
        self.grid_size = grid_size
        self.wall_threshold = wall_threshold
        self.debug = debug
        
        # Optimized parameters from tuning (minimap_tuned_params.txt)
        self.crop_bottom_pct = 21  # Remove HUD
        self.tophat_kernel = 5     # Extract bright structures (walls)
        self.gamma = 3.0           # Strong contrast enhancement
        self.clahe_clip = 3.9      # Local contrast
        self.clahe_grid = 8        # CLAHE grid size
        
        # Morphological kernels (optimized from tuning)
        self.open_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
        self.close_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (8, 8))
        
        if self.debug:
            print(f"[MinimapProcessor] Grid: {grid_size}, Threshold: {wall_threshold}")
            print(f"[MinimapProcessor] Crop: {self.crop_bottom_pct}%, Gamma: {self.gamma}, TopHat: {self.tophat_kernel}")
    
    def process(self, minimap: np.ndarray) -> MinimapGrid:
        """
        Process minimap image to occupancy grid.
        
        Args:
            minimap: Minimap image (BGR numpy array)
            
        Returns:
            MinimapGrid with binary occupancy data
            
        Raises:
            ValueError: If minimap is invalid
        """
        if minimap is None or minimap.size == 0:
            raise ValueError("Minimap is empty or None")
        
        # Convert to grayscale
        if len(minimap.shape) == 3:
            gray = cv2.cvtColor(minimap, cv2.COLOR_BGR2GRAY)
        else:
            gray = minimap
        
        # Debug: save grayscale
        if self.debug:
            from pathlib import Path
            output_dir = Path("data/screenshots/outputs/diagnostic/minimap_steps")
            output_dir.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(output_dir / "step2_grayscale.png"), gray)
        
        # STEP 0: Crop bottom to remove HUD
        if self.crop_bottom_pct > 0:
            h = gray.shape[0]
            crop_pixels = int(h * self.crop_bottom_pct / 100)
            gray = gray[:-crop_pixels, :] if crop_pixels < h else gray
        
        if self.debug:
            cv2.imwrite(str(output_dir / "step0_cropped.png"), gray)
        
        # STEP 1: Top Hat - Extract bright structures (walls)
        kernel_tophat = cv2.getStructuringElement(cv2.MORPH_RECT, (self.tophat_kernel, self.tophat_kernel))
        tophat = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, kernel_tophat)
        gray = cv2.add(gray, tophat)
        
        if self.debug:
            cv2.imwrite(str(output_dir / "step1_tophat.png"), gray)
        
        # STEP 2: Gamma correction (strong contrast enhancement)
        normalized = gray.astype(np.float32) / 255.0
        contrast_enhanced = np.power(normalized, self.gamma)
        contrast_enhanced = (contrast_enhanced * 255).astype(np.uint8)
        
        if self.debug:
            cv2.imwrite(str(output_dir / "step2_gamma.png"), contrast_enhanced)
        
        # STEP 3: CLAHE (local contrast enhancement)
        clahe = cv2.createCLAHE(clipLimit=self.clahe_clip, tileGridSize=(self.clahe_grid, self.clahe_grid))
        contrast_enhanced = clahe.apply(contrast_enhanced)
        
        if self.debug:
            cv2.imwrite(str(output_dir / "step3_clahe.png"), contrast_enhanced)
        
        # STEP 4: Bilateral filter to reduce noise while preserving edges
        filtered = cv2.bilateralFilter(contrast_enhanced, 5, 50, 50)
        
        # Debug: save filtered
        if self.debug:
            cv2.imwrite(str(output_dir / "step4_filtered.png"), filtered)
        
        # Threshold: BRIGHT pixels are walls, DARK pixels are free
        # D2R minimap: walls = white/light gray, walkable = dark
        # Using THRESH_BINARY: pixels > threshold = 255 (walls)
        _, binary = cv2.threshold(
            filtered,
            self.wall_threshold,
            255,
            cv2.THRESH_BINARY  # Changed from THRESH_BINARY_INV
        )
        
        # Debug: save binary threshold
        if self.debug:
            cv2.imwrite(str(output_dir / "step5_binary_threshold.png"), binary)
        
        # Morphological operations to remove noise
        # Opening removes small bright spots (noise in walls)
        opened = cv2.morphologyEx(binary, cv2.MORPH_OPEN, self.open_kernel)
        
        # Debug: save after opening
        if self.debug:
            cv2.imwrite(str(output_dir / "step6_morphology_open.png"), opened)
        
        # Closing fills small dark holes (noise in free space)
        cleaned = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, self.close_kernel)
        
        # Debug: save after closing
        if self.debug:
            cv2.imwrite(str(output_dir / "step7_morphology_close.png"), cleaned)
        
        # Resize to grid size
        resized = cv2.resize(
            cleaned,
            (self.grid_size, self.grid_size),
            interpolation=cv2.INTER_AREA
        )
        
        # Debug: save resized
        if self.debug:
            cv2.imwrite(str(output_dir / "step8_resized_grid.png"), resized)
        
        # Convert to occupancy grid
        # Binary image: 255 = wall (bright in original), 0 = free (dark in original)
        grid = np.where(resized > 127, CellType.WALL, CellType.FREE).astype(np.uint8)
        
        # Debug: Save processed minimap with walls visualization
        if self.debug:
            debug_img = cv2.cvtColor(resized, cv2.COLOR_GRAY2BGR)
            # Color walls in red, free space in green
            wall_mask = grid == CellType.WALL
            debug_img[wall_mask] = [0, 0, 255]  # Red for walls
            debug_img[~wall_mask] = [0, 255, 0]  # Green for free
            # Save to outputs folder
            cv2.imwrite(str(output_dir / "step9_final_grid_colored.png"), debug_img)
            cv2.imwrite("data/screenshots/outputs/minimap_walls_debug.png", debug_img)
        
        # Player is always at center of minimap
        center = (self.grid_size // 2, self.grid_size // 2)
        
        # Calculate cell size (how many pixels per grid cell)
        cell_size = minimap.shape[1] / self.grid_size
        
        if self.debug:
            wall_count = np.sum(grid == CellType.WALL)
            free_count = np.sum(grid == CellType.FREE)
            print(f"[MinimapProcessor] Grid: {free_count} free, {wall_count} walls")
        
        return MinimapGrid(
            grid=grid,
            center=center,
            cell_size=cell_size
        )
    
    def visualize(self, minimap_grid: MinimapGrid) -> np.ndarray:
        """
        Create visualization of occupancy grid for debugging.
        
        Args:
            minimap_grid: Processed minimap grid
            
        Returns:
            BGR image showing the grid with color coding
        """
        # Create color visualization
        vis = np.zeros((minimap_grid.grid.shape[0], minimap_grid.grid.shape[1], 3), dtype=np.uint8)
        
        # Color coding
        vis[minimap_grid.grid == CellType.FREE] = [200, 200, 200]      # Light gray for free
        vis[minimap_grid.grid == CellType.WALL] = [50, 50, 50]         # Dark gray for walls
        vis[minimap_grid.grid == CellType.UNKNOWN] = [0, 0, 0]         # Black for unknown
        
        # Draw player position (center) as green circle
        cx, cy = minimap_grid.center
        cv2.circle(vis, (cx, cy), 2, (0, 255, 0), -1)
        
        # Scale up for better visibility
        vis = cv2.resize(vis, (256, 256), interpolation=cv2.INTER_NEAREST)
        
        return vis
