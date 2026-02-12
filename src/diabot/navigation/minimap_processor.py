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
    - Uses background subtraction to isolate minimap content
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
        wall_threshold: int = 30,
        debug: bool = False,
        use_background_subtraction: bool = True
    ):
        """
        Initialize minimap processor with optimized parameters.
        
        Args:
            grid_size: Size of output grid (grid_size x grid_size)
            wall_threshold: Brightness threshold for wall detection (0-255)
                          With background subtraction: 30 (tuned for difference image)
                          Without: 49 (legacy dungeon optimization)
            debug: Enable debug output
            use_background_subtraction: Enable automatic background subtraction
        """
        self.grid_size = grid_size
        self.wall_threshold = wall_threshold
        self.debug = debug
        self.use_background_subtraction = use_background_subtraction
        
        # Background reference frame (captured without minimap)
        self.background_frame: np.ndarray | None = None
        
        # Optimized parameters for background-subtracted processing
        if self.use_background_subtraction:
            self.blur_size = 3
            self.morph_open = 2
            self.morph_close = 3
            self.gamma = 1.0
            self.clahe_clip = 2.0
        else:
            # Legacy parameters from tuning (minimap_tuned_params.txt)
            self.crop_bottom_pct = 21  # Remove HUD
            self.tophat_kernel = 5     # Extract bright structures (walls)
            self.gamma = 3.0           # Strong contrast enhancement
            self.clahe_clip = 3.9      # Local contrast
            self.clahe_grid = 8        # CLAHE grid size
        
        # Morphological kernels
        if self.use_background_subtraction:
            self.open_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (self.morph_open, self.morph_open))
            self.close_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (self.morph_close, self.morph_close))
        else:
            self.open_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
            self.close_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (8, 8))
        
        if self.debug:
            mode = "BackgroundSub" if self.use_background_subtraction else "Legacy"
            print(f"[MinimapProcessor] Mode: {mode}, Grid: {grid_size}, Threshold: {wall_threshold}")
    
    def set_background(self, frame: np.ndarray):
        """
        Set background reference frame (captured without minimap).
        
        This should be called once when the bot starts, with a frame
        captured while the minimap is hidden (Tab key not pressed).
        
        Args:
            frame: Full screen capture without minimap overlay
        """
        self.background_frame = frame.copy()
        if self.debug:
            print("[MinimapProcessor] Background reference frame set")
    
    def process(self, minimap: np.ndarray, full_frame: np.ndarray | None = None) -> MinimapGrid:
        """
        Process minimap image to occupancy grid.
        
        Args:
            minimap: Minimap image (BGR numpy array)
            full_frame: Full screen frame with minimap (for background subtraction)
            
        Returns:
            MinimapGrid with binary occupancy data
            
        Raises:
            ValueError: If minimap is invalid
        """
        if minimap is None or minimap.size == 0:
            raise ValueError("Minimap is empty or None")
        
        # If using background subtraction and we have both frames
        if self.use_background_subtraction and self.background_frame is not None and full_frame is not None:
            # Subtract background to isolate minimap content
            diff = cv2.absdiff(full_frame, self.background_frame)
            
            # Extract grayscale from difference
            if len(diff.shape) == 3:
                gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
            else:
                gray = diff
            
            if self.debug:
                from pathlib import Path
                output_dir = Path("data/screenshots/outputs/diagnostic/minimap_steps")
                output_dir.mkdir(parents=True, exist_ok=True)
                cv2.imwrite(str(output_dir / "step0_diff.png"), gray)
        else:
            # Legacy mode: use minimap directly
            if len(minimap.shape) == 3:
                gray = cv2.cvtColor(minimap, cv2.COLOR_BGR2GRAY)
            else:
                gray = minimap
            
            if self.debug:
                from pathlib import Path
                output_dir = Path("data/screenshots/outputs/diagnostic/minimap_steps")
                output_dir.mkdir(parents=True, exist_ok=True)
                cv2.imwrite(str(output_dir / "step0_grayscale.png"), gray)
        
        # Branch: Background subtraction or legacy processing
        if self.use_background_subtraction:
            processed = self._process_with_background_sub(gray)
        else:
            processed = self._process_legacy(gray)
        
        # Resize to grid size
        resized = cv2.resize(
            processed,
            (self.grid_size, self.grid_size),
            interpolation=cv2.INTER_AREA
        )
        
        # Debug: save resized
        if self.debug:
            cv2.imwrite(str(output_dir / "step8_resized_grid.png"), resized)
        
        # Convert to occupancy grid
        # Binary image: 255 = wall (bright in difference), 0 = free (dark)
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
    
    def _process_with_background_sub(self, diff_gray: np.ndarray) -> np.ndarray:
        """
        Process background-subtracted minimap (NEW METHOD).
        
        Starting from clean difference image, apply light processing
        to detect walls (bright areas in difference).
        
        Args:
            diff_gray: Grayscale difference image
            
        Returns:
            Binary image (255=wall, 0=free)
        """
        if self.debug:
            from pathlib import Path
            output_dir = Path("data/screenshots/outputs/diagnostic/minimap_steps")
        
        # Step 1: Gamma correction (optional, for visibility)
        if self.gamma != 1.0:
            inv_gamma = 1.0 / self.gamma
            table = np.array([((i / 255.0) ** inv_gamma) * 255 
                             for i in np.arange(0, 256)]).astype("uint8")
            processed = cv2.LUT(diff_gray, table)
            
            if self.debug:
                cv2.imwrite(str(output_dir / "step1_gamma.png"), processed)
        else:
            processed = diff_gray
        
        # Step 2: CLAHE (optional, for local contrast)
        if self.clahe_clip > 0:
            clahe = cv2.createCLAHE(
                clipLimit=self.clahe_clip,
                tileGridSize=(8, 8)
            )
            processed = clahe.apply(processed)
            
            if self.debug:
                cv2.imwrite(str(output_dir / "step2_clahe.png"), processed)
        
        # Step 3: Blur to reduce noise
        if self.blur_size > 1:
            processed = cv2.GaussianBlur(
                processed,
                (self.blur_size, self.blur_size),
                0
            )
            
            if self.debug:
                cv2.imwrite(str(output_dir / "step3_blur.png"), processed)
        
        # Step 4: Threshold (bright = walls)
        _, binary = cv2.threshold(
            processed,
            self.wall_threshold,
            255,
            cv2.THRESH_BINARY
        )
        
        if self.debug:
            cv2.imwrite(str(output_dir / "step4_threshold.png"), binary)
        
        # Step 5: Opening (remove small noise)
        if self.morph_open > 0:
            binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, self.open_kernel)
            
            if self.debug:
                cv2.imwrite(str(output_dir / "step5_open.png"), binary)
        
        # Step 6: Closing (fill small gaps)
        if self.morph_close > 0:
            binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, self.close_kernel)
            
            if self.debug:
                cv2.imwrite(str(output_dir / "step6_close.png"), binary)
        
        return binary
    
    def _process_legacy(self, gray: np.ndarray) -> np.ndarray:
        """
        Process minimap using legacy method (for dungeons).
        
        Uses TopHat, strong gamma, CLAHE - optimized for dungeon walls.
        
        Args:
            gray: Grayscale minimap
            
        Returns:
            Binary image (255=wall, 0=free)
        """
        if self.debug:
            from pathlib import Path
            output_dir = Path("data/screenshots/outputs/diagnostic/minimap_steps")
        
        if self.debug:
            from pathlib import Path
            output_dir = Path("data/screenshots/outputs/diagnostic/minimap_steps")
        
        # STEP 0: Crop bottom to remove HUD
        if hasattr(self, 'crop_bottom_pct') and self.crop_bottom_pct > 0:
            h = gray.shape[0]
            crop_pixels = int(h * self.crop_bottom_pct / 100)
            gray = gray[:-crop_pixels, :] if crop_pixels < h else gray
        
        if self.debug:
            cv2.imwrite(str(output_dir / "step0_legacy_cropped.png"), gray)
        
        # STEP 1: Top Hat - Extract bright structures (walls)
        if hasattr(self, 'tophat_kernel'):
            kernel_tophat = cv2.getStructuringElement(cv2.MORPH_RECT, (self.tophat_kernel, self.tophat_kernel))
            tophat = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, kernel_tophat)
            gray = cv2.add(gray, tophat)
            
            if self.debug:
                cv2.imwrite(str(output_dir / "step1_legacy_tophat.png"), gray)
            if self.debug:
                cv2.imwrite(str(output_dir / "step1_legacy_tophat.png"), gray)
        
        # STEP 2: Gamma correction (strong contrast enhancement)
        normalized = gray.astype(np.float32) / 255.0
        contrast_enhanced = np.power(normalized, self.gamma)
        contrast_enhanced = (contrast_enhanced * 255).astype(np.uint8)
        
        if self.debug:
            cv2.imwrite(str(output_dir / "step2_legacy_gamma.png"), contrast_enhanced)
        
        # STEP 3: CLAHE (local contrast enhancement)
        clahe = cv2.createCLAHE(clipLimit=self.clahe_clip, tileGridSize=(self.clahe_grid, self.clahe_grid))
        contrast_enhanced = clahe.apply(contrast_enhanced)
        
        if self.debug:
            cv2.imwrite(str(output_dir / "step3_legacy_clahe.png"), contrast_enhanced)
        
        # STEP 4: Bilateral filter to reduce noise while preserving edges
        filtered = cv2.bilateralFilter(contrast_enhanced, 5, 50, 50)
        
        if self.debug:
            cv2.imwrite(str(output_dir / "step4_legacy_filtered.png"), filtered)
        
        # Threshold: BRIGHT pixels are walls, DARK pixels are free
        _, binary = cv2.threshold(
            filtered,
            self.wall_threshold,
            255,
            cv2.THRESH_BINARY
        )
        
        if self.debug:
            cv2.imwrite(str(output_dir / "step5_legacy_threshold.png"), binary)
        
        # Morphological operations
        opened = cv2.morphologyEx(binary, cv2.MORPH_OPEN, self.open_kernel)
        
        if self.debug:
            cv2.imwrite(str(output_dir / "step6_legacy_open.png"), opened)
        
        cleaned = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, self.close_kernel)
        
        if self.debug:
            cv2.imwrite(str(output_dir / "step7_legacy_close.png"), cleaned)
        
        return cleaned
    
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
