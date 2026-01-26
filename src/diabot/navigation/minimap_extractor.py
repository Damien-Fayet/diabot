"""
Minimap extraction module.

Extracts the minimap region from a full game frame.
Cross-platform, operates only on numpy arrays.
"""

from __future__ import annotations

from typing import Tuple

import cv2
import numpy as np

from ..vision.screen_regions import UI_REGIONS


class MinimapExtractor:
    """
    Extract minimap region from full game frame.
    
    Uses predefined screen regions to locate and crop the minimap.
    The minimap is assumed to be in the top-right corner of the screen.
    """
    
    def __init__(self, debug: bool = False, fullscreen_mode: bool = True):
        """
        Initialize minimap extractor.
        
        Args:
            debug: Enable debug output
            fullscreen_mode: If True, extract entire screen (Tab fullscreen minimap)
                           If False, extract top-right corner (normal minimap)
        """
        self.debug = debug
        self.fullscreen_mode = fullscreen_mode
        
        # Get minimap region from UI layout (only used if not fullscreen)
        if 'minimap_ui' not in UI_REGIONS:
            raise ValueError("Minimap region not defined in UI_REGIONS")
        
        self.minimap_region = UI_REGIONS['minimap_ui']
        
        if self.debug:
            mode = "FULLSCREEN" if fullscreen_mode else "normal"
            print(f"[MinimapExtractor] Initialized with mode: {mode}")
    
    def extract(self, frame: np.ndarray) -> np.ndarray:
        """
        Extract minimap from full game frame.
        
        Args:
            frame: Full game frame (BGR numpy array)
            
        Returns:
            Cropped minimap image (BGR numpy array)
            
        Raises:
            ValueError: If frame is invalid
        """
        if frame is None or frame.size == 0:
            raise ValueError("Frame is empty or None")
        
        if len(frame.shape) != 3:
            raise ValueError(f"Frame must be 3-channel BGR image, got shape {frame.shape}")
        
        # In fullscreen mode, extract entire screen (minus small margins)
        if self.fullscreen_mode:
            h, w = frame.shape[:2]
            # Remove 5% margins to exclude UI borders
            margin_x = int(w * 0.05)
            margin_y = int(h * 0.05)
            minimap = frame[margin_y:h-margin_y, margin_x:w-margin_x].copy()
        else:
            # Extract minimap using region definition (top-right corner)
            minimap = self.minimap_region.extract_from_frame(frame)
        
        if minimap.size == 0:
            raise ValueError("Extracted minimap is empty")
        
        if self.debug:
            h, w = minimap.shape[:2]
            print(f"[MinimapExtractor] Extracted minimap: {w}x{h}")
            # Save extracted minimap
            from pathlib import Path
            output_dir = Path("data/screenshots/outputs/diagnostic/minimap_steps")
            output_dir.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(output_dir / "step1_extracted.png"), minimap)
        
        return minimap
    
    def get_minimap_bounds(self, frame_height: int, frame_width: int) -> Tuple[int, int, int, int]:
        """
        Get pixel bounds of minimap in frame.
        
        Args:
            frame_height: Frame height in pixels
            frame_width: Frame width in pixels
            
        Returns:
            (x, y, width, height) in pixels
        """
        return self.minimap_region.get_bounds(frame_height, frame_width)
