"""
Screen regions definition for Diablo 2 layout.

Diablo 2 has a specific UI layout:
- Top-left: Health/Mana bars
- Top-right: Minimap
- Bottom: Inventory, spells, etc.
- Center: Playfield (where the action happens)

This module defines the regions so we can separate UI from Environment.
"""

from dataclasses import dataclass
from typing import Tuple


@dataclass
class ScreenRegion:
    """Represents a region of the screen defined by ratios."""
    
    name: str
    x_ratio: float      # 0.0-1.0, where on screen horizontally
    y_ratio: float      # 0.0-1.0, where on screen vertically
    w_ratio: float      # 0.0-1.0, width as % of screen
    h_ratio: float      # 0.0-1.0, height as % of screen
    
    def get_bounds(self, frame_height: int, frame_width: int) -> Tuple[int, int, int, int]:
        """
        Convert ratio-based region to pixel bounds.
        
        Returns:
            (x, y, width, height) in pixels
        """
        x = int(self.x_ratio * frame_width)
        y = int(self.y_ratio * frame_height)
        w = int(self.w_ratio * frame_width)
        h = int(self.h_ratio * frame_height)
        
        return (x, y, w, h)
    
    def extract_from_frame(self, frame):
        """Extract this region from a frame."""
        import numpy as np
        
        if not isinstance(frame, np.ndarray):
            raise ValueError("Frame must be numpy array")
        
        frame_h, frame_w = frame.shape[:2]
        x, y, w, h = self.get_bounds(frame_h, frame_w)
        
        # Clamp to frame bounds
        x = max(0, min(x, frame_w - 1))
        y = max(0, min(y, frame_h - 1))
        x_end = min(x + w, frame_w)
        y_end = min(y + h, frame_h)
        
        return frame[y:y_end, x:x_end]


# Standard Diablo 2 screen layout
# (These are based on typical 1024x768, but use ratios so they scale)

UI_REGIONS = {
    'top_left_ui': ScreenRegion(
        name='top_left_ui',
        x_ratio=0.0,
        y_ratio=0.0,
        w_ratio=0.15, 
        h_ratio=0.4,     
    ),
    
    'minimap_ui': ScreenRegion(
        name='minimap_ui',
        x_ratio=0.68,
        y_ratio=0.1,
        w_ratio=0.31,
        h_ratio=0.35,
    ),
    
    'lifebar_ui': ScreenRegion(
        name='lifebar_ui',
        x_ratio=0.2,      # Text region above HP orb (wider for 999/999)
        y_ratio=0.833,      # Text region above HP orb
        w_ratio=0.1,      # Text region above HP orb (wider for 999/999)
        h_ratio=0.022,      # Text region above HP orb
    ),
    'manabar_ui': ScreenRegion(
        name='manabar_ui',
        x_ratio=0.71,      # Text region above Mana orb (wider for 999/999)
        y_ratio=0.837,      # Text region above Mana orb
        w_ratio=0.1,      # Text region above Mana orb (wider for 999/999)
        h_ratio=0.019,      # Text region above Mana orb
    ),
    'zone_ui': ScreenRegion(
        name='zone_ui',
        x_ratio=0.75,
        y_ratio=0.05,
        w_ratio=0.4,
        h_ratio=0.06,
    ),
}

ENVIRONMENT_REGIONS = {
    'playfield': ScreenRegion(
        name='playfield',
        x_ratio=0.0,
        y_ratio=0.05,
        w_ratio=1.0,      # Full width
        h_ratio=0.85,      
    ),
    'inventory_ui': ScreenRegion(
        name='inventory_ui',
        x_ratio=0.75,
        y_ratio=0.6,
        w_ratio=0.25,
        h_ratio=0.4,
    ),
}

ALL_REGIONS = {**UI_REGIONS, **ENVIRONMENT_REGIONS}
