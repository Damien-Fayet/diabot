"""
Navigation debug visualization utilities.

Provides visual overlays for debugging navigation system:
- Local map visualization
- Player pose and path
- Frontier targets
- Minimap processing results
"""

from __future__ import annotations

from typing import Tuple, Optional, List

import cv2
import numpy as np

from .local_map import LocalMap
from .pose_estimator import Pose
from .frontier_navigator import NavigationState, NavigationAction
from .minimap_processor import MinimapGrid, CellType


class NavigationOverlay:
    """
    Debug visualization for navigation system.
    
    Draws navigation state on top of game frame for debugging:
    - Local map with explored areas
    - Player position and orientation
    - Current path and target
    - Frontier cells
    - Navigation action
    """
    
    def __init__(
        self,
        show_local_map: bool = True,
        show_path: bool = True,
        show_frontiers: bool = True,
        show_minimap_grid: bool = False
    ):
        """
        Initialize navigation overlay.
        
        Args:
            show_local_map: Show local map visualization
            show_path: Show planned path
            show_frontiers: Show frontier cells
            show_minimap_grid: Show processed minimap grid
        """
        self.show_local_map = show_local_map
        self.show_path = show_path
        self.show_frontiers = show_frontiers
        self.show_minimap_grid = show_minimap_grid
    
    def draw(
        self,
        frame: np.ndarray,
        nav_state: NavigationState,
        local_map: Optional[LocalMap] = None,
        minimap_grid: Optional[MinimapGrid] = None
    ) -> np.ndarray:
        """
        Draw navigation overlay on frame.
        
        Args:
            frame: Game frame (BGR)
            nav_state: Current navigation state
            local_map: Local map object (optional)
            minimap_grid: Processed minimap grid (optional)
            
        Returns:
            Frame with overlay drawn
        """
        overlay = frame.copy()
        
        # Draw text info
        self._draw_text_info(overlay, nav_state)
        
        # Draw local map visualization
        if self.show_local_map and local_map is not None:
            self._draw_local_map(overlay, local_map, nav_state)
        
        # Draw minimap grid
        if self.show_minimap_grid and minimap_grid is not None:
            self._draw_minimap_grid(overlay, minimap_grid)
        
        return overlay
    
    def _draw_text_info(
        self,
        frame: np.ndarray,
        nav_state: NavigationState
    ):
        """Draw text information overlay."""
        height, width = frame.shape[:2]
        
        # Background rectangle for text
        cv2.rectangle(
            frame,
            (10, 10),
            (350, 180),
            (0, 0, 0),
            -1
        )
        cv2.rectangle(
            frame,
            (10, 10),
            (350, 180),
            (0, 255, 255),
            2
        )
        
        # Text content
        y_offset = 35
        line_height = 25
        
        texts = [
            f"Action: {nav_state.action.value}",
            f"Position: ({nav_state.current_position[0]}, {nav_state.current_position[1]})",
            f"Angle: {nav_state.current_angle:.1f}°",
            f"Target: {nav_state.target_position if nav_state.target_position else 'None'}",
            f"Frontiers: {nav_state.frontiers_available}",
            f"Progress: {nav_state.exploration_progress * 100:.1f}%",
        ]
        
        for i, text in enumerate(texts):
            cv2.putText(
                frame,
                text,
                (20, y_offset + i * line_height),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 255),
                1,
                cv2.LINE_AA
            )
    
    def _draw_local_map(
        self,
        frame: np.ndarray,
        local_map: LocalMap,
        nav_state: NavigationState
    ):
        """Draw local map visualization in corner."""
        # Create local map visualization
        map_vis = local_map.visualize(show_frontiers=self.show_frontiers)
        
        # Draw path on map if available
        if self.show_path and nav_state.path:
            for i in range(len(nav_state.path) - 1):
                p1 = nav_state.path[i]
                p2 = nav_state.path[i + 1]
                cv2.line(map_vis, p1, p2, (255, 255, 0), 1)
        
        # Draw target on map
        if nav_state.target_position:
            tx, ty = nav_state.target_position
            cv2.circle(map_vis, (tx, ty), 3, (0, 0, 255), -1)
        
        # Scale up for visibility
        display_size = 300
        map_display = cv2.resize(
            map_vis,
            (display_size, display_size),
            interpolation=cv2.INTER_NEAREST
        )
        
        # Position in bottom-left corner
        height, width = frame.shape[:2]
        x_pos = 10
        y_pos = height - display_size - 10
        
        # Add to frame
        frame[y_pos:y_pos + display_size, x_pos:x_pos + display_size] = map_display
        
        # Add border
        cv2.rectangle(
            frame,
            (x_pos, y_pos),
            (x_pos + display_size, y_pos + display_size),
            (0, 255, 0),
            2
        )
        
        # Add label
        cv2.putText(
            frame,
            "Local Map",
            (x_pos + 5, y_pos - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2,
            cv2.LINE_AA
        )
    
    def _draw_minimap_grid(
        self,
        frame: np.ndarray,
        minimap_grid: MinimapGrid
    ):
        """Draw processed minimap grid visualization."""
        # Create visualization
        vis = np.zeros((minimap_grid.grid.shape[0], minimap_grid.grid.shape[1], 3), dtype=np.uint8)
        
        # Color coding
        vis[minimap_grid.grid == CellType.FREE] = [200, 200, 200]
        vis[minimap_grid.grid == CellType.WALL] = [50, 50, 50]
        vis[minimap_grid.grid == CellType.UNKNOWN] = [0, 0, 0]
        
        # Draw player at center
        cx, cy = minimap_grid.center
        cv2.circle(vis, (cx, cy), 2, (0, 255, 0), -1)
        
        # Scale up
        display_size = 200
        grid_display = cv2.resize(
            vis,
            (display_size, display_size),
            interpolation=cv2.INTER_NEAREST
        )
        
        # Position in top-right corner
        height, width = frame.shape[:2]
        x_pos = width - display_size - 10
        y_pos = 10
        
        # Add to frame
        frame[y_pos:y_pos + display_size, x_pos:x_pos + display_size] = grid_display
        
        # Add border
        cv2.rectangle(
            frame,
            (x_pos, y_pos),
            (x_pos + display_size, y_pos + display_size),
            (255, 0, 255),
            2
        )
        
        # Add label
        cv2.putText(
            frame,
            "Minimap Grid",
            (x_pos + 5, y_pos - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 0, 255),
            2,
            cv2.LINE_AA
        )


def draw_pose_arrow(
    frame: np.ndarray,
    pose: Pose,
    position_px: Tuple[int, int],
    length: int = 30,
    color: Tuple[int, int, int] = (0, 255, 0)
):
    """
    Draw arrow showing pose orientation.
    
    Args:
        frame: Frame to draw on
        pose: Pose with angle
        position_px: Position in frame pixels (x, y)
        length: Arrow length in pixels
        color: Arrow color (BGR)
    """
    # Calculate arrow end point
    angle_rad = np.radians(pose.angle)
    end_x = int(position_px[0] + length * np.cos(angle_rad))
    end_y = int(position_px[1] + length * np.sin(angle_rad))
    
    # Draw arrow
    cv2.arrowedLine(
        frame,
        position_px,
        (end_x, end_y),
        color,
        2,
        tipLength=0.3
    )


def draw_path_on_frame(
    frame: np.ndarray,
    path: List[Tuple[int, int]],
    map_to_frame_transform,
    color: Tuple[int, int, int] = (255, 255, 0),
    thickness: int = 2
):
    """
    Draw planned path on frame.
    
    Args:
        frame: Frame to draw on
        path: List of (x, y) waypoints in map coordinates
        map_to_frame_transform: Function to convert map coords to frame pixels
        color: Path color (BGR)
        thickness: Line thickness
    """
    if not path or len(path) < 2:
        return
    
    for i in range(len(path) - 1):
        p1_map = path[i]
        p2_map = path[i + 1]
        
        p1_frame = map_to_frame_transform(p1_map)
        p2_frame = map_to_frame_transform(p2_map)
        
        cv2.line(frame, p1_frame, p2_frame, color, thickness)
