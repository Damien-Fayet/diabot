"""
Visualization utilities for MinimapSLAM debugging and monitoring.

Provides:
- Real-time map overlay on game screen
- SLAM statistics display
- Motion vectors
- POI markers
- Loop closure indicators
- Drift correction visualization
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple
import numpy as np
import cv2

from .minimap_slam import MinimapSLAM, OccupancyCell


class SLAMVisualizer:
    """
    Visualize SLAM state for debugging and monitoring.
    
    Shows:
    - Current minimap with preprocessing steps
    - Global map view
    - Motion vectors
    - POIs
    - Statistics overlay
    """
    
    def __init__(
        self,
        window_name: str = "Minimap SLAM",
        map_view_size: int = 600,
        debug: bool = True
    ):
        """
        Initialize visualizer.
        
        Args:
            window_name: OpenCV window name
            map_view_size: Size of map viewport
            debug: Enable debug output
        """
        self.window_name = window_name
        self.map_view_size = map_view_size
        self.debug = debug
        
        # Motion history for visualization
        self.motion_history = []
        self.max_history = 50
        
        # Colors
        self.COLOR_WALL = (200, 200, 200)
        self.COLOR_FREE = (50, 50, 50)
        self.COLOR_UNKNOWN = (0, 0, 0)
        self.COLOR_PLAYER = (0, 255, 255)  # Yellow
        self.COLOR_MOTION = (0, 255, 0)     # Green
        self.COLOR_LOOP = (255, 0, 255)     # Magenta
    
    def visualize_minimap_processing(
        self,
        original: np.ndarray,
        skeleton: np.ndarray
    ) -> np.ndarray:
        """
        Show minimap preprocessing steps side-by-side.
        
        Args:
            original: Original minimap
            skeleton: Processed skeleton
            
        Returns:
            Combined visualization
        """
        # Convert skeleton to BGR
        skeleton_bgr = cv2.cvtColor(skeleton, cv2.COLOR_GRAY2BGR)
        
        # Resize to same height
        h = 200
        w1 = int(original.shape[1] * h / original.shape[0])
        w2 = int(skeleton.shape[1] * h / skeleton.shape[0])
        
        orig_resized = cv2.resize(original, (w1, h))
        skel_resized = cv2.resize(skeleton_bgr, (w2, h))
        
        # Add labels
        cv2.putText(orig_resized, "Original", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(skel_resized, "Skeleton", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Combine
        combined = np.hstack([orig_resized, skel_resized])
        
        return combined
    
    def visualize_local_map(
        self,
        slam: MinimapSLAM,
        radius: int = 200,
        show_motion: bool = True
    ) -> np.ndarray:
        """
        Visualize local area around player.
        
        Args:
            slam: SLAM instance
            radius: View radius
            show_motion: Draw motion history
            
        Returns:
            Visualization image
        """
        # Get visible area
        local_map = slam.get_visible_area(radius=radius)
        
        # Create BGR visualization
        h, w = local_map.shape
        vis = np.zeros((h, w, 3), dtype=np.uint8)
        
        vis[local_map == OccupancyCell.WALL] = self.COLOR_WALL
        vis[local_map == OccupancyCell.FREE] = self.COLOR_FREE
        vis[local_map == OccupancyCell.UNKNOWN] = self.COLOR_UNKNOWN
        
        # Draw player at center
        center_x, center_y = w // 2, h // 2
        cv2.circle(vis, (center_x, center_y), 8, self.COLOR_PLAYER, -1)
        cv2.circle(vis, (center_x, center_y), 8, (0, 0, 0), 2)
        
        # Draw motion history
        if show_motion and len(self.motion_history) > 1:
            for i in range(1, len(self.motion_history)):
                prev_dx, prev_dy = self.motion_history[i-1]
                curr_dx, curr_dy = self.motion_history[i]
                
                # Scale to local coordinates
                p1 = (center_x + int(prev_dx), center_y + int(prev_dy))
                p2 = (center_x + int(curr_dx), center_y + int(curr_dy))
                
                cv2.line(vis, p1, p2, self.COLOR_MOTION, 2)
        
        # Draw POIs
        for poi in slam.current_level.pois:
            gx, gy = poi.pos
            # Convert to local coordinates
            local_x = gx - slam.player_center[0] - slam.world_offset_x + center_x
            local_y = gy - slam.player_center[1] - slam.world_offset_y + center_y
            
            if 0 <= local_x < w and 0 <= local_y < h:
                color = slam._poi_color(poi.poi_type)
                cv2.circle(vis, (int(local_x), int(local_y)), 5, color, -1)
                cv2.circle(vis, (int(local_x), int(local_y)), 5, (0, 0, 0), 1)
                
                # Label
                label = poi.poi_type[:3].upper()
                cv2.putText(vis, label, (int(local_x) + 8, int(local_y)),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        
        # Scale up for visibility
        scale = 2
        vis = cv2.resize(vis, None, fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)
        
        return vis
    
    def visualize_global_map(
        self,
        slam: MinimapSLAM,
        view_size: Optional[int] = None
    ) -> np.ndarray:
        """
        Visualize entire global map with player position.
        
        Args:
            slam: SLAM instance
            view_size: Output image size (default: map_view_size)
            
        Returns:
            Visualization image
        """
        if view_size is None:
            view_size = self.map_view_size
        
        # Render full map at small scale
        full_map = slam.render_map(scale=1, show_pois=True, show_player=True)
        
        # Find bounding box of known area
        known_mask = slam.global_map != OccupancyCell.UNKNOWN
        if known_mask.any():
            rows, cols = np.where(known_mask)
            y_min, y_max = rows.min(), rows.max()
            x_min, x_max = cols.min(), cols.max()
            
            # Add padding
            padding = 50
            y_min = max(0, y_min - padding)
            y_max = min(slam.map_size, y_max + padding)
            x_min = max(0, x_min - padding)
            x_max = min(slam.map_size, x_max + padding)
            
            # Crop to known area
            cropped = full_map[y_min:y_max, x_min:x_max]
            
            # Resize to view size
            aspect = cropped.shape[1] / cropped.shape[0]
            if aspect > 1:
                w = view_size
                h = int(view_size / aspect)
            else:
                h = view_size
                w = int(view_size * aspect)
            
            vis = cv2.resize(cropped, (w, h), interpolation=cv2.INTER_NEAREST)
        else:
            # No known area yet
            vis = np.zeros((view_size, view_size, 3), dtype=np.uint8)
            cv2.putText(vis, "No map data yet", (view_size//4, view_size//2),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        return vis
    
    def draw_stats_overlay(
        self,
        image: np.ndarray,
        slam: MinimapSLAM,
        fps: Optional[float] = None,
        position: Tuple[int, int] = (10, 30)
    ) -> np.ndarray:
        """
        Draw SLAM statistics on image.
        
        Args:
            image: Input image
            slam: SLAM instance
            fps: Optional FPS counter
            position: Text start position (x, y)
            
        Returns:
            Image with overlay
        """
        overlay = image.copy()
        stats = slam.get_stats()
        
        x, y = position
        line_height = 25
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.6
        thickness = 2
        
        # Create semi-transparent background
        bg_height = 250 if fps else 225
        cv2.rectangle(overlay, (x - 5, y - 20), (x + 350, y + bg_height), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, image, 0.3, 0, image)
        
        # Draw stats
        lines = [
            f"MINIMAP SLAM v1.0",
            f"Frame: {stats['frames']}",
            f"Level: {stats['current_level']} ({stats['levels']} total)",
            f"World Offset: ({stats['world_offset'][0]}, {stats['world_offset'][1]})",
            f"Movement: {stats['total_movement']:.1f} px",
            f"Loop Closures: {stats['loop_closures']}",
            f"Signatures: {stats['signatures']}",
            f"Known Cells: {stats['known_cells']} ({stats['known_cells']/slam.map_size**2*100:.2f}%)",
            f"Walls: {stats['wall_cells']}, Free: {stats['free_cells']}",
            f"POIs: {stats['pois']}",
        ]
        
        if fps is not None:
            lines.insert(1, f"FPS: {fps:.1f}")
        
        for i, line in enumerate(lines):
            color = (0, 255, 255) if i == 0 else (255, 255, 255)
            weight = 2 if i == 0 else 1
            cv2.putText(image, line, (x, y + i * line_height),
                       font, font_scale, color, weight)
        
        return image
    
    def draw_motion_vector(
        self,
        image: np.ndarray,
        dx: float,
        dy: float,
        center: Optional[Tuple[int, int]] = None,
        scale: float = 10.0
    ) -> np.ndarray:
        """
        Draw motion vector on image.
        
        Args:
            image: Input image
            dx, dy: Motion vector
            center: Arrow center (default: image center)
            scale: Vector scaling factor
            
        Returns:
            Image with arrow
        """
        if center is None:
            h, w = image.shape[:2]
            center = (w // 2, h // 2)
        
        # Draw motion arrow
        end_x = int(center[0] + dx * scale)
        end_y = int(center[1] + dy * scale)
        
        cv2.arrowedLine(image, center, (end_x, end_y),
                       self.COLOR_MOTION, 3, tipLength=0.3)
        
        # Draw magnitude text
        magnitude = np.sqrt(dx*dx + dy*dy)
        text = f"{magnitude:.1f} px"
        cv2.putText(image, text, (end_x + 10, end_y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.COLOR_MOTION, 2)
        
        return image
    
    def create_dashboard(
        self,
        slam: MinimapSLAM,
        current_minimap: Optional[np.ndarray] = None,
        skeleton: Optional[np.ndarray] = None,
        dx: float = 0.0,
        dy: float = 0.0,
        fps: Optional[float] = None
    ) -> np.ndarray:
        """
        Create comprehensive SLAM dashboard.
        
        Layout:
        +-------------------+-------------------+
        | Minimap Process   | Local Map View    |
        +-------------------+-------------------+
        | Global Map        | Stats Overlay     |
        +-------------------+-------------------+
        
        Args:
            slam: SLAM instance
            current_minimap: Current raw minimap
            skeleton: Processed skeleton
            dx, dy: Current motion vector
            fps: FPS counter
            
        Returns:
            Dashboard image
        """
        # Panel size
        panel_w = 400
        panel_h = 300
        
        # Create panels
        panels = []
        
        # Panel 1: Minimap processing
        if current_minimap is not None and skeleton is not None:
            panel1 = self.visualize_minimap_processing(current_minimap, skeleton)
            panel1 = cv2.resize(panel1, (panel_w, panel_h))
        else:
            panel1 = np.zeros((panel_h, panel_w, 3), dtype=np.uint8)
            cv2.putText(panel1, "No minimap", (panel_w//3, panel_h//2),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.rectangle(panel1, (0, 0), (panel_w-1, panel_h-1), (128, 128, 128), 2)
        cv2.putText(panel1, "Minimap Processing", (10, 25),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        panels.append(panel1)
        
        # Panel 2: Local map view
        panel2 = self.visualize_local_map(slam, radius=150)
        panel2 = cv2.resize(panel2, (panel_w, panel_h))
        cv2.rectangle(panel2, (0, 0), (panel_w-1, panel_h-1), (128, 128, 128), 2)
        cv2.putText(panel2, "Local Map View", (10, 25),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        
        # Draw motion vector
        if abs(dx) > 0.1 or abs(dy) > 0.1:
            self.draw_motion_vector(panel2, dx, dy, scale=20.0)
        panels.append(panel2)
        
        # Panel 3: Global map
        panel3 = self.visualize_global_map(slam, view_size=panel_w)
        if panel3.shape[0] < panel_h:
            # Pad to panel height
            padding = panel_h - panel3.shape[0]
            panel3 = np.vstack([panel3, np.zeros((padding, panel3.shape[1], 3), dtype=np.uint8)])
        panel3 = cv2.resize(panel3, (panel_w, panel_h))
        cv2.rectangle(panel3, (0, 0), (panel_w-1, panel_h-1), (128, 128, 128), 2)
        cv2.putText(panel3, "Global Map", (10, 25),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        panels.append(panel3)
        
        # Panel 4: Stats
        panel4 = np.zeros((panel_h, panel_w, 3), dtype=np.uint8)
        panel4 = self.draw_stats_overlay(panel4, slam, fps=fps, position=(20, 40))
        
        # Add loop closure indicator
        if slam.loop_closures > 0:
            cv2.putText(panel4, f"Loop Closures: {slam.loop_closures}",
                       (20, panel_h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                       self.COLOR_LOOP, 2)
        
        cv2.rectangle(panel4, (0, 0), (panel_w-1, panel_h-1), (128, 128, 128), 2)
        panels.append(panel4)
        
        # Combine panels
        top_row = np.hstack([panels[0], panels[1]])
        bottom_row = np.hstack([panels[2], panels[3]])
        dashboard = np.vstack([top_row, bottom_row])
        
        # Add title bar
        title_h = 40
        title_bar = np.zeros((title_h, dashboard.shape[1], 3), dtype=np.uint8)
        title = "Diablo II SLAM - Visual Odometry | No Game Coordinates"
        cv2.putText(title_bar, title, (20, 28),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        
        dashboard = np.vstack([title_bar, dashboard])
        
        return dashboard
    
    def show(self, image: np.ndarray, wait_key: int = 1):
        """
        Display image in window.
        
        Args:
            image: Image to display
            wait_key: Wait key timeout (ms)
            
        Returns:
            Pressed key code
        """
        cv2.imshow(self.window_name, image)
        return cv2.waitKey(wait_key)
    
    def close(self):
        """Close visualization window."""
        cv2.destroyWindow(self.window_name)
    
    def add_motion_to_history(self, dx: float, dy: float):
        """Add motion vector to history for visualization."""
        self.motion_history.append((dx, dy))
        if len(self.motion_history) > self.max_history:
            self.motion_history.pop(0)
