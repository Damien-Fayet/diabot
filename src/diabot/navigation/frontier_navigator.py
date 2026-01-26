"""
Navigation controller with frontier-based exploration.

High-level navigation system that:
- Processes minimap observations
- Updates local map and pose
- Selects exploration targets using frontier detection
- Generates navigation actions
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Optional, Tuple, List
import time

import numpy as np

from .minimap_extractor import MinimapExtractor
from .minimap_processor import MinimapProcessor, MinimapGrid
from .local_map import LocalMap, Frontier
from .pose_estimator import PoseEstimator


class NavigationAction(str, Enum):
    """High-level navigation actions."""
    STOP = "stop"
    MOVE_FORWARD = "move_forward"
    TURN_LEFT = "turn_left"
    TURN_RIGHT = "turn_right"
    MOVE_TO_TARGET = "move_to_target"


@dataclass
class NavigationState:
    """
    Current navigation state.
    
    Contains all information about navigation status.
    """
    action: NavigationAction
    current_position: Tuple[int, int]
    current_angle: float
    target_position: Optional[Tuple[int, int]] = None
    target_angle: Optional[float] = None
    path: Optional[List[Tuple[int, int]]] = None
    frontiers_available: int = 0
    exploration_progress: float = 0.0  # 0.0 to 1.0


class FrontierNavigator:
    """
    Vision-based navigator using frontier exploration.
    
    Implements frontier-based exploration for autonomous navigation:
    1. Extract and process minimap from game frame
    2. Update local map with new observations
    3. Track player pose using dead reckoning
    4. Find frontier cells (boundary of known/unknown space)
    5. Plan path to nearest frontier
    6. Generate navigation actions
    
    This is a robotics-inspired approach that works purely from
    visual observations without any game state access.
    """
    
    def __init__(
        self,
        minimap_grid_size: int = 64,
        local_map_size: int = 200,
        movement_speed: float = 2.0,
        debug: bool = False
    ):
        """
        Initialize frontier navigator.
        
        Args:
            minimap_grid_size: Size of minimap processing grid
            local_map_size: Size of local map grid
            movement_speed: Estimated movement speed (cells/sec)
            debug: Enable debug output
        """
        self.debug = debug
        
        # Initialize components
        # Use fullscreen mode for better resolution and precision
        self.minimap_extractor = MinimapExtractor(debug=debug, fullscreen_mode=True)
        self.minimap_processor = MinimapProcessor(
            grid_size=minimap_grid_size,
            debug=debug
        )
        self.local_map = LocalMap(
            map_size=local_map_size,
            debug=debug
        )
        self.pose_estimator = PoseEstimator(
            initial_x=local_map_size // 2,
            initial_y=local_map_size // 2,
            movement_speed=movement_speed,
            debug=debug
        )
        
        # Navigation state
        self.current_target: Optional[Frontier] = None
        self.current_path: Optional[List[Tuple[int, int]]] = None
        self.last_update_time = time.time()
        self.stuck_counter = 0
        self.last_position = None
        
        if self.debug:
            print("[FrontierNavigator] Initialized")
    
    def update(self, frame: np.ndarray) -> NavigationState:
        """
        Update navigation system with new frame.
        
        Complete navigation update cycle:
        1. Extract minimap from frame
        2. Process minimap to occupancy grid
        3. Update local map with new observations
        4. Select or update navigation target
        5. Generate navigation action
        
        Args:
            frame: Full game frame (BGR numpy array)
            
        Returns:
            NavigationState with action and status
        """
        current_time = time.time()
        dt = current_time - self.last_update_time
        self.last_update_time = current_time
        
        try:
            # 1. Extract and process minimap
            minimap = self.minimap_extractor.extract(frame)
            minimap_grid = self.minimap_processor.process(minimap)
            
            # 2. Get current pose estimate
            player_x, player_y = self.pose_estimator.get_position_int()
            player_angle = self.pose_estimator.get_angle()
            
            # 3. Update local map
            self.local_map.update_from_minimap(
                minimap_grid.grid,
                (player_x, player_y)
            )
            self.local_map.mark_visited(player_x, player_y)
            
            # 4. Check if stuck (not moving)
            if self.last_position is not None:
                dx = player_x - self.last_position[0]
                dy = player_y - self.last_position[1]
                distance_moved = np.sqrt(dx * dx + dy * dy)
                
                if distance_moved < 0.5:  # Threshold for "stuck"
                    self.stuck_counter += 1
                else:
                    self.stuck_counter = 0
            
            self.last_position = (player_x, player_y)
            
            # 5. Select or update target
            if self._should_select_new_target():
                self._select_new_target(player_x, player_y)
            
            # 6. Generate navigation action
            action = self._generate_action(player_x, player_y, player_angle)
            
            # 7. Build state
            frontiers = self.local_map.get_frontiers()
            state = NavigationState(
                action=action,
                current_position=(player_x, player_y),
                current_angle=player_angle,
                target_position=self.current_target.position if self.current_target else None,
                path=self.current_path,
                frontiers_available=len(frontiers),
                exploration_progress=self._calculate_exploration_progress()
            )
            
            if self.debug:
                print(f"[FrontierNavigator] Action: {action}, "
                      f"Pos: ({player_x}, {player_y}), "
                      f"Frontiers: {len(frontiers)}")
            
            return state
            
        except Exception as e:
            if self.debug:
                print(f"[FrontierNavigator] Error: {e}")
            
            # Return safe default state on error
            return NavigationState(
                action=NavigationAction.STOP,
                current_position=self.pose_estimator.get_position_int(),
                current_angle=self.pose_estimator.get_angle(),
                frontiers_available=0
            )
    
    def report_movement(self, direction: str, duration: float):
        """
        Report movement command execution for pose tracking.
        
        Call this after executing a movement command to update
        dead reckoning estimate.
        
        Args:
            direction: Movement direction ('forward', 'backward', 'left', 'right')
            duration: Duration of movement in seconds
        """
        self.pose_estimator.update_from_movement(direction, duration)
    
    def report_rotation(self, angle_delta: float):
        """
        Report rotation for pose tracking.
        
        Args:
            angle_delta: Change in angle (degrees, positive = clockwise)
        """
        self.pose_estimator.update_rotation(angle_delta)
    
    def reset(self):
        """Reset navigation state."""
        self.current_target = None
        self.current_path = None
        self.stuck_counter = 0
        self.last_position = None
        
        # Reset local map and pose to initial state
        map_size = self.local_map.map_size
        self.local_map = LocalMap(map_size=map_size, debug=self.debug)
        self.pose_estimator.reset(map_size // 2, map_size // 2)
        
        if self.debug:
            print("[FrontierNavigator] Reset")
    
    def _should_select_new_target(self) -> bool:
        """Check if new target should be selected."""
        # No target
        if self.current_target is None:
            return True
        
        # Reached target
        player_x, player_y = self.pose_estimator.get_position_int()
        tx, ty = self.current_target.position
        dx = tx - player_x
        dy = ty - player_y
        distance = np.sqrt(dx * dx + dy * dy)
        
        if distance < 3.0:  # Close enough threshold
            return True
        
        # Stuck for too long
        if self.stuck_counter > 10:
            if self.debug:
                print("[FrontierNavigator] Stuck, selecting new target")
            self.stuck_counter = 0
            return True
        
        return False
    
    def _select_new_target(self, player_x: int, player_y: int):
        """Select new frontier target for exploration."""
        frontiers = self.local_map.get_frontiers(max_frontiers=20)
        
        if not frontiers:
            if self.debug:
                print("[FrontierNavigator] No frontiers available")
            self.current_target = None
            self.current_path = None
            return
        
        # Select closest frontier
        self.current_target = frontiers[0]
        
        # Plan path to target
        self.current_path = self.local_map.find_path(
            (player_x, player_y),
            self.current_target.position
        )
        
        if self.debug:
            target_x, target_y = self.current_target.position
            print(f"[FrontierNavigator] New target: ({target_x}, {target_y}), "
                  f"Path length: {len(self.current_path) if self.current_path else 0}")
    
    def _generate_action(
        self,
        player_x: int,
        player_y: int,
        player_angle: float
    ) -> NavigationAction:
        """
        Generate navigation action based on current state.
        
        Args:
            player_x: Current X position
            player_y: Current Y position
            player_angle: Current orientation angle
            
        Returns:
            NavigationAction to execute
        """
        # No target or path
        if self.current_target is None or self.current_path is None:
            return NavigationAction.STOP
        
        # No valid path
        if len(self.current_path) < 2:
            return NavigationAction.STOP
        
        # Get next waypoint (skip current position)
        next_waypoint = self.current_path[1] if len(self.current_path) > 1 else self.current_path[0]
        wx, wy = next_waypoint
        
        # Calculate direction to waypoint
        dx = wx - player_x
        dy = wy - player_y
        
        # Calculate target angle
        target_angle = np.degrees(np.arctan2(dy, dx)) % 360
        
        # Calculate angle difference
        angle_diff = (target_angle - player_angle + 180) % 360 - 180
        
        # If we need to turn significantly, turn first
        if abs(angle_diff) > 30:
            if angle_diff > 0:
                return NavigationAction.TURN_RIGHT
            else:
                return NavigationAction.TURN_LEFT
        
        # Otherwise, move forward
        return NavigationAction.MOVE_FORWARD
    
    def _calculate_exploration_progress(self) -> float:
        """
        Calculate exploration progress (0.0 to 1.0).
        
        Rough estimate based on explored vs total visible area.
        """
        total_cells = self.local_map.map_size * self.local_map.map_size
        explored_cells = np.sum(self.local_map.visited)
        
        # Cap at reasonable percentage since we don't explore entire map
        progress = min(1.0, explored_cells / (total_cells * 0.3))
        
        return progress
    
    def get_local_map(self) -> LocalMap:
        """Get local map for visualization."""
        return self.local_map
    
    def get_pose(self):
        """Get current pose estimate."""
        return self.pose_estimator.get_pose()
