"""
Pose estimation module.

Maintains player pose (position and orientation) using dead reckoning.
Updates from movement commands and corrects from minimap observations.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Tuple, Optional

import numpy as np


@dataclass
class Pose:
    """
    Player pose (position and orientation).
    
    Attributes:
        x: X coordinate in map coordinates
        y: Y coordinate in map coordinates
        angle: Orientation angle in degrees (0 = right, 90 = down, 180 = left, 270 = up)
        timestamp: Time of last update
    """
    x: float
    y: float
    angle: float
    timestamp: float = 0.0
    
    def copy(self) -> Pose:
        """Create a copy of this pose."""
        return Pose(
            x=self.x,
            y=self.y,
            angle=self.angle,
            timestamp=self.timestamp
        )


class PoseEstimator:
    """
    Estimate player pose using dead reckoning.
    
    Tracks player position and orientation based on:
    - Movement commands (direction and duration)
    - Estimated movement speed
    - Optional corrections from minimap observations
    
    Dead reckoning accumulates error over time, so periodic
    correction from visual observations is important.
    
    Coordinate system:
    - Positions are in map grid cells
    - Angles in degrees: 0=right, 90=down, 180=left, 270=up
    """
    
    def __init__(
        self,
        initial_x: float = 100.0,
        initial_y: float = 100.0,
        initial_angle: float = 0.0,
        movement_speed: float = 2.0,
        debug: bool = False
    ):
        """
        Initialize pose estimator.
        
        Args:
            initial_x: Starting X position
            initial_y: Starting Y position
            initial_angle: Starting angle in degrees
            movement_speed: Movement speed in grid cells per second
            debug: Enable debug output
        """
        self.debug = debug
        self.movement_speed = movement_speed
        
        # Current pose
        self.pose = Pose(
            x=initial_x,
            y=initial_y,
            angle=initial_angle,
            timestamp=time.time()
        )
        
        # Pose history for correction
        self.last_correction_time = time.time()
        
        if self.debug:
            print(f"[PoseEstimator] Initialized at ({initial_x:.1f}, {initial_y:.1f}), angle={initial_angle:.1f}°")
    
    def update_from_movement(
        self,
        direction: str,
        duration: float
    ):
        """
        Update pose based on movement command.
        
        Uses dead reckoning to estimate new position.
        
        Args:
            direction: Movement direction ('forward', 'backward', 'left', 'right')
            duration: Movement duration in seconds
        """
        current_time = time.time()
        
        # Calculate distance moved
        distance = self.movement_speed * duration
        
        # Update position based on direction
        if direction == 'forward':
            # Move in current facing direction
            rad = np.radians(self.pose.angle)
            dx = distance * np.cos(rad)
            dy = distance * np.sin(rad)
            self.pose.x += dx
            self.pose.y += dy
            
        elif direction == 'backward':
            # Move opposite to facing direction
            rad = np.radians(self.pose.angle)
            dx = distance * np.cos(rad)
            dy = distance * np.sin(rad)
            self.pose.x -= dx
            self.pose.y -= dy
            
        elif direction == 'left':
            # Strafe left (perpendicular to facing)
            rad = np.radians(self.pose.angle - 90)
            dx = distance * np.cos(rad)
            dy = distance * np.sin(rad)
            self.pose.x += dx
            self.pose.y += dy
            
        elif direction == 'right':
            # Strafe right (perpendicular to facing)
            rad = np.radians(self.pose.angle + 90)
            dx = distance * np.cos(rad)
            dy = distance * np.sin(rad)
            self.pose.x += dx
            self.pose.y += dy
        
        self.pose.timestamp = current_time
        
        if self.debug:
            print(f"[PoseEstimator] Moved {direction} for {duration:.2f}s -> ({self.pose.x:.1f}, {self.pose.y:.1f})")
    
    def update_rotation(self, angle_delta: float):
        """
        Update pose orientation.
        
        Args:
            angle_delta: Change in angle (degrees, positive = clockwise)
        """
        self.pose.angle = (self.pose.angle + angle_delta) % 360
        self.pose.timestamp = time.time()
        
        if self.debug:
            print(f"[PoseEstimator] Rotated {angle_delta:.1f}° -> {self.pose.angle:.1f}°")
    
    def set_angle(self, angle: float):
        """
        Set absolute orientation angle.
        
        Args:
            angle: New angle in degrees
        """
        self.pose.angle = angle % 360
        self.pose.timestamp = time.time()
        
        if self.debug:
            print(f"[PoseEstimator] Set angle to {self.pose.angle:.1f}°")
    
    def correct_position(
        self,
        new_x: float,
        new_y: float,
        confidence: float = 1.0
    ):
        """
        Apply correction to position estimate.
        
        Uses weighted average to blend dead reckoning with observation.
        Higher confidence means more trust in the observation.
        
        Args:
            new_x: Observed X position
            new_y: Observed Y position
            confidence: Confidence in observation (0.0 to 1.0)
        """
        # Blend current estimate with observation
        self.pose.x = (1 - confidence) * self.pose.x + confidence * new_x
        self.pose.y = (1 - confidence) * self.pose.y + confidence * new_y
        self.pose.timestamp = time.time()
        self.last_correction_time = time.time()
        
        if self.debug:
            print(f"[PoseEstimator] Corrected position to ({self.pose.x:.1f}, {self.pose.y:.1f})")
    
    def correct_from_minimap_alignment(
        self,
        minimap_grid: np.ndarray,
        local_map_grid: np.ndarray,
        current_map_pos: Tuple[int, int]
    ) -> Optional[Tuple[float, float]]:
        """
        Correct pose by aligning minimap with local map.
        
        Uses template matching to find best alignment between
        current minimap observation and existing local map.
        Returns corrected position if alignment is good enough.
        
        Args:
            minimap_grid: Current minimap occupancy grid
            local_map_grid: Local map occupancy grid
            current_map_pos: Current estimated position in local map
            
        Returns:
            Corrected (x, y) position if successful, None otherwise
        """
        # This is a placeholder for future implementation
        # Full implementation would use:
        # - Template matching (cv2.matchTemplate)
        # - Cross-correlation to find best offset
        # - Confidence threshold to accept/reject correction
        
        # For now, return None (no correction)
        return None
    
    def get_pose(self) -> Pose:
        """
        Get current pose estimate.
        
        Returns:
            Current Pose object
        """
        return self.pose.copy()
    
    def get_position(self) -> Tuple[float, float]:
        """
        Get current position estimate.
        
        Returns:
            (x, y) tuple
        """
        return (self.pose.x, self.pose.y)
    
    def get_angle(self) -> float:
        """
        Get current orientation angle.
        
        Returns:
            Angle in degrees
        """
        return self.pose.angle
    
    def get_position_int(self) -> Tuple[int, int]:
        """
        Get current position as integer coordinates.
        
        Returns:
            (x, y) tuple with integer coordinates
        """
        return (int(round(self.pose.x)), int(round(self.pose.y)))
    
    def time_since_last_correction(self) -> float:
        """
        Get time elapsed since last position correction.
        
        Returns:
            Time in seconds
        """
        return time.time() - self.last_correction_time
    
    def reset(self, x: float, y: float, angle: float = 0.0):
        """
        Reset pose to new position.
        
        Args:
            x: New X position
            y: New Y position
            angle: New angle in degrees
        """
        self.pose.x = x
        self.pose.y = y
        self.pose.angle = angle
        self.pose.timestamp = time.time()
        self.last_correction_time = time.time()
        
        if self.debug:
            print(f"[PoseEstimator] Reset to ({x:.1f}, {y:.1f}), angle={angle:.1f}°")
