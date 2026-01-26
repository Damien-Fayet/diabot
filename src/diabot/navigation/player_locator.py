"""
Player position detector for minimap.

Detects the white cross at the center of the minimap to determine player position.
"""

from __future__ import annotations

from typing import Optional, Tuple

import cv2
import numpy as np


class PlayerLocator:
    """
    Detects player position on minimap using the white cross marker.
    
    The player is indicated by a white cross (+) at the center of the minimap.
    This class detects this cross to confirm player position and orientation.
    """
    
    def __init__(self, debug: bool = False):
        """
        Initialize player locator.
        
        Args:
            debug: Enable debug output
        """
        self.debug = debug
    
    def detect_player_cross(self, minimap: np.ndarray) -> Optional[Tuple[int, int]]:
        """
        Detect white cross (+) marking player position on minimap.
        
        Args:
            minimap: Minimap image (BGR numpy array)
            
        Returns:
            (x, y) position of cross center, or None if not detected
        """
        if minimap is None or minimap.size == 0:
            return None
        
        h, w = minimap.shape[:2]
        center_x, center_y = w // 2, h // 2
        
        # Extract center region (cross should be near center)
        search_radius = min(w, h) // 8
        y1 = max(0, center_y - search_radius)
        y2 = min(h, center_y + search_radius)
        x1 = max(0, center_x - search_radius)
        x2 = min(w, center_x + search_radius)
        
        roi = minimap[y1:y2, x1:x2]
        
        # Convert to grayscale
        if len(roi.shape) == 3:
            gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        else:
            gray = roi
        
        # Detect white pixels (cross is white/bright)
        # Use high threshold to isolate bright white cross
        _, white_mask = cv2.threshold(gray, 220, 255, cv2.THRESH_BINARY)
        
        # Find contours
        contours, _ = cv2.findContours(white_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            # No white pixels found - assume center
            if self.debug:
                print("[PlayerLocator] No white cross detected, assuming center")
            return (center_x, center_y)
        
        # Find the contour closest to ROI center
        roi_cx = (x2 - x1) // 2
        roi_cy = (y2 - y1) // 2
        
        best_contour = None
        min_dist = float('inf')
        
        for cnt in contours:
            M = cv2.moments(cnt)
            if M["m00"] == 0:
                continue
            
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            
            dist = np.sqrt((cx - roi_cx)**2 + (cy - roi_cy)**2)
            
            if dist < min_dist:
                min_dist = dist
                best_contour = cnt
        
        if best_contour is not None:
            M = cv2.moments(best_contour)
            if M["m00"] > 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                
                # Convert back to full minimap coordinates
                cross_x = x1 + cx
                cross_y = y1 + cy
                
                if self.debug:
                    print(f"[PlayerLocator] Cross detected at ({cross_x}, {cross_y})")
                
                return (cross_x, cross_y)
        
        # Fallback to center
        if self.debug:
            print("[PlayerLocator] Cross detection uncertain, using center")
        return (center_x, center_y)
    
    def get_orientation(self, minimap: np.ndarray) -> Optional[float]:
        """
        Estimate player orientation from cross shape.
        
        Args:
            minimap: Minimap image
            
        Returns:
            Angle in degrees (0 = north), or None if not detected
            
        Note:
            Currently not implemented - requires cross arm detection.
            Returns None for now.
        """
        # TODO: Analyze cross arms to determine facing direction
        # This requires detecting the longer arm of the cross
        return None
    
    def visualize_detection(self, minimap: np.ndarray) -> np.ndarray:
        """
        Create debug visualization showing detected player position.
        
        Args:
            minimap: Original minimap image
            
        Returns:
            Visualization image with cross marked
        """
        vis = minimap.copy()
        pos = self.detect_player_cross(minimap)
        
        if pos:
            x, y = pos
            # Draw green circle at detected position
            cv2.circle(vis, (x, y), 5, (0, 255, 0), 2)
            cv2.drawMarker(vis, (x, y), (0, 255, 255), 
                          cv2.MARKER_CROSS, 15, 2)
            
            # Draw text
            cv2.putText(vis, f"Player: ({x}, {y})", (10, 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        
        return vis
