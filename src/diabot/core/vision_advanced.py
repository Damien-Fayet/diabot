"""Advanced vision module with real game detection."""

import cv2
import numpy as np

from diabot.core.interfaces import VisionModule, Perception


class DiabloVisionModule(VisionModule):
    """
    Advanced vision module specifically tuned for Diablo 2 game perception.
    
    Detects:
    - Health bar (red coloring at top-left)
    - Mana bar (blue coloring at top-left)
    - Enemies (red/orange objects in playfield)
    - Items (yellow/gold highlights)
    - Player position (estimated from center)
    """
    
    def __init__(self, debug=False):
        """Initialize the vision module.
        
        Args:
            debug: Enable debug output
        """
        self.debug = debug
    
    def perceive(self, frame: np.ndarray) -> Perception:
        """Extract game perception from frame."""
        # Convert to HSV for better color detection
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        h, w = frame.shape[:2]
        
        # Detect health and mana bars (usually top-left UI)
        hp_ratio = self._detect_health_bar(frame, hsv)
        mana_ratio = self._detect_mana_bar(frame, hsv)
        
        # Detect enemies (reddish objects)
        enemy_count, enemy_types = self._detect_enemies(frame, hsv)
        
        # Detect items (golden/yellow highlights)
        items = self._detect_items(frame, hsv)
        
        # Estimate player position (usually center of game view)
        player_pos = self._estimate_player_position(frame, hsv)
        
        return Perception(
            hp_ratio=hp_ratio,
            mana_ratio=mana_ratio,
            enemy_count=enemy_count,
            enemy_types=enemy_types,
            visible_items=items,
            player_position=player_pos,
            raw_data={
                "frame_shape": frame.shape,
                "detection_method": "color_thresholding",
            },
        )
    
    def _detect_health_bar(self, frame: np.ndarray, hsv: np.ndarray) -> float:
        """Detect health bar (red color, top-left area)."""
        h, w = frame.shape[:2]
        
        # Focus on top-left UI area (typical Diablo 2 layout)
        ui_region = hsv[:max(1, h//6), :max(1, w//4)]
        
        # Red range in HSV: H: 0-10 or 170-180
        red_mask1 = cv2.inRange(ui_region, np.array([0, 100, 100]), np.array([10, 255, 255]))
        red_mask2 = cv2.inRange(ui_region, np.array([170, 100, 100]), np.array([180, 255, 255]))
        red_mask = cv2.bitwise_or(red_mask1, red_mask2)
        
        # Look for horizontal red bar pattern
        red_pixels = cv2.countNonZero(red_mask)
        ui_area = ui_region.shape[0] * ui_region.shape[1]
        
        # Health is red, estimate ratio from density
        health_ratio = min(1.0, (red_pixels / (ui_area * 0.3)))  # Normalize
        
        return max(0.0, health_ratio)
    
    def _detect_mana_bar(self, frame: np.ndarray, hsv: np.ndarray) -> float:
        """Detect mana bar (blue color, top-left area)."""
        h, w = frame.shape[:2]
        
        # Focus on top-left UI area
        ui_region = hsv[:max(1, h//6), :max(1, w//4)]
        
        # Blue range in HSV: H: 100-140
        blue_mask = cv2.inRange(ui_region, np.array([100, 100, 100]), np.array([140, 255, 255]))
        
        blue_pixels = cv2.countNonZero(blue_mask)
        ui_area = ui_region.shape[0] * ui_region.shape[1]
        
        mana_ratio = min(1.0, (blue_pixels / (ui_area * 0.3)))
        
        return max(0.0, mana_ratio)
    
    def _detect_enemies(self, frame: np.ndarray, hsv: np.ndarray) -> tuple[int, list[str]]:
        """Detect enemies (reddish moving objects)."""
        h, w = frame.shape[:2]
        
        # Red range (enemies are typically reddish)
        red_mask1 = cv2.inRange(hsv, np.array([0, 80, 80]), np.array([10, 255, 255]))
        red_mask2 = cv2.inRange(hsv, np.array([170, 80, 80]), np.array([180, 255, 255]))
        red_mask = cv2.bitwise_or(red_mask1, red_mask2)
        
        # Orange range (also enemies)
        orange_mask = cv2.inRange(hsv, np.array([10, 80, 80]), np.array([25, 255, 255]))
        
        # Combine masks
        threat_mask = cv2.bitwise_or(red_mask, orange_mask)
        
        # Find contours (potential enemies)
        contours, _ = cv2.findContours(threat_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        
        # Filter by size and position (avoid UI elements)
        enemy_count = 0
        enemy_types = []
        
        for contour in contours:
            area = cv2.contourArea(contour)
            # Filter by area (enemies typically have min area)
            if 50 < area < (h * w * 0.1):  # Between 50 and 10% of frame
                x, y, cw, ch = cv2.boundingRect(contour)
                # Skip if in UI area (top 20%)
                if y > h * 0.2:
                    enemy_count += 1
                    # Simple classification by size
                    if area > 500:
                        enemy_types.append("large_enemy")
                    else:
                        enemy_types.append("small_enemy")
        
        return min(enemy_count, 10), enemy_types  # Cap at 10 for sanity
    
    def _detect_items(self, frame: np.ndarray, hsv: np.ndarray) -> list[str]:
        """Detect items (golden/yellow highlights)."""
        h, w = frame.shape[:2]
        
        # Yellow/gold range in HSV
        yellow_mask = cv2.inRange(hsv, np.array([15, 100, 100]), np.array([35, 255, 255]))
        
        # Find contours
        contours, _ = cv2.findContours(yellow_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        
        items = []
        for contour in contours:
            area = cv2.contourArea(contour)
            # Filter by size
            if 20 < area < (h * w * 0.05):
                x, y, cw, ch = cv2.boundingRect(contour)
                # Skip if in UI area
                if y > h * 0.15:
                    items.append("item")
        
        return items[:5]  # Limit to 5 items
    
    def _estimate_player_position(self, frame: np.ndarray, hsv: np.ndarray) -> tuple[int, int]:
        """Estimate player position (usually center of playfield)."""
        h, w = frame.shape[:2]
        
        # Player is typically in the center of the playable area
        # Diablo 2 has isometric view, player is roughly at center
        player_x = w // 2
        player_y = h // 2
        
        # Slight bias upward as UI takes bottom space
        player_y = int(h * 0.45)
        
        return (player_x, player_y)


class FastVisionModule(VisionModule):
    """Optimized lightweight vision module for real-time processing."""
    
    def perceive(self, frame: np.ndarray) -> Perception:
        """Quick perception using brightness and basic thresholding."""
        h, w = frame.shape[:2]
        
        # Convert to HSV
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # Sample key areas
        top_left = hsv[:h//6, :w//4]
        playfield = hsv[h//6:5*h//6, w//4:3*w//4]
        
        # Quick health/mana estimation
        red_in_ui = np.count_nonzero((top_left[:,:,0] < 20) | (top_left[:,:,0] > 160)) / (top_left.shape[0] * top_left.shape[1])
        blue_in_ui = np.count_nonzero((top_left[:,:,0] > 100) & (top_left[:,:,0] < 140)) / (top_left.shape[0] * top_left.shape[1])
        
        # Count red objects in playfield
        red_in_play = np.count_nonzero((playfield[:,:,0] < 20) | (playfield[:,:,0] > 160))
        enemy_count = min(10, red_in_play // 500)  # Rough estimate
        
        return Perception(
            hp_ratio=min(1.0, red_in_ui * 3),
            mana_ratio=min(1.0, blue_in_ui * 3),
            enemy_count=enemy_count,
            enemy_types=["unknown"] * enemy_count if enemy_count > 0 else [],
            visible_items=[],
            player_position=(w // 2, h // 2),
        )
