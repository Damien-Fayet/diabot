"""Screen/menu state detection for Diablo 2.

Detects which screen the game is currently showing:
- Gameplay (main game state)
- Menu (main menu, pause menu)
- Character selection
- Dead (player death screen)
- Loading
- Cinematic
- Error/Dialog
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Optional

import cv2
import numpy as np


class GameScreen(str, Enum):
    """Enum of possible game screens."""
    GAMEPLAY = "gameplay"
    MAIN_MENU = "main_menu"
    CHAR_SELECT = "char_select"
    DEAD = "dead"
    LOADING = "loading"
    CINEMATIC = "cinematic"
    ERROR = "error"
    PAUSE = "pause"
    UNKNOWN = "unknown"


@dataclass
class ScreenDetectionResult:
    """Result of screen detection."""
    screen_type: GameScreen
    confidence: float  # 0.0-1.0
    details: dict = None  # Additional context (e.g., menu items, buttons visible)
    
    def __post_init__(self):
        if self.details is None:
            self.details = {}


class ScreenDetector:
    """Detects current game screen from frame."""
    
    def __init__(self):
        """Initialize detector with screen-specific color signatures."""
        # Color signatures for different screens (BGR format)
        # These are heuristics that should be refined with real screenshots
        
        # Main menu: usually has dark background with buttons
        self.main_menu_colors = {
            'background': (20, 20, 20),  # Very dark
            'button_gold': (0, 215, 255),  # Gold/yellow buttons
        }
        
        # Character select: grid of portraits
        self.char_select_colors = {
            'background': (40, 40, 60),  # Dark blue-ish
            'portrait_border': (100, 100, 100),  # Gray borders
        }
        
        # Gameplay: visible minimap, UI elements
        self.gameplay_colors = {
            'minimap': (50, 50, 50),  # Dark gray minimap
            'health_bar': (0, 0, 255),  # Red health
            'mana_bar': (0, 0, 255),  # Blue mana
        }
        
        # Dead screen: "YOU ARE DEAD" text, skeletal graphics
        self.dead_colors = {
            'skull': (150, 150, 150),  # Light gray skull
            'blood': (0, 0, 200),  # Red blood spatters
        }
    
    def detect(self, frame: np.ndarray) -> ScreenDetectionResult:
        """
        Detect current screen type from frame.
        
        Args:
            frame: Game frame (BGR, H x W x 3)
            
        Returns:
            ScreenDetectionResult with screen type and confidence
        """
        if frame is None or frame.size == 0:
            return ScreenDetectionResult(GameScreen.UNKNOWN, 0.0)
        
        h, w = frame.shape[:2]
        
        # Try detections in order of priority
        # Check loading first (very dark with progress bar)
        result = self._detect_loading(frame)
        if result.confidence > 0.6:
            return result
        
        # Check death (dark + lots of red)
        result = self._detect_dead(frame)
        if result.confidence > 0.7:
            return result
        
        # Check main menu (very dark + golden buttons)
        result = self._detect_main_menu(frame)
        if result.confidence > 0.6:
            return result
        
        # Check character select (edges in grid pattern)
        result = self._detect_char_select(frame)
        if result.confidence > 0.7:
            return result
        
        # Check gameplay (minimap + bars + variety)
        result = self._detect_gameplay(frame)
        if result.confidence > 0.5:
            return result
        
        return ScreenDetectionResult(GameScreen.UNKNOWN, 0.0)
    
    def _detect_dead(self, frame: np.ndarray) -> ScreenDetectionResult:
        """Detect death screen."""
        h, w = frame.shape[:2]
        
        # Death screen characteristics:
        # 1. VERY dark overall (avg brightness < 20)
        # 2. Significant red content (blood splatters, 20%+)
        # 3. NOT just a gradient (some structure)
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        avg_brightness = np.mean(gray)
        
        # Death screen is EXTREMELY dark
        if avg_brightness > 30:
            return ScreenDetectionResult(GameScreen.UNKNOWN, 0.0)
        
        # Count red pixels
        red_channel = frame[:,:,2]
        red_pixels = np.sum(red_channel > 100)
        total_pixels = frame.shape[0] * frame.shape[1]
        
        red_ratio = red_pixels / max(total_pixels, 1)
        
        # Death screen needs significant red (15%+)
        if red_ratio > 0.15:
            return ScreenDetectionResult(
                GameScreen.DEAD,
                0.8 + min(red_ratio / 2, 0.2),  # 0.8-1.0
                {"red_ratio": red_ratio, "brightness": avg_brightness}
            )
        
        return ScreenDetectionResult(GameScreen.UNKNOWN, 0.0)
    
    def _detect_loading(self, frame: np.ndarray) -> ScreenDetectionResult:
        """Detect loading screen."""
        h, w = frame.shape[:2]
        
        # Loading screen is mostly black with progress bar
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Calculate darkness
        dark_pixels = np.sum(gray < 50)
        total_pixels = gray.size
        
        darkness = dark_pixels / max(total_pixels, 1)
        
        # Loading screen is >80% dark
        if darkness > 0.8:
            # Look for progress bar (lighter horizontal line in lower half)
            lower_half = frame[h//2:, :]
            lighter_pixels = np.sum(gray[h//2:] > 100)
            lower_total = lower_half.shape[0] * lower_half.shape[1]
            
            lighter_ratio = lighter_pixels / max(lower_total, 1)
            
            # If mostly dark but some lighter pixels in lower half, likely loading
            if lighter_ratio > 0.05 and lighter_ratio < 0.3:
                return ScreenDetectionResult(
                    GameScreen.LOADING,
                    0.8,
                    {"darkness": darkness, "lighter_ratio": lighter_ratio}
                )
        
        return ScreenDetectionResult(GameScreen.UNKNOWN, 0.0)
    
    def _detect_main_menu(self, frame: np.ndarray) -> ScreenDetectionResult:
        """Detect main menu screen."""
        h, w = frame.shape[:2]
        
        # Main menu characteristics:
        # 1. Very dark background (avg brightness 40-70, NOT < 20)
        # 2. Central golden/yellow buttons
        # 3. NOT lots of red (dead screen)
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        avg_brightness = np.mean(gray)
        
        # Main menu is dark but not death-dark
        if avg_brightness < 35 or avg_brightness > 80:
            return ScreenDetectionResult(GameScreen.UNKNOWN, 0.0)
        
        # Verify NOT death screen (too much red)
        red_channel = frame[:,:,2]
        red_pixels = np.sum(red_channel > 100)
        total_pixels = frame.shape[0] * frame.shape[1]
        red_ratio = red_pixels / max(total_pixels, 1)
        
        if red_ratio > 0.15:
            return ScreenDetectionResult(GameScreen.UNKNOWN, 0.0)
        
        # Look for golden elements
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # Gold/yellow hue range
        lower_gold = np.array([15, 100, 100])
        upper_gold = np.array([40, 255, 255])
        
        gold_mask = cv2.inRange(hsv, lower_gold, upper_gold)
        gold_pixels = np.sum(gold_mask > 0)
        
        # Menu typically has 5-15% golden pixels
        gold_ratio = gold_pixels / max(frame.size // 3, 1)
        
        if 0.03 < gold_ratio < 0.20:
            return ScreenDetectionResult(
                GameScreen.MAIN_MENU,
                0.6 + min(gold_ratio * 2, 0.4),
                {"gold_ratio": gold_ratio, "brightness": avg_brightness}
            )
        
        return ScreenDetectionResult(GameScreen.UNKNOWN, 0.0)
    
    def _detect_char_select(self, frame: np.ndarray) -> ScreenDetectionResult:
        """Detect character select screen."""
        h, w = frame.shape[:2]
        
        # Character select has grid of portrait boxes
        # Look for specific color patterns and structure
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Character portraits have borders and skin tones
        # Look for edges (borders of portraits)
        edges = cv2.Canny(gray, 50, 150)
        
        # Count edges in grid-like pattern
        edge_density = np.sum(edges > 0) / edges.size
        
        # Character select should have moderate edge density from portrait boxes
        if 0.05 < edge_density < 0.25:
            # Also check for centered layout (most edges in center region)
            center_region = edges[h//4:3*h//4, w//4:3*w//4]
            center_edges = np.sum(center_region > 0)
            total_edges = np.sum(edges > 0)
            
            center_ratio = center_edges / max(total_edges, 1)
            
            if center_ratio > 0.6:
                return ScreenDetectionResult(
                    GameScreen.CHAR_SELECT,
                    min(edge_density * 2, 1.0),
                    {"edge_density": edge_density, "center_ratio": center_ratio}
                )
        
        return ScreenDetectionResult(GameScreen.UNKNOWN, 0.0)
    
    def _detect_gameplay(self, frame: np.ndarray) -> ScreenDetectionResult:
        """Detect normal gameplay screen."""
        h, w = frame.shape[:2]
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        avg_brightness = np.mean(gray)
        
        # Gameplay is neither very dark (like menu) nor extremely bright
        if avg_brightness < 50 or avg_brightness > 180:
            return ScreenDetectionResult(GameScreen.UNKNOWN, 0.0)
        
        # Check for minimap (dark region in top-right)
        minimap_region = frame[0:h//4, 3*w//4:w]
        minimap_gray = cv2.cvtColor(minimap_region, cv2.COLOR_BGR2GRAY)
        minimap_darkness = np.mean(minimap_gray)
        
        # Minimap should be relatively dark (avg < 120)
        has_minimap = minimap_darkness < 120
        
        # Check for health/mana bars (distinct colors on left/right bottom)
        left_region = frame[3*h//4:h, 0:w//6]
        right_region = frame[3*h//4:h, 5*w//6:w]
        
        left_hsv = cv2.cvtColor(left_region, cv2.COLOR_BGR2HSV)
        right_hsv = cv2.cvtColor(right_region, cv2.COLOR_BGR2HSV)
        
        # Health bar is red (hue 0 or 180)
        red_mask_left = cv2.inRange(left_hsv, np.array([0, 100, 100]), np.array([10, 255, 255]))
        red_pixels_left = np.sum(red_mask_left > 0)
        
        # Mana bar is blue (hue ~100-130)
        blue_mask_right = cv2.inRange(right_hsv, np.array([100, 100, 100]), np.array([140, 255, 255]))
        blue_pixels_right = np.sum(blue_mask_right > 0)
        
        has_bars = (red_pixels_left > 50) and (blue_pixels_right > 50)
        
        # Check for variety in center (not uniform color, typical of playfield)
        center = frame[h//4:3*h//4, w//4:3*w//4]
        center_std = np.std(cv2.cvtColor(center, cv2.COLOR_BGR2GRAY))
        
        has_variety = center_std > 15
        
        # Gameplay requires at least some variety and reasonable brightness
        confidence = 0.0
        if has_minimap:
            confidence += 0.25
        if has_bars:
            confidence += 0.25
        if has_variety:
            confidence += 0.5
        
        # Also check if it's NOT a dead screen or menu
        is_not_dead = avg_brightness > 60 or True  # Not all red
        is_not_menu = avg_brightness > 50 or True  # Not all dark
        
        if confidence > 0.4 and is_not_dead and is_not_menu:
            return ScreenDetectionResult(
                GameScreen.GAMEPLAY,
                confidence,
                {
                    "has_minimap": has_minimap,
                    "has_bars": has_bars,
                    "has_variety": has_variety,
                    "center_std": center_std,
                    "brightness": avg_brightness
                }
            )
        
        return ScreenDetectionResult(GameScreen.UNKNOWN, 0.0)
