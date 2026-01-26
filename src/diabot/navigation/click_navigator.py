"""
Click-based navigation for Diablo 2 bot.

This module handles:
- Translating navigation goals to click positions
- Sending clicks to the game window
- Tracking navigation state during exploration
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from enum import Enum
from typing import Optional, Tuple

import sys

# Windows-specific imports for mouse control
if sys.platform == 'win32':
    try:
        import pyautogui
        import win32gui
        import win32con
        MOUSE_CONTROL_AVAILABLE = True
    except ImportError:
        MOUSE_CONTROL_AVAILABLE = False
else:
    MOUSE_CONTROL_AVAILABLE = False


class NavigationMode(Enum):
    """Navigation modes for the bot."""
    IDLE = "idle"                      # No navigation
    EXPLORING = "exploring"            # Exploring current zone
    MOVING_TO_WAYPOINT = "waypoint"   # Moving to waypoint
    MOVING_TO_EXIT = "exit"            # Moving to zone exit


@dataclass
class NavigationGoal:
    """Represents a navigation goal."""
    mode: NavigationMode
    target_position: Optional[Tuple[int, int]] = None  # (x, y) in game window
    zone_name: Optional[str] = None
    notes: str = ""


class ClickNavigator:
    """
    Handles click-based navigation in Diablo 2.
    
    Converts high-level navigation goals (e.g., "go to waypoint")
    into mouse clicks to the game window.
    """

    def __init__(self, window_hwnd: Optional[int] = None, window_title: str = "Diablo II: Resurrected", debug: bool = True):
        """
        Initialize click navigator.

        Args:
            window_hwnd: Windows handle for the game window (optional, will find by title if None)
            window_title: Title of the game window to find
            debug: Enable debug output
        """
        if not MOUSE_CONTROL_AVAILABLE:
            raise ImportError(
                "ClickNavigator requires pyautogui. Install with: pip install pyautogui"
            )

        self.window_title = window_title
        self.window_hwnd = window_hwnd
        self.debug = debug
        self.current_goal: Optional[NavigationGoal] = None
        self.goal_start_time: Optional[float] = None
        self.click_count = 0
        
        # Window position cache
        self.window_rect: Optional[Tuple[int, int, int, int]] = None
        self.last_window_check = 0.0
        
        # Safety parameters
        self.min_click_interval = 0.1  # Minimum time between clicks (seconds)
        self.max_click_distance = 800  # Max distance to click from center
        self.goal_timeout = 30.0  # Timeout for a navigation goal (seconds)

    def set_goal(self, goal: NavigationGoal) -> None:
        """
        Set a new navigation goal.

        Args:
            goal: The navigation goal to pursue
        """
        if self.debug:
            print(
                f"[CLICK_NAV] New goal: mode={goal.mode.value}, "
                f"target={goal.target_position}, zone={goal.zone_name}"
            )
        
        self.current_goal = goal
        self.goal_start_time = time.time()

    def _find_window(self) -> bool:
        """Find game window by title and cache hwnd."""
        if sys.platform != 'win32':
            return False
        
        try:
            hwnd = win32gui.FindWindow(None, self.window_title)
            if hwnd == 0:
                if self.debug:
                    print(f"[CLICK_NAV] Window '{self.window_title}' not found")
                return False
            
            self.window_hwnd = hwnd
            return True
        except Exception as e:
            if self.debug:
                print(f"[CLICK_NAV] Error finding window: {e}")
            return False

    def _get_window_position(self) -> Optional[Tuple[int, int, int, int]]:
        """Get window position (left, top, right, bottom) in screen coordinates."""
        if sys.platform != 'win32':
            return None
        
        # Cache window position for 1 second to avoid excessive calls
        now = time.time()
        if self.window_rect and (now - self.last_window_check) < 1.0:
            return self.window_rect
        
        if not self.window_hwnd:
            if not self._find_window():
                return None
        
        try:
            rect = win32gui.GetWindowRect(self.window_hwnd)
            self.window_rect = rect
            self.last_window_check = now
            return rect
        except Exception as e:
            if self.debug:
                print(f"[CLICK_NAV] Error getting window position: {e}")
            return None

    def _activate_window(self) -> bool:
        """Bring game window to foreground."""
        if sys.platform != 'win32':
            return False
        
        if not self.window_hwnd:
            if not self._find_window():
                return False
        
        try:
            # Check if window is minimized
            if win32gui.IsIconic(self.window_hwnd):
                win32gui.ShowWindow(self.window_hwnd, win32con.SW_RESTORE)
            
            # Bring to foreground
            win32gui.SetForegroundWindow(self.window_hwnd)
            time.sleep(0.1)  # Short delay to let window activate
            return True
        except Exception as e:
            if self.debug:
                print(f"[CLICK_NAV] Error activating window: {e}")
            return False

    def click_to_explore(self, x: int, y: int, description: str = "", window_relative: bool = True) -> bool:
        """
        Click at a position to explore/move there.

        Args:
            x: X coordinate (window-relative if window_relative=True, else screen)
            y: Y coordinate
            description: Description of what we're clicking (for logging)
            window_relative: If True, x/y are relative to window client area

        Returns:
            bool: True if click was sent successfully
        """
        try:
            # Activate window first
            if not self._activate_window():
                print(f"[CLICK_NAV] FAIL Could not activate game window")
                return False
            
            # Convert window-relative to screen coordinates if needed
            if window_relative:
                rect = self._get_window_position()
                if not rect:
                    print(f"[CLICK_NAV] FAIL Could not get window position")
                    return False
                
                left, top, right, bottom = rect
                screen_x = left + x
                screen_y = top + y
                
                if self.debug:
                    print(f"[CLICK_NAV] Window rect: {rect}")
                    print(f"[CLICK_NAV] Window coords: ({x}, {y}) -> Screen coords: ({screen_x}, {screen_y})")
            else:
                screen_x = x
                screen_y = y
            
            if self.debug:
                print(f"[CLICK_NAV] Clicking @({screen_x}, {screen_y}) - {description}")
            
            # Safety: Check distance is reasonable
            if abs(screen_x) > 5000 or abs(screen_y) > 5000:
                print(f"[CLICK_NAV] WARNING Sanity check failed: coords ({screen_x}, {screen_y}) too large")
                return False
            
            # Send click
            pyautogui.click(screen_x, screen_y)
            
            self.click_count += 1
            time.sleep(self.min_click_interval)
            
            return True
        
        except Exception as e:
            print(f"[CLICK_NAV] FAIL Error clicking: {e}")
            import traceback
            traceback.print_exc()
            return False

    def click_waypoint(self, wp_position: Tuple[int, int]) -> bool:
        """
        Click on a waypoint to use it.

        Args:
            wp_position: (x, y) position of waypoint on minimap

        Returns:
            bool: True if click successful
        """
        # In Diablo 2, clicking on a waypoint on the minimap teleports you there
        # The waypoint position is in minimap coordinates, which we need to translate
        # to screen coordinates for clicking
        
        x, y = wp_position
        
        # TODO: Implement minimap-to-screen coordinate transformation
        # For now, this is a placeholder
        
        return self.click_to_explore(x, y, f"waypoint @{wp_position}")

    def click_exit(self, exit_position: Tuple[int, int]) -> bool:
        """
        Click on a zone exit to move to the next area.

        Args:
            exit_position: (x, y) position of exit on screen/minimap

        Returns:
            bool: True if click successful
        """
        x, y = exit_position
        return self.click_to_explore(x, y, f"exit @{exit_position}")

    def explore_randomly(self) -> bool:
        """
        Click at a random position to explore the current area.

        Returns:
            bool: True if click successful
        """
        import random
        
        # Get window size to center clicks
        rect = self._get_window_position()
        if rect:
            left, top, right, bottom = rect
            width = right - left
            height = bottom - top
            
            # Click in center region of window (playfield area)
            # Avoid edges which might have UI elements
            center_x = width // 2
            center_y = height // 2
            
            offset_x = random.randint(-300, 300)
            offset_y = random.randint(-200, 200)
            
            x = center_x + offset_x
            y = center_y + offset_y
        else:
            # Fallback: assume 1920x1080 window
            base_x, base_y = 960, 540
            offset_x = random.randint(-300, 300)
            offset_y = random.randint(-200, 200)
            x = base_x + offset_x
            y = base_y + offset_y
        
        return self.click_to_explore(x, y, "random exploration", window_relative=True)

    def check_goal_timeout(self) -> bool:
        """
        Check if current navigation goal has timed out.

        Returns:
            bool: True if goal has timed out
        """
        if not self.current_goal or not self.goal_start_time:
            return False
        
        elapsed = time.time() - self.goal_start_time
        
        if elapsed > self.goal_timeout:
            if self.debug:
                print(f"[CLICK_NAV] WARNING Goal timeout after {elapsed:.1f}s")
            return True
        
        return False

    def get_status(self) -> dict:
        """
        Get current navigation status.

        Returns:
            dict: Status information
        """
        elapsed = None
        if self.goal_start_time:
            elapsed = time.time() - self.goal_start_time
        
        return {
            "current_goal": self.current_goal.mode.value if self.current_goal else None,
            "target": self.current_goal.target_position if self.current_goal else None,
            "elapsed": elapsed,
            "click_count": self.click_count,
        }
