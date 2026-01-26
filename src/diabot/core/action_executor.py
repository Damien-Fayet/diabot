"""Action executor for bot - implements game interactions."""

from __future__ import annotations

import sys
import time
from typing import Optional

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


class ActionExecutor:
    """Executes bot actions (clicks, delays, etc.) on the game.
    
    Handles mouse clicks at specific coordinates with window management.
    """
    
    def __init__(self, window_title: str = "Diablo II: Resurrected", debug: bool = False):
        """Initialize action executor.
        
        Args:
            window_title: Game window title
            debug: Enable debug logging
        """
        if not MOUSE_CONTROL_AVAILABLE:
            raise ImportError("ActionExecutor requires pyautogui and win32gui")
        
        self.window_title = window_title
        self.debug = debug
        self.last_click_time = 0.0
        self.click_count = 0
        
        # Pyautogui settings
        pyautogui.FAILSAFE = True
        pyautogui.PAUSE = 0.1  # Safety delay between actions
    
    def _find_window(self) -> Optional[int]:
        """Find game window handle by title.
        
        Returns:
            Window handle or None if not found
        """
        hwnd = None
        try:
            hwnd = win32gui.FindWindow(None, self.window_title)
        except Exception as e:
            if self.debug:
                print(f"⚠️  Error finding window: {e}")
            return None
        
        return hwnd if hwnd else None
    
    def _activate_window(self, hwnd: int) -> bool:
        """Activate game window.
        
        Args:
            hwnd: Window handle
            
        Returns:
            True if successful
        """
        try:
            # Bring to foreground
            win32gui.SetForegroundWindow(hwnd)
            time.sleep(0.05)
            return True
        except Exception as e:
            if self.debug:
                print(f"⚠️  Error activating window: {e}")
            return False
    
    def _get_window_rect(self, hwnd: int) -> Optional[tuple[int, int, int, int]]:
        """Get window rect (x, y, right, bottom).
        
        Args:
            hwnd: Window handle
            
        Returns:
            (x, y, right, bottom) or None
        """
        try:
            rect = win32gui.GetWindowRect(hwnd)
            return rect
        except Exception as e:
            if self.debug:
                print(f"⚠️  Error getting window rect: {e}")
            return None
    
    def click_at_screen_position(self, screen_x: int, screen_y: int, 
                                 button: str = "left", delay: float = 0.1) -> bool:
        """Click at absolute screen position.
        
        Args:
            screen_x: Screen X coordinate
            screen_y: Screen Y coordinate
            button: "left" or "right"
            delay: Delay after click (seconds)
            
        Returns:
            True if successful
        """
        try:
            # Move and click
            pyautogui.moveTo(screen_x, screen_y, duration=0.05)
            pyautogui.click(button=button)
            
            if delay > 0:
                time.sleep(delay)
            
            self.click_count += 1
            self.last_click_time = time.time()
            
            if self.debug:
                print(f"✓ Clicked at screen ({screen_x}, {screen_y})")
            
            return True
        except Exception as e:
            if self.debug:
                print(f"❌ Click failed: {e}")
            return False
    
    def click_on_npc(self, screen_x: int, screen_y: int, 
                    offset_down: int = 30) -> bool:
        """Click on an NPC.
        
        Typically, quest markers appear above NPCs. This clicks slightly
        below the marker to hit the NPC itself.
        
        Args:
            screen_x: Screen X coordinate (marker position)
            screen_y: Screen Y coordinate (marker position)
            offset_down: Pixels to offset downward to hit NPC
            
        Returns:
            True if successful
        """
        # Adjust click position down to hit NPC below marker
        click_x = screen_x
        click_y = screen_y + offset_down
        
        return self.click_at_screen_position(click_x, click_y, button="left", delay=0.3)
    
    def interact_with_object(self, screen_x: int, screen_y: int) -> bool:
        """Generic interaction with game object (NPC, item, etc).
        
        Args:
            screen_x: Screen X coordinate
            screen_y: Screen Y coordinate
            
        Returns:
            True if successful
        """
        return self.click_at_screen_position(screen_x, screen_y, button="left", delay=0.2)
    
    def move_to(self, screen_x: int, screen_y: int) -> bool:
        """Move character to position (right-click).
        
        Args:
            screen_x: Target screen X
            screen_y: Target screen Y
            
        Returns:
            True if successful
        """
        return self.click_at_screen_position(screen_x, screen_y, button="right", delay=0.1)
    
    def use_skill(self, skill_key: str, target_x: int, target_y: int) -> bool:
        """Use a skill at a position.
        
        Args:
            skill_key: Keyboard shortcut for skill (e.g., 'e', 't')
            target_x: Target screen X
            target_y: Target screen Y
            
        Returns:
            True if successful
        """
        try:
            # Press skill key
            pyautogui.press(skill_key)
            time.sleep(0.1)
            
            # Click target
            pyautogui.click(target_x, target_y)
            time.sleep(0.1)
            
            if self.debug:
                print(f"✓ Used skill '{skill_key}' at ({target_x}, {target_y})")
            
            return True
        except Exception as e:
            if self.debug:
                print(f"❌ Skill use failed: {e}")
            return False
    
    def drink_potion(self) -> bool:
        """Drink a health potion (press 'h').
        
        Returns:
            True if successful
        """
        try:
            pyautogui.press('h')
            time.sleep(0.2)
            
            if self.debug:
                print(f"✓ Drank potion")
            
            return True
        except Exception as e:
            if self.debug:
                print(f"❌ Potion drink failed: {e}")
            return False
    
    def get_stats(self) -> dict:
        """Get executor stats."""
        return {
            "total_clicks": self.click_count,
            "last_click_time": self.last_click_time,
        }
