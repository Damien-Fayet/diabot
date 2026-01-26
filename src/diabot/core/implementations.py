"""Concrete implementations of core interfaces."""

from pathlib import Path
from typing import Any, Dict, Optional
import sys

import cv2
import numpy as np

from diabot.core.interfaces import (
    ImageSource,
    VisionModule,
    ActionExecutor,
    StateBuilder,
    Perception,
)
from diabot.models.state import GameState, Action

# Windows-specific imports
if sys.platform == 'win32':
    try:
        import win32gui
        import win32ui
        import win32con
        from ctypes import windll
        WINDOWS_CAPTURE_AVAILABLE = True
    except ImportError:
        WINDOWS_CAPTURE_AVAILABLE = False
else:
    WINDOWS_CAPTURE_AVAILABLE = False


class ScreenshotFileSource(ImageSource):
    """Load game frames from a file on disk (for developer mode)."""
    
    def __init__(self, image_path: str | Path):
        """
        Initialize with path to an image file.
        
        Args:
            image_path: Path to a .png or .jpg screenshot
        """
        self.image_path = Path(image_path)
        if not self.image_path.exists():
            raise FileNotFoundError(f"Image not found: {self.image_path}")
        
        self._frame = None
        self._load_frame()
    
    def _load_frame(self):
        """Load the frame from disk."""
        self._frame = cv2.imread(str(self.image_path))
        if self._frame is None:
            raise ValueError(f"Failed to load image: {self.image_path}")
    
    def get_frame(self) -> np.ndarray:
        """Return the loaded frame."""
        return self._frame.copy()


class WindowsScreenCapture(ImageSource):
    """Capture live screen from Diablo 2 window on Windows."""
    
    def __init__(self, window_title: str = "Diablo II: Resurrected"):
        """
        Initialize Windows screen capture.
        
        Args:
            window_title: Window title to capture (default: "Diablo II: Resurrected")
        """
        if not WINDOWS_CAPTURE_AVAILABLE:
            raise NotImplementedError(
                "WindowsScreenCapture requires pywin32. Install with: pip install pywin32"
            )
        
        self.window_title = window_title
        self.hwnd: Optional[int] = None
        self._find_window()
    
    def _find_window(self):
        """Find Diablo 2 window handle."""
        self.hwnd = win32gui.FindWindow(None, self.window_title)
        if not self.hwnd:
            raise ValueError(
                f"Window '{self.window_title}' not found. "
                "Make sure Diablo 2 is running and the window title matches."
            )
        print(f"[CAPTURE] ✓ Found window: {self.window_title} (hwnd={self.hwnd})")
    
    def get_frame(self, retry_count: int = 0) -> np.ndarray:
        """
        Capture current frame from Diablo 2 window.
        Args:
            retry_count: Internal, do not use (for recursion limit)
        Returns:
            BGR image as numpy array
        Raises:
            RuntimeError if capture fails after several attempts
        """
        MAX_RETRIES = 3
        if not self.hwnd:
            self._find_window()

        # Get window dimensions
        left, top, right, bottom = win32gui.GetWindowRect(self.hwnd)
        width = right - left
        height = bottom - top

        # Get window device context
        hwndDC = win32gui.GetWindowDC(self.hwnd)
        mfcDC = win32ui.CreateDCFromHandle(hwndDC)
        saveDC = mfcDC.CreateCompatibleDC()

        # Create bitmap
        saveBitMap = win32ui.CreateBitmap()
        saveBitMap.CreateCompatibleBitmap(mfcDC, width, height)
        saveDC.SelectObject(saveBitMap)

        # Copy window content to bitmap
        result = windll.user32.PrintWindow(self.hwnd, saveDC.GetSafeHdc(), 3)

        # Convert to numpy array
        bmpinfo = saveBitMap.GetInfo()
        bmpstr = saveBitMap.GetBitmapBits(True)

        img = np.frombuffer(bmpstr, dtype=np.uint8)
        img.shape = (height, width, 4)  # BGRA format

        # Convert BGRA to BGR
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)

        # Cleanup
        win32gui.DeleteObject(saveBitMap.GetHandle())
        saveDC.DeleteDC()
        mfcDC.DeleteDC()
        win32gui.ReleaseDC(self.hwnd, hwndDC)

        if result != 1:
            print("[CAPTURE] ⚠️  PrintWindow failed, attempting to refind window...")
            self.hwnd = None
            if retry_count < MAX_RETRIES:
                return self.get_frame(retry_count=retry_count+1)
            else:
                raise RuntimeError("[CAPTURE] ERROR: PrintWindow failed after multiple attempts.")

        return img
    
    def get_window_rect(self) -> tuple[int, int, int, int]:
        """Get window rectangle (x, y, right, bottom).
        
        Returns:
            (left, top, right, bottom) in screen coordinates
        """
        if not self.hwnd:
            self._find_window()
        return win32gui.GetWindowRect(self.hwnd)


class RuleBasedVisionModule(VisionModule):
    """Simple rule-based vision module for perception."""
    
    def perceive(self, frame: np.ndarray) -> Perception:
        """
        Extract basic perception from frame.
        
        For now, this is a dummy implementation. Later will use:
        - Color thresholding for HP/Mana bars
        - Object detection for enemies
        - Semantic segmentation for environment
        
        Args:
            frame: Game frame
            
        Returns:
            Perception: Extracted data
        """
        # Placeholder implementation
        return Perception(
            hp_ratio=0.75,
            mana_ratio=0.50,
            enemy_count=0,
            enemy_types=[],
            visible_items=[],
            player_position=(frame.shape[1] // 2, frame.shape[0] // 2),
            raw_data={"frame_shape": frame.shape},
        )


class SimpleStateBuilder(StateBuilder):
    """Convert perception into abstract game state."""
    
    def __init__(self, frame_counter: int = 0):
        """Initialize state builder."""
        self.frame_counter = frame_counter
    
    def build(self, perception: Perception) -> GameState:
        """
        Convert Perception to GameState.
        
        Args:
            perception: Raw perception data
            
        Returns:
            GameState: Abstract game state
        """
        return GameState(
            health_percent=perception.hp_ratio * 100.0,
            mana_percent=perception.mana_ratio * 100.0,
            enemy_count=perception.enemy_count,
            visible_items=len(perception.visible_items),
            current_location="dungeon",  # Placeholder
            frame_number=self.frame_counter,
        )


class RuleBasedDecisionEngine:
    """Rule-based decision making (placeholder for future ML)."""
    
    def decide(self, state: GameState) -> Action:
        """
        Make a decision based on game state.
        
        Simple rules for now:
        - If threatened, attack
        - If low health, drink potion
        - Otherwise, explore
        
        Args:
            state: Current game state
            
        Returns:
            Action: Decided action
        """
        if state.needs_potion:
            return Action(action_type="drink_potion")
        elif state.is_threatened:
            return Action(action_type="attack", target="enemy_0")
        else:
            return Action(action_type="explore", target="north")


class DummyActionExecutor(ActionExecutor):
    """Placeholder action executor (doesn't actually do anything)."""
    
    def execute_action(self, action_type: str, params: Dict[str, Any] = None) -> bool:
        """
        Dummy execution - just logs the action.
        
        Args:
            action_type: Type of action
            params: Action parameters
            
        Returns:
            bool: Always returns True (simulated success)
        """
        print(f"[ACTION] {action_type} with params {params or {}}")
        return True
    
    def execute(self, action, frame) -> bool:
        """
        Execute action (compatible with new Action dataclass).
        
        Args:
            action: Action object with action_type, target, params
            frame: Current game frame
            
        Returns:
            bool: Always returns True (simulated success)
        """
        params_dict = getattr(action, 'params', {})
        if hasattr(action, 'target') and action.target:
            params_dict['target'] = action.target
        return self.execute_action(action.action_type, params_dict)

class WindowsActionExecutor(ActionExecutor):
    """Real action executor using pyautogui for mouse/keyboard on Windows."""
    
    def __init__(self, debug: bool = False, image_source=None):
        """
        Initialize Windows action executor.
        
        Args:
            debug: Enable debug logging
            image_source: Optional WindowsScreenCapture for window bounds
        """
        self.debug = debug
        self.image_source = image_source
        try:
            import pyautogui
            self.pyautogui = pyautogui
            # Set safety settings
            self.pyautogui.PAUSE = 0.05  # Small pause between actions
            self.pyautogui.FAILSAFE = True  # Move mouse to top-left to abort
            if self.debug:
                print("[WindowsActionExecutor] Initialized with pyautogui")
        except ImportError:
            raise ImportError("pyautogui is required for WindowsActionExecutor")
    
    def _clamp_to_window(self, x: int, y: int) -> tuple[int, int]:
        """
        Clamp screen coordinates to game window bounds.
        
        Args:
            x, y: Absolute screen coordinates
            
        Returns:
            Clamped coordinates within game window
        """
        if not self.image_source or not hasattr(self.image_source, 'get_window_rect'):
            # No window bounds available, return as-is
            return x, y
        
        try:
            left, top, right, bottom = self.image_source.get_window_rect()
            width = right - left
            height = bottom - top
            
            # Clamp to window bounds (with 5% margin)
            margin_x = int(width * 0.05)
            margin_y = int(height * 0.05)
            
            x_clamped = max(left + margin_x, min(x, right - margin_x))
            y_clamped = max(top + margin_y, min(y, bottom - margin_y))
            
            if self.debug and (x != x_clamped or y != y_clamped):
                print(f"[CLICK] Clamped ({x},{y}) → ({x_clamped},{y_clamped}) [Window: {left},{top} → {right},{bottom}]")
            
            return x_clamped, y_clamped
        except Exception as e:
            if self.debug:
                print(f"[CLICK] Warning: Failed to clamp coordinates: {e}")
            return x, y
    
    def get_window_click_position(self, rel_x: float, rel_y: float) -> tuple[int, int]:
        """
        Convert relative coordinates (0.0-1.0) to absolute screen coordinates
        within the game window.
        
        Args:
            rel_x: Relative X position (0.0 = left, 1.0 = right)
            rel_y: Relative Y position (0.0 = top, 1.0 = bottom)
            
        Returns:
            Absolute screen coordinates (x, y) clamped to window bounds
        """
        if not self.image_source or not hasattr(self.image_source, 'get_window_rect'):
            # Fallback to absolute coordinates if no window bounds
            return (int(rel_x * 1920), int(rel_y * 1080))
        
        try:
            left, top, right, bottom = self.image_source.get_window_rect()
            width = right - left
            height = bottom - top
            
            # Convert relative to absolute within window
            x = left + int(width * rel_x)
            y = top + int(height * rel_y)
            
            if self.debug:
                print(f"[CLICK_COORD] Window rect: ({left},{top})->({right},{bottom}), size: {width}x{height}")
                print(f"[CLICK_COORD] Relative ({rel_x:.2f},{rel_y:.2f}) → Absolute ({x},{y})")
            
            return x, y
        except Exception as e:
            if self.debug:
                print(f"[CLICK] Warning: Failed to convert relative coordinates: {e}")
            # Fallback
            return (int(rel_x * 1920), int(rel_y * 1080))
    
    def execute_action(self, action_type: str, params: Dict[str, Any] = None) -> bool:
        """
        Execute action with real mouse/keyboard input.
        
        Args:
            action_type: Type of action (move, attack, pickup, etc.)
            params: Action parameters (position, target, etc.)
            
        Returns:
            bool: True if action executed successfully
        """
        if params is None:
            params = {}
        
        try:
            if action_type == "move":
                position = params.get("position")
                if position:
                    x, y = position
                    # Clamp to window bounds
                    x, y = self._clamp_to_window(x, y)
                    # Click at position (left click for movement)
                    self.pyautogui.click(x, y)
                    if self.debug:
                        print(f"[CLICK] Left click at ({x}, {y})")
                    return True
            
            elif action_type == "attack":
                position = params.get("position")
                if position:
                    x, y = position
                    # Clamp to window bounds
                    x, y = self._clamp_to_window(x, y)
                    # Right click for attack/interact
                    self.pyautogui.click(x, y, button='right')
                    if self.debug:
                        print(f"[CLICK] Right click at ({x}, {y})")
                    return True
            
            elif action_type == "pickup":
                position = params.get("position")
                if position:
                    x, y = position
                    # Clamp to window bounds
                    x, y = self._clamp_to_window(x, y)
                    # Left click to pick up item
                    self.pyautogui.click(x, y)
                    if self.debug:
                        print(f"[CLICK] Pickup at ({x}, {y})")
                    return True
            
            elif action_type == "skill":
                key = params.get("key", "q")
                position = params.get("position")
                # Press skill key then click position
                self.pyautogui.press(key)
                if position:
                    x, y = position
                    self.pyautogui.click(x, y)
                if self.debug:
                    print(f"[SKILL] Pressed {key}, clicked {position}")
                return True
            
            else:
                if self.debug:
                    print(f"[WARNING] Unknown action type: {action_type}")
                return False
                
        except Exception as e:
            if self.debug:
                print(f"[ERROR] Failed to execute {action_type}: {e}")
            return False
    
    def execute(self, action, frame) -> bool:
        """
        Execute action (compatible with new Action dataclass).
        
        Args:
            action: Action object with action_type, target, params
            frame: Current game frame
            
        Returns:
            bool: True if action executed successfully
        """
        params_dict = getattr(action, 'params', {})
        if hasattr(action, 'target') and action.target:
            params_dict['target'] = action.target
        return self.execute_action(action.action_type, params_dict)