"""Concrete implementations of core interfaces."""

from pathlib import Path
from typing import Any, Dict

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
    """Capture live screen on Windows (placeholder - not implemented on macOS)."""
    
    def __init__(self):
        """Initialize Windows screen capture."""
        raise NotImplementedError(
            "WindowsScreenCapture not available on this platform. "
            "Use ScreenshotFileSource for developer mode on macOS."
        )
    
    def get_frame(self) -> np.ndarray:
        """Not implemented."""
        pass


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
