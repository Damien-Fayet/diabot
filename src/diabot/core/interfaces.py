"""Core abstract interfaces for the Diablo 2 bot architecture."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict

import numpy as np


@dataclass
class Perception:
    """Raw perception data from computer vision module."""
    
    hp_ratio: float  # 0.0 to 1.0
    mana_ratio: float  # 0.0 to 1.0
    enemy_count: int
    enemy_types: list[str]
    visible_items: list[str]
    player_position: tuple[int, int]
    raw_data: Dict[str, Any] = None  # For debugging


class ImageSource(ABC):
    """Abstract interface for acquiring game frames."""
    
    @abstractmethod
    def get_frame(self) -> np.ndarray:
        """
        Return a game frame as numpy array (H, W, 3) in BGR format.
        
        Returns:
            np.ndarray: Frame with shape (height, width, 3)
        """
        pass


class VisionModule(ABC):
    """Abstract interface for game state perception."""
    
    @abstractmethod
    def perceive(self, frame: np.ndarray) -> Perception:
        """
        Process a frame and extract game state information.
        
        Args:
            frame: Game frame as numpy array
            
        Returns:
            Perception: Extracted game information
        """
        pass


class ActionExecutor(ABC):
    """Abstract interface for executing game actions."""
    
    @abstractmethod
    def execute_action(self, action_type: str, params: Dict[str, Any] = None) -> bool:
        """
        Execute an action in the game.
        
        Args:
            action_type: Type of action (e.g., 'move', 'attack', 'use_skill')
            params: Action parameters
            
        Returns:
            bool: Success status
        """
        pass


class StateBuilder(ABC):
    """Abstract interface for converting perception into game state."""
    
    @abstractmethod
    def build(self, perception: Perception) -> "GameState":
        """
        Convert raw perception into abstract game state.
        
        Args:
            perception: Raw perception data
            
        Returns:
            GameState: Abstract game state
        """
        pass


class DecisionEngine(ABC):
    """Abstract interface for making decisions based on game state."""
    
    @abstractmethod
    def decide(self, state: "GameState") -> "Action":
        """
        Make a decision given the current game state.
        
        Args:
            state: Current game state
            
        Returns:
            Action: Decided action to perform
        """
        pass
