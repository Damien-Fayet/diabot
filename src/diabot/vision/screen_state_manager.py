"""Game screen state machine and screen-aware action handler."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Callable

import numpy as np

from diabot.core.interfaces import ActionExecutor
from diabot.models import BotState
from diabot.vision.screen_detector import ScreenDetector, GameScreen, ScreenDetectionResult


@dataclass
class ScreenStateTransition:
    """Represents a transition between screens."""
    from_screen: GameScreen
    to_screen: GameScreen
    action_required: Optional[str] = None  # Action to take during transition


class ScreenStateManager:
    """
    Manages game screen state and coordinates actions.
    
    Handles:
    - Screen detection and state tracking
    - State transitions
    - Screen-specific action dispatch
    - Death/resurrection flow
    """
    
    def __init__(
        self,
        bot_state: BotState,
        executor: Optional[ActionExecutor] = None,
        screen_timeout_sec: float = 10.0,
    ):
        self.bot_state = bot_state
        self.executor = executor
        self.screen_timeout_sec = screen_timeout_sec
        
        self.detector = ScreenDetector()
        
        # Current screen state
        self.current_screen = GameScreen.UNKNOWN
        self.screen_changes = 0
        self.frames_on_screen = 0
        
        # Screen-specific handlers
        self.handlers: dict[GameScreen, Callable] = {
            GameScreen.GAMEPLAY: self._handle_gameplay,
            GameScreen.DEAD: self._handle_dead,
            GameScreen.CHAR_SELECT: self._handle_char_select,
            GameScreen.MAIN_MENU: self._handle_main_menu,
            GameScreen.LOADING: self._handle_loading,
        }
    
    def update(self, frame: np.ndarray) -> ScreenDetectionResult:
        """
        Update screen state from frame.
        
        Args:
            frame: Current game frame
            
        Returns:
            ScreenDetectionResult with current screen info
        """
        detection = self.detector.detect(frame)
        
        # Track state changes
        if detection.screen_type != self.current_screen:
            self.screen_changes += 1
            self.frames_on_screen = 0
            
            # Log transition
            print(
                f"[SCREEN] {self.current_screen.value} → {detection.screen_type.value} "
                f"(confidence: {detection.confidence:.2f})"
            )
            self.current_screen = detection.screen_type
        else:
            self.frames_on_screen += 1
        
        return detection
    
    def handle_screen_action(self, frame: np.ndarray) -> Optional[str]:
        """
        Execute appropriate action for current screen.
        
        Args:
            frame: Current game frame
            
        Returns:
            Action name if executed, None otherwise
        """
        handler = self.handlers.get(self.current_screen)
        
        if handler:
            return handler(frame)
        
        return None
    
    def _handle_gameplay(self, frame: np.ndarray) -> Optional[str]:
        """Handle normal gameplay - navigation continues."""
        # No special action needed, orchestrator handles navigation
        return None
    
    def _handle_dead(self, frame: np.ndarray) -> Optional[str]:
        """Handle death screen - options: respawn, reload, quit."""
        print("[SCREEN] Player is dead! Handling resurrection...")
        
        # Update bot state
        self.bot_state.hp_ratio = 0.0
        
        if self.executor:
            # Execute respawn action (click OK or similar)
            action_name = "respawn"
            success = self.executor.execute_action(
                action_name,
                {"params": {"screen": "death"}}
            )
            
            if success:
                print("[ACTION] Respawn executed")
                return action_name
        
        return None
    
    def _handle_char_select(self, frame: np.ndarray) -> Optional[str]:
        """Handle character selection screen."""
        print("[SCREEN] On character select screen")
        
        if self.executor:
            # Select character (click on first available)
            action_name = "select_character"
            success = self.executor.execute_action(
                action_name,
                {"params": {"char_index": 0}}
            )
            
            if success:
                print("[ACTION] Character selected")
                return action_name
        
        return None
    
    def _handle_main_menu(self, frame: np.ndarray) -> Optional[str]:
        """Handle main menu screen."""
        print("[SCREEN] On main menu")
        
        if self.executor:
            # Click play/continue button
            action_name = "menu_play"
            success = self.executor.execute_action(
                action_name,
                {"params": {"button": "play"}}
            )
            
            if success:
                print("[ACTION] Menu action executed")
                return action_name
        
        return None
    
    def _handle_loading(self, frame: np.ndarray) -> Optional[str]:
        """Handle loading screen - wait and do nothing."""
        # Just wait for loading to finish, no action needed
        # Orchestrator will skip navigation during loading
        return None
    
    def is_actively_playing(self) -> bool:
        """Check if game is in active gameplay state."""
        return self.current_screen == GameScreen.GAMEPLAY
    
    def is_dead(self) -> bool:
        """Check if player is dead."""
        return self.current_screen == GameScreen.DEAD
    
    def is_loading(self) -> bool:
        """Check if game is loading."""
        return self.current_screen == GameScreen.LOADING
    
    def is_in_menu(self) -> bool:
        """Check if in any menu screen."""
        return self.current_screen in (
            GameScreen.MAIN_MENU,
            GameScreen.CHAR_SELECT,
            GameScreen.PAUSE,
        )
