"""Debug visualization utilities."""

from typing import Optional

import cv2
import numpy as np

from diabot.models.state import GameState, Action
from diabot.core.interfaces import Perception


class BrainOverlay:
    """
    Advanced visual debug system to understand agent perception and decisions.
    
    Purpose:
    - Show what the agent perceives (perception data)
    - Show what the agent thinks (FSM state, threat level)
    - Show what the agent decides (action + reasoning)
    
    Design:
    - Purely visual (no game interaction)
    - Works with static screenshots (developer mode)
    - Toggleable via configuration
    - No coupling to vision logic
    """
    
    def __init__(self, enabled: bool = True):
        """
        Initialize BrainOverlay.
        
        Args:
            enabled: Whether overlay is active
        """
        self.enabled = enabled
        self.font = cv2.FONT_HERSHEY_SIMPLEX
        self.font_scale = 0.6
        self.thickness = 2
        
        # Colors (BGR format)
        self.COLOR_SAFE = (0, 255, 0)      # Green
        self.COLOR_DANGER = (0, 0, 255)    # Red
        self.COLOR_WARNING = (0, 165, 255) # Orange
        self.COLOR_INFO = (255, 255, 255)  # White
        self.COLOR_TARGET = (255, 0, 0)    # Blue
    
    def draw(
        self,
        frame: np.ndarray,
        perception: Optional[Perception] = None,
        state: Optional[GameState] = None,
        action: Optional[Action] = None,
        fsm_state: Optional[str] = None,
    ) -> np.ndarray:
        """
        Draw complete overlay on frame.
        
        Args:
            frame: Original game frame
            perception: Perception data from vision
            state: Game state
            action: Decided action
            fsm_state: Current FSM state name
            
        Returns:
            Frame with overlay drawn
        """
        if not self.enabled:
            return frame
        
        output = frame.copy()
        
        # Draw different sections
        y_offset = 30
        
        if fsm_state:
            y_offset = self._draw_fsm_state(output, fsm_state, y_offset)
        
        if action:
            y_offset = self._draw_action(output, action, y_offset)
        
        if perception:
            y_offset = self._draw_perception(output, perception, y_offset)
        
        if state:
            y_offset = self._draw_state_info(output, state, y_offset)
            self._draw_health_bar(output, state)
            self._draw_threat_indicator(output, state)
        
        return output
    
    def _draw_fsm_state(self, frame: np.ndarray, fsm_state: str, y: int) -> int:
        """Draw FSM state at top."""
        # Choose color based on state
        color = self.COLOR_INFO
        if fsm_state == "PANIC":
            color = self.COLOR_DANGER
        elif fsm_state in ["KITE", "ENGAGE"]:
            color = self.COLOR_WARNING
        elif fsm_state == "EXPLORE":
            color = self.COLOR_SAFE
        
        text = f"FSM: {fsm_state}"
        cv2.putText(frame, text, (10, y), self.font, 0.8, color, self.thickness)
        return y + 35
    
    def _draw_action(self, frame: np.ndarray, action: Action, y: int) -> int:
        """Draw decided action."""
        text = f"Action: {action.action_type}"
        if action.target:
            text += f" -> {action.target}"
        
        cv2.putText(frame, text, (10, y), self.font, self.font_scale, self.COLOR_TARGET, self.thickness)
        return y + 30
    
    def _draw_perception(self, frame: np.ndarray, perception: Perception, y: int) -> int:
        """Draw perception data."""
        texts = [
            f"HP: {perception.hp_ratio*100:.1f}%",
            f"Mana: {perception.mana_ratio*100:.1f}%",
            f"Enemies: {perception.enemy_count}",
        ]
        
        for text in texts:
            cv2.putText(frame, text, (10, y), self.font, self.font_scale, self.COLOR_INFO, 1)
            y += 25
        
        return y + 5
    
    def _draw_state_info(self, frame: np.ndarray, state: GameState, y: int) -> int:
        """Draw game state information."""
        threat_level = state.debug_info.get("threat_level", "none")
        
        # Threat level with color
        color = self.COLOR_SAFE
        if threat_level == "critical":
            color = self.COLOR_DANGER
        elif threat_level in ["high", "medium"]:
            color = self.COLOR_WARNING
        
        text = f"Threat: {threat_level.upper()}"
        cv2.putText(frame, text, (10, y), self.font, self.font_scale, color, self.thickness)
        y += 30
        
        # Location
        text = f"Location: {state.current_location}"
        cv2.putText(frame, text, (10, y), self.font, self.font_scale, self.COLOR_INFO, 1)
        y += 25
        
        return y
    
    def _draw_health_bar(self, frame: np.ndarray, state: GameState):
        """Draw health bar (bottom-left)."""
        h, w = frame.shape[:2]
        
        bar_width = 200
        bar_height = 25
        bar_x, bar_y = 10, h - 40
        
        # Background
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_width, bar_y + bar_height), (0, 0, 0), -1)
        
        # Health fill
        health_width = int(bar_width * (state.health_percent / 100.0))
        health_color = (
            0,
            int(255 * (state.health_percent / 100.0)),
            int(255 * (1 - state.health_percent / 100.0))
        )
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + health_width, bar_y + bar_height), health_color, -1)
        
        # Border
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_width, bar_y + bar_height), self.COLOR_INFO, 2)
        
        # Text
        text = f"HP: {state.health_percent:.0f}%"
        cv2.putText(frame, text, (bar_x + 5, bar_y + 18), self.font, 0.5, (255, 255, 255), 1)
    
    def _draw_threat_indicator(self, frame: np.ndarray, state: GameState):
        """Draw threat indicator (top-right corner)."""
        h, w = frame.shape[:2]
        
        threat_level = state.debug_info.get("threat_level", "none")
        
        # Circle indicator
        center_x = w - 50
        center_y = 50
        radius = 30
        
        # Color based on threat
        if threat_level == "critical":
            color = self.COLOR_DANGER
        elif threat_level == "high":
            color = self.COLOR_WARNING
        elif threat_level == "medium":
            color = (0, 200, 255)
        else:
            color = self.COLOR_SAFE
        
        cv2.circle(frame, (center_x, center_y), radius, color, -1)
        cv2.circle(frame, (center_x, center_y), radius, (255, 255, 255), 2)
        
        # Enemy count in circle
        if state.enemy_count > 0:
            text = str(state.enemy_count)
            text_size = cv2.getTextSize(text, self.font, 0.8, 2)[0]
            text_x = center_x - text_size[0] // 2
            text_y = center_y + text_size[1] // 2
            cv2.putText(frame, text, (text_x, text_y), self.font, 0.8, (255, 255, 255), 2)


class DebugOverlay:
    """Legacy debug overlay (kept for backward compatibility)."""
    
    @staticmethod
    def draw_state(frame: np.ndarray, state: GameState) -> np.ndarray:
        """
        Draw game state information as overlay on frame.
        
        Args:
            frame: Input game frame (BGR)
            state: Game state to visualize
            
        Returns:
            np.ndarray: Frame with overlay drawn
        """
        output = frame.copy()
        
        # Font settings
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.6
        color = (0, 255, 0)  # Green
        thickness = 1
        y_offset = 30
        
        # Draw text overlays
        texts = [
            f"Health: {state.health_percent:.1f}%",
            f"Mana: {state.mana_percent:.1f}%",
            f"Enemies: {state.enemy_count}",
            f"Location: {state.current_location}",
            f"Threatened: {state.is_threatened}",
            f"Frame: {state.frame_number}",
        ]
        
        for i, text in enumerate(texts):
            y = y_offset + (i * 25)
            cv2.putText(
                output,
                text,
                (10, y),
                font,
                font_scale,
                color,
                thickness,
            )
        
        # Draw health bar (top-left area)
        bar_width = 200
        bar_height = 20
        bar_x, bar_y = 10, y_offset + len(texts) * 25 + 10
        
        # Background
        cv2.rectangle(output, (bar_x, bar_y), (bar_x + bar_width, bar_y + bar_height), (0, 0, 0), -1)
        
        # Health bar
        health_width = int(bar_width * (state.health_percent / 100.0))
        health_color = (0, int(255 * (1 - state.health_percent / 100.0)), int(255 * (state.health_percent / 100.0)))
        cv2.rectangle(output, (bar_x, bar_y), (bar_x + health_width, bar_y + bar_height), health_color, -1)
        
        # Border
        cv2.rectangle(output, (bar_x, bar_y), (bar_x + bar_width, bar_y + bar_height), color, 2)
        
        return output
