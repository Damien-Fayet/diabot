"""Debug visualization utilities."""

from typing import Optional, List

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
    
    def __init__(self, enabled: bool = True, show_boxes: bool = True, show_indicators: bool = True):
        """
        Initialize BrainOverlay.
        
        Args:
            enabled: Whether overlay is active
            show_boxes: Draw detections and targets
            show_indicators: Draw danger/safe indicators
        """
        self.enabled = enabled
        self.show_boxes = show_boxes
        self.show_indicators = show_indicators
        self.font = cv2.FONT_HERSHEY_SIMPLEX
        self.font_scale = 0.6
        self.thickness = 2
        
        # Colors (BGR format)
        self.COLOR_SAFE = (0, 255, 0)      # Green
        self.COLOR_DANGER = (0, 0, 255)    # Red
        self.COLOR_WARNING = (0, 165, 255) # Orange
        self.COLOR_INFO = (255, 255, 255)  # White
        self.COLOR_TARGET = (255, 0, 0)    # Blue
        self.COLOR_TEMPLATE = (255, 255, 0) # Cyan (for templates/NPCs)
        self.COLOR_ZONE = (180, 105, 255)  # Light purple for zone text
    
    def draw(
        self,
        frame: np.ndarray,
        perception: Optional[Perception] = None,
        state: Optional[GameState] = None,
        action: Optional[Action] = None,
        fsm_state: Optional[str] = None,
        detections: Optional[List] = None,
        zone_name: Optional[str] = None,
        minimap_crop: Optional[np.ndarray] = None,
        minimap_pois: Optional[List] = None,
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
            y_offset = self._draw_state_info(output, state, y_offset, zone_name)
            self._draw_health_bar(output, state)
            if self.show_indicators:
                self._draw_threat_indicator(output, state)
                self._draw_decision_driver(output, state)

        # Draw entities (enemies/items) when provided
        if perception and perception.raw_data:
            self._draw_entities(output, perception)
        # YOLO detections (live mode)
        # Si perception.raw_data['yolo_boxes'] existe, on affiche les bounding boxes YOLO
        if self.show_boxes:
            yolo_boxes = None
            if perception and perception.raw_data and 'yolo_boxes' in perception.raw_data:
                yolo_boxes = perception.raw_data['yolo_boxes']
            if yolo_boxes:
                for box in yolo_boxes:
                    x1, y1, x2, y2 = map(int, box['bbox'])
                    class_name = box.get('class_name', 'enemy')
                    conf = box.get('confidence', 0.0)
                    color = self.COLOR_DANGER if class_name == 'enemy' else self.COLOR_INFO
                    cv2.rectangle(output, (x1, y1), (x2, y2), color, 2)
                    label = f"{class_name} {conf:.2f}"
                    cv2.putText(output, label, (x1, max(12, y1 - 5)), self.font, 0.45, color, 1)
            elif detections:
                self._draw_detections(output, detections, action)
        
        # Affichage de la minimap et des POIs compris par le bot (en bas à droite)
        if minimap_crop is not None:
            h, w = output.shape[:2]
            # Redimensionne la minimap pour affichage (ex: 200x200)
            mini = cv2.resize(minimap_crop, (200, 200))
            # Dessine les POIs sur la minimap
            if minimap_pois:
                for poi in minimap_pois:
                    px, py = int(poi.position[0] * 200 / minimap_crop.shape[1]), int(poi.position[1] * 200 / minimap_crop.shape[0])
                    color = (0, 255, 255) if poi.poi_type == 'waypoint' else (255, 0, 255)
                    cv2.circle(mini, (px, py), 6, color, 2)
                    cv2.putText(mini, poi.poi_type[:2].upper(), (px+8, py), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
            # Place la minimap en bas à droite de l'overlay
            x_offset = w - 210
            y_offset = h - 210
            output[y_offset:y_offset+200, x_offset:x_offset+200] = mini
            cv2.rectangle(output, (x_offset, y_offset), (x_offset+200, y_offset+200), (255,255,255), 2)
            cv2.putText(output, "Minimap", (x_offset+10, y_offset+25), self.font, 0.7, (255,255,255), 2)
        return output

    def _draw_entities(self, frame: np.ndarray, perception: Perception):
        """Draw detected enemies/items on the frame using perception raw data."""
        env_state = perception.raw_data.get("env_state") if perception.raw_data else None
        playfield_bounds = perception.raw_data.get("playfield_bounds") if perception.raw_data else None

        if env_state is None:
            return

        pf_x, pf_y = 0, 0
        if playfield_bounds:
            pf_x, pf_y, _, _ = playfield_bounds

        # Enemies
        for idx, enemy in enumerate(env_state.enemies, 1):
            ex, ey, ew, eh = enemy.bbox
            ex += pf_x
            ey += pf_y

            color = self.COLOR_DANGER if enemy.enemy_type == "large_enemy" else self.COLOR_TARGET

            cv2.rectangle(frame, (ex, ey), (ex + ew, ey + eh), color, 2)
            cv2.putText(
                frame,
                f"#{idx} {enemy.confidence:.2f}",
                (ex, max(10, ey - 5)),
                self.font,
                0.4,
                color,
                1,
            )

        # Items with bbox information
        if hasattr(env_state, "items"):
            for item in env_state.items:
                if not isinstance(item, dict) or "bbox" not in item:
                    continue
                ix, iy, iw, ih = item["bbox"]
                ix += pf_x
                iy += pf_y
                cv2.rectangle(frame, (ix, iy), (ix + iw, iy + ih), self.COLOR_WARNING, 1)
                cv2.putText(frame, "item", (ix, max(10, iy - 3)), self.font, 0.35, self.COLOR_WARNING, 1)
        
        # Template-detected objects (NPCs, waypoints, quests)
        if hasattr(env_state, "template_objects"):
            for idx, obj in enumerate(env_state.template_objects, 1):
                ox, oy = obj.position
                # Draw filled circle at detection point
                cv2.circle(frame, (ox, oy), 12, self.COLOR_TEMPLATE, -1)  # Filled
                cv2.circle(frame, (ox, oy), 12, (0, 0, 0), 2)  # Black outline
                # Draw index number inside circle
                cv2.putText(
                    frame,
                    str(idx),
                    (ox - 5, oy + 5),
                    self.font,
                    0.5,
                    (0, 0, 0),  # Black text
                    2,
                )
                # Draw label with object type (larger, more readable)
                label = f"{obj.object_type.upper()}"
                # Add background rectangle for better readability
                label_size = cv2.getTextSize(label, self.font, 0.6, 2)[0]
                label_x = ox + 18
                label_y = oy - 10
                cv2.rectangle(
                    frame,
                    (label_x - 2, label_y - label_size[1] - 2),
                    (label_x + label_size[0] + 2, label_y + 4),
                    (0, 0, 0),
                    -1,
                )
                cv2.putText(
                    frame,
                    label,
                    (label_x, label_y),
                    self.font,
                    0.6,
                    self.COLOR_TEMPLATE,
                    2,
                )
                # Draw confidence below
                conf_label = f"{obj.confidence:.2f}"
                cv2.putText(
                    frame,
                    conf_label,
                    (ox + 18, oy + 10),
                    self.font,
                    0.4,
                    self.COLOR_TEMPLATE,
                    1,
                )
    
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
    
    def _draw_state_info(self, frame: np.ndarray, state: GameState, y: int, zone_name: Optional[str]) -> int:
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
        
        # Location / zone
        zone_text = zone_name or state.current_location
        text = f"Zone: {zone_text}"
        cv2.putText(frame, text, (10, y), self.font, self.font_scale, self.COLOR_ZONE, 1)
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

    def _draw_decision_driver(self, frame: np.ndarray, state: GameState):
        """Show main decision driver (danger vs safe)."""
        h, w = frame.shape[:2]
        label = "Safe Exploration"
        color = self.COLOR_SAFE
        if state.needs_potion or state.health_percent < 30:
            label = "Low HP"
            color = self.COLOR_DANGER
        elif state.enemy_count > 0:
            label = "Enemy Nearby"
            color = self.COLOR_TARGET
        cv2.putText(frame, label, (w - 220, h - 20), self.font, 0.6, color, 2)

    def _draw_detections(self, frame: np.ndarray, detections: List, action: Optional[Action]):
        """Draw YOLO detections with color coding."""
        target_pos = None
        if action and action.params.get("position"):
            target_pos = action.params.get("position")
        for det in detections:
            x1, y1, x2, y2 = map(int, det.bbox)
            color = self.COLOR_TARGET if det.class_name in ["quest", "waypoint"] else self.COLOR_INFO
            if det.class_name in ["zombie", "fallen", "quill rat", "corps"]:
                color = self.COLOR_DANGER
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            label = f"{det.class_name} {det.confidence:.2f}"
            cv2.putText(frame, label, (x1, max(12, y1 - 5)), self.font, 0.45, color, 1)
        if target_pos:
            tx, ty = map(int, target_pos)
            cv2.circle(frame, (tx, ty), 10, self.COLOR_TARGET, 2)


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
