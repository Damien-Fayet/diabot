"""Main bot orchestrator - combines vision, FSM, quests, and actions."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Optional

import cv2

import numpy as np

from diabot.core.implementations import WindowsScreenCapture
from diabot.core.action_executor import ActionExecutor
from diabot.decision.diablo_fsm import DiabloFSM, FSMState
from diabot.models.state import GameState, EnemyInfo, Action
from diabot.progression import QuestManager
from diabot.vision.yolo_detector import YOLODetector, Detection
from diabot.vision.ui_vision import UIVisionModule, UIState
from diabot.vision.minimap_parser import MinimapParser
from diabot.navigation.map_system import GameMap, MapNavigator
from diabot.debug.overlay import BrainOverlay


class DiabloBot:
    """Main bot orchestrator.
    
    Integrates:
    - Screen capture (Windows-specific)
    - YOLO vision
    - Quest progression
    - FSM decision making
    - Action execution (TODO)
    """
    
    def __init__(
        self,
        yolo_model_path: Path | str,
        progression_file: Path | str = "data/game_progression.json",
        window_title: str = "Diablo II: Resurrected",
        debug: bool = False,
        overlay_enabled: bool = False,
        overlay_dir: Path | str = "data/screenshots/overlays",
        overlay_show: bool = False,
    ):
        """Initialize bot.
        
        Args:
            yolo_model_path: Path to trained YOLO model
            progression_file: Path to game_progression.json
            window_title: Game window title
            debug: Enable debug logging
        """
        self.debug = debug
        self.window_title = window_title
        self.overlay_enabled = overlay_enabled
        self.overlay_dir = Path(overlay_dir)
        self.overlay_show = overlay_show
        
        # Initialize components
        print("🤖 Initializing Diablo Bot...")
        
        # Vision
        self.detector = YOLODetector(yolo_model_path, confidence_threshold=0.35, debug=debug)
        self.screen_capture = WindowsScreenCapture(window_title=window_title)
        self.ui_vision = UIVisionModule(debug=debug)
        self.minimap_parser = MinimapParser()
        self.last_ui_state: Optional[UIState] = None
        self.last_minimap = None
        self.brain_overlay = BrainOverlay(enabled=overlay_enabled)
        
        # Actions
        self.executor = ActionExecutor(window_title=window_title, debug=debug)
        
        # Game state
        self.fsm = DiabloFSM(
            initial_state=FSMState.IDLE,
            progression_file=Path(progression_file),
        )
        self.quest_manager = self.fsm.quest_manager
        
        # Navigation and mapping
        self.game_map = GameMap()
        self.navigator = MapNavigator(self.game_map)
        self.current_zone_from_ui: Optional[str] = None  # Zone detected from UI
        self.current_zone_from_minimap: Optional[str] = None  # Zone detected from minimap
        
        # Frame tracking
        self.frame_count = 0
        self.start_time = time.time()
        
        # Distance tracking for debugging
        self.last_hero_pos: Optional[tuple[float, float]] = None
        self.last_npc_pos: Optional[tuple[float, float]] = None
        self.last_distance: float = 0.0
        self.distance_decreasing_frames = 0  # Frames where distance decreased
        
        # Movement optimization
        self.current_movement_target: Optional[tuple[float, float]] = None
        self.movement_target_distance: float = 0.0
        self.frames_without_progress = 0  # Frames where distance didn't decrease
        self.max_frames_without_progress = 15  # Reclick if stuck for N frames
        
        print("✓ Bot initialized")
        print(f"  Map loaded: {len(self.game_map.zones)} zones")
        print(f"  UI Vision: ready")
        print(f"  Minimap Parser: ready")
    
    def _calculate_distance(self, pos1: tuple[float, float], 
                           pos2: tuple[float, float]) -> float:
        """Calculate Euclidean distance between two points.
        
        Args:
            pos1: First point (x, y)
            pos2: Second point (x, y)
            
        Returns:
            Distance in pixels
        """
        x1, y1 = pos1
        x2, y2 = pos2
        return ((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5
    
    def _update_distance_tracking(self, hero_det: Optional[Detection], 
                                 npc_det: Optional[Detection]) -> None:
        """Track distance between hero and NPC for debugging.
        
        Args:
            hero_det: Hero detection
            npc_det: Current quest NPC detection
        """
        if not hero_det or not npc_det:
            return
        
        current_distance = self._calculate_distance(hero_det.center, npc_det.center)
        
        # Track if distance is decreasing (moving towards NPC)
        if self.last_distance > 0 and current_distance < self.last_distance:
            self.distance_decreasing_frames += 1
        elif current_distance >= self.last_distance:
            self.distance_decreasing_frames = 0
        
        self.last_distance = current_distance
        self.last_hero_pos = hero_det.center
        self.last_npc_pos = npc_det.center
    
    def _find_hero(self, detections: list[Detection]) -> Optional[Detection]:
        """Find hero in detections.
        
        Args:
            detections: All detections
            
        Returns:
            Hero detection or None
        """
        for det in detections:
            if det.class_name == "hero":
                return det
        return None
    
    def _should_reclick_movement_target(self, hero_pos: tuple[float, float],
                                       target_pos: tuple[float, float]) -> bool:
        """Check if we should reclick movement target.
        
        Don't reclick while getting closer. Only reclick if:
        - Target has changed
        - Distance stopped decreasing for too many frames (stuck)
        
        Args:
            hero_pos: Current hero position
            target_pos: Current movement target
            
        Returns:
            True if should reclick, False to continue current movement
        """
        # Target changed
        if self.current_movement_target is None:
            self.current_movement_target = target_pos
            self.movement_target_distance = self._calculate_distance(hero_pos, target_pos)
            self.frames_without_progress = 0
            return True
        
        if target_pos != self.current_movement_target:
            # New target
            self.current_movement_target = target_pos
            self.movement_target_distance = self._calculate_distance(hero_pos, target_pos)
            self.frames_without_progress = 0
            return True
        
        # Same target - check if we're making progress
        current_distance = self._calculate_distance(hero_pos, target_pos)
        
        # Are we getting closer?
        if current_distance < self.movement_target_distance:
            # Good! Getting closer
            self.movement_target_distance = current_distance
            self.frames_without_progress = 0
            return False  # Don't reclick
        else:
            # Not getting closer - increment counter
            self.frames_without_progress += 1
            
            # If stuck too long, reclick
            if self.frames_without_progress > self.max_frames_without_progress:
                return True
            
            return False  # Still hope, don't reclick yet
    
    def _capture_frame(self) -> Optional[np.ndarray]:
        """Capture current game frame.
        
        Returns:
            Frame array or None if capture failed
        """
        frame = self.screen_capture.get_frame()
        if frame is None:
            print("⚠️  Failed to capture frame")
            return None
        return frame
    
    def _run_detection(self, frame: np.ndarray) -> list[Detection]:
        """Run YOLO detection on frame.
        
        Args:
            frame: Input frame
            
        Returns:
            List of detections
        """
        detections = self.detector.detect(frame)
        return detections
    
    def _update_game_state(self, detections: list[Detection], ui_state: Optional[UIState] = None) -> GameState:
        """Convert YOLO detections into abstract game state.
        
        Args:
            detections: YOLO detections
            
        Returns:
            GameState object
        """
        # Filter enemies
        enemy_classes = ["zombie", "fallen", "quill rat", "corps"]
        enemy_detections = self.detector.filter_by_class(detections, enemy_classes)
        
        # Create EnemyInfo objects
        enemies = []
        for det in enemy_detections:
            enemy = EnemyInfo(
                type=det.class_name,
                position=det.bbox[:2],  # (x1, y1)
                distance=None,
            )
            enemies.append(enemy)
        
        # Use UI readings when available
        hp_ratio = ui_state.hp_ratio if ui_state else 0.95
        mana_ratio = ui_state.mana_ratio if ui_state else 0.80
        # Use normalized zones for consistency
        current_zone = (
            self._normalize_zone_name(self.current_zone_from_ui)
            or self._normalize_zone_name(self.current_zone_from_minimap)
            or self.game_map.current_zone
            or "unknown"
        )
        
        state = GameState(
            hp_ratio=hp_ratio,
            mana_ratio=mana_ratio,
            enemies=enemies,
            current_location=current_zone,
            frame_number=self.frame_count,
            debug_info={
                "detected_objects": len(detections),
                "detected_enemies": len(enemies),
                "zone_ui": self._normalize_zone_name(self.current_zone_from_ui),
                "zone_minimap": self._normalize_zone_name(self.current_zone_from_minimap),
            }
        )
        
        return state
    
    def _get_quest_targets(self) -> list[str]:
        """Get list of NPCs/objects needed for current quest.
        
        Returns:
            Class names to look for
        """
        current_quest = self.quest_manager.get_current_quest()
        if not current_quest:
            return []
        
        # Map quest names to NPC targets
        quest_targets = {
            "Den of Evil": [],  # No NPC, just clear area
            "Sisters' Burial Grounds": [],
            "Search for Cain": ["waypoint"],
            "Andariel": [],
            "Radament's Lair": [],
            "The Horadric Staff": ["stash"],
            "Tainted Sun": [],
            "Arcane Sanctuary": ["waypoint"],
            "The Summoner": [],
            "Duriel": [],
            "The Golden Bird": [],
            "Khalim's Will": [],
            "Mephisto": [],
            "The Fallen Angel": [],
            "Hell's Brood": [],
            "Diablo": [],
            "Siege on Harrogath": [],
            "Rescue on Mount Arreat": [],
            "Prison of Ice": [],
            "Baal": [],
        }
        
        return quest_targets.get(current_quest.name, [])
    
    def _get_quest_target_zone(self, quest) -> Optional[str]:
        """Get zone ID for current quest objective.
        
        Maps quest names to target zones for navigation.
        
        Args:
            quest: Current quest object
            
        Returns:
            Zone ID or None
        """
        # Simple mapping of quests to zones
        # In a real game, this would be more sophisticated
        quest_zones = {
            "Sisters' Burial Grounds": "burial_grounds",
            "Andariel": "barracks",
            "Radament's Lair": "stony_field",
            "Countess": "catacombs",
            "Mephisto": "stony_field",
        }
        
        return quest_zones.get(quest.name if quest else None)
    
    def _map_minimap_to_zone(self, minimap_result) -> Optional[str]:
        """Map minimap occupancy grid to game zone.
        
        Uses the minimap grid pattern to determine which zone the player is in.
        Different zones have distinctive occupancy patterns on the minimap.
        
        Args:
            minimap_result: MinimapParseResult with grid (96x96) and player_pos
            
        Returns:
            Zone ID or None if unable to determine
        """
        if not minimap_result or not hasattr(minimap_result, 'grid'):
            return None
        
        # For now, we'll use a simple heuristic based on occupied cells
        # In a real system, we could train a classifier on minimap patterns
        
        # Count dark cells (occupied/walls) vs light cells (open)
        # Different zones have different wall/open ratios
        occupied = (minimap_result.grid > 128).sum()
        total = minimap_result.grid.size
        occupancy_ratio = occupied / total if total > 0 else 0.5
        
        # Very rough heuristic for Act 1 zones:
        # - Rogue Encampment: ~30% occupied (mostly open camp)
        # - Blood Moor: ~25% occupied (mostly grass)
        # - Cold Plains: ~20% occupied (very open)
        # - Stony Field: ~35% occupied (more cliffs)
        # - Burial Grounds: ~40% occupied (many tombs/obstacles)
        # - Catacombs: ~50% occupied (very dense dungeon)
        
        if occupancy_ratio < 0.22:
            return "cold_plains"
        elif occupancy_ratio < 0.27:
            return "blood_moor"
        elif occupancy_ratio < 0.32:
            return "rogue_encampment"
        elif occupancy_ratio < 0.42:
            return "stony_field"
        elif occupancy_ratio < 0.48:
            return "burial_grounds"
        else:
            return "catacombs"

    def _normalize_zone_name(self, name: Optional[str]) -> Optional[str]:
        """Normalize OCR/minimap zone text to internal zone_id.
        
        - Lowercase
        - Replace spaces with underscores
        - Strip quotes/punctuation
        - Validate against known zones
        """
        if not name:
            return None
        normalized = name.lower().replace(" ", "_")
        normalized = normalized.replace("'", "").replace('"', "")
        # common OCR artifacts
        normalized = normalized.replace("rogue_encampment", "rogue_encampment")
        # Accept only known zones
        if normalized in self.game_map.zones:
            return normalized
        return None
    
    def _find_quest_npcs(self, detections: list[Detection]) -> list[Detection]:
        """Find NPCs/quest markers relevant to current quest.
        
        Args:
            detections: All detections
            
        Returns:
            Relevant quest-related detections
        """
        targets = self._get_quest_targets()
        if not targets:
            return []
        
        return self.detector.filter_by_class(detections, targets)
    
    def _detect_quest_marker(self, detections: list[Detection]) -> Optional[Detection]:
        """Detect quest marker above NPC.
        
        Quest markers appear as yellow/gold indicators above NPCs.
        
        Args:
            detections: All detections
            
        Returns:
            Quest marker detection or None
        """
        quest_detections = self.detector.filter_by_class(detections, ["quest"])
        if not quest_detections:
            return None
        
        # Return highest confidence quest marker
        return quest_detections[0]
    
    def _find_npc_under_marker(self, quest_marker: Detection, 
                              detections: list[Detection]) -> Optional[Detection]:
        """Find NPC directly under quest marker.
        
        Quest markers appear above NPCs. This finds the NPC that's
        closest below the marker.
        
        Args:
            quest_marker: Quest marker detection
            detections: All detections
            
        Returns:
            NPC detection or None
        """
        # Get marker position and dimensions
        marker_x, marker_y = quest_marker.center
        marker_x1, marker_y1, marker_x2, marker_y2 = quest_marker.bbox
        marker_width = marker_x2 - marker_x1
        marker_bottom = marker_y2
        
        # Find NPCs that are below the marker and roughly aligned horizontally
        npc_classes = ["akara", "kashya", "warriv", "cain", "charsi", 
                      "stash", "waypoint", "hero"]  # Common quest NPCs
        
        npcs = self.detector.filter_by_class(detections, npc_classes)
        
        best_npc = None
        best_score = float('inf')
        
        for npc in npcs:
            npc_x, npc_y = npc.center
            npc_x1, npc_y1, npc_x2, npc_y2 = npc.bbox
            npc_top = npc_y1
            npc_height = npc_y2 - npc_y1
            
            # NPC should be below marker (with some tolerance)
            vertical_gap = npc_top - marker_bottom
            if vertical_gap < -20:  # Allow slight overlap
                continue
            
            # Prefer NPCs horizontally aligned with marker
            # Check if NPC horizontal center is within marker horizontal range (with margin)
            horizontal_overlap = abs(npc_x - marker_x) 
            
            # Score: prefer closer vertically and horizontally aligned
            # Penalize NPCs too far horizontally
            vertical_score = max(0, vertical_gap)  # 0 to gap
            horizontal_score = horizontal_overlap * 0.5  # Weight less than vertical
            
            score = vertical_score + horizontal_score
            
            if self.debug and self.frame_count % 20 == 0:
                print(f"    - Candidate: {npc.class_name} at ({npc_x:.0f},{npc_y:.0f}), score={score:.1f}")
            
            if score < best_score:
                best_score = score
                best_npc = npc
        
        if best_npc and self.debug and self.frame_count % 20 == 0:
            print(f"  ✓ Best NPC under marker: {best_npc.class_name} (score={best_score:.1f})")
        
        return best_npc
    
    def _decide_action(self, game_state: GameState,
                      detections: list[Detection],
                      quest_npcs: list[Detection],
                      quest_marker: Optional[Detection],
                      hero_det: Optional[Detection] = None,
                      npc_det: Optional[Detection] = None) -> Action:
        """Decide next action based on game state and quests.
        
        Args:
            game_state: Current game state
            detections: All detections from YOLO
            quest_npcs: Detected quest NPCs
            quest_marker: Detected quest marker (if any)
            hero_det: Hero detection for distance tracking
            npc_det: NPC found under marker (if any)
            
        Returns:
            Action to take
        """
        # Update FSM with new state
        new_fsm_state = self.fsm.update(game_state)
        
        # If FSM says to engage/kite, use combat logic
        if new_fsm_state in [FSMState.ENGAGE, FSMState.KITE, FSMState.PANIC]:
            action = self.fsm.decide_action(game_state)
        else:
            # Priority 1: Quest marker visible - interact with NPC under it
            if quest_marker:
                # Use provided NPC detection or find it
                target_npc = npc_det if npc_det else self._find_npc_under_marker(quest_marker, detections)
                if target_npc:
                    action = Action(
                        action_type="interact_with_npc",
                        target=target_npc.class_name,
                        params={
                            "npc_position": target_npc.center,
                        }
                    )
                else:
                    # Marker visible but no NPC found - move to marker
                    action = Action(
                        action_type="move_to",
                        target="quest_marker",
                        params={"position": quest_marker.center}
                    )
            # Priority 2: Quest NPCs visible - move to them
            elif quest_npcs:
                nearest = quest_npcs[0]
                action = Action(
                    action_type="move_to",
                    target=nearest.class_name,
                    params={"position": nearest.center}
                )
            # Priority 3: Quest target not on screen - navigate to zone
            else:
                current_quest = self.quest_manager.get_current_quest()
                if current_quest:
                    # For now, use a simple target zone based on quest
                    # TODO: Map quests to target zones more intelligently
                    target_zone = self._get_quest_target_zone(current_quest)
                    if target_zone and hero_det:
                        # Use navigator to find next waypoint
                        target_pos = self.navigator.navigate_to_quest_target(target_zone)
                        if target_pos:
                            action = Action(
                                action_type="move_to",
                                target=f"zone:{target_zone}",
                                params={"position": target_pos}
                            )
                        else:
                            action = Action(action_type="explore")
                    else:
                        action = Action(action_type="explore")
                else:
                    # No active quest - explore
                    action = Action(action_type="explore")
        
        return action
    
    def run_step(self) -> bool:
        """Execute one bot cycle.
        
        Returns:
            True to continue, False to stop
        """
        # Capture
        frame = self._capture_frame()
        if frame is None:
            return False
        
        # Detect
        detections = self._run_detection(frame)
        
        # Read UI vision (zone name via OCR, HP/Mana/etc)
        try:
            ui_state = self.ui_vision.analyze(frame)
            self.current_zone_from_ui = ui_state.zone_name
            self.last_ui_state = ui_state
        except Exception as e:
            if self.debug:
                print(f"⚠️ UIVisionModule error: {e}")
            self.current_zone_from_ui = None
            self.last_ui_state = None
        
        # Read minimap (player position and occupancy grid)
        try:
            minimap_result = self.minimap_parser.parse(frame)
            self.current_zone_from_minimap = self._map_minimap_to_zone(minimap_result)
            self.last_minimap = minimap_result
        except Exception as e:
            if self.debug:
                print(f"⚠️ MinimapParser error: {e}")
            self.current_zone_from_minimap = None
            self.last_minimap = None
        
        # Normalize and update game map with detected zone (prefer UI, fallback to minimap)
        normalized_ui_zone = self._normalize_zone_name(self.current_zone_from_ui)
        normalized_mm_zone = self._normalize_zone_name(self.current_zone_from_minimap)
        if normalized_ui_zone:
            self.game_map.current_zone = normalized_ui_zone
        elif normalized_mm_zone:
            self.game_map.current_zone = normalized_mm_zone
        
        # Build game state
        game_state = self._update_game_state(detections, ui_state)
        
        # Find quest targets and markers
        quest_npcs = self._find_quest_npcs(detections)
        quest_marker = self._detect_quest_marker(detections)
        hero_det = self._find_hero(detections)
        
        # Find NPC under marker
        npc_under_marker = None
        if quest_marker:
            npc_under_marker = self._find_npc_under_marker(quest_marker, detections)
        
        # Track distance
        self._update_distance_tracking(hero_det, npc_under_marker)
        
        # Decide action
        action = self._decide_action(game_state, detections, quest_npcs, quest_marker, hero_det, npc_under_marker)
        
        # Execute action
        self._execute_action(action, frame, hero_det)

        # Draw overlay for debugging (optional)
        if self.overlay_enabled:
            try:
                overlay_frame = self.brain_overlay.draw(
                    frame,
                    perception=None,
                    state=game_state,
                    action=action,
                    fsm_state=self.fsm.get_state_name(),
                    detections=detections,
                    zone_name=self.game_map.current_zone,
                )
                # Save overlay frames if a directory is configured
                if self.overlay_dir:
                    self.overlay_dir.mkdir(parents=True, exist_ok=True)
                    out_path = self.overlay_dir / f"frame_{self.frame_count:05d}.png"
                    cv2.imwrite(str(out_path), overlay_frame)

                # Optional onscreen display
                if self.overlay_show:
                    cv2.imshow("Diabot BrainOverlay", overlay_frame)
                    cv2.waitKey(1)
            except Exception as e:
                if self.debug:
                    print(f"⚠️ Overlay error: {e}")
        
        # Log
        if self.debug or self.frame_count % 10 == 0:
            self._log_step(detections, game_state, quest_npcs, action, quest_marker, 
                          hero_det, npc_under_marker)
        
        self.frame_count += 1
        return True
    
    def _execute_action(self, action: Action, frame: np.ndarray,
                       hero_det: Optional[Detection] = None) -> None:
        """Execute an action on the game.
        
        Optimized movement: only click if target changed or we're stuck.
        
        Args:
            action: Action to execute
            frame: Current frame (for coordinates)
            hero_det: Hero detection for distance tracking
        """
        if action.action_type == "interact_with_npc":
            # Click on NPC (use NPC bbox center, not marker)
            npc_pos = action.params.get("npc_position")
            
            if npc_pos:
                # Convert from window-relative to screen-absolute
                rect = self.screen_capture.get_window_rect()
                if rect:
                    window_x, window_y = rect[:2]
                    screen_x = window_x + int(npc_pos[0])
                    screen_y = window_y + int(npc_pos[1])
                    
                    # Click on NPC center
                    self.executor.interact_with_object(screen_x, screen_y)
        
        elif action.action_type == "move_to":
            # Right-click to move - with optimization to avoid re-clicking
            pos = action.params.get("position")
            if pos and hero_det:
                # Check if we should reclick
                if self._should_reclick_movement_target(hero_det.center, pos):
                    # Reclick needed
                    rect = self.screen_capture.get_window_rect()
                    if rect:
                        window_x, window_y = rect[:2]
                        screen_x = window_x + int(pos[0])
                        screen_y = window_y + int(pos[1])
                        
                        self.executor.move_to(screen_x, screen_y)
                        if self.debug:
                            print(f"  ➡️  Clicking move target at {pos}")
                else:
                    # Already moving towards target, don't reclick
                    if self.debug and self.frame_count % 30 == 0:
                        print(f"  ➡️  Continuing towards {pos} "
                              f"(dist={self.movement_target_distance:.0f}px, "
                              f"stuck_frames={self.frames_without_progress})")
        
        elif action.action_type == "explore":
            # Random explore - no action needed yet
            pass
    
    def _log_step(self, detections: list[Detection], 
                 game_state: GameState, 
                 quest_npcs: list[Detection],
                 action: Action,
                 quest_marker: Optional[Detection] = None,
                 hero_det: Optional[Detection] = None,
                 npc_det: Optional[Detection] = None) -> None:
        """Log bot state and decisions.
        
        Args:
            detections: All detections
            game_state: Current game state
            quest_npcs: Quest-related detections
            action: Decided action
        """
        elapsed = time.time() - self.start_time
        fps = self.frame_count / elapsed if elapsed > 0 else 0
        
        print(f"\n[Frame {self.frame_count} | FPS: {fps:.1f}]")
        print(f"  FSM State: {self.fsm.get_state_name()}")
        print(f"  Quest: {self.fsm.get_quest_guidance()}")
        
        # Zone detection info
        zone_str = f"Zone: {self.game_map.current_zone}"
        if self.current_zone_from_ui:
            zone_str += f" (UI: {self._normalize_zone_name(self.current_zone_from_ui) or self.current_zone_from_ui})"
        if self.current_zone_from_minimap:
            zone_str += f" (MM: {self._normalize_zone_name(self.current_zone_from_minimap) or self.current_zone_from_minimap})"
        print(f"  {zone_str}")
        
        if self.last_ui_state:
            print(f"  HP={self.last_ui_state.hp_ratio*100:.0f}%  Mana={self.last_ui_state.mana_ratio*100:.0f}%")
        
        print(f"  Detections: {len(detections)} total, {len(quest_npcs)} quest-related")
        
        if quest_marker:
            print(f"  🎯 Quest marker at {quest_marker.center} (conf={quest_marker.confidence:.2f})")
            if npc_det:
                print(f"    → NPC under marker: {npc_det.class_name} at {npc_det.center}")
        
        # Movement optimization info
        if action.action_type == "move_to" and self.movement_target_distance > 0:
            zone_info = f" [zone: {self.game_map.current_zone}]" if self.game_map.current_zone else ""
            status = "continuing" if self.frames_without_progress == 0 else f"stuck {self.frames_without_progress}f"
            print(f"  ➡️  Moving to {action.target}: dist={self.movement_target_distance:.0f}px ({status}){zone_info}")
        # Distance tracking
        if hero_det and npc_det:
            distance = self._calculate_distance(hero_det.center, npc_det.center)
            dist_str = f"Distance: {distance:.0f}px"
            if self.last_distance > 0:
                delta = distance - self.last_distance
                arrow = "↓" if delta < -5 else "↑" if delta > 5 else "="
                dist_str += f" {arrow} (Δ{delta:+.0f}px, {self.distance_decreasing_frames} frames closer)"
            print(f"  📍 {dist_str}")
        
        if detections:
            print(f"  Detected objects:")
            for det in detections[:5]:  # Show top 5
                print(f"    - {det.class_name} (conf={det.confidence:.2f})")
        
        print(f"  Action: {action.action_type} → {action.target}")
    
    def run_loop(self, max_frames: Optional[int] = None, fps_limit: int = 5) -> None:
        """Run main bot loop.
        
        Args:
            max_frames: Max frames to run (None = infinite)
            fps_limit: Target FPS
        """
        frame_time = 1.0 / fps_limit
        
        print(f"\n{'='*70}")
        print(f"DIABLO BOT RUNNING")
        print(f"{'='*70}")
        print(f"Game window: {self.window_title}")
        print(f"FPS limit: {fps_limit}")
        print(f"YOLO model: {self.detector.model_path}")
        print(f"\nPress Ctrl+C to stop")
        print(f"{'='*70}\n")
        
        try:
            last_step_time = time.time()
            
            while True:
                current_time = time.time()
                
                # Rate limiting
                if current_time - last_step_time < frame_time:
                    time.sleep(0.01)
                    continue
                
                # Run bot step
                if not self.run_step():
                    print("⚠️  Bot step failed, continuing...")
                    time.sleep(0.1)
                    continue
                
                # Check frame limit
                if max_frames and self.frame_count >= max_frames:
                    print(f"\n✓ Reached max frames ({max_frames})")
                    break
                
                last_step_time = current_time
        
        except KeyboardInterrupt:
            print("\n\n👋 Bot stopped by user")
        except Exception as e:
            print(f"\n❌ Bot error: {e}")
            import traceback
            traceback.print_exc()
        finally:
            self._print_summary()
    
    def _print_summary(self) -> None:
        """Print final summary."""
        elapsed = time.time() - self.start_time
        avg_fps = self.frame_count / elapsed if elapsed > 0 else 0
        
        print(f"\n{'='*70}")
        print(f"BOT SUMMARY")
        print(f"{'='*70}")
        print(f"Frames processed: {self.frame_count}")
        print(f"Time elapsed: {elapsed:.1f}s")
        print(f"Average FPS: {avg_fps:.1f}")
        print(f"Progression: {self.quest_manager.get_progress_summary()}")
        print(f"{'='*70}\n")
