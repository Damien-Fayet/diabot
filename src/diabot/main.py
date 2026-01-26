#!/usr/bin/env python3
# Unified Diabot Main Loop
# Core bot orchestration tying together:
#   1. Image Source (frame acquisition)
#   2. Vision Module (perception)
#   3. State Builder (abstraction)
#   4. Orchestrator (navigation + decision)
#   5. Action Executor (game interaction)
#   6. Persistence (bot state)
from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Optional

# Setup path
sys.path.insert(0, str(Path(__file__).parent / "src"))

import cv2
import numpy as np

from diabot.core.implementations import (
    ScreenshotFileSource,
    DummyActionExecutor,
    WindowsActionExecutor,
)
from diabot.builders.state_builder import EnhancedStateBuilder
from diabot.core.vision_advanced import DiabloVisionModule
from diabot.vision.ui_vision import UIVisionModule
from diabot.decision.orchestrator import Orchestrator
from diabot.models import BotState
from diabot.models.state import Action
from diabot.persistence import save_bot_state, load_bot_state
from diabot.debug.overlay import BrainOverlay
from diabot.navigation import WorldMapManager, Navigator, MinimapPOIDetector
from diabot.navigation.click_navigator import ClickNavigator
from diabot.navigation import FrontierNavigator, NavigationAction, NavigationOverlay
from diabot.navigation.minimap_processor import MinimapProcessor
from diabot.navigation.player_locator import PlayerLocator
from diabot.navigation.map_accumulator import MapAccumulator
from diabot.navigation.exit_navigator import ExitNavigator



class DiabotRunner:
    """Main bot runner that orchestrates the full pipeline."""

    def __init__(
        self,
        image_source_path: Optional[str] = None,
        executor: Optional[object] = None,
        state_file: Optional[str] = None,
        debug: bool = True,
    ):
        # Initialize bot runner.
        # Initialize components
        self.state_file = state_file
        self.debug = debug
        self.last_frame = None
        self.last_perception = None
        print("[BOT] Initializing components...")
        # --- Quest navigation state ---
        self.current_route = None  # NavigationPath
        self.quest_target_zone = None  # str

        # 1. Image source
        if image_source_path:
            from diabot.core.implementations import ScreenshotFileSource
            self.image_source = ScreenshotFileSource(image_source_path)
            print(f"[BOT] OK Loaded screenshot: {image_source_path}")
        else:
            from diabot.core.implementations import WindowsScreenCapture
            self.image_source = WindowsScreenCapture()
            print("[BOT] OK Live screen capture enabled")

        # 2. Vision modules
        self.vision_module = DiabloVisionModule(debug=debug)  # Playfield only
        self.ui_vision = UIVisionModule(debug=debug)  # UI OCR/ScreenRegion
        print("[BOT] OK Vision modules ready")

        # 3. State builder
        self.state_builder = EnhancedStateBuilder()
        print("[BOT] OK State builder ready")

        # 4. Action executor
        if executor is None:
            try:
                # Use real Windows executor for actual mouse clicks
                self.executor = WindowsActionExecutor(debug=debug, image_source=self.image_source)
            except ImportError:
                # Fallback to dummy if pyautogui not available
                print("[WARNING] pyautogui not available, using DummyActionExecutor")
                self.executor = DummyActionExecutor()
        else:
            self.executor = executor
        print("[BOT] OK Action executor ready")

        # 5. Load or initialize bot state
        self.bot_state = self._load_bot_state()
        print(f"[BOT] OK Bot state loaded (mode: {self.bot_state.mode.value})")

        # 6. Orchestrator
        self.orchestrator = Orchestrator(
            bot_state=self.bot_state,
            executor=self.executor,
            dispatch_full_path=True,
        )
        print("[BOT] OK Orchestrator ready")

        # 7. Debug overlay
        self.overlay = BrainOverlay(enabled=debug, show_boxes=True, show_indicators=True) if debug else None
        print("[BOT] OK Debug overlay ready" if debug else "")

        # 8. Navigation system
        self.navigator = Navigator(debug=debug)
        self.world_map_manager = self.navigator.world_map  # Reference to internal world map
        self.minimap_detector = MinimapPOIDetector(debug=debug)
        self.click_navigator = ClickNavigator(debug=debug)
        
        # 9. Frontier-based navigation (vision-only exploration)
        self.frontier_navigator = FrontierNavigator(
            minimap_grid_size=64,
            local_map_size=200,
            movement_speed=2.0,
            debug=debug
        )
        self.nav_overlay = NavigationOverlay(
            show_local_map=True,
            show_path=True,
            show_frontiers=True,
            show_minimap_grid=False
        ) if debug else None
        self.use_frontier_nav = True  # Toggle for frontier-based exploration (enabled for quest)
        
        # 10. Optimized minimap processing with tuned parameters
        self.minimap_processor = MinimapProcessor(
            grid_size=64,
            wall_threshold=49,  # From minimap_tuned_params.txt
            debug=debug
        )
        self.player_locator = PlayerLocator(debug=debug)
        self.map_accumulator = MapAccumulator(
            map_size=2048,
            cell_size=1.0,
            save_dir=Path("data/maps"),
            debug=debug
        )
        self.exit_navigator = ExitNavigator(debug=debug)
        print("[BOT] OK Navigation system ready (FrontierNavigator + MapAccumulator ACTIVE)")

        # Minimap fullscreen toggle state
        self.minimap_fullscreen_active = False  # Track Tab minimap state
        self.frames_since_minimap_toggle = 0     # Cooldown between toggles
        self.last_player_minimap_pos = None      # Track player movement
        self.navigation_mode = "explore"         # "explore" or "exit_seek"

        # Statistics
        self.frame_count = 0
        self.start_time = time.time()
        self.last_zone_name: Optional[str] = None
        self.last_minimap_crop = None
        self.last_minimap_pois = []
        self.last_overlay_frame = None  # Store last overlay for display
    
    def _load_bot_state(self) -> BotState:
        # Load bot state from file or create new.
        if Path(self.state_file).exists():
            try:
                state = load_bot_state(self.state_file)
                print(f"[BOT] Loaded persistent state from {self.state_file}")
                return state
            except Exception as e:
                print(f"[BOT] Error loading state: {e}, creating new")
        
        return BotState()
    
    def _save_bot_state(self):
        # Save bot state to file.
        try:
            save_bot_state(self.bot_state, self.state_file)
            if self.debug:
                print(f"[BOT] Saved state to {self.state_file}")
        except Exception as e:
            print(f"[BOT] Error saving state: {e}")
    
    def _extract_minimap_and_pois(self, frame: np.ndarray):
        """Extract minimap region and POIs for debug visualization."""
        try:
            from diabot.vision.screen_regions import UI_REGIONS
            minimap_region = UI_REGIONS.get("minimap_ui")
            if not minimap_region:
                self.last_minimap_crop = None
                self.last_minimap_pois = []
                return
            frame_h, frame_w = frame.shape[:2]
            x, y, w, h = minimap_region.get_bounds(frame_h, frame_w)
            x_end = min(x + w, frame_w)
            y_end = min(y + h, frame_h)
            minimap = frame[y:y_end, x:x_end].copy()
            pois = self.minimap_detector.detect(minimap)
            # Dessiner les POIs sur la minicarte
            for poi in pois:
                px, py = int(poi.position[0]), int(poi.position[1])
                cv2.circle(minimap, (px, py), 6, (0, 255, 255), 2)
                cv2.putText(minimap, poi.poi_type[:2].upper(), (px+8, py), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,255), 1)
            self.last_minimap_crop = minimap
            self.last_minimap_pois = pois
        except Exception as e:
            if self.debug:
                print(f"[DEBUG] Minimap extraction failed: {e}")
            self.last_minimap_crop = None
            self.last_minimap_pois = []

    def step(self, overlay_show: bool = False) -> bool:
        # Execute one bot cycle.
        # Args:
        #     overlay_show: Whether to display overlay window with cv2.imshow
        # Returns:
        #     bool: True if cycle completed successfully
        try:
            # 1. Acquire frame
            frame = self.image_source.get_frame()
            if frame is None or frame.size == 0:
                print("[BOT] Error: Failed to get frame")
                return False
            self.last_frame = frame
            self.frame_count += 1

            # 2. UI Vision (OCR/ScreenRegion)
            ui_state = self.ui_vision.analyze(frame)
            # 3. Playfield vision (enemies/items/pos)
            perception = self.vision_module.perceive(frame)
            # Inject UI info into perception
            perception.hp_ratio = ui_state.hp_ratio
            perception.mana_ratio = ui_state.mana_ratio
            perception.current_zone = ui_state.zone_name or "UNKNOWN"
            self.last_perception = perception
            if self.debug:
                print(
                    f"[VISION] HP={perception.hp_ratio:.0%}, "
                    f"Mana={perception.mana_ratio:.0%}, "
                    f"Enemies={perception.enemy_count}"
                )

            # 4. Perception → GameState
            game_state = self.state_builder.build(perception)
            if self.debug:
                print(f"[STATE] Threat={game_state.threat_level}, Location={game_state.current_location}")

            # 4b. Ensure minimap is active before navigation analysis
            # CRITICAL: Always verify minimap is displayed before extracting it
            if self.use_frontier_nav:
                import pyautogui
                import time
                
                # Step 1: Check if minimap is already active by verifying Zone OCR
                current_ui_state = self.ui_vision.analyze(frame)
                is_valid_zone = (current_ui_state.zone_name and 
                               current_ui_state.zone_name != "UNKNOWN" and 
                               len(current_ui_state.zone_name) > 2)  # At least 3 chars
                
                if is_valid_zone:
                    # Minimap already active, good to go
                    self.minimap_fullscreen_active = True
                    if self.debug and self.frames_since_minimap_toggle >= 5:
                        print(f"[MINIMAP] ✓ Minimap already active (Zone: '{current_ui_state.zone_name}')")
                        self.frames_since_minimap_toggle = 0
                else:
                    # Step 2: Minimap not active, press Tab to activate it
                    if self.debug:
                        print(f"[MINIMAP] Minimap not active (Zone OCR: '{current_ui_state.zone_name or 'empty'}'), pressing Tab...")
                    
                    # Activate game window to ensure Tab press is received
                    if hasattr(self.image_source, 'hwnd'):
                        import win32gui
                        try:
                            win32gui.SetForegroundWindow(self.image_source.hwnd)
                            time.sleep(0.05)  # Wait for window activation
                        except Exception:
                            pass  # Ignore if window activation fails
                    
                    pyautogui.press('tab')
                    time.sleep(0.1)  # Wait for Tab animation to complete
                    
                    # Step 3: Wait and verify minimap is now displayed
                    max_attempts = 5
                    for attempt in range(max_attempts):
                        time.sleep(0.1)  # Wait for screen update
                        test_frame = self.image_source.get_frame()
                        if test_frame is None:
                            continue
                        
                        # Verify zone name is now readable
                        test_ui_state = self.ui_vision.analyze(test_frame)
                        is_valid = (test_ui_state.zone_name and 
                                  test_ui_state.zone_name != "UNKNOWN" and 
                                  len(test_ui_state.zone_name) > 2)
                        
                        if is_valid:
                            # Zone OCR succeeded = fullscreen minimap is now displayed!
                            frame = test_frame
                            self.minimap_fullscreen_active = True
                            if self.debug:
                                print(f"[MINIMAP] ✓ Minimap activated (Zone: '{test_ui_state.zone_name}')")
                                # Save debug screenshot
                                from pathlib import Path
                                import cv2 as cv2_lib
                                debug_path = Path("data/screenshots/outputs/minimap_fullscreen_capture.png")
                                debug_path.parent.mkdir(parents=True, exist_ok=True)
                                cv2_lib.imwrite(str(debug_path), frame)
                                print(f"[MINIMAP] Saved debug screenshot: {debug_path}")
                            break
                        else:
                            if self.debug:
                                zone_text = test_ui_state.zone_name or "empty"
                                print(f"[MINIMAP] Attempt {attempt+1}/{max_attempts}: Zone OCR still invalid ('{zone_text}'), waiting...")
                    else:
                        # Failed to activate minimap after all attempts
                        if self.debug:
                            print("[MINIMAP] WARNING: Failed to activate fullscreen minimap, skipping navigation")
                        self.minimap_fullscreen_active = False
                
                # Only extract minimap if it's confirmed active
                if self.minimap_fullscreen_active:
                    self._extract_minimap_and_pois(frame)
                    self.frames_since_minimap_toggle += 1
                else:
                    # Skip navigation this frame
                    if self.debug:
                        print("[MINIMAP] Skipping minimap extraction (not active)")
            else:
                # No navigation, extract normal minimap
                self._extract_minimap_and_pois(frame)
            if self.debug and self.last_minimap_crop is not None:
                import cv2 as cv2_debug
                cv2_debug.imshow("Minimap Debug", self.last_minimap_crop)
                cv2_debug.waitKey(1)

            # 5a. OPTIMIZED Navigation: Map-aware exit seeking with accumulation
            nav_state = None
            # Allow navigation in safe zones (towns) even with NPCs detected
            safe_zones = ["ROGUE ENCAMPMENT", "LUT GHOLEIN", "KURAST DOCKS", "PANDEMONIUM FORTRESS", "HARROGATH"]
            is_safe_zone = perception.current_zone in safe_zones
            can_navigate = game_state.threat_level == "none" or (is_safe_zone and game_state.threat_level == "low")
            
            if self.use_frontier_nav and can_navigate and self.last_minimap_crop is not None:
                current_zone = perception.current_zone or "UNKNOWN"
                
                # STEP 1: Process minimap with optimized parameters
                try:
                    minimap_grid = self.minimap_processor.process(self.last_minimap_crop)
                    
                    # STEP 2: Detect player position (white cross)
                    player_pos = self.player_locator.detect_player_cross(self.last_minimap_crop)
                    
                    # STEP 3: Calculate movement since last frame
                    player_offset = (0, 0)
                    if self.last_player_minimap_pos and player_pos:
                        dx = player_pos[0] - self.last_player_minimap_pos[0]
                        dy = player_pos[1] - self.last_player_minimap_pos[1]
                        player_offset = (dx, dy)
                    self.last_player_minimap_pos = player_pos
                    
                    # STEP 4: Update accumulated map
                    self.map_accumulator.update(minimap_grid, player_offset)
                    self.map_accumulator.current_zone = current_zone
                    
                    # STEP 5: Decide navigation mode
                    should_explore = self.exit_navigator.should_explore_instead(
                        self.map_accumulator,
                        exploration_threshold=0.3  # Explore 30% before seeking exit
                    )
                    
                    if should_explore:
                        self.navigation_mode = "explore"
                    else:
                        self.navigation_mode = "exit_seek"
                    
                    # STEP 6: Execute navigation based on mode
                    if self.navigation_mode == "exit_seek":
                        # Find and navigate to best exit candidate
                        exit_candidates = self.exit_navigator.find_exit_candidates(
                            self.map_accumulator,
                            max_candidates=3
                        )
                        
                        if exit_candidates:
                            best_exit = exit_candidates[0]
                            target_pos = self.exit_navigator.get_navigation_target(
                                best_exit,
                                self.map_accumulator,
                                minimap_grid
                            )
                            
                            if target_pos:
                                rel_x, rel_y = target_pos
                                x, y = self.executor.get_window_click_position(rel_x, rel_y)
                                action = Action(action_type="move", target="exit", params={"position": (x, y)})
                                self.executor.execute(action, frame)
                                
                                if self.debug:
                                    print(f"[EXIT_NAV] Moving to exit @ ({rel_x:.2f}, {rel_y:.2f}) "
                                          f"score={best_exit.score:.2f}")
                                
                                # Save map periodically
                                if self.frame_count % 50 == 0:
                                    self.map_accumulator.save_map(current_zone)
                        else:
                            # No exits found, explore
                            self.navigation_mode = "explore"
                    
                    if self.navigation_mode == "explore":
                        # Frontier-based exploration for unmapped areas
                        try:
                            nav_state = self.frontier_navigator.update(frame)
                            if self.debug:
                                print(
                                    f"[EXPLORE] Action={nav_state.action.value}, "
                                    f"Pos={nav_state.current_position}, "
                                    f"Angle={nav_state.current_angle:.1f}°, "
                                    f"Frontiers={nav_state.frontiers_available}"
                                )
                            
                            # Execute frontier navigation action via ActionExecutor
                            if nav_state.action == NavigationAction.MOVE_FORWARD:
                                # Click at center-bottom (relative coords: 0.5, 0.6)
                                x, y = self.executor.get_window_click_position(0.5, 0.6)
                                action = Action(action_type="move", target="forward", params={"position": (x, y)})
                                self.executor.execute(action, frame)
                                self.frontier_navigator.report_movement("forward", 0.5)
                                if self.debug:
                                    print(f"[NAV_ACTION] Moving forward to ({x}, {y})")
                            elif nav_state.action == NavigationAction.TURN_LEFT:
                                # Click at left-center (relative coords: 0.3, 0.5)
                                x, y = self.executor.get_window_click_position(0.3, 0.5)
                                action = Action(action_type="move", target="left", params={"position": (x, y)})
                                self.executor.execute(action, frame)
                                self.frontier_navigator.report_rotation(-30)
                                if self.debug:
                                    print(f"[NAV_ACTION] Turning left to ({x}, {y})")
                            elif nav_state.action == NavigationAction.TURN_RIGHT:
                                # Click at right-center (relative coords: 0.7, 0.5)
                                x, y = self.executor.get_window_click_position(0.7, 0.5)
                                action = Action(action_type="move", target="right", params={"position": (x, y)})
                                self.executor.execute(action, frame)
                                self.frontier_navigator.report_rotation(30)
                                if self.debug:
                                    print(f"[NAV_ACTION] Turning right to ({x}, {y})")
                            elif nav_state.action == NavigationAction.STOP:
                                # No frontiers: move randomly to explore
                                if self.debug:
                                    print(f"[NAV_ACTION] No frontiers, moving randomly")
                                import random
                                # Random click in lower half of screen (relative: 0.3-0.7 x, 0.4-0.7 y)
                                rel_x = random.uniform(0.3, 0.7)
                                rel_y = random.uniform(0.4, 0.7)
                                x, y = self.executor.get_window_click_position(rel_x, rel_y)
                                action = Action(action_type="move", target="random", params={"position": (x, y)})
                                self.executor.execute(action, frame)
                        except Exception as e:
                            if self.debug:
                                print(f"[EXPLORE] Error: {e}")
                
                except Exception as e:
                    if self.debug:
                        print(f"[MAP_NAV] Processing error: {e}")
                        import traceback
                        traceback.print_exc()
            
            # Hide minimap after navigation to allow normal detection
            if self.minimap_fullscreen_active and self.frames_since_minimap_toggle >= 5:
                import pyautogui
                pyautogui.press('tab')
                self.minimap_fullscreen_active = False
                self.frames_since_minimap_toggle = 0
                if self.debug:
                    print("[MINIMAP] Deactivated fullscreen minimap (Tab)")
                # Wait a bit for minimap to disappear
                import time
                time.sleep(0.1)
            
            # 5b. Orchestrate: Vision → Navigation → Decision → Action
            # DISABLED: Using FrontierNavigator instead for now
            # result = self.orchestrator.step(frame)

            # if self.debug:
            #     # Debug screen detection
            #     screen_detection = self.orchestrator.screen_manager.detector.detect(frame)
            #     print(
            #         f"[SCREEN] Type={screen_detection.screen_type}, "
            #         f"Confidence={screen_detection.confidence:.2f}"
            #     )
            #     print(
            #         f"[ORCHESTRATOR] Screen={result.screen_type}, "
            #         f"CanNavigate={result.can_navigate}, "
            #         f"Goal={result.goal_kind}, "
            #         f"PathLen={len(result.path)}"
            #     )
            #     if result.screen_action:
            #         print(f"[SCREEN_ACTION] {result.screen_action}")
            #     if result.dispatched_action:
            #         print(
            #             f"[ACTION] {result.dispatched_action} -> "
            #             f"{'OK' if result.action_success else 'FAIL'}"
            #         )

            # 6. Update bot state with current perception
            self.bot_state.hp_ratio = perception.hp_ratio
            self.bot_state.mana_ratio = perception.mana_ratio

            # 7. AUTO-REGISTER ZONES ON VISIT
            current_zone = perception.current_zone or "UNKNOWN"
            if current_zone != self.last_zone_name:
                self.last_zone_name = current_zone
                # Register zone if not already known
                if current_zone not in self.world_map_manager.zones:
                    act = self._infer_act_from_zone(current_zone)
                    self.world_map_manager.register_zone(current_zone, act=act)
                    self.navigator.visit_zone(current_zone, act=act)
                    if self.debug:
                        print(f"[BOT] NAV Auto-registered zone: {current_zone} ({act})")
                # Mark as explored
                self.navigator.visit_zone(current_zone)
                # Extract minimap POIs for this zone if available
                self._auto_detect_zone_pois(frame)

            # Periodic save
            if self.frame_count % 30 == 0:  # Every 30 frames (~1 sec at 30 FPS)
                self._save_bot_state()
                self.world_map_manager.save_all_maps()  # Save maps too

            # 8. Debug visualization
            if self.overlay:
                output = self.overlay.draw(
                    frame,
                    perception=perception,
                    state=game_state,
                    action=Action(action_type="explore" if self.use_frontier_nav else "idle"),
                    fsm_state="EXPLORING" if self.use_frontier_nav else "IDLE",
                    minimap_crop=self.last_minimap_crop,
                    minimap_pois=self.last_minimap_pois,
                )
                # Add frontier navigation overlay (only if enabled and active)
                if self.nav_overlay and nav_state and self.use_frontier_nav:
                    try:
                        output = self.nav_overlay.draw(
                            output,
                            nav_state,
                            local_map=self.frontier_navigator.get_local_map()
                        )
                    except Exception as e:
                        if self.debug:
                            print(f"[DEBUG] Nav overlay error: {e}")
                
                # Always save overlay to disk for inspection
                from pathlib import Path
                output_path = Path(__file__).parent.parent.parent / "data" / "screenshots" / "outputs" / "brain_overlay.png"
                cv2.imwrite(str(output_path), output)
                
                # Store last overlay for display (used by run() method)
                self.last_overlay_frame = output
                
                # Show window if requested
                if overlay_show:
                    cv2.imshow("Diabot BrainOverlay", output)
                    cv2.waitKey(1)

            return True
        except Exception as e:
            print(f"[BOT] Error in step: {e}")
            import traceback
            traceback.print_exc()
            return False
    


    def run(self, max_frames: Optional[int] = None, fps_limit: int = 5, overlay_show: bool = False) -> None:
        """Unified main run loop for the bot, with synchronized overlay and debug output."""
        print("=" * 60 + "\n")
        frame_delay = 1.0 / fps_limit
        try:
            while max_frames is None or self.frame_count < max_frames:
                loop_start = time.time()
                # Execute one bot cycle
                try:
                    success = self.step(overlay_show=overlay_show)
                    if not success:
                        if self.debug:
                            print("[BOT] Step failed, continuing...")
                    
                    # Frame rate control
                    elapsed = time.time() - loop_start
                    if elapsed < frame_delay:
                        time.sleep(frame_delay - elapsed)
                except KeyboardInterrupt:
                    print("\n[BOT] Interrupted by user")
                    break
                except Exception as e:
                    print(f"[BOT] Error in main loop: {e}")
                    import traceback
                    traceback.print_exc()
                    continue
        finally:
            self._shutdown()

    def _infer_act_from_zone(self, zone_name: str) -> str:
        # Infer act number from zone name.
        # Args:
        #     zone_name: Zone name from perception
        # Returns:
        #     str: Act identifier (a1, a2, a3, a4, a5)
        zone_lower = zone_name.lower()
        # Act 1
        if any(z in zone_lower for z in ['rogue', 'blood moor', 'burial', 'catacombs', 'tristram', 'barracks']):
            return "a1"
        # Act 2
        if any(z in zone_lower for z in ['lut gholein', 'desert', 'sewers', 'harem', 'maggot']):
            return "a2"
        # Act 3
        if any(z in zone_lower for z in ['kurast', 'spider', 'dungeon', 'upper', 'travincal']):
            return "a3"
        # Act 4
        if any(z in zone_lower for z in ['pandemonium', 'diablo', 'river', 'city']):
            return "a4"
        # Act 5
        if any(z in zone_lower for z in ['harrogath', 'worldstone', 'temple', 'freeze', 'ancient', 'baal']):
            return "a5"
        return "unknown"
    
    def _auto_detect_zone_pois(self, frame: np.ndarray) -> None:
        # Automatically detect and register zone POIs from minimap.
        # Args:
        #     frame: Current game frame
        try:
            # Extract minimap region
            from diabot.vision.screen_regions import UI_REGIONS
            
            minimap_region = UI_REGIONS.get("minimap_ui")
            if not minimap_region or not self.last_zone_name:
                return
            
            # Get pixel bounds from ScreenRegion
            frame_h, frame_w = frame.shape[:2]
            x, y, w, h = minimap_region.get_bounds(frame_h, frame_w)
            x_end = min(x + w, frame_w)
            y_end = min(y + h, frame_h)
            minimap = frame[y:y_end, x:x_end]
            
            # Detect POIs
            pois = self.minimap_detector.detect(minimap)
            
            # Register in world map
            for poi in pois:
                self.world_map_manager.add_poi(
                    zone_name=self.last_zone_name,
                    poi_name=f"{poi.poi_type.capitalize()}",
                    poi_type=poi.poi_type,
                    position=poi.position,
                    target_zone=None,
                )
            
            if self.debug and pois:
                print(f"[BOT] MAP Detected {len(pois)} POIs in {self.last_zone_name}")
        
        except Exception as e:
            if self.debug:
                print(f"[BOT] WARNING Error detecting zone POIs: {e}")
    
    def _shutdown(self):
        # Graceful shutdown.
        print("\n" + "=" * 60)
        print("[BOT] Shutting down...")
        
        # Save final state
        self._save_bot_state()
        self.world_map_manager.save_all_maps()
        
        # Statistics
        elapsed = time.time() - self.start_time
        fps = self.frame_count / elapsed if elapsed > 0 else 0
        
        print(f"[BOT] Frames processed: {self.frame_count}")
        print(f"[BOT] Duration: {elapsed:.1f}s")
        print(f"[BOT] Average FPS: {fps:.1f}")
        print(f"[BOT] Final state saved")
        print(f"[BOT] Zones discovered: {len(self.world_map_manager.zones)}")
        print("=" * 60)


def main():
    # CLI entry point.
    import argparse
    
    parser = argparse.ArgumentParser(description="Diabot - Vision-based Diablo 2 Bot")
    parser.add_argument(
        "--image",
        type=str,
        default=None,
        help="Path to screenshot for dev mode (if not set, uses live capture)"
    )
    parser.add_argument(
        "--state",
        type=str,
        default="bot_state.json",
        help="Path to persistent bot state file"
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="Max frames to process (None = infinite)"
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=30.0,
        help="Target FPS"
    )
    parser.add_argument(
        "--overlay-show",
        action="store_true",
        help="Show BrainOverlay onscreen (OpenCV window)"
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug output"
    )
    args = parser.parse_args()

    # Create and run bot
    bot = DiabotRunner(
        image_source_path=args.image,
        state_file=args.state,
        debug=args.debug,
    )
    bot.run(max_frames=args.max_frames, fps_limit=int(args.fps), overlay_show=args.overlay_show)


if __name__ == "__main__":
    main()
