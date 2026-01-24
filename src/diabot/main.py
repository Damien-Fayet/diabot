#!/usr/bin/env python3
"""
Unified Diabot Main Loop
═════════════════════════

Core bot orchestration tying together:
  1. Image Source (frame acquisition)
  2. Vision Module (perception)
  3. State Builder (abstraction)
  4. Orchestrator (navigation + decision)
  5. Action Executor (game interaction)
  6. Persistence (bot state)
"""
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
)
from diabot.builders.state_builder import EnhancedStateBuilder
from diabot.core.vision_advanced import DiabloVisionModule
from diabot.decision.orchestrator import Orchestrator
from diabot.models import BotState
from diabot.persistence import save_bot_state, load_bot_state
from diabot.debug.overlay import BrainOverlay


class DiabotRunner:
    """Main bot runner that orchestrates the full pipeline."""

    def __init__(
        self,
        image_source_path: Optional[str] = None,
        executor: Optional[object] = None,
        state_file: Optional[str] = None,
        debug: bool = True,
    ):
        """
        Initialize bot runner.

        Args:
            image_source_path: Path to screenshot (dev mode) or None for live capture (runtime)
            executor: ActionExecutor implementation (None = DummyActionExecutor)
            state_file: Path to persist BotState (if None, uses default)
            debug: Enable debug output and visualization
        """
        self.debug = debug
        self.state_file = state_file or "bot_state.json"
        
        # Initialize components
        print("[BOT] Initializing components...")
        
        # 1. Image source
        if image_source_path:
            from diabot.core.implementations import ScreenshotFileSource
            self.image_source = ScreenshotFileSource(image_source_path)
            print(f"[BOT] ✓ Loaded screenshot: {image_source_path}")
        else:
            from diabot.core.implementations import WindowsScreenCapture
            self.image_source = WindowsScreenCapture()
            print("[BOT] ✓ Live screen capture enabled")
        
        # 2. Vision module
        self.vision_module = DiabloVisionModule(debug=debug)
        print("[BOT] ✓ Vision module ready")
        
        # 3. State builder
        self.state_builder = EnhancedStateBuilder()
        print("[BOT] ✓ State builder ready")
        
        # 4. Action executor
        self.executor = executor or DummyActionExecutor()
        print("[BOT] ✓ Action executor ready")
        
        # 5. Load or initialize bot state
        self.bot_state = self._load_bot_state()
        print(f"[BOT] ✓ Bot state loaded (mode: {self.bot_state.mode.value})")
        
        # 6. Orchestrator
        self.orchestrator = Orchestrator(
            bot_state=self.bot_state,
            executor=self.executor,
            dispatch_full_path=True,
        )
        print("[BOT] ✓ Orchestrator ready")
        
        # 7. Debug overlay
        self.overlay = BrainOverlay() if debug else None
        print("[BOT] ✓ Debug overlay ready" if debug else "")
        
        # Statistics
        self.frame_count = 0
        self.start_time = time.time()
    
    def _load_bot_state(self) -> BotState:
        """Load bot state from file or create new."""
        if Path(self.state_file).exists():
            try:
                state = load_bot_state(self.state_file)
                print(f"[BOT] Loaded persistent state from {self.state_file}")
                return state
            except Exception as e:
                print(f"[BOT] Error loading state: {e}, creating new")
        
        return BotState()
    
    def _save_bot_state(self):
        """Save bot state to file."""
        try:
            save_bot_state(self.bot_state, self.state_file)
            if self.debug:
                print(f"[BOT] Saved state to {self.state_file}")
        except Exception as e:
            print(f"[BOT] Error saving state: {e}")
    
    def step(self) -> bool:
        """
        Execute one bot cycle.

        Returns:
            bool: True if cycle completed successfully
        """
        try:
            # 1. Acquire frame
            frame = self.image_source.get_frame()
            if frame is None or frame.size == 0:
                print("[BOT] Error: Failed to get frame")
                return False
            
            self.frame_count += 1
            
            # 2. Vision → Perception
            perception = self.vision_module.perceive(frame)
            if self.debug:
                print(
                    f"[VISION] HP={perception.hp_ratio:.0%}, "
                    f"Mana={perception.mana_ratio:.0%}, "
                    f"Enemies={perception.enemy_count}"
                )
            
            # 3. Perception → GameState
            game_state = self.state_builder.build(perception)
            if self.debug:
                print(f"[STATE] Threat={game_state.threat_level}, Location={game_state.current_location}")
            
            # 4. Orchestrate: Vision → Navigation → Decision → Action
            result = self.orchestrator.step(frame)
            
            if self.debug:
                print(
                    f"[ORCHESTRATOR] Screen={result.screen_type}, "
                    f"CanNavigate={result.can_navigate}, "
                    f"Goal={result.goal_kind}, "
                    f"PathLen={len(result.path)}"
                )
                
                if result.screen_action:
                    print(f"[SCREEN_ACTION] {result.screen_action}")
                
                if result.dispatched_action:
                    print(
                        f"[ACTION] {result.dispatched_action} → "
                        f"{'✓' if result.action_success else '✗'}"
                    )
            
            # 5. Update bot state with current perception
            self.bot_state.hp_ratio = perception.hp_ratio
            self.bot_state.mana_ratio = perception.mana_ratio
            
            # 6. Periodic save
            if self.frame_count % 30 == 0:  # Every 30 frames (~1 sec at 30 FPS)
                self._save_bot_state()
            
            # 7. Debug visualization
            if self.overlay and result.can_navigate:
                output = self.overlay.draw_on_frame(
                    frame,
                    fsm_state="NAVIGATE",  # Would get from FSM
                    action_type=result.goal_kind or "idle",
                    perception=perception,
                    threat_level=game_state.threat_level,
                )
                
                # Save output for inspection
                if self.frame_count % 100 == 0:
                    output_path = "bot_debug_overlay.png"
                    cv2.imwrite(output_path, output)
                    if self.debug:
                        print(f"[DEBUG] Saved overlay to {output_path}")
            
            return True
        
        except Exception as e:
            print(f"[BOT] Error in step: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def run_loop(self, max_frames: Optional[int] = None, fps: float = 30.0):
        """
        Run main bot loop.

        Args:
            max_frames: Max frames to run (None = infinite)
            fps: Target frames per second
        """
        print("\n" + "=" * 60)
        print("[BOT] Starting main loop...")
        print("=" * 60 + "\n")
        
        frame_delay = 1.0 / fps
        
        try:
            while max_frames is None or self.frame_count < max_frames:
                loop_start = time.time()
                
                # Execute one cycle
                success = self.step()
                if not success:
                    print("[BOT] Cycle failed, continuing...")
                
                # Frame rate control
                elapsed = time.time() - loop_start
                if elapsed < frame_delay:
                    time.sleep(frame_delay - elapsed)
        
        except KeyboardInterrupt:
            print("\n[BOT] Interrupted by user")
        
        except Exception as e:
            print(f"\n[BOT] Fatal error: {e}")
            import traceback
            traceback.print_exc()
        
        finally:
            self._shutdown()
    
    def _shutdown(self):
        """Graceful shutdown."""
        print("\n" + "=" * 60)
        print("[BOT] Shutting down...")
        
        # Save final state
        self._save_bot_state()
        
        # Statistics
        elapsed = time.time() - self.start_time
        fps = self.frame_count / elapsed if elapsed > 0 else 0
        
        print(f"[BOT] Frames processed: {self.frame_count}")
        print(f"[BOT] Duration: {elapsed:.1f}s")
        print(f"[BOT] Average FPS: {fps:.1f}")
        print(f"[BOT] Final state saved")
        print("=" * 60)


def main():
    """CLI entry point."""
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
        "--quiet",
        action="store_true",
        help="Disable debug output"
    )
    
    args = parser.parse_args()
    
    # Create and run bot
    bot = DiabotRunner(
        image_source_path=args.image,
        state_file=args.state,
        debug=not args.quiet,
    )
    
    bot.run_loop(max_frames=args.max_frames, fps=args.fps)


if __name__ == "__main__":
    main()
