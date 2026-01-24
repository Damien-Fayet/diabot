#!/usr/bin/env python3
"""Updated developer mode with advanced vision module."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import cv2
import numpy as np

from diabot.core.vision_advanced import DiabloVisionModule
from diabot.core.implementations import (
    ScreenshotFileSource,
    SimpleStateBuilder,
    DummyActionExecutor,
)
from diabot.builders.state_builder import EnhancedStateBuilder
from diabot.decision.diablo_fsm import DiabloFSM
from diabot.debug.overlay import BrainOverlay


def main(screenshot_path: str = None):
    """Run bot with advanced vision on a screenshot."""
    
    if screenshot_path is None:
        screenshot_path = "/Users/damien/PersoLocal/diabot/data/screenshots/inputs/game.jpg"
        print(f"Using default screenshot: {screenshot_path}")
    
    print(f"\n🎮 Diabot - Developer Mode with Advanced Vision")
    print("=" * 60)
    print(f"Screenshot: {Path(screenshot_path).name}\n")
    
    try:
        # Initialize components
        image_source = ScreenshotFileSource(screenshot_path)
        vision_module = DiabloVisionModule(debug=False)
        state_builder = EnhancedStateBuilder(frame_counter=0)
        fsm = DiabloFSM()
        action_executor = DummyActionExecutor()
        brain_overlay = BrainOverlay(enabled=True)
        
        # Get frame
        frame = image_source.get_frame()
        print(f"✓ Loaded frame: {frame.shape}")
        
        # Perception - using advanced vision
        perception = vision_module.perceive(frame)
        print(f"✓ Perception:")
        print(f"    HP: {perception.hp_ratio*100:.1f}%")
        print(f"    Mana: {perception.mana_ratio*100:.1f}%")
        print(f"    Enemies: {perception.enemy_count}")
        print(f"    Items: {len(perception.visible_items)}")
        
        # State building
        state = state_builder.build(perception)
        print(f"✓ State built:")
        print(f"    Health: {state.health_percent:.1f}%")
        print(f"    Threatened: {state.is_threatened}")
        print(f"    Needs Potion: {state.needs_potion}")
        print(f"    Threat Level: {state.debug_info.get('threat_level', 'unknown')}")
        
        # FSM update and decision
        fsm_state = fsm.update(state)
        print(f"✓ FSM State: {fsm_state.name}")
        
        action = fsm.decide_action(state)
        print(f"✓ Decision: {action.action_type}")
        if fsm.transition_history:
            print(f"    Transition: {fsm.get_transition_summary()}")
        
        # Execute action
        action_executor.execute_action(action.action_type, action.params)
        
        # Visualize with BrainOverlay
        output_frame = brain_overlay.draw(
            frame=frame,
            perception=perception,
            state=state,
            action=action,
            fsm_state=fsm_state.name,
        )
        
        # Save debug output
        output_path = Path(__file__).parent.parent / "data" / "screenshots" / "outputs" / "brain_overlay.png"
        cv2.imwrite(str(output_path), output_frame)
        print(f"\n✓ Debug overlay saved: {output_path.name}")
        
        # Display if possible
        try:
            cv2.imshow("Diabot - Advanced Vision", output_frame)
            print("Press any key to close window...")
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        except Exception as e:
            print(f"Note: Display unavailable ({e})")
        
        print("\n✅ Bot cycle complete!\n")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    screenshot_arg = sys.argv[1] if len(sys.argv) > 1 else None
    sys.exit(main(screenshot_arg))
