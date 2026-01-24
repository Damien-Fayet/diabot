"""Integration tests for the complete perception-decision pipeline."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import cv2
import numpy as np

from diabot.core.vision_advanced import DiabloVisionModule
from diabot.builders.state_builder import EnhancedStateBuilder
from diabot.decision.diablo_fsm import DiabloFSM
from diabot.debug.overlay import BrainOverlay


def test_full_pipeline(screenshot_path: str, description: str = ""):
    """Test the complete pipeline: vision → state → decision."""
    print(f"\n{'='*70}")
    print(f"📸 TEST: {description or Path(screenshot_path).name}")
    print(f"{'='*70}")
    
    # Load image
    frame = cv2.imread(screenshot_path)
    if frame is None:
        print(f"❌ Failed to load image")
        return False
    
    print(f"Frame shape: {frame.shape}")
    
    # Step 1: Vision
    vision = DiabloVisionModule()
    perception = vision.perceive(frame)
    print(f"\n1️⃣ PERCEPTION:")
    print(f"   HP: {perception.hp_ratio*100:.1f}%")
    print(f"   Mana: {perception.mana_ratio*100:.1f}%")
    print(f"   Enemies: {perception.enemy_count}")
    print(f"   Items: {len(perception.visible_items)}")
    
    # Step 2: State Building
    state_builder = EnhancedStateBuilder()
    state = state_builder.build(perception)
    print(f"\n2️⃣ GAME STATE:")
    print(f"   Health: {state.health_percent:.1f}%")
    print(f"   Mana: {state.mana_percent:.1f}%")
    print(f"   Location: {state.current_location}")
    print(f"   Threatened: {state.is_threatened}")
    print(f"   Needs Potion: {state.needs_potion}")
    threat_level = state.debug_info.get("threat_level", "unknown")
    print(f"   Threat Level: {threat_level}")
    
    # Step 3: FSM Decision
    fsm = DiabloFSM()
    fsm_state = fsm.update(state)
    action = fsm.decide_action(state)
    print(f"\n3️⃣ DECISION:")
    print(f"   FSM State: {fsm_state.name}")
    print(f"   Action: {action.action_type}")
    print(f"   Target: {action.target}")
    print(f"   Params: {action.params}")
    if fsm.transition_history:
        print(f"   Transition: {fsm.get_transition_summary()}")
    
    # Step 4: Visualization
    brain_overlay = BrainOverlay(enabled=True)
    output_frame = brain_overlay.draw(
        frame=frame,
        perception=perception,
        state=state,
        action=action,
        fsm_state=fsm_state.name,
    )
    
    # Save output
    output_name = f"integration_{Path(screenshot_path).stem}.png"
    output_path = Path(__file__).parent.parent / "data" / "screenshots" / "outputs" / output_name
    cv2.imwrite(str(output_path), output_frame)
    print(f"\n4️⃣ VISUALIZATION:")
    print(f"   Saved: {output_name}")
    
    print(f"\n✅ Pipeline test passed!")
    return True


def main():
    """Run all integration tests."""
    screenshot_dir = Path(__file__).parent.parent / "data" / "screenshots" / "inputs"
    
    tests = [
        (screenshot_dir / "char_menu.jpg", "Character Menu - Checking UI"),
        (screenshot_dir / "game.jpg", "Game Screen - In Dungeon"),
    ]
    
    print("🎮 INTEGRATION TESTS - Perception → State → Decision Pipeline")
    print("=" * 70)
    
    passed = 0
    failed = 0
    
    for test_path, description in tests:
        if test_path.exists():
            try:
                if test_full_pipeline(str(test_path), description):
                    passed += 1
                else:
                    failed += 1
            except Exception as e:
                print(f"❌ Test failed with error: {e}")
                import traceback
                traceback.print_exc()
                failed += 1
        else:
            print(f"⚠️  Skipped (not found): {test_path.name}")
    
    print(f"\n{'='*70}")
    print(f"📊 RESULTS: {passed} passed, {failed} failed")
    print(f"{'='*70}\n")
    
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
