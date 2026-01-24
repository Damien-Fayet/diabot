#!/usr/bin/env python3
"""Developer mode: Process a static screenshot and visualize the bot's perception."""

import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import cv2
import numpy as np

from diabot.core.implementations import (
    ScreenshotFileSource,
    RuleBasedVisionModule,
    SimpleStateBuilder,
    RuleBasedDecisionEngine,
    DummyActionExecutor,
)
from diabot.debug.overlay import DebugOverlay


def main(screenshot_path: str = None):
    """
    Run the bot in developer mode on a static screenshot.
    
    Args:
        screenshot_path: Path to screenshot file. If None, uses a default.
    """
    # Use default test image or provided path
    if screenshot_path is None:
        # For now, we'll create a dummy image if no path provided
        print("No screenshot path provided. Creating a dummy image for testing...")
        dummy_img = (255 * np.ones((600, 800, 3), dtype=np.uint8))
        cv2.putText(dummy_img, "Dummy Game Frame", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
        
        # Save it temporarily
        test_path = Path(__file__).parent.parent / "data" / "screenshots" / "test_frame.png"
        test_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(test_path), dummy_img)
        screenshot_path = str(test_path)
    
    print(f"Loading screenshot from: {screenshot_path}")
    
    try:
        # Initialize components
        image_source = ScreenshotFileSource(screenshot_path)
        vision_module = RuleBasedVisionModule()
        state_builder = SimpleStateBuilder(frame_counter=0)
        decision_engine = RuleBasedDecisionEngine()
        action_executor = DummyActionExecutor()
        
        # Get frame
        frame = image_source.get_frame()
        print(f"✓ Loaded frame shape: {frame.shape}")
        
        # Perception
        perception = vision_module.perceive(frame)
        print(f"✓ Perception: HP={perception.hp_ratio*100:.1f}%, Enemies={perception.enemy_count}")
        
        # State building
        state = state_builder.build(perception)
        print(f"✓ State built: {state}")
        
        # Decision making
        action = decision_engine.decide(state)
        print(f"✓ Decision: {action.action_type}")
        
        # Execute action
        action_executor.execute_action(action.action_type, action.params)
        
        # Visualize with overlay
        output_frame = DebugOverlay.draw_state(frame, state)
        
        # Display
        output_path = Path(__file__).parent.parent / "data" / "screenshots" / "output_debug.png"
        cv2.imwrite(str(output_path), output_frame)
        print(f"✓ Debug overlay saved to: {output_path}")
        
        # Show frame if display available
        try:
            cv2.imshow("Diabot - Developer Mode", output_frame)
            print("\nPress any key to close the window...")
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        except Exception as e:
            print(f"Note: Could not display window ({e}). Image saved to {output_path}")
        
        print("\n✅ Bot cycle complete!")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    screenshot_arg = sys.argv[1] if len(sys.argv) > 1 else None
    sys.exit(main(screenshot_arg))
