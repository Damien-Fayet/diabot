"""
Demo: Separated UI vs Environment Vision

Shows how the refactored vision module works with clear separation.
"""

from diabot.vision import UIVisionModule, EnvironmentVisionModule, UI_REGIONS, ENVIRONMENT_REGIONS
import numpy as np
import cv2


def demo_vision_architecture():
    """Demonstrate the separated vision architecture."""
    
    print("\n" + "="*70)
    print("VISION ARCHITECTURE: UI vs Environment Separation")
    print("="*70)
    
    # Create a dummy frame (in real usage, this comes from screenshot)
    frame = np.zeros((768, 1024, 3), dtype=np.uint8)
    
    # Simulate some game content
    # - Red bar at top-left (health)
    frame[10:30, 10:100] = [0, 0, 200]  # BGR format, so red
    # - Blue bar below it (mana)
    frame[40:60, 10:100] = [200, 0, 0]  # BGR format, so blue
    # - Red/orange enemy in center
    frame[350:380, 500:530] = [0, 100, 255]  # BGR orange
    # - Yellow item on ground
    frame[400:415, 480:495] = [0, 200, 255]  # BGR yellow/gold
    
    # Initialize modules
    ui_module = UIVisionModule(debug=True)
    env_module = EnvironmentVisionModule(debug=True)
    
    print("\n" + "─"*70)
    print("STEP 1: Extract UI Region (Top-Left)")
    print("─"*70)
    
    ui_region = UI_REGIONS['top_left_ui']
    x, y, w, h = ui_region.get_bounds(frame.shape[0], frame.shape[1])
    print(f"UI Region bounds: x={x}, y={y}, w={w}, h={h}")
    print(f"  → Position: {ui_region.x_ratio*100:.0f}% from left, {ui_region.y_ratio*100:.0f}% from top")
    print(f"  → Size: {ui_region.w_ratio*100:.0f}% width, {ui_region.h_ratio*100:.0f}% height")
    
    ui_state = ui_module.analyze(frame)
    
    print(f"\n✓ UI Analysis Results:")
    print(f"  Health: {ui_state.hp_ratio:.1%}")
    print(f"  Mana: {ui_state.mana_ratio:.1%}")
    print(f"  Potions: {ui_state.potions_available}")
    
    print("\n" + "─"*70)
    print("STEP 2: Extract Playfield Region (Center)")
    print("─"*70)
    
    playfield_region = ENVIRONMENT_REGIONS['playfield']
    x, y, w, h = playfield_region.get_bounds(frame.shape[0], frame.shape[1])
    print(f"Playfield Region bounds: x={x}, y={y}, w={w}, h={h}")
    print(f"  → Position: {playfield_region.x_ratio*100:.0f}% from left, {playfield_region.y_ratio*100:.0f}% from top")
    print(f"  → Size: {playfield_region.w_ratio*100:.0f}% width, {playfield_region.h_ratio*100:.0f}% height")
    
    env_state = env_module.analyze(frame)
    
    print(f"\n✓ Environment Analysis Results:")
    print(f"  Enemies: {len(env_state.enemies)}")
    for enemy in env_state.enemies:
        print(f"    - {enemy.enemy_type} at {enemy.position}, confidence {enemy.confidence:.0%}")
    print(f"  Items: {len(env_state.items)}")
    for item in env_state.items:
        print(f"    - {item}")
    print(f"  Player Position: {env_state.player_position}")
    
    print("\n" + "="*70)
    print("ARCHITECTURE BENEFITS")
    print("="*70)
    
    print("""
    ✓ Clear Separation:
      - UIVisionModule handles ONLY UI elements (health, mana, potions)
      - EnvironmentVisionModule handles ONLY game elements (enemies, items)
      - Never confusion between the two
    
    ✓ Independent Development:
      - Can improve UI detection without touching environment detection
      - Can tune environment without breaking UI
      - Can test each independently
    
    ✓ Scalable:
      - Easy to add new UI elements (buffs, debuffs, cooldowns)
      - Easy to add environment elements (obstacles, doors, traps)
      - No code refactoring needed, just add new detection methods
    
    ✓ Maintainable:
      - UIVisionModule is ~100 LOC
      - EnvironmentVisionModule is ~100 LOC
      - Each is focused and easy to understand
      - Easy to debug individual components
    
    ✓ Testable:
      - Can test UI detection on UI screenshots
      - Can test environment detection on environment screenshots
      - No coupling between tests
    """)
    
    print("\n" + "="*70)
    print("NEXT STEPS: FIABILISATION")
    print("="*70)
    
    print("""
    To make this RELIABLE:
    
    1. Create Configuration File
       - vision_config.yaml with HSV ranges
       - Parameterize all magic numbers
       - Easy to tune without changing code
    
    2. Calibration Tool
       - Interactive UI to adjust HSV ranges
       - Test on real screenshots
       - Export working parameters
    
    3. Test Suite
       - test_ui_vision.py with various screenshots
       - test_environment_vision.py with various scenarios
       - Verify no false positives / negatives
    
    4. Debug Visualizer
       - Show detected masks
       - Show extracted regions
       - Show final detections
       - Make debugging visual and easy
    
    5. Logging
       - Log confidence scores
       - Log detection method used
       - Log timing
       - Help identify problems
    """)


if __name__ == "__main__":
    demo_vision_architecture()
