"""Tests for vision module on real screenshots."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import cv2
from diabot.core.vision_advanced import DiabloVisionModule, FastVisionModule
from diabot.core.implementations import SimpleStateBuilder


def test_vision_on_screenshot(screenshot_path: str, vision_module, name: str):
    """Test vision module on a screenshot."""
    print(f"\n📸 Testing {name} on: {Path(screenshot_path).name}")
    print("=" * 60)
    
    # Load image
    frame = cv2.imread(screenshot_path)
    if frame is None:
        print(f"❌ Failed to load {screenshot_path}")
        return
    
    print(f"  Frame shape: {frame.shape}")
    
    # Perceive
    perception = vision_module.perceive(frame)
    
    print(f"  HP Ratio: {perception.hp_ratio:.2%}")
    print(f"  Mana Ratio: {perception.mana_ratio:.2%}")
    print(f"  Enemy Count: {perception.enemy_count}")
    print(f"  Enemy Types: {perception.enemy_types}")
    print(f"  Items: {len(perception.visible_items)} visible")
    print(f"  Player Pos: {perception.player_position}")
    
    # Build state
    builder = SimpleStateBuilder(frame_counter=0)
    state = builder.build(perception)
    
    print(f"\n  GameState:")
    print(f"    Health: {state.health_percent:.1f}%")
    print(f"    Mana: {state.mana_percent:.1f}%")
    print(f"    Threatened: {state.is_threatened}")
    print(f"    Needs Potion: {state.needs_potion}")
    
    return state


def main():
    """Run vision tests on all screenshots."""
    screenshot_dir = Path(__file__).parent.parent / "data" / "screenshots" / "inputs"
    
    # Get test images
    images = [
        screenshot_dir / "char_menu.jpg",
        screenshot_dir / "game.jpg",
    ]
    
    print("🎮 Vision Module Testing")
    print("=" * 60)
    
    # Test both modules
    modules = [
        (DiabloVisionModule(debug=False), "DiabloVisionModule (Advanced)"),
        (FastVisionModule(), "FastVisionModule (Optimized)"),
    ]
    
    for vision_module, module_name in modules:
        print(f"\n🔬 {module_name}")
        print("=" * 60)
        
        for img_path in images:
            if img_path.exists():
                test_vision_on_screenshot(str(img_path), vision_module, module_name)
    
    print("\n✅ Vision testing complete!")


if __name__ == "__main__":
    main()
