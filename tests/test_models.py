"""Basic tests for the diabot architecture."""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np
from diabot.core.implementations import (
    SimpleStateBuilder,
    RuleBasedVisionModule,
)
from diabot.core.interfaces import Perception
from diabot.models.state import GameState, Action


def test_gamestate_creation():
    """Test GameState dataclass."""
    state = GameState(
        health_percent=50.0,
        mana_percent=75.0,
        enemy_count=2,
        visible_items=1,
        current_location="dungeon",
    )
    
    assert state.health_percent == 50.0
    assert state.is_threatened is True  # 2 enemies
    assert state.needs_potion is False  # 50% health >= 30%
    print("✓ test_gamestate_creation passed")


def test_gamestate_needs_potion():
    """Test needs_potion threshold."""
    state_low = GameState(
        health_percent=25.0,
        mana_percent=50.0,
        enemy_count=0,
        visible_items=0,
        current_location="dungeon",
    )
    
    state_ok = GameState(
        health_percent=50.0,
        mana_percent=50.0,
        enemy_count=0,
        visible_items=0,
        current_location="dungeon",
    )
    
    assert state_low.needs_potion is True
    assert state_ok.needs_potion is False
    print("✓ test_gamestate_needs_potion passed")


def test_state_builder():
    """Test StateBuilder."""
    perception = Perception(
        hp_ratio=0.8,
        mana_ratio=0.6,
        enemy_count=1,
        enemy_types=["skeleton"],
        visible_items=[],
        player_position=(400, 300),
    )
    
    builder = SimpleStateBuilder(frame_counter=42)
    state = builder.build(perception)
    
    assert state.health_percent == 80.0
    assert state.mana_percent == 60.0
    assert state.enemy_count == 1
    assert state.is_threatened is True
    assert state.frame_number == 42
    print("✓ test_state_builder passed")


def test_action_creation():
    """Test Action dataclass."""
    action = Action(action_type="attack", target="enemy_0")
    
    assert action.action_type == "attack"
    assert action.target == "enemy_0"
    assert action.params == {}
    print("✓ test_action_creation passed")


if __name__ == "__main__":
    print("Running unit tests...\n")
    test_gamestate_creation()
    test_gamestate_needs_potion()
    test_state_builder()
    test_action_creation()
    print("\n✅ All tests passed!")
