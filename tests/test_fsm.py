"""Tests for the Diablo FSM."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from diabot.decision.diablo_fsm import DiabloFSM, FSMState
from diabot.models.state import GameState


def test_fsm_panic_transition():
    """Test transition to PANIC state."""
    fsm = DiabloFSM()
    
    # Low HP + threatened = PANIC
    state = GameState(
        health_percent=20.0,
        mana_percent=50.0,
        enemy_count=3,
        visible_items=0,
        current_location="dungeon",
    )
    
    new_state = fsm.update(state)
    assert new_state == FSMState.PANIC, f"Expected PANIC, got {new_state}"
    
    action = fsm.decide_action(state)
    assert action.action_type == "drink_potion", f"Expected drink_potion, got {action.action_type}"
    
    print("✓ test_fsm_panic_transition passed")


def test_fsm_engage_transition():
    """Test transition to ENGAGE state."""
    fsm = DiabloFSM()
    
    # Healthy + enemies = ENGAGE
    state = GameState(
        health_percent=80.0,
        mana_percent=60.0,
        enemy_count=2,
        visible_items=0,
        current_location="dungeon",
        debug_info={"threat_level": "medium"},
    )
    
    new_state = fsm.update(state)
    assert new_state == FSMState.ENGAGE, f"Expected ENGAGE, got {new_state}"
    
    action = fsm.decide_action(state)
    assert action.action_type == "attack", f"Expected attack, got {action.action_type}"
    
    print("✓ test_fsm_engage_transition passed")


def test_fsm_kite_transition():
    """Test transition to KITE state."""
    fsm = DiabloFSM()
    
    # Many enemies = KITE
    state = GameState(
        health_percent=70.0,
        mana_percent=50.0,
        enemy_count=8,
        visible_items=0,
        current_location="deep_dungeon",
        debug_info={"threat_level": "high"},
    )
    
    new_state = fsm.update(state)
    assert new_state == FSMState.KITE, f"Expected KITE, got {new_state}"
    
    action = fsm.decide_action(state)
    assert action.action_type == "kite", f"Expected kite, got {action.action_type}"
    
    print("✓ test_fsm_kite_transition passed")


def test_fsm_explore_transition():
    """Test transition to EXPLORE state."""
    fsm = DiabloFSM()
    
    # Safe in dungeon = EXPLORE
    state = GameState(
        health_percent=90.0,
        mana_percent=80.0,
        enemy_count=0,
        visible_items=0,
        current_location="dungeon",
    )
    
    new_state = fsm.update(state)
    assert new_state == FSMState.EXPLORE, f"Expected EXPLORE, got {new_state}"
    
    action = fsm.decide_action(state)
    assert action.action_type == "explore", f"Expected explore, got {action.action_type}"
    
    print("✓ test_fsm_explore_transition passed")


def test_fsm_transition_history():
    """Test transition history tracking."""
    fsm = DiabloFSM()
    
    # Start safe
    state1 = GameState(
        health_percent=90.0,
        mana_percent=80.0,
        enemy_count=0,
        visible_items=0,
        current_location="dungeon",
    )
    
    fsm.update(state1)
    assert len(fsm.transition_history) == 1, "Should have 1 transition (IDLE->EXPLORE)"
    
    # Encounter enemies
    state2 = GameState(
        health_percent=90.0,
        mana_percent=80.0,
        enemy_count=3,
        visible_items=0,
        current_location="dungeon",
        debug_info={"threat_level": "medium"},
    )
    
    fsm.update(state2)
    assert len(fsm.transition_history) == 2, "Should have 2 transitions"
    assert fsm.current_state == FSMState.ENGAGE, "Should be in ENGAGE state"
    
    # Get hurt
    state3 = GameState(
        health_percent=25.0,
        mana_percent=50.0,
        enemy_count=3,
        visible_items=0,
        current_location="dungeon",
    )
    
    fsm.update(state3)
    assert len(fsm.transition_history) == 3, "Should have 3 transitions"
    assert fsm.current_state == FSMState.PANIC, "Should be in PANIC state"
    
    print("✓ test_fsm_transition_history passed")


def test_fsm_state_duration():
    """Test state duration tracking."""
    fsm = DiabloFSM()
    
    state = GameState(
        health_percent=90.0,
        mana_percent=80.0,
        enemy_count=0,
        visible_items=0,
        current_location="dungeon",
    )
    
    # Stay in same state
    fsm.update(state)
    assert fsm.state_duration == 0, "Duration should be 0 after first frame"
    
    fsm.update(state)
    assert fsm.state_duration == 1, "Duration should be 1 after second frame"
    
    fsm.update(state)
    assert fsm.state_duration == 2, "Duration should be 2 after third frame"
    
    print("✓ test_fsm_state_duration passed")


if __name__ == "__main__":
    print("Running FSM tests...\n")
    test_fsm_panic_transition()
    test_fsm_engage_transition()
    test_fsm_kite_transition()
    test_fsm_explore_transition()
    test_fsm_transition_history()
    test_fsm_state_duration()
    print("\n✅ All FSM tests passed!")
