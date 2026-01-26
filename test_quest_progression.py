#!/usr/bin/env python3
"""Test quest progression integration with DiabloFSM.

Demonstrates:
1. Loading quest progression from JSON
2. Starting and completing quests
3. Automatic progression to next quest
4. FSM state tracking with quest context
"""

from pathlib import Path
from src.diabot.decision.diablo_fsm import DiabloFSM, FSMState
from src.diabot.models.state import GameState, Action


def test_quest_progression() -> None:
    """Test quest progression system."""
    print("="*70)
    print("QUEST PROGRESSION SYSTEM TEST")
    print("="*70)
    
    # Initialize FSM with quest manager
    fsm = DiabloFSM(
        initial_state=FSMState.IDLE,
        progression_file=Path("data/game_progression.json")
    )
    
    print(f"\n✓ FSM initialized with quest manager")
    print(f"  {fsm.quest_manager.get_progress_summary()}\n")
    
    # Test 1: Get current quest guidance
    print("Test 1: Quest Guidance")
    print("-" * 70)
    guidance = fsm.get_quest_guidance()
    print(f"Guidance: {guidance}\n")
    
    # Test 2: Simulate quest flow
    print("Test 2: Quest Flow Simulation")
    print("-" * 70)
    
    quest_chain = fsm.quest_manager.get_quest_chain()
    print(f"Quest chain has {len(quest_chain)} quests:\n")
    
    for i, quest in enumerate(quest_chain[:5], 1):  # Show first 5
        print(f"  {i}. {quest.name} (Act {quest.act})")
        print(f"     → {quest.objective}")
        print(f"     Reward: {quest.reward}\n")
    
    # Test 3: Complete first quest
    print("Test 3: Complete First Quest")
    print("-" * 70)
    
    first_quest = quest_chain[0]
    print(f"Starting: {first_quest.name}")
    fsm.quest_manager.start_quest(first_quest.id)
    
    print(f"Status before: {fsm.quest_manager.get_current_quest().status.value}")
    fsm.complete_current_quest()
    print(f"Status after: {fsm.quest_manager.get_current_quest().status.value if fsm.quest_manager.get_current_quest() else 'Completed'}")
    
    print(f"\nCurrent state:\n{fsm.get_progress_summary()}\n")
    
    # Test 4: FSM with quest context
    print("Test 4: FSM State Transitions with Quest Context")
    print("-" * 70)
    
    # Simulate some game states
    test_states = [
        GameState(
            hp_ratio=1.0,
            mana_ratio=1.0,
            enemies=[],
            current_location="tristram",
        ),
        GameState(
            hp_ratio=1.0,
            mana_ratio=1.0,
            enemies=[
                "zombie", "skeleton", "fallen"  # Just count, not full objects for test
            ],
            current_location="barracks",
        ),
        GameState(
            hp_ratio=0.25,
            mana_ratio=0.5,
            enemies=[
                "zombie", "skeleton", "fallen", "quill_rat", "demon"  # 5 enemies
            ],
            current_location="barracks",
        ),
    ]
    
    for i, state in enumerate(test_states, 1):
        fsm.update(state)
        print(f"\nScenario {i}:")
        print(f"  HP: {state.health_percent}% | Enemies: {state.enemy_count} | Location: {state.current_location}")
        print(f"  FSM State: {fsm.get_state_name()}")
        print(f"  Reason: {fsm.get_transition_summary()}")
        print(f"  Quest: {fsm.get_quest_guidance()}")
    
    print("\n" + "="*70)
    print("✓ All tests completed")
    print("="*70)


if __name__ == "__main__":
    test_quest_progression()
