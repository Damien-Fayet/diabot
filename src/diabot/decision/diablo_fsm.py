"""Diablo-inspired Finite State Machine for decision making."""

from enum import Enum, auto
from dataclasses import dataclass
from typing import Optional
from pathlib import Path

from diabot.models.state import GameState, Action
from diabot.progression import QuestManager


class FSMState(Enum):
    """FSM states reflecting Diablo gameplay patterns."""
    
    IDLE = auto()       # No enemies, waiting/scanning
    EXPLORE = auto()    # Moving to unexplored area, no danger
    ENGAGE = auto()     # Enemies detected, actively attacking
    KITE = auto()       # Enemies too close, repositioning while attacking
    PANIC = auto()      # Low HP or surrounded, emergency
    RECOVER = auto()    # Regaining resources after danger


@dataclass
class FSMTransition:
    """Record of a state transition with reasoning."""
    
    from_state: FSMState
    to_state: FSMState
    reason: str
    trigger_values: dict


class DiabloFSM:
    """
    Finite State Machine for Diablo 2 bot decision making.
    
    Design philosophy:
    - States reflect human gameplay intuition
    - Survival prioritized over optimization
    - Easy to extend with new states
    
    State priorities (highest to lowest):
    1. PANIC - Survival emergency
    2. RECOVER - Post-danger recovery
    3. KITE - Tactical repositioning
    4. ENGAGE - Active combat
    5. EXPLORE - Safe exploration
    6. IDLE - Default/waiting
    """
    
    def __init__(self, initial_state: FSMState = FSMState.IDLE, progression_file: Optional[Path] = None):
        """Initialize FSM.
        
        Args:
            initial_state: Starting FSM state
            progression_file: Path to game_progression.json (defaults to data/game_progression.json)
        """
        self.current_state = initial_state
        self.previous_state: Optional[FSMState] = None
        self.state_duration = 0  # Frames in current state
        self.transition_history: list[FSMTransition] = []
        
        # Quest management
        if progression_file is None:
            progression_file = Path("data/game_progression.json")
        self.quest_manager = QuestManager(progression_file, debug=False)
        
        # Thresholds for state transitions
        self.PANIC_HP_THRESHOLD = 30.0
        self.RECOVER_HP_THRESHOLD = 60.0
        self.KITE_ENEMY_COUNT = 5
        self.ENGAGE_ENEMY_COUNT = 1
    
    def update(self, game_state: GameState) -> FSMState:
        """
        Update FSM based on current game state.
        
        Args:
            game_state: Current game state
            
        Returns:
            New FSM state (may be same as current)
        """
        new_state = self._evaluate_transitions(game_state)
        
        if new_state != self.current_state:
            # State transition
            transition = FSMTransition(
                from_state=self.current_state,
                to_state=new_state,
                reason=self._get_transition_reason(game_state, new_state),
                trigger_values={
                    "hp": game_state.health_percent,
                    "enemies": game_state.enemy_count,
                    "threatened": game_state.is_threatened,
                },
            )
            self.transition_history.append(transition)
            
            self.previous_state = self.current_state
            self.current_state = new_state
            self.state_duration = 0
        else:
            self.state_duration += 1
        
        return self.current_state
    
    def _evaluate_transitions(self, state: GameState) -> FSMState:
        """
        Evaluate which state to transition to.
        
        Priority order (highest first):
        1. PANIC if critical danger
        2. RECOVER if recovering from danger
        3. KITE if need repositioning
        4. ENGAGE if enemies present
        5. EXPLORE if safe and moving
        6. IDLE otherwise
        """
        threat_level = state.debug_info.get("threat_level", "none")
        
        # PANIC: Critical situation
        if state.needs_potion or (state.health_percent < self.PANIC_HP_THRESHOLD and state.is_threatened):
            return FSMState.PANIC
        
        # RECOVER: Just survived danger, need to stabilize
        if self.current_state == FSMState.PANIC and state.health_percent < self.RECOVER_HP_THRESHOLD:
            return FSMState.RECOVER
        
        if self.current_state == FSMState.RECOVER:
            # Stay in RECOVER until HP > 60% or no threats
            if state.health_percent < self.RECOVER_HP_THRESHOLD and state.is_threatened:
                return FSMState.RECOVER
        
        # KITE: Too many enemies or critical threat
        if state.is_threatened:
            if state.enemy_count >= self.KITE_ENEMY_COUNT or threat_level == "critical":
                return FSMState.KITE
        
        # ENGAGE: Enemies present but manageable
        if state.is_threatened and state.enemy_count > 0:
            return FSMState.ENGAGE
        
        # EXPLORE: Safe, look for content
        if state.current_location in ["dungeon", "deep_dungeon"]:
            return FSMState.EXPLORE
        
        # IDLE: Default state
        return FSMState.IDLE
    
    def _get_transition_reason(self, state: GameState, new_state: FSMState) -> str:
        """Get human-readable reason for transition."""
        reasons = {
            FSMState.PANIC: f"Critical: HP={state.health_percent:.0f}%, Enemies={state.enemy_count}",
            FSMState.RECOVER: f"Recovering: HP={state.health_percent:.0f}%",
            FSMState.KITE: f"Too many enemies: {state.enemy_count}",
            FSMState.ENGAGE: f"Enemy engaged: {state.enemy_count}",
            FSMState.EXPLORE: f"Safe exploration: {state.current_location}",
            FSMState.IDLE: "No activity",
        }
        return reasons.get(new_state, "Unknown")
    
    def decide_action(self, game_state: GameState) -> Action:
        """
        Decide action based on current FSM state.
        
        Args:
            game_state: Current game state
            
        Returns:
            Action to take
        """
        # Map FSM states to actions
        if self.current_state == FSMState.PANIC:
            return self._panic_action(game_state)
        
        elif self.current_state == FSMState.RECOVER:
            return self._recover_action(game_state)
        
        elif self.current_state == FSMState.KITE:
            return self._kite_action(game_state)
        
        elif self.current_state == FSMState.ENGAGE:
            return self._engage_action(game_state)
        
        elif self.current_state == FSMState.EXPLORE:
            return self._explore_action(game_state)
        
        else:  # IDLE
            return self._idle_action(game_state)
    
    def _panic_action(self, state: GameState) -> Action:
        """Emergency action: drink potion or flee."""
        if state.needs_potion:
            return Action(action_type="drink_potion", params={"potion_type": "health"})
        return Action(action_type="flee", target="town_portal")
    
    def _recover_action(self, state: GameState) -> Action:
        """Recovery action: potion or retreat to safe area."""
        if state.health_percent < 50:
            return Action(action_type="drink_potion", params={"potion_type": "health"})
        return Action(action_type="retreat", target="safe_area")
    
    def _kite_action(self, state: GameState) -> Action:
        """Kiting action: move while attacking."""
        return Action(action_type="kite", target="enemy_group", params={"direction": "backward"})
    
    def _engage_action(self, state: GameState) -> Action:
        """Combat action: attack enemies."""
        return Action(action_type="attack", target="nearest_enemy")
    
    def _explore_action(self, state: GameState) -> Action:
        """Exploration action: move to unexplored areas."""
        return Action(action_type="explore", target="unexplored")
    
    def _idle_action(self, state: GameState) -> Action:
        """Idle action: wait or scan."""
        return Action(action_type="idle")
    
    def get_state_name(self) -> str:
        """Get current state name for display."""
        return self.current_state.name
    
    def get_transition_summary(self) -> str:
        """Get summary of recent transitions for debugging."""
        if not self.transition_history:
            return "No transitions"
        
        last = self.transition_history[-1]
        return f"{last.from_state.name} → {last.to_state.name}: {last.reason}"
    
    def get_quest_guidance(self) -> str:
        """Get guidance on what to do next based on quest progression.
        
        Returns:
            Human-readable quest guidance string
        """
        current_quest = self.quest_manager.get_current_quest()
        if not current_quest:
            next_quest = self.quest_manager.get_next_quest()
            if next_quest:
                self.quest_manager.start_quest(next_quest.id)
                return f"Starting: {next_quest.name} - {next_quest.objective}"
            return "No quests available (game complete?)"
        
        return f"{current_quest.name}: {current_quest.objective}"
    
    def complete_current_quest(self) -> bool:
        """Mark current quest as complete and advance to next.
        
        Returns:
            True if quest was completed, False otherwise
        """
        if not self.quest_manager.current_quest_id:
            return False
        
        return self.quest_manager.complete_quest(self.quest_manager.current_quest_id)
    
    def get_progress_summary(self) -> str:
        """Get full game progress summary for display.
        
        Returns:
            Human-readable progress string
        """
        return self.quest_manager.get_progress_summary()
