"""Enhanced decision engine integrating FSM, Skills, and Inventory."""

from typing import Optional
from dataclasses import dataclass, field

from diabot.models.state import GameState, Action
from diabot.models.skills import SkillBar
from diabot.models.inventory import Inventory
from diabot.skills import SkillManager
from diabot.decision.diablo_fsm import DiabloFSM, FSMState


@dataclass
class EnhancedDecisionContext:
    """
    Complete decision context for the bot.
    
    Integrates:
    - FSM for high-level state management
    - Skill system for combat decisions
    - Inventory for resource management
    """
    
    fsm: DiabloFSM = field(default_factory=DiabloFSM)
    skill_manager: Optional[SkillManager] = None
    inventory: Inventory = field(default_factory=Inventory)
    
    def __post_init__(self):
        """Initialize with default skill manager if not provided."""
        if self.skill_manager is None:
            # Create a basic skill bar (can be customized later)
            from diabot.models.skills import SORCERESS_SKILLS, SkillBar
            skill_bar = SkillBar(
                left_click=SORCERESS_SKILLS["glacial_spike"],
                right_click=SORCERESS_SKILLS["frozen_orb"],
                hotkeys=[
                    SORCERESS_SKILLS["teleport"],
                    None, None, None, None, None, None, None
                ],
            )
            self.skill_manager = SkillManager(skill_bar)


class EnhancedDecisionEngine:
    """
    Advanced decision engine combining FSM, skills, and inventory.
    
    This engine decides:
    1. High-level behavior (via FSM)
    2. Which skills to use (via SkillManager)
    3. When to use items (via Inventory)
    """
    
    def __init__(self, context: Optional[EnhancedDecisionContext] = None):
        """
        Initialize decision engine.
        
        Args:
            context: Decision context with FSM, skills, inventory
        """
        self.context = context or EnhancedDecisionContext()
    
    def decide(self, game_state: GameState) -> Action:
        """
        Make a decision based on current game state.
        
        Decision flow:
        1. Check if emergency item use needed (potions)
        2. Update FSM state
        3. Get FSM-recommended action
        4. Enhance action with skill selection
        
        Args:
            game_state: Current game state
            
        Returns:
            Action to execute
        """
        # 1. Emergency potion check (overrides everything)
        if self._should_use_health_potion(game_state):
            return self._create_potion_action("health")
        
        if self._should_use_mana_potion(game_state):
            return self._create_potion_action("mana")
        
        # 2. Update FSM
        self.context.fsm.update(game_state)
        
        # 3. Get FSM action
        fsm_action = self.context.fsm.decide_action(game_state)
        
        # 4. Enhance with skill selection
        enhanced_action = self._enhance_with_skills(
            fsm_action,
            game_state,
        )
        
        return enhanced_action
    
    def _should_use_health_potion(self, game_state: GameState) -> bool:
        """
        Check if we should drink health potion.
        
        Args:
            game_state: Current game state
            
        Returns:
            True if should use health potion
        """
        # Critical HP
        if game_state.hp_ratio < 0.3:
            return self.context.inventory.has_health_potion()
        
        # Low HP and in danger
        if game_state.hp_ratio < 0.5 and len(game_state.enemies) > 0:
            return self.context.inventory.has_health_potion()
        
        return False
    
    def _should_use_mana_potion(self, game_state: GameState) -> bool:
        """
        Check if we should drink mana potion.
        
        Args:
            game_state: Current game state
            
        Returns:
            True if should use mana potion
        """
        # Only in combat
        if len(game_state.enemies) == 0:
            return False
        
        # Low mana during combat
        if game_state.mana_ratio < 0.2:
            return self.context.inventory.has_mana_potion()
        
        # Or if we have plenty of potions and mana is < 50%
        if game_state.mana_ratio < 0.5:
            mana_pots = self.context.inventory.get_mana_potions()
            return len(mana_pots) > 3
        
        return False
    
    def _create_potion_action(self, potion_type: str) -> Action:
        """
        Create action to use a potion.
        
        Args:
            potion_type: "health" or "mana"
            
        Returns:
            Action to drink potion
        """
        if potion_type == "health":
            potion = self.context.inventory.get_best_health_potion()
        else:
            potion = self.context.inventory.get_best_mana_potion()
        
        if potion:
            # Use the potion
            self.context.inventory.use_item(potion)
            
            return Action(
                action_type="drink_potion",
                target="self",
                params={"potion_type": potion_type},
                item_name=potion.name if potion else None,
            )
        
        return Action(action_type="idle")
    
    def _enhance_with_skills(
        self,
        base_action: Action,
        game_state: GameState,
    ) -> Action:
        """
        Enhance FSM action with skill selection.
        
        Args:
            base_action: Base action from FSM
            game_state: Current game state
            
        Returns:
            Enhanced action with skill info
        """
        fsm_state = self.context.fsm.current_state
        
        # Only enhance attack-related actions
        if base_action.action_type not in ["attack", "engage", "kite"]:
            return base_action
        
        # Select appropriate skill based on FSM state
        skill = None
        
        if fsm_state == FSMState.PANIC:
            # Try escape skill first
            skill = self.context.skill_manager.select_escape_skill(game_state)
            if skill:
                return Action(
                    action_type="use_skill",
                    target="escape",
                    params={"skill": skill.name},
                    skill_name=skill.name,
                )
        
        if fsm_state in [FSMState.ENGAGE, FSMState.KITE]:
            # Combat skill
            skill = self.context.skill_manager.select_combat_skill(game_state)
        
        # If no specific skill or should use basic
        if skill is None or self.context.skill_manager.should_use_basic_attack(game_state):
            return Action(
                action_type="attack",
                target="nearest_enemy",
                params={"attack_type": "basic"},
            )
        
        # Use selected skill
        if skill:
            skill.use()  # Mark as used
            return Action(
                action_type="use_skill",
                target="enemies",
                params={"skill": skill.name},
                skill_name=skill.name,
            )
        
        return base_action
    
    def get_status_summary(self) -> dict:
        """
        Get comprehensive status of decision engine.
        
        Returns:
            Dictionary with FSM state, skills, inventory status
        """
        return {
            "fsm_state": self.context.fsm.current_state.name,
            "state_duration": self.context.fsm.state_duration,
            "skills": self.context.skill_manager.get_skill_usage_summary(),
            "inventory": self.context.inventory.get_inventory_summary(),
            "recent_transitions": [
                {
                    "from": t.from_state.name,
                    "to": t.to_state.name,
                    "reason": t.reason,
                }
                for t in self.context.fsm.transition_history[-3:]
            ],
        }
