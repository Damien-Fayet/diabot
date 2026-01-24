"""Enhanced state builder with perception analysis."""

from diabot.core.interfaces import StateBuilder, Perception
from diabot.models.state import GameState


class EnhancedStateBuilder(StateBuilder):
    """
    Convert Perception into GameState with enhanced analysis.
    
    Analyzes perception data to infer additional state information:
    - Threat level based on enemy composition
    - Location estimation from environmental cues
    - Item significance
    """
    
    def __init__(self, frame_counter: int = 0):
        """Initialize state builder."""
        self.frame_counter = frame_counter
        self.previous_state = None
    
    def build(self, perception: Perception) -> GameState:
        """
        Convert Perception to GameState with analysis.
        
        Args:
            perception: Raw perception data from vision module
            
        Returns:
            GameState: Enhanced game state with analysis
        """
        from diabot.models.state import EnemyInfo, ItemInfo
        
        # Convert enemies
        enemies = [
            EnemyInfo(type=e_type, position=(0, 0))
            for e_type in perception.enemy_types
        ]
        
        # Convert items
        items = [
            ItemInfo(type=item_name, position=(0, 0))
            for item_name in perception.visible_items
        ]
        
        # Calculate threat level
        threat_level = self._calculate_threat_level(perception)
        
        # Basic conversion using ratio-based API
        state = GameState(
            hp_ratio=perception.hp_ratio,
            mana_ratio=perception.mana_ratio,
            enemies=enemies,
            items=items,
            threat_level=threat_level,
            current_location=self._estimate_location(perception),
            frame_number=self.frame_counter,
        )
        
        # Add debug info
        state.debug_info = {
            "threat_level": threat_level,
            "item_types": perception.visible_items,
            "player_pos": perception.player_position,
        }
        
        self.frame_counter += 1
        return state
    
    def _calculate_threat_level(self, perception: Perception) -> str:
        """
        Calculate threat level based on enemy composition.
        
        Returns: "none", "low", "medium", "high", "critical"
        """
        if perception.enemy_count == 0:
            return "none"
        
        large_enemies = perception.enemy_types.count("large_enemy")
        total_enemies = perception.enemy_count
        
        if large_enemies > 2 or total_enemies > 8:
            return "critical"
        elif large_enemies > 0 or total_enemies > 5:
            return "high"
        elif total_enemies > 2:
            return "medium"
        else:
            return "low"
    
    def _estimate_location(self, perception: Perception) -> str:
        """
        Estimate current location based on environmental cues.
        
        For now, simple heuristics. Could be enhanced with ML later.
        """
        # Simple heuristic: more enemies = dungeon, no enemies = town
        if perception.enemy_count > 5:
            return "deep_dungeon"
        elif perception.enemy_count > 0:
            return "dungeon"
        else:
            return "town"
    
    def update_frame_counter(self):
        """Increment frame counter."""
        self.frame_counter += 1


class AdvancedDecisionEngine:
    """
    Enhanced decision engine with threat-aware behavior.
    """
    
    def __init__(self):
        """Initialize decision engine."""
        self.threat_history = []
    
    def decide(self, state: GameState):
        """
        Make decision based on game state with threat awareness.
        
        Decision hierarchy:
        1. Critical HP + threat → flee + drink potion
        2. Low HP → drink potion
        3. Threat + low mana → drink mana potion
        4. High threat → attack/kite
        5. Medium threat → fight defensively
        6. Low/no threat → explore
        
        Args:
            state: Current game state
            
        Returns:
            Action: Decided action
        """
        from diabot.models.state import Action
        
        threat_level = state.debug_info.get("threat_level", "none")
        
        # Critical situation: low health + threats
        if state.needs_potion and state.is_threatened:
            return Action(action_type="drink_potion", params={"potion_type": "health"})
        
        # Low health
        if state.needs_potion:
            return Action(action_type="drink_potion", params={"potion_type": "health"})
        
        # Low mana + threat
        if state.mana_percent < 20.0 and state.is_threatened:
            return Action(action_type="drink_potion", params={"potion_type": "mana"})
        
        # Threat response
        if state.is_threatened:
            if threat_level == "critical":
                return Action(action_type="flee", target="town")
            elif threat_level == "high":
                return Action(action_type="attack_kite", target="enemy_0")
            else:
                return Action(action_type="attack", target="enemy_0")
        
        # Peaceful - explore
        return Action(action_type="explore", target="deeper")
