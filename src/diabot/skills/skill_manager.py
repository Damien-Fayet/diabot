"""Skill selection and management logic."""

from typing import Optional
from ..models.skills import Skill, SkillBar, SkillType
from ..models.state import GameState


class SkillManager:
    """
    Manages skill selection based on game state.
    
    Responsibilities:
    - Select best skill for current situation
    - Track cooldowns
    - Manage mana efficiency
    """
    
    def __init__(self, skill_bar: SkillBar):
        """
        Initialize skill manager.
        
        Args:
            skill_bar: Character's skill bar configuration
        """
        self.skill_bar = skill_bar
        
    def select_best_skill(
        self,
        game_state: GameState,
        prefer_defensive: bool = False,
    ) -> Optional[Skill]:
        """
        Select the best skill for current situation.
        
        Args:
            game_state: Current game state
            prefer_defensive: Prioritize defensive skills
            
        Returns:
            Best skill to use, or None if no skill available
        """
        # Get skills we can afford
        affordable = self.skill_bar.get_affordable_skills(game_state.mana_ratio)
        
        if not affordable:
            return None
        
        # Filter by preference
        if prefer_defensive:
            defensive = [s for s in affordable if s.skill_type == SkillType.DEFENSIVE]
            if defensive:
                affordable = defensive
        
        # Calculate effectiveness scores
        scores = [
            (
                skill,
                skill.get_effectiveness_score(
                    enemy_count=len(game_state.enemies),
                    current_hp_ratio=game_state.hp_ratio,
                )
            )
            for skill in affordable
        ]
        
        # Sort by score (highest first)
        scores.sort(key=lambda x: x[1], reverse=True)
        
        if scores:
            return scores[0][0]
        
        return None
    
    def select_combat_skill(self, game_state: GameState) -> Optional[Skill]:
        """
        Select best offensive skill for combat.
        
        Args:
            game_state: Current game state
            
        Returns:
            Best combat skill
        """
        affordable = self.skill_bar.get_affordable_skills(game_state.mana_ratio)
        
        # Filter combat skills
        combat_skills = [
            s for s in affordable
            if s.skill_type in [SkillType.ATTACK, SkillType.AOE, SkillType.DOT]
        ]
        
        if not combat_skills:
            return None
        
        enemy_count = len(game_state.enemies)
        
        # Prefer AOE when many enemies
        if enemy_count >= 3:
            aoe_skills = [s for s in combat_skills if s.is_aoe]
            if aoe_skills:
                # Return highest damage AOE
                return max(aoe_skills, key=lambda s: s.damage * s.level)
        
        # Single target for few enemies
        return max(combat_skills, key=lambda s: s.damage * s.level)
    
    def select_escape_skill(self, game_state: GameState) -> Optional[Skill]:
        """
        Select best skill for escaping danger.
        
        Args:
            game_state: Current game state
            
        Returns:
            Best escape skill (teleport, etc.)
        """
        affordable = self.skill_bar.get_affordable_skills(game_state.mana_ratio)
        
        # Look for utility skills
        utility = [
            s for s in affordable
            if s.skill_type == SkillType.UTILITY
        ]
        
        if utility:
            # Prefer teleport-like skills (highest range)
            return max(utility, key=lambda s: s.range_)
        
        return None
    
    def select_defensive_skill(self, game_state: GameState) -> Optional[Skill]:
        """
        Select best defensive skill.
        
        Args:
            game_state: Current game state
            
        Returns:
            Best defensive skill
        """
        affordable = self.skill_bar.get_affordable_skills(game_state.mana_ratio)
        
        defensive = [
            s for s in affordable
            if s.skill_type == SkillType.DEFENSIVE
        ]
        
        if defensive:
            # Prefer skills with shorter cooldowns when in danger
            if game_state.hp_ratio < 0.3:
                return min(defensive, key=lambda s: s.cooldown)
            return defensive[0]
        
        return None
    
    def get_mana_reserve(self) -> float:
        """
        Calculate minimum mana to keep in reserve.
        
        Returns:
            Mana ratio to keep (0.0-1.0)
        """
        # Always keep enough for defensive/escape skills
        all_skills = self.skill_bar.get_all_skills()
        
        defensive_costs = [
            s.mana_cost for s in all_skills
            if s.skill_type in [SkillType.DEFENSIVE, SkillType.UTILITY]
        ]
        
        if not defensive_costs:
            return 0.1  # 10% reserve
        
        # Reserve enough for most expensive defensive skill
        max_cost = max(defensive_costs)
        return min(0.3, max_cost / 100.0)  # Cap at 30%
    
    def should_use_basic_attack(self, game_state: GameState) -> bool:
        """
        Decide if we should use basic attack instead of skills.
        
        Args:
            game_state: Current game state
            
        Returns:
            True if basic attack is better choice
        """
        # Low mana - conserve for defensive skills
        if game_state.mana_ratio < self.get_mana_reserve():
            return True
        
        # No affordable skills
        affordable = self.skill_bar.get_affordable_skills(game_state.mana_ratio)
        if not affordable:
            return True
        
        # Weak enemies - save mana
        if len(game_state.enemies) == 1 and game_state.hp_ratio > 0.7:
            return True
        
        return False
    
    def get_skill_usage_summary(self) -> dict:
        """
        Get summary of skill availability.
        
        Returns:
            Dictionary with skill stats
        """
        all_skills = self.skill_bar.get_all_skills()
        ready = self.skill_bar.get_ready_skills()
        
        return {
            "total_skills": len(all_skills),
            "ready_skills": len(ready),
            "on_cooldown": len(all_skills) - len(ready),
            "skill_types": {
                "attack": len([s for s in all_skills if s.skill_type == SkillType.ATTACK]),
                "defensive": len([s for s in all_skills if s.skill_type == SkillType.DEFENSIVE]),
                "utility": len([s for s in all_skills if s.skill_type == SkillType.UTILITY]),
                "aoe": len([s for s in all_skills if s.skill_type == SkillType.AOE]),
            },
        }
