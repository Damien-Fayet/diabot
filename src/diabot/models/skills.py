"""Skill system models for Diablo 2 characters."""

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Optional
import time


class SkillType(Enum):
    """Types of skills."""
    
    ATTACK = auto()      # Direct damage
    DEFENSIVE = auto()   # Buffs, shields
    UTILITY = auto()     # Movement, teleport
    SUMMON = auto()      # Minions
    AOE = auto()         # Area damage
    DOT = auto()         # Damage over time


class CharacterClass(Enum):
    """Diablo 2 character classes."""
    
    AMAZON = auto()
    ASSASSIN = auto()
    BARBARIAN = auto()
    DRUID = auto()
    NECROMANCER = auto()
    PALADIN = auto()
    SORCERESS = auto()


@dataclass
class Skill:
    """Represents a character skill."""
    
    name: str
    skill_type: SkillType
    char_class: CharacterClass
    
    # Resource costs
    mana_cost: int = 0
    
    # Cooldown (seconds)
    cooldown: float = 0.0
    
    # Last use timestamp (0 means never used)
    last_used: float = 0.0
    
    # Skill level (1-20 in D2)
    level: int = 1
    
    # Effectiveness
    damage: int = 0
    range_: int = 0  # Range in game units
    
    # Special properties
    is_aoe: bool = False
    is_synergy: bool = False  # Synergy with other skills
    requires_target: bool = True
    
    def is_ready(self) -> bool:
        """Check if skill is off cooldown."""
        if self.cooldown == 0:
            return True
        
        if self.last_used == 0.0:
            return True  # Never used
        
        return (time.time() - self.last_used) >= self.cooldown
    
    def get_cooldown_remaining(self) -> float:
        """Get remaining cooldown in seconds."""
        if self.cooldown == 0 or self.last_used == 0.0:
            return 0.0
        
        elapsed = time.time() - self.last_used
        remaining = self.cooldown - elapsed
        return max(0.0, remaining)
    
    def use(self) -> bool:
        """
        Mark skill as used.
        
        Returns:
            True if skill was ready and used, False otherwise
        """
        if not self.is_ready():
            return False
        
        self.last_used = time.time()
        return True
    
    def can_afford(self, current_mana: float) -> bool:
        """Check if we have enough mana."""
        return current_mana >= self.mana_cost
    
    def get_effectiveness_score(self, enemy_count: int, current_hp_ratio: float) -> float:
        """
        Calculate skill effectiveness for current situation.
        
        Args:
            enemy_count: Number of enemies
            current_hp_ratio: Current HP ratio (0.0-1.0)
            
        Returns:
            Effectiveness score (higher is better)
        """
        score = 0.0
        
        # Base damage
        score += self.damage * self.level
        
        # AOE bonus when many enemies
        if self.is_aoe and enemy_count > 1:
            score *= (1 + enemy_count * 0.2)
        
        # Defensive skills more valuable at low HP
        if self.skill_type == SkillType.DEFENSIVE:
            if current_hp_ratio < 0.3:
                score *= 2.0
        
        # Utility skills (teleport) valuable for escape
        if self.skill_type == SkillType.UTILITY:
            if current_hp_ratio < 0.3:
                score *= 1.5
        
        # Single target better against few enemies
        if not self.is_aoe and enemy_count == 1:
            score *= 1.3
        
        return score


@dataclass
class SkillBar:
    """Character skill bar with hotkeys."""
    
    left_click: Optional[Skill] = None   # Primary attack
    right_click: Optional[Skill] = None  # Secondary attack
    
    # Hotkeys (F1-F8 in D2)
    hotkeys: list[Optional[Skill]] = field(default_factory=lambda: [None] * 8)
    
    def get_skill_by_hotkey(self, key_index: int) -> Optional[Skill]:
        """Get skill assigned to hotkey (0-7 for F1-F8)."""
        if 0 <= key_index < len(self.hotkeys):
            return self.hotkeys[key_index]
        return None
    
    def get_all_skills(self) -> list[Skill]:
        """Get all assigned skills."""
        skills = []
        
        if self.left_click:
            skills.append(self.left_click)
        if self.right_click:
            skills.append(self.right_click)
        
        skills.extend([s for s in self.hotkeys if s is not None])
        
        return skills
    
    def get_ready_skills(self) -> list[Skill]:
        """Get all skills that are ready to use."""
        return [s for s in self.get_all_skills() if s.is_ready()]
    
    def get_affordable_skills(self, current_mana: float) -> list[Skill]:
        """
        Get all skills we can afford with current mana.
        
        Args:
            current_mana: Current mana as ratio (0.0-1.0) OR absolute value
            
        Returns:
            List of affordable skills
        """
        # If current_mana is a ratio (0-1), convert to percentage for comparison
        # We assume max mana is ~100 for simplicity
        if current_mana <= 1.0:
            current_mana = current_mana * 100.0
        
        return [
            s for s in self.get_ready_skills()
            if s.can_afford(current_mana)
        ]


# Predefined skill templates for common D2 skills
SORCERESS_SKILLS = {
    "frozen_orb": Skill(
        name="Frozen Orb",
        skill_type=SkillType.AOE,
        char_class=CharacterClass.SORCERESS,
        mana_cost=25,
        cooldown=1.0,
        damage=100,
        range_=20,
        is_aoe=True,
        requires_target=False,
    ),
    "blizzard": Skill(
        name="Blizzard",
        skill_type=SkillType.AOE,
        char_class=CharacterClass.SORCERESS,
        mana_cost=40,
        cooldown=2.0,
        damage=150,
        range_=15,
        is_aoe=True,
        requires_target=False,
    ),
    "teleport": Skill(
        name="Teleport",
        skill_type=SkillType.UTILITY,
        char_class=CharacterClass.SORCERESS,
        mana_cost=35,
        cooldown=1.0,
        damage=0,
        range_=30,
        is_aoe=False,
        requires_target=True,
    ),
    "glacial_spike": Skill(
        name="Glacial Spike",
        skill_type=SkillType.ATTACK,
        char_class=CharacterClass.SORCERESS,
        mana_cost=15,
        cooldown=0.5,
        damage=80,
        range_=15,
        is_aoe=False,
        requires_target=True,
    ),
}

NECROMANCER_SKILLS = {
    "bone_spear": Skill(
        name="Bone Spear",
        skill_type=SkillType.ATTACK,
        char_class=CharacterClass.NECROMANCER,
        mana_cost=10,
        cooldown=0.0,
        damage=70,
        range_=20,
        is_aoe=False,
        requires_target=True,
    ),
    "corpse_explosion": Skill(
        name="Corpse Explosion",
        skill_type=SkillType.AOE,
        char_class=CharacterClass.NECROMANCER,
        mana_cost=15,
        cooldown=0.5,
        damage=120,
        range_=10,
        is_aoe=True,
        requires_target=True,
    ),
}

PALADIN_SKILLS = {
    "holy_bolt": Skill(
        name="Holy Bolt",
        skill_type=SkillType.ATTACK,
        char_class=CharacterClass.PALADIN,
        mana_cost=8,
        cooldown=0.0,
        damage=60,
        range_=15,
        is_aoe=False,
        requires_target=True,
    ),
}
