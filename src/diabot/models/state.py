"""Game state models."""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Tuple, Optional


@dataclass
class EnemyInfo:
    """Information about a detected enemy."""
    
    type: str  # e.g., "zombie", "skeleton", "boss"
    position: Tuple[int, int]  # (x, y) in image coordinates
    distance: Optional[float] = None  # Distance from player
    health_ratio: Optional[float] = None  # Enemy HP if visible


@dataclass
class ItemInfo:
    """Information about a detected item."""
    
    type: str  # e.g., "potion", "weapon", "gold"
    position: Tuple[int, int]
    value: Optional[int] = None  # Estimated value


@dataclass
class GameState:
    """
    Abstract representation of game state.
    
    This is the main state object used by decision engines.
    """
    
    # Character state (0.0-1.0 ratios)
    hp_ratio: float
    mana_ratio: float
    
    # Environment
    enemies: List[EnemyInfo] = field(default_factory=list)
    items: List[ItemInfo] = field(default_factory=list)
    
    # Current threat level
    threat_level: str = "none"  # none, low, medium, high, critical
    
    # Location context
    current_location: str = "unknown"
    
    # Metadata
    frame_number: int = 0
    debug_info: Dict[str, Any] = field(default_factory=dict)
    
    # Derived state (computed in __post_init__)
    is_threatened: bool = field(init=False)
    is_full_health: bool = field(init=False)
    needs_potion: bool = field(init=False)
    
    # Legacy compatibility (percent values)
    health_percent: float = field(init=False)
    mana_percent: float = field(init=False)
    enemy_count: int = field(init=False)
    visible_items: int = field(init=False)
    
    def __post_init__(self):
        """Compute derived state."""
        # Convert ratios to percentages for legacy code
        self.health_percent = self.hp_ratio * 100.0
        self.mana_percent = self.mana_ratio * 100.0
        
        # Counts
        self.enemy_count = len(self.enemies)
        self.visible_items = len(self.items)
        
        # Threat assessment
        self.is_threatened = len(self.enemies) > 0
        self.is_full_health = self.hp_ratio >= 0.95
        self.needs_potion = self.hp_ratio < 0.3


@dataclass
class Action:
    """Represents an action to execute."""
    
    action_type: str  # e.g., 'move', 'attack', 'use_skill', 'drink_potion', 'idle'
    target: Optional[str] = None  # e.g., 'north', 'enemy_0', None for self
    params: Dict[str, Any] = field(default_factory=dict)
    
    # Optional skill/item info
    skill_name: Optional[str] = None
    item_name: Optional[str] = None

