"""Inventory and item management models."""

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Optional


class ItemType(Enum):
    """Types of items in inventory."""
    
    HEALTH_POTION = auto()
    MANA_POTION = auto()
    REJUV_POTION = auto()
    WEAPON = auto()
    ARMOR = auto()
    SCROLL = auto()
    GOLD = auto()
    QUEST_ITEM = auto()
    RUNE = auto()
    GEM = auto()


class PotionSize(Enum):
    """Potion sizes in Diablo 2."""
    
    MINOR = auto()      # Small red/blue
    LIGHT = auto()      # Medium
    REGULAR = auto()    # Normal
    GREATER = auto()    # Large
    SUPER = auto()      # Purple
    FULL_REJUV = auto() # Full restore


@dataclass
class Item:
    """Represents an item in inventory."""
    
    item_type: ItemType
    name: str
    quantity: int = 1
    slot: Optional[int] = None  # Position in inventory (0-39 for belt, etc.)
    
    # Potion specific
    potion_size: Optional[PotionSize] = None
    
    # Stats
    value: int = 0  # Gold value or effectiveness
    
    def is_usable(self) -> bool:
        """Check if item can be used."""
        return self.item_type in [
            ItemType.HEALTH_POTION,
            ItemType.MANA_POTION,
            ItemType.REJUV_POTION,
            ItemType.SCROLL,
        ]


@dataclass
class Inventory:
    """Player inventory management."""
    
    # Belt slots (quick access potions)
    belt_slots: list[Optional[Item]] = field(default_factory=lambda: [None] * 4)
    
    # Main inventory (simplified - not full grid)
    backpack: list[Item] = field(default_factory=list)
    
    # Equipped items
    weapon: Optional[Item] = None
    armor: Optional[Item] = None
    
    # Resources
    gold: int = 0
    
    def get_health_potions(self) -> list[Item]:
        """Get all health potions (belt + backpack)."""
        potions = []
        
        # Check belt
        for item in self.belt_slots:
            if item and item.item_type == ItemType.HEALTH_POTION:
                potions.append(item)
        
        # Check backpack
        for item in self.backpack:
            if item.item_type == ItemType.HEALTH_POTION:
                potions.append(item)
        
        return potions
    
    def get_mana_potions(self) -> list[Item]:
        """Get all mana potions."""
        potions = []
        
        for item in self.belt_slots:
            if item and item.item_type == ItemType.MANA_POTION:
                potions.append(item)
        
        for item in self.backpack:
            if item.item_type == ItemType.MANA_POTION:
                potions.append(item)
        
        return potions
    
    def get_best_health_potion(self) -> Optional[Item]:
        """Get best available health potion."""
        potions = self.get_health_potions()
        if not potions:
            return None
        
        # Prefer larger potions
        size_priority = {
            PotionSize.FULL_REJUV: 6,
            PotionSize.SUPER: 5,
            PotionSize.GREATER: 4,
            PotionSize.REGULAR: 3,
            PotionSize.LIGHT: 2,
            PotionSize.MINOR: 1,
        }
        
        return max(potions, key=lambda p: size_priority.get(p.potion_size, 0))
    
    def get_best_mana_potion(self) -> Optional[Item]:
        """Get best available mana potion."""
        potions = self.get_mana_potions()
        if not potions:
            return None
        
        size_priority = {
            PotionSize.SUPER: 5,
            PotionSize.GREATER: 4,
            PotionSize.REGULAR: 3,
            PotionSize.LIGHT: 2,
            PotionSize.MINOR: 1,
        }
        
        return max(potions, key=lambda p: size_priority.get(p.potion_size, 0))
    
    def use_item(self, item: Item) -> bool:
        """
        Use an item from inventory.
        
        Returns:
            True if item was used, False otherwise
        """
        if not item.is_usable():
            return False
        
        # Remove from belt if present
        for i, belt_item in enumerate(self.belt_slots):
            if belt_item is item:
                self.belt_slots[i] = None
                return True
        
        # Remove from backpack
        if item in self.backpack:
            self.backpack.remove(item)
            return True
        
        return False
    
    def has_health_potion(self) -> bool:
        """Check if any health potion is available."""
        return len(self.get_health_potions()) > 0
    
    def has_mana_potion(self) -> bool:
        """Check if any mana potion is available."""
        return len(self.get_mana_potions()) > 0
    
    def is_belt_full(self) -> bool:
        """Check if belt is full."""
        return all(slot is not None for slot in self.belt_slots)
    
    def add_to_belt(self, item: Item, slot: Optional[int] = None) -> bool:
        """
        Add item to belt.
        
        Args:
            item: Item to add
            slot: Specific slot (0-3) or None for first empty
            
        Returns:
            True if added successfully
        """
        if slot is not None:
            if 0 <= slot < len(self.belt_slots) and self.belt_slots[slot] is None:
                self.belt_slots[slot] = item
                item.slot = slot
                return True
        else:
            # Find first empty slot
            for i, belt_item in enumerate(self.belt_slots):
                if belt_item is None:
                    self.belt_slots[i] = item
                    item.slot = i
                    return True
        
        return False
    
    def get_inventory_summary(self) -> dict:
        """Get summary of inventory state."""
        return {
            "health_potions": len(self.get_health_potions()),
            "mana_potions": len(self.get_mana_potions()),
            "belt_slots_used": sum(1 for s in self.belt_slots if s is not None),
            "backpack_items": len(self.backpack),
            "gold": self.gold,
        }
