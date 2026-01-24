"""Item classification and tiering system."""

import json
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, List, Optional, Any
from enum import Enum


class ItemTier(Enum):
    """Item tier classification."""
    
    S = "S"  # Best items
    A = "A"  # Very good
    B = "B"  # Good
    C = "C"  # Functional
    D = "D"  # Low value


@dataclass
class ClassificationRule:
    """Rule for item classification."""
    
    name: str                  # Exact item name
    tier: str                  # Target tier (S, A, B, C, D)
    item_type: Optional[str] = None  # Weapon, armor, charm, etc.
    quality: Optional[str] = None    # Unique, rare, magic, etc.
    min_stats: Optional[Dict[str, int]] = None  # Minimum stat requirements


class ItemClassifier:
    """
    Classifies items into tiers based on configurable rules.
    
    Default database path: data/items_database.json
    
    Database format:
    {
        "items": {
            "Harlequin Crest": {
                "tier": "S",
                "type": "helm",
                "quality": "unique"
            },
            "Mara's Kaleidoscope": {
                "tier": "S",
                "type": "amulet",
                "quality": "unique"
            }
        },
        "runewords": {
            "Enigma": {
                "tier": "S",
                "effect": "teleport"
            }
        },
        "default_tiers": {
            "unique": "A",
            "set": "B",
            "rare": "C",
            "magic": "D"
        }
    }
    """
    
    def __init__(self, database_path: Optional[str] = None):
        """
        Initialize classifier with database.
        
        Args:
            database_path: Path to items_database.json (default: data/items_database.json)
        """
        self.database_path = Path(database_path) if database_path else Path("data/items_database.json")
        self.database = self._load_database()
        self.rules = self._build_rules()
    
    def _load_database(self) -> Dict[str, Any]:
        """
        Load items database from JSON.
        
        Returns:
            Database dictionary
        """
        if not self.database_path.exists():
            # Return default database
            return {
                "items": self._get_default_items(),
                "runewords": self._get_default_runewords(),
                "default_tiers": {
                    "unique": "A",
                    "set": "B",
                    "rare": "C",
                    "magic": "D",
                    "normal": "D",
                }
            }
        
        try:
            with open(self.database_path, 'r') as f:
                return json.load(f)
        except Exception as e:
            print(f"Error loading database: {e}")
            return {"items": {}, "runewords": {}, "default_tiers": {}}
    
    def _get_default_items(self) -> Dict[str, Dict[str, Any]]:
        """Get default S/A tier unique items."""
        return {
            # S Tier - Best in slot
            "Harlequin Crest": {"tier": "S", "type": "helm", "quality": "unique"},
            "Mara's Kaleidoscope": {"tier": "S", "type": "amulet", "quality": "unique"},
            "Stone of Jordan": {"tier": "S", "type": "ring", "quality": "unique"},
            "The Oculus": {"tier": "S", "type": "orb", "quality": "unique"},
            "Tal Rasha's Wrappings": {"tier": "S", "type": "armor", "quality": "set"},
            "Annihilus": {"tier": "S", "type": "charm", "quality": "unique"},
            "Hellfire Torch": {"tier": "S", "type": "charm", "quality": "unique"},
            
            # A Tier - Very useful
            "Shako": {"tier": "A", "type": "helm", "quality": "unique"},
            "Homunculus": {"tier": "A", "type": "shield", "quality": "unique"},
            "Dracs Grasp": {"tier": "A", "type": "gloves", "quality": "unique"},
            "Unique Ring": {"tier": "A", "type": "ring", "quality": "unique"},
            "War Traveler": {"tier": "A", "type": "boots", "quality": "unique"},
            "String of Ears": {"tier": "A", "type": "amulet", "quality": "unique"},
        }
    
    def _get_default_runewords(self) -> Dict[str, Dict[str, str]]:
        """Get default runeword classifications."""
        return {
            # S Tier
            "Enigma": {"tier": "S", "type": "armor"},
            "Chains of Honor": {"tier": "S", "type": "armor"},
            "Spirit": {"tier": "S", "type": "weapon"},
            
            # A Tier
            "Infinity": {"tier": "A", "type": "weapon"},
            "Insight": {"tier": "A", "type": "weapon"},
            "Harmony": {"tier": "A", "type": "weapon"},
            "Holy Thunder": {"tier": "A", "type": "weapon"},
        }
    
    def _build_rules(self) -> List[ClassificationRule]:
        """Build classification rules from database."""
        rules = []
        
        # Add item rules
        for name, data in self.database.get("items", {}).items():
            rule = ClassificationRule(
                name=name,
                tier=data.get("tier", "D"),
                item_type=data.get("type"),
                quality=data.get("quality"),
                min_stats=data.get("min_stats"),
            )
            rules.append(rule)
        
        return rules
    
    def classify(
        self,
        item_name: str,
        item_type: Optional[str] = None,
        quality: Optional[str] = None,
        stats: Optional[Dict[str, int]] = None,
    ) -> ItemTier:
        """
        Classify an item into a tier.
        
        Args:
            item_name: Name of the item
            item_type: Type (helm, armor, weapon, etc.)
            quality: Quality (unique, rare, magic, etc.)
            stats: Dictionary of item stats
            
        Returns:
            ItemTier classification
        """
        # Check exact name match in database first
        for name, data in self.database.get("items", {}).items():
            if name.lower() == item_name.lower():
                # Check type and quality if specified
                if item_type and data.get("type", "").lower() != item_type.lower():
                    continue
                if quality and data.get("quality", "").lower() != quality.lower():
                    continue
                
                return ItemTier(data.get("tier", "D"))
        
        # Check runeword
        tier = self._classify_runeword(item_name)
        if tier:
            return tier
        
        # Default based on quality
        if quality:
            default_tiers = self.database.get("default_tiers", {})
            tier_str = default_tiers.get(quality.lower(), "D")
            return ItemTier(tier_str)
        
        return ItemTier.D
    
    def _classify_runeword(self, name: str) -> Optional[ItemTier]:
        """Check if item is a classified runeword."""
        for rw_name, data in self.database.get("runewords", {}).items():
            if rw_name.lower() == name.lower():
                return ItemTier(data.get("tier", "D"))
        return None
    
    def _meets_stat_requirements(
        self,
        item_stats: Dict[str, int],
        requirements: Dict[str, int],
    ) -> bool:
        """Check if item stats meet requirements."""
        for stat, min_value in requirements.items():
            if stat not in item_stats or item_stats[stat] < min_value:
                return False
        return True
    
    def get_tier_color(self, tier: ItemTier) -> tuple:
        """
        Get color for tier (for drawing).
        
        Args:
            tier: Item tier
            
        Returns:
            (B, G, R) color tuple for OpenCV
        """
        colors = {
            ItemTier.S: (0, 0, 255),      # Red (highest value)
            ItemTier.A: (0, 165, 255),    # Orange
            ItemTier.B: (0, 255, 255),    # Yellow
            ItemTier.C: (0, 255, 0),      # Green
            ItemTier.D: (128, 128, 128),  # Gray
        }
        return colors.get(tier, (128, 128, 128))
    
    def save_database(self) -> bool:
        """
        Save current database to JSON.
        
        Returns:
            True if successful
        """
        try:
            self.database_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.database_path, 'w') as f:
                json.dump(self.database, f, indent=2)
            return True
        except Exception as e:
            print(f"Error saving database: {e}")
            return False
    
    def add_item_rule(
        self,
        name: str,
        tier: str,
        item_type: Optional[str] = None,
        quality: Optional[str] = None,
    ) -> bool:
        """
        Add a new item classification rule.
        
        Args:
            name: Item name
            tier: Tier (S, A, B, C, D)
            item_type: Item type
            quality: Item quality
            
        Returns:
            True if successful
        """
        if tier not in ["S", "A", "B", "C", "D"]:
            return False
        
        item_data = {"tier": tier}
        if item_type:
            item_data["type"] = item_type
        if quality:
            item_data["quality"] = quality
        
        self.database["items"][name] = item_data
        self._build_rules()
        return True
    
    def get_statistics(self) -> Dict[str, int]:
        """Get statistics about database."""
        return {
            "total_items": len(self.database.get("items", {})),
            "total_runewords": len(self.database.get("runewords", {})),
            "tier_s_count": len([
                d for d in self.database.get("items", {}).values()
                if d.get("tier") == "S"
            ]),
            "tier_a_count": len([
                d for d in self.database.get("items", {}).values()
                if d.get("tier") == "A"
            ]),
        }
