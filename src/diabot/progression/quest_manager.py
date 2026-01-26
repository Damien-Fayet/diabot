"""Quest and game progression management.

Handles loading, tracking, and updating quest progression from game_progression.json.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional


class QuestStatus(str, Enum):
    """Quest status enum."""
    NOT_STARTED = "not_started"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"


@dataclass
class Quest:
    """Represents a single quest."""
    id: str
    name: str
    giver: str
    objective: str
    reward: str
    status: QuestStatus = QuestStatus.NOT_STARTED
    act: int = 0


@dataclass
class Zone:
    """Represents a zone in the game."""
    id: str
    name: str
    zone_type: str  (town, dungeon, outdoor)
    npcs: List[str] = field(default_factory=list)
    enemies: List[str] = field(default_factory=list)
    objectives: List[str] = field(default_factory=list)


@dataclass
class Act:
    """Represents an act."""
    id: int
    name: str
    town: str
    zones: Dict[str, Zone] = field(default_factory=dict)
    quests: Dict[str, Quest] = field(default_factory=dict)


class QuestManager:
    """Manages quest progression and game state.
    
    Loads progression from data/game_progression.json, tracks current quests,
    and persists changes.
    """
    
    def __init__(self, progression_file: Path = Path("data/game_progression.json"), debug: bool = False):
        """Initialize quest manager.
        
        Args:
            progression_file: Path to game_progression.json
            debug: Enable debug logging
        """
        self.progression_file = progression_file
        self.debug = debug
        
        self.current_act: int = 1
        self.current_quest_id: Optional[str] = None
        self.current_zone_id: Optional[str] = None
        self.completed_quests: List[str] = []
        
        self.acts: Dict[int, Act] = {}
        self.all_quests: Dict[str, Quest] = {}
        
        self.load()
    
    def load(self) -> None:
        """Load progression from JSON file."""
        if not self.progression_file.exists():
            raise FileNotFoundError(f"Progression file not found: {self.progression_file}")
        
        with open(self.progression_file, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        # Load current progression
        prog = data.get("progression", {})
        self.current_act = prog.get("current_act", 1)
        self.current_quest_id = prog.get("current_quest")
        self.current_zone_id = prog.get("current_zone")
        self.completed_quests = prog.get("completed_quests", [])
        
        # Load acts and quests
        for act_data in data.get("acts", []):
            act = Act(
                id=act_data["id"],
                name=act_data["name"],
                town=act_data["town"]
            )
            
            # Load zones
            for zone_data in act_data.get("zones", []):
                zone = Zone(
                    id=zone_data["id"],
                    name=zone_data["name"],
                    zone_type=zone_data["type"],
                    npcs=zone_data.get("npcs", []),
                    enemies=zone_data.get("enemies", []),
                    objectives=zone_data.get("objectives", [])
                )
                act.zones[zone.id] = zone
            
            # Load quests
            for quest_data in act_data.get("quests", []):
                quest = Quest(
                    id=quest_data["id"],
                    name=quest_data["name"],
                    giver=quest_data["giver"],
                    objective=quest_data["objective"],
                    reward=quest_data["reward"],
                    status=QuestStatus(quest_data.get("status", "not_started")),
                    act=act.id
                )
                act.quests[quest.id] = quest
                self.all_quests[quest.id] = quest
            
            self.acts[act.id] = act
        
        if self.debug:
            print(f"✓ Loaded progression: Act {self.current_act}, Quest {self.current_quest_id}")
            print(f"  Completed: {len(self.completed_quests)} quests")
    
    def save(self) -> None:
        """Save current progression back to JSON file."""
        if not self.progression_file.exists():
            return
        
        with open(self.progression_file, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        # Update progression
        data["progression"] = {
            "current_act": self.current_act,
            "current_quest": self.current_quest_id,
            "current_zone": self.current_zone_id,
            "completed_quests": self.completed_quests
        }
        
        # Update quest statuses in acts
        for act_id, act in self.acts.items():
            act_data = next((a for a in data["acts"] if a["id"] == act_id), None)
            if act_data:
                for quest_id, quest in act.quests.items():
                    quest_data = next((q for q in act_data["quests"] if q["id"] == quest_id), None)
                    if quest_data:
                        quest_data["status"] = quest.status.value
        
        with open(self.progression_file, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        
        if self.debug:
            print(f"✓ Saved progression")
    
    def get_current_act(self) -> Optional[Act]:
        """Get current act."""
        return self.acts.get(self.current_act)
    
    def get_current_quest(self) -> Optional[Quest]:
        """Get current quest."""
        if not self.current_quest_id:
            return None
        return self.all_quests.get(self.current_quest_id)
    
    def get_next_quest(self) -> Optional[Quest]:
        """Get next quest to complete."""
        act = self.get_current_act()
        if not act:
            return None
        
        # Find first incomplete quest in current act
        for quest in act.quests.values():
            if quest.status != QuestStatus.COMPLETED:
                return quest
        
        # If all quests in act completed, move to next act
        if self.current_act < 5:
            self.current_act += 1
            return self.get_next_quest()
        
        return None
    
    def start_quest(self, quest_id: str) -> bool:
        """Start a quest.
        
        Args:
            quest_id: Quest ID to start
            
        Returns:
            True if quest started, False if not found
        """
        quest = self.all_quests.get(quest_id)
        if not quest:
            return False
        
        quest.status = QuestStatus.IN_PROGRESS
        self.current_quest_id = quest_id
        self.current_act = quest.act
        
        if self.debug:
            print(f"→ Started quest: {quest.name} (Act {quest.act})")
        
        self.save()
        return True
    
    def complete_quest(self, quest_id: str) -> bool:
        """Complete a quest.
        
        Args:
            quest_id: Quest ID to complete
            
        Returns:
            True if quest completed, False if not found
        """
        quest = self.all_quests.get(quest_id)
        if not quest:
            return False
        
        quest.status = QuestStatus.COMPLETED
        if quest_id not in self.completed_quests:
            self.completed_quests.append(quest_id)
        
        if self.debug:
            print(f"✓ Completed quest: {quest.name} ({quest.reward})")
        
        self.save()
        
        # Auto-start next quest
        next_quest = self.get_next_quest()
        if next_quest:
            self.start_quest(next_quest.id)
        
        return True
    
    def set_current_zone(self, zone_id: str) -> bool:
        """Set current zone.
        
        Args:
            zone_id: Zone ID
            
        Returns:
            True if zone found, False otherwise
        """
        act = self.get_current_act()
        if not act or zone_id not in act.zones:
            return False
        
        self.current_zone_id = zone_id
        if self.debug:
            zone = act.zones[zone_id]
            print(f"→ Entered zone: {zone.name}")
        
        self.save()
        return True
    
    def get_current_zone(self) -> Optional[Zone]:
        """Get current zone."""
        act = self.get_current_act()
        if not act or not self.current_zone_id:
            return None
        return act.zones.get(self.current_zone_id)
    
    def get_quest_chain(self) -> List[Quest]:
        """Get quests in recommended completion order."""
        # Get the "quest_completion_order" from JSON
        with open(self.progression_file, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        order = data.get("bot_strategy", {}).get("quest_completion_order", [])
        
        result = []
        for quest_id in order:
            quest = self.all_quests.get(quest_id)
            if quest:
                result.append(quest)
        
        return result
    
    def get_progress_summary(self) -> str:
        """Get human-readable progress summary."""
        total_quests = len(self.all_quests)
        completed = len(self.completed_quests)
        
        current_act = self.get_current_act()
        current_quest = self.get_current_quest()
        
        summary = f"Act {self.current_act}/5 | Quests {completed}/{total_quests}\n"
        if current_act:
            summary += f"  Location: {current_act.name}\n"
        if current_quest:
            summary += f"  Current: {current_quest.name} ({current_quest.objective})"
        
        return summary
