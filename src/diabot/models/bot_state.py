"""Bot state and quest/farming tracking models."""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional
import json


class BotMode(str, Enum):
    SAFETY = "safety"
    QUESTING = "questing"
    FARMING = "farming"
    EXPLORATION = "exploration"
    IDLE = "idle"


class QuestStatus(str, Enum):
    LOCKED = "locked"
    AVAILABLE = "available"
    IN_PROGRESS = "in_progress"
    DONE = "done"
    SKIPPED = "skipped"


@dataclass
class QuestState:
    quest_id: str
    name: str
    act: int
    status: QuestStatus = QuestStatus.AVAILABLE
    current_step: Optional[str] = None
    notes: str = ""


@dataclass
class RunLogEntry:
    run_name: str
    duration_sec: float
    loot_count: int = 0
    xp_gain: float = 0.0
    deaths: int = 0


@dataclass
class BotState:
    act: int = 1
    zone: str = ""
    mode: BotMode = BotMode.EXPLORATION
    hp_ratio: float = 1.0
    mana_ratio: float = 1.0
    gold: int = 0
    inventory_full: bool = False
    waypoint_unlocked: Dict[str, bool] = field(default_factory=dict)
    quests: Dict[str, QuestState] = field(default_factory=dict)
    run_logs: List[RunLogEntry] = field(default_factory=list)

    def to_json(self) -> str:
        def encode(obj):
            if isinstance(obj, Enum):
                return obj.value
            if hasattr(obj, "__dict__"):
                return obj.__dict__
            raise TypeError(f"Type not serializable: {type(obj)}")

        return json.dumps(self, default=encode, indent=2)

    @staticmethod
    def from_json(data: str) -> "BotState":
        raw = json.loads(data)
        quests = {qid: QuestState(**q) for qid, q in raw.get("quests", {}).items()}
        run_logs = [RunLogEntry(**r) for r in raw.get("run_logs", [])]
        return BotState(
            act=raw.get("act", 1),
            zone=raw.get("zone", ""),
            mode=BotMode(raw.get("mode", BotMode.EXPLORATION)),
            hp_ratio=raw.get("hp_ratio", 1.0),
            mana_ratio=raw.get("mana_ratio", 1.0),
            gold=raw.get("gold", 0),
            inventory_full=raw.get("inventory_full", False),
            waypoint_unlocked=raw.get("waypoint_unlocked", {}),
            quests=quests,
            run_logs=run_logs,
        )
