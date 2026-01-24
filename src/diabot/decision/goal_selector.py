"""Goal selection logic based on bot state, quests, and navigation."""
from __future__ import annotations

from typing import List, Optional, Tuple

from diabot.models.bot_state import BotMode, BotState, QuestStatus
from diabot.navigation.planner import GoalManager, NavigationGoal, Navigator


class GoalSelector:
    """Selects the next navigation/interaction goal based on current state."""

    def __init__(self):
        self.goal_manager = GoalManager()

    def choose_goal(self,
                    bot_state: BotState,
                    navigator: Navigator,
                    landmarks: List[Tuple[str, Tuple[int, int]]]) -> Optional[NavigationGoal]:
        # Safety override
        if bot_state.hp_ratio < 0.2:
            return NavigationGoal(kind="safety", position=navigator.occ.frontier_cells()[0]
                                  if navigator.occ.frontier_cells() else bot_state.quests.get("town", (0, 0)),
                                  priority=0)

        # Questing mode: pick first available/in-progress quest target if landmarks provided
        if bot_state.mode == BotMode.QUESTING:
            quest_target = self._pick_quest_target(bot_state)
            if quest_target:
                return NavigationGoal(kind="quest", position=quest_target, priority=0)

        # Farming mode: prefer landmarks passed in (e.g., boss room/waypoint)
        if bot_state.mode == BotMode.FARMING and landmarks:
            kind, pos = landmarks[0]
            return NavigationGoal(kind=kind, position=pos, priority=1)

        # Otherwise, exploration/frontier
        return self.goal_manager.choose_goal(start=(navigator.occ.w // 2, navigator.occ.h // 2),
                                             landmarks=landmarks,
                                             navigator=navigator)

    @staticmethod
    def _pick_quest_target(bot_state: BotState) -> Optional[Tuple[int, int]]:
        # Placeholder: future mapping from quest to coordinates
        for q in bot_state.quests.values():
            if q.status in (QuestStatus.AVAILABLE, QuestStatus.IN_PROGRESS):
                # Return None until we have a map from quest id to target coords
                return None
        return None
