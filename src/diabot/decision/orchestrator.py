"""High-level orchestrator tying perception, navigation, and goal selection."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from diabot.core.interfaces import ActionExecutor
from diabot.decision import GoalSelector
from diabot.models import BotState
from diabot.navigation.planner import Navigator
from diabot.vision import MinimapParser, MinimapParseResult
from diabot.vision.screen_detector import GameScreen
from diabot.vision.screen_state_manager import ScreenStateManager


@dataclass
class OrchestratorResult:
    parse: MinimapParseResult
    path: list  # list of (x, y) grid points (may be empty)
    goal_kind: Optional[str]
    dispatched_action: Optional[str]
    action_success: Optional[bool]
    screen_type: Optional[str] = None
    screen_action: Optional[str] = None
    can_navigate: bool = True  # False if on non-gameplay screen


class Orchestrator:
    """Single-step orchestrator for vision-only navigation.

    - Detects current game screen (gameplay, menu, dead, etc.)
    - Handles screen-specific actions (respawn, menu navigation)
    - When in gameplay:
      - Parses minimap → grid + player pos
      - Selects goal (quest/farm/frontier)
      - Plans path with A*
    """

    def __init__(
        self,
        bot_state: BotState,
        grid_size: int = 96,
        executor: Optional[ActionExecutor] = None,
        dispatch_full_path: bool = True,
    ):
        self.bot_state = bot_state
        self.minimap_parser = MinimapParser(grid_size=grid_size)
        self.goal_selector = GoalSelector()
        self.executor = executor
        self.dispatch_full_path = dispatch_full_path
        
        # Add screen state manager
        self.screen_manager = ScreenStateManager(bot_state, executor)

    def step(self, frame: np.ndarray) -> OrchestratorResult:
        # 1) Detect current game screen
        screen_detection = self.screen_manager.update(frame)
        current_screen = self.screen_manager.current_screen
        
        path = []
        goal_kind = None
        dispatched_action = None
        action_success: Optional[bool] = None
        screen_action = None
        can_navigate = True
        
        # 2) Handle screen-specific actions (if not in gameplay)
        if current_screen != GameScreen.GAMEPLAY:
            can_navigate = False
            screen_action = self.screen_manager.handle_screen_action(frame)
            
            # Parse minimap anyway for later reference, but don't navigate
            parse = self.minimap_parser.parse(frame)
        else:
            # 3) If in gameplay, do normal navigation
            parse = self.minimap_parser.parse(frame)
            
            # 4) Navigation planner on current grid
            navigator = Navigator(parse.grid)
            
            # 5) Choose goal
            landmarks = [(lm.kind, lm.position) for lm in parse.landmarks]
            goal = self.goal_selector.choose_goal(self.bot_state, navigator, landmarks)
            
            if goal:
                goal_kind = goal.kind
                plan = navigator.plan_to(parse.player_pos, goal)
                if plan.success:
                    path = plan.path
            
            # 6) Dispatch path action if we have a path
            if self.executor and path:
                dispatched_action = "follow_path" if self.dispatch_full_path else "move"
                payload = {"path": path, "goal_kind": goal_kind}
                action_success = self.executor.execute_action(dispatched_action, payload)
        
        return OrchestratorResult(
            parse=parse,
            path=path,
            goal_kind=goal_kind,
            dispatched_action=dispatched_action,
            action_success=action_success,
            screen_type=current_screen.value,
            screen_action=screen_action,
            can_navigate=can_navigate,
        )
