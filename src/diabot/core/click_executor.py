"""
Click-based action executor for Diablo 2 bot.

Executes navigation and exploration actions using mouse clicks.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

from diabot.core.interfaces import ActionExecutor
from diabot.navigation.click_navigator import ClickNavigator, NavigationGoal, NavigationMode


class ClickActionExecutor(ActionExecutor):
    """
    Executes actions by clicking in the game window.
    
    Supported actions:
    - explore_random: Click randomly to explore
    - click_position: Click at specific coordinates
    - move_to_waypoint: Click on waypoint
    - move_to_exit: Click on exit to next zone
    """

    def __init__(self, click_navigator: Optional[ClickNavigator] = None, debug: bool = True):
        """
        Initialize click action executor.

        Args:
            click_navigator: ClickNavigator instance (creates if None)
            debug: Enable debug output
        """
        self.debug = debug
        
        if click_navigator is None:
            try:
                self.click_nav = ClickNavigator(debug=debug)
            except ImportError:
                print("[CLICK_EXECUTOR] ⚠️  pyautogui not available, click actions will fail")
                self.click_nav = None
        else:
            self.click_nav = click_navigator

    def execute_action(self, action_type: str, params: Dict[str, Any] = None) -> bool:
        """
        Execute an action by clicking.

        Args:
            action_type: Type of action to execute
            params: Action parameters

        Returns:
            bool: True if action succeeded
        """
        if not self.click_nav:
            print("[CLICK_EXECUTOR] ✗ No click navigator available")
            return False

        params = params or {}

        try:
            if action_type == "explore_random":
                return self.click_nav.explore_randomly()

            elif action_type == "click_position":
                x = params.get("x")
                y = params.get("y")
                desc = params.get("description", "")
                
                if x is None or y is None:
                    print("[CLICK_EXECUTOR] ✗ Missing x or y in click_position")
                    return False
                
                return self.click_nav.click_to_explore(int(x), int(y), desc)

            elif action_type == "move_to_waypoint":
                position = params.get("position")
                if not position:
                    print("[CLICK_EXECUTOR] ✗ Missing position in move_to_waypoint")
                    return False
                
                return self.click_nav.click_waypoint(tuple(position))

            elif action_type == "move_to_exit":
                position = params.get("position")
                if not position:
                    print("[CLICK_EXECUTOR] ✗ Missing position in move_to_exit")
                    return False
                
                return self.click_nav.click_exit(tuple(position))

            elif action_type == "set_navigation_goal":
                mode_str = params.get("mode", "exploring")
                target = params.get("target")
                zone = params.get("zone")
                
                try:
                    mode = NavigationMode(mode_str)
                except ValueError:
                    mode = NavigationMode.EXPLORING
                
                goal = NavigationGoal(
                    mode=mode,
                    target_position=target,
                    zone_name=zone,
                    notes=params.get("notes", "")
                )
                
                self.click_nav.set_goal(goal)
                return True

            else:
                if self.debug:
                    print(f"[CLICK_EXECUTOR] Unknown action: {action_type}")
                return False

        except Exception as e:
            print(f"[CLICK_EXECUTOR] ✗ Error executing {action_type}: {e}")
            import traceback
            traceback.print_exc()
            return False
