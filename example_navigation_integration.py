"""
Integration example: Using FrontierNavigator with existing bot.

This shows how to integrate the vision-based navigation system
with the existing Diablo 2 bot architecture.
"""

from pathlib import Path
import cv2
import numpy as np

# Existing bot imports
from src.diabot.navigation import FrontierNavigator, NavigationAction, NavigationOverlay
from src.diabot.decision.diablo_fsm import State as FSMState
from src.diabot.models.abstract_state import AbstractState


class NavigationIntegration:
    """
    Integration layer between FrontierNavigator and bot FSM.
    
    Translates navigation actions to bot commands and manages
    navigation state within the bot's state machine.
    """
    
    def __init__(self, debug: bool = False):
        """
        Initialize navigation integration.
        
        Args:
            debug: Enable debug visualization
        """
        self.navigator = FrontierNavigator(
            minimap_grid_size=64,
            local_map_size=200,
            movement_speed=2.0,
            debug=debug
        )
        
        self.overlay = NavigationOverlay(
            show_local_map=True,
            show_path=True,
            show_frontiers=True,
            show_minimap_grid=False
        ) if debug else None
        
        self.debug = debug
        self.enabled = True
        self.last_action = None
    
    def should_navigate(self, current_fsm_state: FSMState) -> bool:
        """
        Determine if navigation should be active based on FSM state.
        
        Args:
            current_fsm_state: Current bot FSM state
            
        Returns:
            True if navigation should be active
        """
        # Only navigate in EXPLORE state
        # Don't navigate during combat, recovery, etc.
        navigation_states = [FSMState.IDLE, FSMState.EXPLORE]
        return current_fsm_state in navigation_states and self.enabled
    
    def update(self, frame: np.ndarray, abstract_state: AbstractState) -> dict:
        """
        Update navigation and get action recommendation.
        
        Args:
            frame: Game frame
            abstract_state: Current abstract state from bot
            
        Returns:
            Dictionary with navigation recommendations:
            {
                'action': 'move_forward' | 'turn_left' | 'turn_right' | 'stop',
                'duration': float,  # seconds
                'angle': float,     # degrees (for rotation)
                'target': (x, y),   # target position (optional)
                'confidence': float # 0-1 confidence in action
            }
        """
        if not self.enabled:
            return {'action': 'stop', 'confidence': 0.0}
        
        try:
            # Get navigation state
            nav_state = self.navigator.update(frame)
            
            # Translate navigation action to bot command
            recommendation = self._translate_action(nav_state)
            
            # Add visualization if debug mode
            if self.debug and self.overlay:
                vis_frame = self.overlay.draw(
                    frame,
                    nav_state,
                    local_map=self.navigator.get_local_map()
                )
                # Display or save visualization
                cv2.imshow("Navigation Debug", vis_frame)
                cv2.waitKey(1)
            
            self.last_action = recommendation
            return recommendation
            
        except Exception as e:
            if self.debug:
                print(f"[NavigationIntegration] Error: {e}")
            return {'action': 'stop', 'confidence': 0.0}
    
    def report_action_executed(self, action: str, duration: float = 0.0, angle: float = 0.0):
        """
        Report that a navigation action was executed.
        
        This updates the pose estimator with actual movement.
        
        Args:
            action: Action that was executed
            duration: Duration of movement (seconds)
            angle: Angle of rotation (degrees)
        """
        if action == 'move_forward':
            self.navigator.report_movement("forward", duration)
        elif action == 'move_backward':
            self.navigator.report_movement("backward", duration)
        elif action == 'strafe_left':
            self.navigator.report_movement("left", duration)
        elif action == 'strafe_right':
            self.navigator.report_movement("right", duration)
        elif action in ['turn_left', 'turn_right']:
            self.navigator.report_rotation(angle)
    
    def reset(self):
        """Reset navigation state (e.g., when entering new zone)."""
        self.navigator.reset()
        if self.debug:
            print("[NavigationIntegration] Navigation reset")
    
    def enable(self):
        """Enable navigation."""
        self.enabled = True
    
    def disable(self):
        """Disable navigation."""
        self.enabled = False
    
    def get_exploration_progress(self) -> float:
        """Get current exploration progress (0.0 to 1.0)."""
        # Could track this from last nav_state
        return 0.0
    
    def _translate_action(self, nav_state) -> dict:
        """
        Translate NavigationAction to bot command.
        
        Args:
            nav_state: NavigationState from FrontierNavigator
            
        Returns:
            Bot command dictionary
        """
        action = nav_state.action
        
        if action == NavigationAction.STOP:
            return {
                'action': 'stop',
                'confidence': 1.0,
                'reason': 'No navigation targets'
            }
        
        elif action == NavigationAction.MOVE_FORWARD:
            return {
                'action': 'move_forward',
                'duration': 0.5,  # Move for 0.5 seconds
                'confidence': 0.8
            }
        
        elif action == NavigationAction.TURN_LEFT:
            return {
                'action': 'turn_left',
                'angle': -30,  # Turn 30 degrees left
                'duration': 0.3,
                'confidence': 0.9
            }
        
        elif action == NavigationAction.TURN_RIGHT:
            return {
                'action': 'turn_right',
                'angle': 30,  # Turn 30 degrees right
                'duration': 0.3,
                'confidence': 0.9
            }
        
        else:
            return {
                'action': 'stop',
                'confidence': 0.5
            }


# Example usage in bot's main loop
def example_integration_in_bot():
    """
    Example of how to integrate navigation into bot's main loop.
    """
    from src.diabot.decision.diablo_fsm import DiabloFSM
    from src.diabot.builders.state_builder import StateBuilder
    
    # Initialize components
    navigation = NavigationIntegration(debug=True)
    fsm = DiabloFSM()
    state_builder = StateBuilder()
    
    print("Bot running with vision-based navigation...")
    
    while True:
        # Get frame (from screen capture or file)
        frame = capture_screen_or_load_image()
        
        # Build abstract state
        perception = process_perception(frame)
        abstract_state = state_builder.build(perception)
        
        # Update FSM
        current_fsm_state = fsm.get_current_state()
        
        # Check if navigation should be active
        if navigation.should_navigate(current_fsm_state):
            # Get navigation recommendation
            nav_command = navigation.update(frame, abstract_state)
            
            if nav_command['action'] != 'stop':
                # Execute navigation command
                execute_command(nav_command)
                
                # Report back to navigation
                navigation.report_action_executed(
                    action=nav_command['action'],
                    duration=nav_command.get('duration', 0.0),
                    angle=nav_command.get('angle', 0.0)
                )
        
        else:
            # FSM is in combat, recovery, etc.
            # Use regular decision engine
            action = fsm.decide_action(abstract_state)
            execute_command(action)
        
        # Check for zone transition
        if detect_zone_change():
            navigation.reset()


# Example: Adding navigation to FSM EXPLORE state
class ExploreStateWithNavigation:
    """
    Enhanced EXPLORE state using vision-based navigation.
    """
    
    def __init__(self):
        self.navigation = NavigationIntegration(debug=True)
    
    def enter(self):
        """Called when entering EXPLORE state."""
        print("[EXPLORE] Entering exploration mode")
        self.navigation.enable()
    
    def update(self, frame: np.ndarray, abstract_state: AbstractState) -> str:
        """
        Update exploration state.
        
        Returns:
            Next action to execute
        """
        # Get navigation recommendation
        nav_command = self.navigation.update(frame, abstract_state)
        
        # Check for threats while exploring
        if abstract_state.danger_flag:
            # Transition to combat
            return "TRANSITION_TO_ENGAGE"
        
        # Check for low resources
        if abstract_state.hp_ratio < 0.3:
            return "TRANSITION_TO_RECOVER"
        
        # Execute navigation
        if nav_command['action'] != 'stop':
            return nav_command['action']
        else:
            # No more frontiers, exploration complete
            return "TRANSITION_TO_IDLE"
    
    def exit(self):
        """Called when exiting EXPLORE state."""
        print("[EXPLORE] Exiting exploration mode")
        self.navigation.disable()


def capture_screen_or_load_image():
    """Placeholder for screen capture."""
    # In real bot, this would capture screen
    # For testing, load a static image
    return cv2.imread("data/screenshots/inputs/game_frame.png")


def process_perception(frame):
    """Placeholder for perception processing."""
    return {}


def execute_command(command):
    """Placeholder for command execution."""
    print(f"Executing: {command}")


def detect_zone_change():
    """Placeholder for zone change detection."""
    return False


if __name__ == "__main__":
    print("=" * 60)
    print("Navigation Integration Example")
    print("=" * 60)
    print("\nThis script shows how to integrate FrontierNavigator")
    print("with the existing Diablo 2 bot architecture.")
    print("\nKey integration points:")
    print("  1. NavigationIntegration wraps FrontierNavigator")
    print("  2. Check FSM state to enable/disable navigation")
    print("  3. Translate NavigationAction to bot commands")
    print("  4. Report executed actions back to navigator")
    print("  5. Reset on zone transitions")
    print("\nSee code for detailed examples.")
    print("=" * 60)
