#!/usr/bin/env python3
"""Test screen detection and orchestrator with screen awareness."""

import sys
from pathlib import Path

# Setup path
sys.path.insert(0, str(Path(__file__).parent / "src"))

import cv2
import numpy as np

from diabot.core.implementations import DummyActionExecutor
from diabot.decision.orchestrator import Orchestrator
from diabot.models.bot_state import BotState
from diabot.vision.screen_detector import ScreenDetector, GameScreen


def test_screen_detector():
    """Test screen detection on dummy frames."""
    print("=" * 60)
    print("TEST: Screen Detector")
    print("=" * 60)
    
    detector = ScreenDetector()
    
    # Create dummy frames for different screens
    test_cases = [
        ("Gameplay", _create_gameplay_frame()),
        ("Dead Screen", _create_dead_frame()),
        ("Main Menu", _create_menu_frame()),
        ("Loading", _create_loading_frame()),
        ("Char Select", _create_char_select_frame()),
    ]
    
    for name, frame in test_cases:
        result = detector.detect(frame)
        print(f"\n{name}:")
        print(f"  Screen Type: {result.screen_type.value}")
        print(f"  Confidence: {result.confidence:.2f}")
        print(f"  Details: {result.details}")


def test_orchestrator_with_screens():
    """Test orchestrator handling different screens."""
    print("\n" + "=" * 60)
    print("TEST: Orchestrator with Screen Awareness")
    print("=" * 60)
    
    bot_state = BotState()
    executor = DummyActionExecutor()
    
    orchestrator = Orchestrator(
        bot_state=bot_state,
        executor=executor,
        dispatch_full_path=True,
    )
    
    test_scenarios = [
        ("Gameplay", _create_gameplay_frame()),
        ("Dead Screen", _create_dead_frame()),
        ("Main Menu", _create_menu_frame()),
        ("Loading", _create_loading_frame()),
    ]
    
    for scenario_name, frame in test_scenarios:
        print(f"\nScenario: {scenario_name}")
        print("-" * 40)
        
        result = orchestrator.step(frame)
        
        print(f"  Screen Type: {result.screen_type}")
        print(f"  Can Navigate: {result.can_navigate}")
        print(f"  Goal: {result.goal_kind}")
        print(f"  Path Length: {len(result.path)}")
        print(f"  Dispatched Action: {result.dispatched_action}")
        print(f"  Action Success: {result.action_success}")
        print(f"  Screen Action: {result.screen_action}")


def _create_gameplay_frame() -> np.ndarray:
    """Create a dummy gameplay frame."""
    frame = np.ones((768, 1024, 3), dtype=np.uint8) * 50  # Dark background
    
    # Add minimap area (top-right)
    minimap = frame[0:192, 768:1024]
    minimap[:] = [60, 60, 60]
    # Add some "map" details
    cv2.circle(frame, (896, 96), 20, [100, 100, 100], -1)
    
    # Add health bar area (left-bottom)
    health_area = frame[600:750, 180:300]
    health_area[:] = [0, 0, 200]  # Red-ish
    
    # Add mana bar area (right-bottom)
    mana_area = frame[600:750, 720:840]
    mana_area[:] = [200, 0, 0]  # Blue-ish
    
    return frame


def _create_dead_frame() -> np.ndarray:
    """Create a dummy death screen."""
    frame = np.zeros((768, 1024, 3), dtype=np.uint8)
    
    # Add red blood splatter effect
    cv2.circle(frame, (512, 384), 150, [0, 0, 200], -1)  # Red
    cv2.circle(frame, (300, 200), 100, [0, 0, 180], -1)
    cv2.circle(frame, (700, 500), 80, [0, 0, 150], -1)
    
    # Add some gray (skull area)
    cv2.ellipse(frame, (512, 300), (80, 100), 0, 0, 360, [150, 150, 150], -1)
    
    return frame


def _create_menu_frame() -> np.ndarray:
    """Create a dummy main menu frame."""
    frame = np.ones((768, 1024, 3), dtype=np.uint8) * 30  # Very dark
    
    # Add golden buttons (centered)
    button_positions = [
        (512, 300),
        (512, 400),
        (512, 500),
    ]
    
    for x, y in button_positions:
        # Gold/yellow color (BGR)
        cv2.rectangle(frame, (x-80, y-20), (x+80, y+20), (0, 215, 255), -1)
        cv2.rectangle(frame, (x-85, y-25), (x+85, y+25), (100, 150, 200), 2)
    
    return frame


def _create_loading_frame() -> np.ndarray:
    """Create a dummy loading screen."""
    frame = np.zeros((768, 1024, 3), dtype=np.uint8)
    
    # Almost entirely black (loading background)
    frame[:] = [10, 10, 10]
    
    # Add progress bar in lower half
    bar_y = 650
    cv2.rectangle(frame, (300, bar_y), (700, bar_y+30), [100, 100, 100], -1)
    cv2.rectangle(frame, (300, bar_y), (550, bar_y+30), [0, 215, 255], -1)  # Progress
    
    return frame


def _create_char_select_frame() -> np.ndarray:
    """Create a dummy character select screen."""
    frame = np.ones((768, 1024, 3), dtype=np.uint8) * 50
    
    # Add portrait boxes in grid
    positions = [
        (200, 200), (450, 200), (700, 200),
        (200, 500), (450, 500), (700, 500),
    ]
    
    for x, y in positions:
        # Portrait border
        cv2.rectangle(frame, (x-60, y-80), (x+60, y+80), [100, 100, 100], 2)
        # Skin tone inside
        cv2.rectangle(frame, (x-50, y-70), (x+50, y+70), [150, 120, 100], -1)
    
    return frame


if __name__ == "__main__":
    test_screen_detector()
    test_orchestrator_with_screens()
    
    print("\n" + "=" * 60)
    print("All tests completed!")
    print("=" * 60)
