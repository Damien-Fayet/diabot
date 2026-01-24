---
applyTo: '**'
---
You are a senior Python engineer designing a clean, modular AI agent.

Goal:
Create a first working skeleton of a Diablo 2 bot that:
- perceives the game ONLY through images (no memory access)
- supports two modes:
  1) runtime mode (Windows only): live screen capture
  2) developer mode (cross-platform): processing a static screenshot file
- is designed to be extended later with computer vision and RL

Constraints:
- The code must run on macOS in developer mode without any Windows-specific dependency
- Windows-specific code must be isolated behind interfaces
- Use clean architecture and dependency inversion
- Favor readability and debuggability over performance
- No actual cheating or game-specific logic yet, only scaffolding

Architecture to implement:
- ImageSource interface with get_frame() -> np.ndarray
- Two implementations:
  - ScreenshotFileSource (loads an image from disk)
  - WindowsScreenCapture (placeholder, not fully implemented)
- Vision module that:
  - receives an image
  - returns a dummy perception dict (hp_ratio, enemy_count, etc.)
- StateBuilder that converts perception into an abstract state object
- RuleBasedDecisionEngine that decides a high-level action
- ActionExecutor interface with a WindowsInputExecutor placeholder
- Debug overlay utility to draw state info on the frame

Deliverables:
- Python package structure
- Base classes and interfaces
- Minimal runnable main.py in developer mode
- Clear docstrings explaining responsibilities
- No deep learning yet, no actual Diablo-specific detection

Use:
- Python 3.11+
- OpenCV for image handling
- Dataclasses where relevant

Assume this is a research / experimental project.
Focus on extensibility and clean separation of concerns.

## Add a visual debug system called "BrainOverlay".

Purpose:
- Help developers understand what the agent perceives and decides
- Must work in developer mode using a static screenshot
- Must NOT depend on Windows-specific libraries

Requirements:
- Implement a BrainOverlay class using OpenCV
- It receives:
  - the original frame (np.ndarray)
  - perception data
  - abstract state
  - chosen action
- It draws:
  - text overlay (top-left):
      - current FSM state
      - chosen action
      - hp_ratio / mana_ratio
      - enemy_count_near
  - optional bounding boxes for detected entities
  - optional colored indicators:
      - red = danger
      - green = safe
      - blue = target

Design constraints:
- Overlay must be purely visual (no game interaction)
- Drawing must be toggleable via configuration
- Code must be isolated in a debug/overlay.py module
- Overlay logic must NOT be coupled to vision logic

Deliverables:
- BrainOverlay class
- draw(frame, perception, state, action) method
- Example usage in main.py in developer mode
- Clear docstrings explaining each visual element


## Implement a Diablo-inspired Finite State Machine (FSM) for the decision engine.

Design philosophy:
- States must reflect human gameplay intuition
- Prioritize survival over optimization
- FSM must be easy to extend with new states later

Required FSM states:
- IDLE:
    - No enemies nearby
    - Waiting or scanning environment
- EXPLORE:
    - Moving toward unexplored area
    - No immediate danger
- ENGAGE:
    - Enemies detected within engagement range
    - Actively attacking
- KITE:
    - Enemies too close or dangerous
    - Moving while attacking or repositioning
- PANIC:
    - Low HP or surrounded
    - Emergency behavior (potion, escape)
- RECOVER:
    - Regaining resources after danger
    - Safe zone behavior

State transitions:
- Must be driven by abstract state only (not raw pixels)
- Use clear, readable conditions (hp_ratio thresholds, enemy_count, danger flag)
- Transitions must be logged for debugging

Implementation requirements:
- Create a State enum
- Create a DiabloFSM class
- FSM must expose:
    - update(state_data) -> new_state
    - decide_action(state_data) -> action string
- FSM logic must be deterministic and testable
- No reinforcement learning at this stage

Deliverables:
- FSM implementation in decision/diablo_fsm.py
- Simple transition table or readable if/else logic
- Example integration in main.py
- Inline comments explaining gameplay reasoning

The BrainOverlay must visually display the current FSM state and highlight
the main decision driver (e.g. "Low HP", "Enemy Nearby", "Safe Exploration").
