# Diabot - Vision-based Diablo 2 Game Agent

## 🎮 Project Overview

Diabot is a research project exploring perception and decision-making in video games using only image-based inputs. It's designed as a **clean, modular architecture** for eventual ML/RL integration.

**Current Status**: Phase 1 (Foundation) - Basic scaffolding complete ✅

## 🏗️ Architecture

The project follows **clean architecture** principles with clear separation of concerns:

```
ImageSource (get frame)
    ↓
VisionModule (perceive) → Perception
    ↓
StateBuilder (build) → GameState
    ↓
DecisionEngine (decide) → Action
    ↓
ActionExecutor (execute)
    ↓
DebugOverlay (visualize)
```

### Core Components

- **ImageSource**: Abstract interface for acquiring frames
  - `ScreenshotFileSource`: Load images from disk (macOS developer mode)
  - `WindowsScreenCapture`: Placeholder for runtime mode (Windows only)

- **VisionModule**: Extract game information from images
  - `RuleBasedVisionModule`: Dummy implementation for now

- **GameState**: Abstract representation of game conditions
  - Health, mana, enemies, location, etc.

- **DecisionEngine**: Make decisions based on state
  - `RuleBasedDecisionEngine`: Simple rules (POC)

- **ActionExecutor**: Execute decisions
  - `DummyActionExecutor`: Placeholder (no actual game interaction yet)

- **DebugOverlay**: Visualize bot perception and state

## 🚀 Quick Start

### 1. Setup Environment
```bash
cd /Users/damien/PersoLocal/diabot
source .venv/bin/activate
```

### 2. Run Developer Mode
```bash
# With default test image
python scripts/run_dev.py

# With custom screenshot
python scripts/run_dev.py /path/to/screenshot.png
```

### 3. Run Tests
```bash
python tests/test_models.py

# Or with pytest
pytest tests/ -v
```

## 📁 Project Structure
```
diabot/
├── src/diabot/                 # Main package
│   ├── core/
│   │   ├── interfaces.py      # Abstract interfaces
│   │   └── implementations.py # Concrete implementations
│   ├── models/
│   │   └── state.py           # GameState, Action dataclasses
│   ├── debug/
│   │   └── overlay.py         # Visualization utilities
│   └── main.py               # (future) Main bot loop
├── scripts/
│   └── run_dev.py            # Developer mode entry point
├── tests/
│   └── test_models.py        # Unit tests
├── data/
│   └── screenshots/          # Test images
├── DEVELOPMENT_PLAN.md       # Phase 1-6 roadmap
├── pyproject.toml
└── requirements.txt
```

## 🔧 Technologies

- **Python 3.14** (uses 3.11+ compatible code)
- **OpenCV** (4.13+): Image processing
- **NumPy** (2.4+): Numerical operations
- **Pytest** (9.0+): Testing framework
- **Dataclasses**: Type-safe data structures

## 🎯 Next Steps

See [DEVELOPMENT_PLAN.md](DEVELOPMENT_PLAN.md) for detailed roadmap.

Currently at: **Étape 2 - Interfaces abstraites** ✅  
Next: **Étape 3 - État et Perception**

## 🤖 Design Principles

- ✅ **Clean Architecture**: Clear separation of concerns
- ✅ **Dependency Inversion**: Interfaces first
- ✅ **Testability**: Units are independently testable
- ✅ **Platform Agnostic**: macOS development, Windows runtime
- ✅ **Extensibility**: Easy to add ML/RL later
- ✅ **Readability**: Code over performance (for now)

## 📝 Notes

This is an **experimental research project**. Focus is on:
- Learning game state through images only
- Building robust decision frameworks
- Creating a foundation for future ML integration

**NOT** focused on:
- Actual game cheating or exploitation
- Performance optimization
- Real-time gameplay

## 👤 Author

Damien @ Michelin
