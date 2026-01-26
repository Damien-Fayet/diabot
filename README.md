# Unified Navigation Architecture

The navigation system is composed of modular components for clean separation of concerns and extensibility:

- **click_navigator.py**: Executes navigation actions by sending mouse clicks to the game window (Windows-only).
- **grid.py**: Provides grid-based pathfinding utilities (A* planner, occupancy grid).
- **planner.py**: Selects navigation goals (frontier, landmark, target) and plans local paths using the grid.
- **navigator.py**: Handles global navigation, zone-to-zone pathfinding, and route planning using world map and POI data.
- **world_map.py**: Manages zone metadata, Points of Interest (POIs), connections, and persistent storage.
- **minimap_detector.py**: Detects POIs (waypoints, player, monsters, exits) on the minimap using color analysis.
- **map_geometry.py**: Extracts minimap geometry to build a local occupancy grid and locate the player for pathfinding.

### Recommended Usage
- High-level navigation logic (main loop/orchestrator) should use `navigator.py` and `planner.py` for route and goal selection.
- Perception modules (`minimap_detector.py`, `map_geometry.py`) provide real-time map and POI data.
- All movement actions should be executed via `click_navigator.py` for reliability and platform isolation.

Legacy modules (e.g., `map_system.py`) have been removed to ensure a single, coherent navigation pipeline.

---

# diabot


## Rerun yolo training
If you relabel or add images, rerun prepare_yolo_dataset.py --clean to refresh labels and data.yaml.

Train this first model (tiny, for a sanity check):
```
python scripts/train_yolo.py --data data/ml/data.yaml --model yolo11n.pt --epochs 50 --imgsz 640 --batch 8
```

Then test on a screenshot:
```
python scripts/run_yolo_inference.py --model runs/train/diablo-yolo/weights/best.pt --image data/screenshots/inputs/game.jpg --output data/screenshots/outputs/yolo_inference.jpg --conf 0.35
```