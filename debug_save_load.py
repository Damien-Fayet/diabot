"""Debug script to check save/load."""
import numpy as np
from pathlib import Path
from src.diabot.navigation.minimap_slam import MinimapSLAM

# Save a map
print("Creating and saving map...")
slam = MinimapSLAM(map_size=2048, debug=True)
minimap = np.random.randint(0, 255, (200, 200, 3), dtype=np.uint8)
slam.update(minimap)
slam.save_map("debug_map.npz")

# Check what was saved
print("\nChecking saved data...")
save_path = Path("data/slam_maps/debug_map.npz")
data = np.load(save_path, allow_pickle=True)
print(f"Keys in saved file: {list(data.keys())}")
metadata = data['metadata'].item()
print(f"Metadata: {metadata}")

# Try loading
print("\nLoading map...")
slam2 = MinimapSLAM(map_size=2048, debug=True)
slam2.load_map("debug_map.npz")
print(f"Loaded levels: {list(slam2.levels.keys())}")
print(f"Current level ID: {slam2.current_level_id}")
