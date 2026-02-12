"""
Calculate the true screen-to-grid conversion accounting for minimap resize.

The minimap is resized from ~1459x869 to 64x64 grid.
This means the actual "world scale" is different from what we calculated.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent / "src"))

import cv2
import numpy as np


def calculate_true_scale():
    """Calculate the true conversion ratio."""
    
    print("\n" + "="*70)
    print("🎯 TRUE SCALE CALCULATION")
    print("="*70 + "\n")
    
    # Load minimap to get actual size
    minimap_path = Path("data/screenshots/outputs/diagnostic/minimap_steps/step1_extracted.png")
    if minimap_path.exists():
        minimap = cv2.imread(str(minimap_path))
        minimap_h, minimap_w = minimap.shape[:2]
        print(f"📐 Original minimap: {minimap_w}x{minimap_h} pixels")
    else:
        minimap_w, minimap_h = 1459, 869  # From previous test
        print(f"📐 Estimated minimap: {minimap_w}x{minimap_h} pixels")
    
    # Grid size after resize
    grid_size = 64
    print(f"📐 Resized grid: {grid_size}x{grid_size} cells\n")
    
    # Minimap scale
    minimap_scale_w = minimap_w / grid_size
    minimap_scale_h = minimap_h / grid_size
    
    print(f"🔍 Minimap resize ratio:")
    print(f"   Width: {minimap_scale_w:.1f}x ({minimap_w} → {grid_size})")
    print(f"   Height: {minimap_scale_h:.1f}x ({minimap_h} → {grid_size})")
    print(f"   Average: {(minimap_scale_w + minimap_scale_h)/2:.1f}x\n")
    
    # Key insight: The minimap shows a certain portion of the game world
    # That portion is represented by 64x64 cells
    # The game screen shows approximately the same area (or slightly less)
    
    # Load game frame
    game_path = Path("data/screenshots/inputs/game_screenshot.png")
    if game_path.exists():
        game = cv2.imread(str(game_path))
        game_h, game_w = game.shape[:2]
        print(f"📐 Game screen: {game_w}x{game_h} pixels")
    else:
        game_w, game_h = 1922, 1114  # From previous test
        print(f"📐 Estimated game screen: {game_w}x{game_h} pixels")
    
    print()
    
    # The game viewport and minimap show approximately the same world area
    # If minimap shows 64 cells, game screen also shows ~64 cells
    # Therefore: game_pixels / 64 = pixels_per_cell on screen
    
    viewport_cells = 64  # Same as minimap grid
    
    screen_px_per_cell_w = game_w / viewport_cells
    screen_px_per_cell_h = game_h / viewport_cells
    
    print(f"✅ TRUE SCREEN-TO-GRID CONVERSION:")
    print(f"   Viewport shows: ~{viewport_cells}x{viewport_cells} cells")
    print(f"   Screen width: {game_w}px / {viewport_cells} cells = {screen_px_per_cell_w:.1f} px/cell")
    print(f"   Screen height: {game_h}px / {viewport_cells} cells = {screen_px_per_cell_h:.1f} px/cell")
    print(f"   Average: {(screen_px_per_cell_w + screen_px_per_cell_h)/2:.1f} px/cell\n")
    
    # Test examples
    print("="*70)
    print("📊 TEST CONVERSIONS:")
    print("="*70)
    
    avg_ratio = (screen_px_per_cell_w + screen_px_per_cell_h) / 2
    
    test_cases = [
        ("Quest 200px left, 100px up", -200, -100),
        ("Waypoint 500px right, 100px down", 500, 100),
        ("NPC 100px left, 200px down", -100, 200),
    ]
    
    for desc, dx, dy in test_cases:
        cells_x = int(dx / avg_ratio)
        cells_y = int(dy / avg_ratio)
        print(f"\n{desc}:")
        print(f"  Screen: ({dx:+4d}, {dy:+4d}) px")
        print(f"  Grid:   ({cells_x:+3d}, {cells_y:+3d}) cells")
        print(f"  World:  ({1024+cells_x}, {1024+cells_y})")
    
    print("\n" + "="*70)
    print("💡 RECOMMENDED VALUE:")
    print("="*70)
    print(f"SCREEN_PIXELS_PER_CELL = {avg_ratio:.1f}")
    print("="*70 + "\n")
    
    return avg_ratio


if __name__ == "__main__":
    calculate_true_scale()
