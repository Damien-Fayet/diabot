#!/usr/bin/env python3
"""
Test vision modules on game.jpg screenshot.
Shows what the modules detect.
"""

import cv2
import numpy as np
from pathlib import Path
from src.diabot.vision import UIVisionModule, EnvironmentVisionModule
from src.diabot.vision.screen_regions import UI_REGIONS, ENVIRONMENT_REGIONS

def draw_ui_state(frame, ui_state, ui_region):
    """Draw UI state information on frame."""
    h, w = frame.shape[:2]
    x, y, region_w, region_h = ui_region.get_bounds(h, w)
    
    # Draw results below the region
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.7
    color = (0, 255, 0)
    
    y_text = y + region_h + 30
    
    cv2.putText(frame, f"HP: {ui_state.hp_ratio:.1%}", (x, y_text),
                font, font_scale, color, 2)
    cv2.putText(frame, f"Mana: {ui_state.mana_ratio:.1%}", (x, y_text + 30),
                font, font_scale, color, 2)
    
    if ui_state.buffs:
        cv2.putText(frame, f"Buffs: {ui_state.buffs}", (x, y_text + 60),
                    font, font_scale, color, 2)
    
    if ui_state.debuffs:
        cv2.putText(frame, f"Debuffs: {ui_state.debuffs}", (x, y_text + 90),
                    font, font_scale, (0, 0, 255), 2)

def draw_environment_state(frame, env_state, env_region):
    """Draw environment detections on frame."""
    h, w = frame.shape[:2]
    x_offset, y_offset, _, _ = env_region.get_bounds(h, w)
    
    # Draw enemies
    for enemy in env_state.enemies:
        ex, ey, ew, eh = enemy.bbox
        # Convert from region-local to frame coords
        ex += x_offset
        ey += y_offset
        
        # Color based on enemy type
        if enemy.enemy_type == "red":
            color = (0, 0, 255)
        elif enemy.enemy_type == "orange":
            color = (0, 165, 255)
        else:
            color = (255, 0, 0)
        
        # Draw bbox
        cv2.rectangle(frame, (ex, ey), (ex + ew, ey + eh), color, 2)
        
        # Draw label
        label = f"{enemy.enemy_type}"
        cv2.putText(frame, label, (ex, ey - 5),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    
    # Draw items (optional bbox info)
    for item in env_state.items:
        # Some detectors return simple strings; skip if no bbox
        if not isinstance(item, dict) or 'bbox' not in item:
            continue

        ix, iy, iw, ih = item['bbox']
        # Convert from region-local to frame coords
        ix += x_offset
        iy += y_offset
        
        color = (0, 255, 255)  # Cyan for items
        cv2.rectangle(frame, (ix, iy), (ix + iw, iy + ih), color, 2)
        cv2.putText(frame, "ITEM", (ix, iy - 5),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

def main():
    print("=" * 60)
    print("🔍 VISION TEST ON game.jpg")
    print("=" * 60)
    print()
    
    # Load image
    image_path = Path("data/screenshots/inputs/game.jpg")
    frame = cv2.imread(str(image_path))
    
    if frame is None:
        print(f"❌ Failed to load {image_path}")
        return
    
    h, w = frame.shape[:2]
    print(f"📸 Image: {image_path}")
    print(f"   Size: {w}x{h}px")
    print()
    
    # Create modules
    print("🎯 Initializing vision modules...")
    ui_module = UIVisionModule()
    env_module = EnvironmentVisionModule()
    print("   ✓ UIVisionModule ready")
    print("   ✓ EnvironmentVisionModule ready")
    print()
    
    # Analyze UI
    print("🔎 Analyzing UI...")
    ui_state = ui_module.analyze(frame)
    print(f"   Health: {ui_state.hp_ratio:.1%}")
    print(f"   Mana: {ui_state.mana_ratio:.1%}")
    print(f"   Potions available: {ui_state.potions_available}")
    print()
    
    # Analyze environment
    print("🔎 Analyzing environment...")
    env_state = env_module.analyze(frame)
    print(f"   Enemies detected: {len(env_state.enemies)}")
    for i, enemy in enumerate(env_state.enemies, 1):
        print(f"     {i}. {enemy.enemy_type} at {enemy.position}")
    print(f"   Items detected: {len(env_state.items)}")
    for i, item in enumerate(env_state.items, 1):
        print(f"     {i}. {item}")
    print(f"   Obstacles detected: {len(env_state.obstacles)}")
    print(f"   Player position: {env_state.player_position}")
    print()
    
    # Draw results on frame
    print("🎨 Drawing results on image...")
    frame_result = frame.copy()
    
    # Draw regions using actual ratios
    top_left_ui = UI_REGIONS['top_left_ui']
    ui_x, ui_y, ui_w, ui_h = top_left_ui.get_bounds(h, w)
    cv2.rectangle(frame_result, (ui_x, ui_y), (ui_x + ui_w, ui_y + ui_h), (0, 255, 255), 3)
    cv2.putText(frame_result, "UI Region", (ui_x + 10, ui_y + ui_h - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

    playfield_region = ENVIRONMENT_REGIONS['playfield']
    pf_x, pf_y, pf_w, pf_h = playfield_region.get_bounds(h, w)
    cv2.rectangle(frame_result, (pf_x, pf_y), (pf_x + pf_w, pf_y + pf_h), (0, 255, 0), 2)
    cv2.putText(frame_result, "Playfield Region", (pf_x + 10, pf_y + 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    
    # Draw UI state
    draw_ui_state(frame_result, ui_state, UI_REGIONS['top_left_ui'])
    
    # Draw environment state
    draw_environment_state(frame_result, env_state, ENVIRONMENT_REGIONS['playfield'])
    
    # Save
    output_path = Path("data/screenshots/outputs/game_vision_analysis.jpg")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), frame_result)
    print(f"✅ Saved analysis to: {output_path}")
    print()
    
    # Statistics
    print("=" * 60)
    print("📊 SUMMARY")
    print("=" * 60)
    print(f"UI Detections:        HP={ui_state.hp_ratio:.0%} Mana={ui_state.mana_ratio:.0%}")
    print(f"Environment:          {len(env_state.enemies)} enemies, {len(env_state.items)} items")
    print(f"Output:               {output_path}")
    print()

if __name__ == "__main__":
    main()
