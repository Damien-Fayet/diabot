#!/usr/bin/env python3
"""
Visualize what the bot 'sees' - brain overlay showing perception data.
"""

import cv2
import numpy as np
from pathlib import Path
from src.diabot.vision.ui_vision import UIVisionModule
from src.diabot.vision.screen_regions import ALL_REGIONS

def draw_brain_overlay(image_path: str, output_path: str = None):
    """Draw what the bot perceives on the image."""
    img = cv2.imread(image_path)
    if img is None:
        print(f"❌ Cannot load: {image_path}")
        return
    
    height, width = img.shape[:2]
    print(f"Image size: {width}x{height}\n")
    
    # Create vision module
    vision = UIVisionModule(debug=True)
    
    # Run perception
    print("Running vision analysis...")
    ui_state = vision.analyze(img)
    
    # Convert UIState to dict for display
    perception = {
        'hp_ratio': ui_state.hp_ratio,
        'mana_ratio': ui_state.mana_ratio,
        'zone_name': ui_state.zone_name,
        'is_dead': ui_state.is_dead,
        'potions_available': ui_state.potions_available,
        'buffs': ui_state.buffs,
        'debuffs': ui_state.debuffs,
        'enemy_count': 0,  # Placeholder
        'nearest_enemy_distance': None,
        'items_nearby': []
    }
    
    print("\n" + "=" * 80)
    print("PERCEPTION DATA")
    print("=" * 80)
    for key, value in perception.items():
        print(f"  {key}: {value}")
    
    # Create overlay
    overlay = img.copy()
    vis = img.copy()
    
    # Colors
    COLOR_HP = (0, 0, 255)      # Red
    COLOR_MANA = (255, 0, 0)    # Blue
    COLOR_MINIMAP = (0, 255, 0) # Green
    COLOR_UI = (128, 128, 128)  # Gray
    
    # Draw all regions with transparency
    for region_name, region_def in ALL_REGIONS.items():
        x, y, w, h = region_def.get_bounds(height, width)
        
        # Choose color
        if 'lifebar' in region_name:
            color = COLOR_HP
        elif 'manabar' in region_name:
            color = COLOR_MANA
        elif 'minimap' in region_name:
            color = COLOR_MINIMAP
        else:
            color = COLOR_UI
        
        # Semi-transparent fill
        cv2.rectangle(overlay, (x, y), (x + w, y + h), color, -1)
        
        # Solid border
        cv2.rectangle(vis, (x, y), (x + w, y + h), color, 2)
        
        # Label
        label = region_name.replace('_ui', '').replace('_', ' ').title()
        cv2.putText(vis, label, (x + 5, y + 15),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)
    
    # Blend overlay
    alpha = 0.2
    vis = cv2.addWeighted(overlay, alpha, vis, 1 - alpha, 0)
    
    # Draw perception info panel
    panel_height = 200
    panel = np.zeros((panel_height, width, 3), dtype=np.uint8)
    
    # Title
    cv2.putText(panel, "BOT PERCEPTION", (10, 30),
               cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2, cv2.LINE_AA)
    
    # HP bar
    hp_ratio = perception.get('hp_ratio', 0.0)
    hp_bar_width = int(300 * hp_ratio)
    cv2.rectangle(panel, (10, 50), (310, 70), (64, 64, 64), -1)
    cv2.rectangle(panel, (10, 50), (10 + hp_bar_width, 70), COLOR_HP, -1)
    cv2.putText(panel, f"HP: {hp_ratio:.1%}", (320, 65),
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
    
    # Mana bar
    mana_ratio = perception.get('mana_ratio', 0.0)
    mana_bar_width = int(300 * mana_ratio)
    cv2.rectangle(panel, (10, 80), (310, 100), (64, 64, 64), -1)
    cv2.rectangle(panel, (10, 80), (10 + mana_bar_width, 100), COLOR_MANA, -1)
    cv2.putText(panel, f"Mana: {mana_ratio:.1%}", (320, 95),
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
    
    # Other stats
    y_pos = 120
    stats = [
        f"Zone: {perception.get('zone_name', 'Unknown')}",
        f"Enemy Count: {perception.get('enemy_count', 0)}",
        f"Nearest Enemy: {perception.get('nearest_enemy_distance', 'N/A')}",
        f"Items Nearby: {len(perception.get('items_nearby', []))}",
    ]
    
    for stat in stats:
        cv2.putText(panel, stat, (10, y_pos),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1, cv2.LINE_AA)
        y_pos += 25
    
    # Combine image and panel
    result = np.vstack([vis, panel])
    
    # Save
    if output_path is None:
        output_path = "data/screenshots/outputs/brain_overlay.png"
    
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(output_path, result)
    
    print("\n" + "=" * 80)
    print(f"✓ Saved brain overlay to: {output_path}")
    print("=" * 80)
    
    return result


if __name__ == "__main__":
    import sys
    
    image_path = "data/screenshots/inputs/game_a1.png"
    if len(sys.argv) > 1:
        image_path = sys.argv[1]
    
    print(f"\nGenerating brain overlay for: {image_path}\n")
    draw_brain_overlay(image_path)
