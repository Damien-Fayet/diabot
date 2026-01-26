"""
Calibrate OCR regions interactively.
Displays game.png with HP/Mana/Zone regions highlighted.
You can adjust the coordinates and test OCR in real-time.
"""

import cv2
import numpy as np
from pathlib import Path
from src.diabot.vision.screen_regions import UI_REGIONS

# Load game image
game_path = Path("data/screenshots/inputs/game.png")
if not game_path.exists():
    print(f"❌ {game_path} not found")
    exit(1)

frame = cv2.imread(str(game_path))
h, w = frame.shape[:2]

print(f"📸 Image: {w}x{h}px")
print("\n" + "="*80)
print("🎯 RÉGIONS OCR ACTUELLES")
print("="*80)

# Display current regions
regions_to_check = ['lifebar_ui', 'manabar_ui', 'zone_ui']

for region_name in regions_to_check:
    region = UI_REGIONS[region_name]
    x, y, rw, rh = region.get_bounds(h, w)
    
    print(f"\n{region_name}:")
    print(f"  Ratios : x={region.x_ratio:.2f} y={region.y_ratio:.2f} w={region.w_ratio:.2f} h={region.h_ratio:.2f}")
    print(f"  Pixels : x={x} y={y} w={rw} h={rh}")
    print(f"  Bounds : ({x}, {y}) → ({x+rw}, {y+rh})")
    
    # Extract and show region
    region_img = frame[y:y+rh, x:x+rw]
    
    # Save individual region
    output_path = f"data/screenshots/outputs/diagnostic/calibrate_{region_name}.png"
    cv2.imwrite(output_path, region_img)
    print(f"  Saved  : {output_path}")

# Draw all regions on frame
frame_annotated = frame.copy()

for region_name in regions_to_check:
    region = UI_REGIONS[region_name]
    x, y, rw, rh = region.get_bounds(h, w)
    
    # Different colors for each region
    if 'lifebar' in region_name:
        color = (0, 0, 255)  # Red for HP
    elif 'manabar' in region_name:
        color = (255, 0, 0)  # Blue for Mana
    else:
        color = (0, 255, 255)  # Yellow for Zone
    
    cv2.rectangle(frame_annotated, (x, y), (x+rw, y+rh), color, 2)
    cv2.putText(frame_annotated, region_name, (x, y-5), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

# Save annotated image
output_full = "data/screenshots/outputs/diagnostic/calibrate_full_annotated.png"
cv2.imwrite(output_full, frame_annotated)
print(f"\n✓ Annotated image: {output_full}")

print("\n" + "="*80)
print("📋 INSTRUCTIONS")
print("="*80)
print("""
1. Ouvrez 'calibrate_full_annotated.png' pour voir où sont les régions
2. Ouvrez 'calibrate_lifebar_ui.png' et 'calibrate_manabar_ui.png'
3. Si les régions ne contiennent pas le texte HP/Mana, ajustez les ratios dans:
   src/diabot/vision/screen_regions.py
   
4. Pour ajuster:
   - x_ratio/y_ratio : déplace la position (0.0-1.0)
   - w_ratio/h_ratio : change la taille (0.0-1.0)
   
5. Relancez ce script pour vérifier les nouvelles coordonnées
""")
