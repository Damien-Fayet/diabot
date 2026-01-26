#!/usr/bin/env python3
"""
Diagnostic complet de la détection pour game.png
Affiche tous les détails sur ce qui est utilisé et détecté.
"""

import sys
import argparse
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent / "src"))

import cv2
import numpy as np
from diabot.vision import UIVisionModule, EnvironmentVisionModule
from diabot.vision.screen_regions import UI_REGIONS, ENVIRONMENT_REGIONS
from diabot.debug.overlay import BrainOverlay
from diabot.builders.state_builder import EnhancedStateBuilder
from diabot.decision.diablo_fsm import DiabloFSM

def main():
    parser = argparse.ArgumentParser(description="Diagnostic complet vision + BrainOverlay")
    parser.add_argument("--show-regions", action="store_true", help="Dessine les rectangles de régions dans l'overlay")
    args = parser.parse_args()

    print("=" * 80)
    print("🔍 DIAGNOSTIC COMPLET - game.png")
    print("=" * 80)
    print()
    
    # Load image
    image_path = Path("data/screenshots/inputs/game.png")
    frame = cv2.imread(str(image_path))
    
    if frame is None:
        print(f"❌ Image non trouvée: {image_path}")
        return
    
    h, w = frame.shape[:2]
    print(f"📸 Image: {w}x{h}px")
    print()
    
    # Check Tesseract
    print("=" * 80)
    print("🔧 TESSERACT OCR")
    print("=" * 80)
    try:
        import pytesseract
        print("✓ Module pytesseract importé")
        try:
            version = pytesseract.get_tesseract_version()
            print(f"✓ Tesseract OCR v{version} installé")
        except Exception as e:
            print(f"❌ Tesseract OCR non installé ou non accessible")
            print(f"   Erreur: {e}")
            print(f"   → Installer depuis: https://github.com/UB-Mannheim/tesseract/wiki")
    except ImportError:
        print("❌ Module pytesseract non installé")
    print()
    
    # Show regions
    print("=" * 80)
    print("📐 RÉGIONS DÉFINIES")
    print("=" * 80)
    print()
    print("Régions UI:")
    for name, region in UI_REGIONS.items():
        x, y, rw, rh = region.get_bounds(h, w)
        print(f"  {name:20s}: x={x:4d} y={y:4d} w={rw:4d} h={rh:4d}")
    print()
    print("Régions Environment:")
    for name, region in ENVIRONMENT_REGIONS.items():
        x, y, rw, rh = region.get_bounds(h, w)
        print(f"  {name:20s}: x={x:4d} y={y:4d} w={rw:4d} h={rh:4d}")
    print()
    
    # Extract and save regions
    print("=" * 80)
    print("🎨 EXTRACTION DES RÉGIONS")
    print("=" * 80)
    output_dir = Path("data/screenshots/outputs/diagnostic")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save HP region
    hp_region = UI_REGIONS['lifebar_ui'].extract_from_frame(frame)
    if hp_region.size > 0:
        hp_path = output_dir / "region_lifebar.png"
        cv2.imwrite(str(hp_path), hp_region)
        print(f"✓ Lifebar région: {hp_region.shape} → {hp_path}")
    else:
        print("❌ Lifebar région vide")
    
    # Save Mana region
    mana_region = UI_REGIONS['manabar_ui'].extract_from_frame(frame)
    if mana_region.size > 0:
        mana_path = output_dir / "region_manabar.png"
        cv2.imwrite(str(mana_path), mana_region)
        print(f"✓ Manabar région: {mana_region.shape} → {mana_path}")
    else:
        print("❌ Manabar région vide")
    
    # Save playfield region
    playfield_region = ENVIRONMENT_REGIONS['playfield'].extract_from_frame(frame)
    if playfield_region.size > 0:
        playfield_path = output_dir / "region_playfield.png"
        cv2.imwrite(str(playfield_path), playfield_region)
        print(f"✓ Playfield région: {playfield_region.shape} → {playfield_path}")
    else:
        print("❌ Playfield région vide")
    print()
    
    # Analyze UI
    print("=" * 80)
    print("🔎 ANALYSE UI")
    print("=" * 80)
    ui_module = UIVisionModule(debug=True)
    ui_state = ui_module.analyze(frame)
    print(f"HP ratio: {ui_state.hp_ratio:.1%}")
    print(f"Mana ratio: {ui_state.mana_ratio:.1%}")
    print(f"Zone: '{ui_state.zone_name}'")
    print()
    
    # Analyze Environment
    print("=" * 80)
    print("🔎 ANALYSE ENVIRONMENT")
    print("=" * 80)
    env_module = EnvironmentVisionModule(debug=True)
    env_state = env_module.analyze(frame)
    print(f"Ennemis détectés: {len(env_state.enemies)}")
    print()
    for i, enemy in enumerate(env_state.enemies, 1):  # TOUS les ennemis
        print(f"  {i:3d}. {enemy.enemy_type:15s} à {str(enemy.position):20s} confidence={enemy.confidence:.2f}  bbox={enemy.bbox}")
    print()
    print(f"Items détectés: {len(env_state.items)}")
    print(f"Position joueur: {env_state.player_position}")
    print()

    # Build Perception and GameState for BrainOverlay
    from diabot.core.interfaces import Perception
    from diabot.models.state import GameState, EnemyInfo as ModelEnemy, ItemInfo as ModelItem, Action

    playfield_region = ENVIRONMENT_REGIONS['playfield']
    pf_x, pf_y, pf_w, pf_h = playfield_region.get_bounds(h, w)

    perception = Perception(
        hp_ratio=ui_state.hp_ratio,
        mana_ratio=ui_state.mana_ratio,
        enemy_count=len(env_state.enemies),
        enemy_types=[e.enemy_type for e in env_state.enemies],
        visible_items=[str(item) for item in env_state.items],
        player_position=env_state.player_position,
        raw_data={
            "env_state": env_state,
            "playfield_bounds": (pf_x, pf_y, pf_w, pf_h),
        },
    )

    # Map enemies/items to GameState model types
    mapped_enemies = [
        ModelEnemy(
            type=e.enemy_type,
            position=(e.bbox[0] + pf_x + e.bbox[2] // 2, e.bbox[1] + pf_y + e.bbox[3] // 2),
        )
        for e in env_state.enemies
    ]

    mapped_items = [
        ModelItem(type=str(item), position=(0, 0))
        for item in env_state.items
    ]

    game_state = GameState(
        hp_ratio=ui_state.hp_ratio,
        mana_ratio=ui_state.mana_ratio,
        enemies=mapped_enemies,
        items=mapped_items,
        current_location=ui_state.zone_name or "unknown",
        frame_number=0,
    )

    # Build a simple action placeholder
    action = Action(action_type="diagnostic", target=None)
    fsm_state_name = "DIAGNOSTIC"

    # Create visualization with BrainOverlay (includes entities)
    print("=" * 80)
    print("🎨 VISUALISATION AVEC BRAIN OVERLAY")
    print("=" * 80)
    overlay = BrainOverlay(enabled=True)
    vis = overlay.draw(
        frame=frame,
        perception=perception,
        state=game_state,
        action=action,
        fsm_state=fsm_state_name,
    )

    vis_path = output_dir / "full_diagnostic_with_brain.jpg"
    cv2.imwrite(str(vis_path), vis)
    print(f"✓ Visualisation BrainOverlay: {vis_path}")
    print(f"  FSM State: {fsm_state_name}")
    print(f"  Action: {action.action_type}")
    print()

    # Optionally draw region rectangles on top of overlay
    if args.show_regions:
        vis_regions = vis.copy()
        font = cv2.FONT_HERSHEY_SIMPLEX

        for name, region in UI_REGIONS.items():
            x, y, rw, rh = region.get_bounds(h, w)
            cv2.rectangle(vis_regions, (x, y), (x + rw, y + rh), (0, 255, 255), 1)
            cv2.putText(vis_regions, name, (x + 5, max(15, y + 15)), font, 0.4, (0, 255, 255), 1)

        for name, region in ENVIRONMENT_REGIONS.items():
            x, y, rw, rh = region.get_bounds(h, w)
            cv2.rectangle(vis_regions, (x, y), (x + rw, y + rh), (0, 255, 0), 1)
            cv2.putText(vis_regions, name, (x + 5, max(20, y + 20)), font, 0.5, (0, 255, 0), 2)

        vis_regions_path = output_dir / "full_diagnostic_with_brain_regions.jpg"
        cv2.imwrite(str(vis_regions_path), vis_regions)
        print(f"✓ Visualisation BrainOverlay + régions: {vis_regions_path}")
        print()
    
    print("=" * 80)
    print("✅ DIAGNOSTIC TERMINÉ")
    print("=" * 80)
    print(f"Tous les fichiers dans: {output_dir}")

if __name__ == "__main__":
    main()
