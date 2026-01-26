"""
Test live screen capture from Diablo 2 window.

This script:
1. Finds the Diablo 2 window
2. Captures a frame
3. Runs vision analysis (OCR + enemy detection)
4. Displays BrainOverlay
5. Saves the result

Usage:
    python test_live_capture.py
    python test_live_capture.py --window-title "Diablo II: Resurrected"
"""

import argparse
import cv2
from pathlib import Path

from src.diabot.core.implementations import WindowsScreenCapture
from src.diabot.vision.ui_vision import UIVisionModule
from src.diabot.vision.environment_vision import EnvironmentVisionModule
from src.diabot.vision.screen_regions import UI_REGIONS
from src.diabot.debug.overlay import BrainOverlay
from src.diabot.models.state import GameState, Action
from src.diabot.core.interfaces import Perception


def draw_regions(frame, show_regions=False):
    """Draw UI regions on frame if requested."""
    if not show_regions:
        return frame
    
    output = frame.copy()
    h, w = frame.shape[:2]
    
    region_colors = {
        'lifebar_ui': (0, 0, 255),      # Red for HP
        'manabar_ui': (255, 0, 0),      # Blue for Mana
        'zone_ui': (0, 255, 255),       # Yellow for Zone
        'playfield': (0, 255, 0),       # Green for playfield
    }
    
    for region_name, color in region_colors.items():
        if region_name not in UI_REGIONS:
            continue
        
        region = UI_REGIONS[region_name]
        x, y, rw, rh = region.get_bounds(h, w)
        
        # Draw rectangle
        cv2.rectangle(output, (x, y), (x + rw, y + rh), color, 2)
        
        # Draw label
        cv2.putText(
            output,
            region_name,
            (x, y - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            color,
            2
        )
    
    return output


def main():
    parser = argparse.ArgumentParser(description='Test live Diablo 2 screen capture')
    parser.add_argument(
        '--window-title',
        type=str,
        default='Diablo II: Resurrected',
        help='Window title to capture (default: "Diablo II: Resurrected")'
    )
    parser.add_argument(
        '--continuous',
        action='store_true',
        help='Continuous capture mode (press Q to quit)'
    )
    parser.add_argument(
        '--show-regions',
        action='store_true',
        help='Draw UI regions on output image'
    )
    args = parser.parse_args()
    
    print("="*80)
    print("🎮 DIABLO 2 LIVE CAPTURE TEST")
    print("="*80)
    
    # Initialize capture
    try:
        capture = WindowsScreenCapture(window_title=args.window_title)
    except Exception as e:
        print(f"❌ Failed to initialize capture: {e}")
        print("\nMake sure:")
        print("1. Diablo 2 is running")
        print(f"2. Window title matches: '{args.window_title}'")
        print("3. pywin32 is installed: pip install pywin32")
        return
    
    # Initialize vision modules
    ui_vision = UIVisionModule(debug=True)
    env_vision = EnvironmentVisionModule(debug=True)
    overlay = BrainOverlay()
    
    # Output directory
    output_dir = Path("data/screenshots/outputs/live_capture")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if args.continuous:
        print("\n🔄 Continuous mode - Press 'Q' to quit")
        print("Positioning window next to game...")
        frame_count = 0
        
        import time
        start_time = time.time()
        last_fps_update = start_time
        fps = 0.0
        
        # Create window and position it
        window_name = 'Bot Brain (Press Q to quit)'
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.moveWindow(window_name, 100, 100)  # Position near top-left
        
        while True:
            loop_start = time.time()
            
            # Capture frame
            frame = capture.get_frame()
            frame_count += 1
            
            # Analyze with timing
            t0 = time.time()
            ui_state = ui_vision.analyze(frame)
            ui_time = time.time() - t0
            
            # No enemy detection (disabled), only templates on demand
            t0 = time.time()
            env_state = env_vision.analyze(
                frame, 
                current_zone=ui_state.zone_name,
                detect_templates=False,  # Disabled by default
                detect_enemies=False     # Disabled by default
            )
            env_time = time.time() - t0
            
            # Create dummy game state for overlay
            game_state = GameState(
                hp_ratio=ui_state.hp_ratio,
                mana_ratio=ui_state.mana_ratio,
                current_location=ui_state.zone_name,
                enemies=env_state.enemies,
                items=env_state.items,
                threat_level="low" if len(env_state.enemies) < 5 else "high",
            )
            
            # Create Perception
            perception = Perception(
                hp_ratio=ui_state.hp_ratio,
                mana_ratio=ui_state.mana_ratio,
                enemy_count=len(env_state.enemies),
                enemy_types=[],
                visible_items=[],
                player_position=env_state.player_position,
                raw_data={'ui_state': ui_state, 'env_state': env_state}
            )
            
            # Draw overlay
            t0 = time.time()
            annotated = overlay.draw(
                frame=frame,
                perception=perception,
                state=game_state,
                action=Action(action_type="ANALYZE"),
                fsm_state="LIVE"
            )
            draw_time = time.time() - t0
            
            # Draw regions if requested
            if args.show_regions:
                annotated = draw_regions(annotated, show_regions=True)
            
            # Calculate and display FPS
            current_time = time.time()
            if current_time - last_fps_update >= 1.0:
                fps = frame_count / (current_time - start_time)
                last_fps_update = current_time
            
            # Draw FPS counter
            fps_text = f"FPS: {fps:.1f}"
            cv2.putText(annotated, fps_text, (annotated.shape[1] - 150, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # Draw template count
            template_count = len(env_state.template_objects)
            cv2.putText(annotated, f"Templates: {template_count}", 
                       (annotated.shape[1] - 150, 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
            
            # Draw timing info
            timing_y = 90
            cv2.putText(annotated, f"UI: {ui_time*1000:.1f}ms", 
                       (annotated.shape[1] - 150, timing_y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
            cv2.putText(annotated, f"Env: {env_time*1000:.1f}ms", 
                       (annotated.shape[1] - 150, timing_y + 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
            cv2.putText(annotated, f"Draw: {draw_time*1000:.1f}ms", 
                       (annotated.shape[1] - 150, timing_y + 40),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
            
            # Display
            cv2.imshow(window_name, annotated)
            
            # Check for quit
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        cv2.destroyAllWindows()
        elapsed = time.time() - start_time
        avg_fps = frame_count / elapsed if elapsed > 0 else 0
        print(f"\n✓ Captured {frame_count} frames in {elapsed:.1f}s")
        print(f"✓ Average FPS: {avg_fps:.1f}")
    
    else:
        # Single capture
        print("\n📸 Capturing single frame...")
        frame = capture.get_frame()
        h, w = frame.shape[:2]
        print(f"✓ Captured: {w}x{h}px")
        
        print("\n" + "="*80)
        print("🔎 ANALYZING FRAME")
        print("="*80)
        
        # Analyze UI
        print("\n[UI ANALYSIS]")
        ui_state = ui_vision.analyze(frame)
        print(f"HP: {ui_state.hp_ratio*100:.1f}%")
        print(f"Mana: {ui_state.mana_ratio*100:.1f}%")
        print(f"Zone: {ui_state.zone_name}")
        
        # Analyze environment (pass zone for template detection)
        print("\n[ENVIRONMENT ANALYSIS]")
        env_state = env_vision.analyze(frame, current_zone=ui_state.zone_name)
        print(f"Enemies: {len(env_state.enemies)}")
        print(f"Items: {len(env_state.items)}")
        print(f"Template objects: {len(env_state.template_objects)}")
        if env_state.template_objects:
            print("\n  Detected objects:")
            for obj in env_state.template_objects:
                print(f"    - {obj.object_type}: {obj.template_name} ({obj.confidence:.2f}) @ {obj.position}")
        print(f"Player position: {env_state.player_position}")
        
        # Create dummy game state
        game_state = GameState(
            hp_ratio=ui_state.hp_ratio,
            mana_ratio=ui_state.mana_ratio,
            current_location=ui_state.zone_name,
            enemies=env_state.enemies,
            items=env_state.items,
            threat_level="low" if len(env_state.enemies) < 5 else "high",
        )
        
        # Create Perception object
        perception = Perception(
            hp_ratio=ui_state.hp_ratio,
            mana_ratio=ui_state.mana_ratio,
            enemy_count=len(env_state.enemies),
            enemy_types=[],
            visible_items=[],
            player_position=env_state.player_position,
            raw_data={'ui_state': ui_state, 'env_state': env_state}
        )
        
        # Draw overlay
        annotated = overlay.draw(
            frame=frame,
            perception=perception,
            state=game_state,
            action=Action(action_type="ANALYZE"),
            fsm_state="DIAGNOSTIC"
        )
        
        # Draw regions if requested
        if args.show_regions:
            annotated = draw_regions(annotated, show_regions=True)
        
        # Save result
        output_path = output_dir / "live_capture_with_brain.jpg"
        cv2.imwrite(str(output_path), annotated)
        print(f"\n✓ Saved: {output_path}")
        
        # Save raw frame
        raw_path = output_dir / "live_capture_raw.jpg"
        cv2.imwrite(str(raw_path), frame)
        print(f"✓ Raw frame: {raw_path}")
    
    print("\n" + "="*80)
    print("✅ TEST COMPLETE")
    print("="*80)


if __name__ == '__main__':
    main()
