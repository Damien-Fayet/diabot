#!/usr/bin/env python3
"""
Test script for static map localization debugging.
Captures game screen, extracts minimap, processes it, and attempts localization.
Saves all intermediate steps as debug images.
"""
import sys
from pathlib import Path
import cv2
import numpy as np
from datetime import datetime
import time
import pyautogui

# Setup path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from diabot.core.implementations import WindowsScreenCapture
from diabot.navigation.minimap_extractor import MinimapExtractor
from diabot.navigation.minimap_processor import MinimapProcessor
from diabot.navigation.player_locator import PlayerLocator
from diabot.navigation.static_map_localizer import StaticMapLocalizer, load_zone_static_map
from diabot.vision.ui_vision import UIVisionModule


def save_debug_image(img, name, output_dir):
    """Save a debug image with timestamp."""
    if img is None or img.size == 0:
        print(f"⚠️  Cannot save {name}: image is empty")
        return None
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{name}_{timestamp}.png"
    filepath = output_dir / filename
    cv2.imwrite(str(filepath), img)
    print(f"✓ Saved: {filepath}")
    return filepath

def remove_path_stones(minimap):
    """Remove brown path stones from minimap."""
    # Convert to HSV for better color detection
    hsv = cv2.cvtColor(minimap, cv2.COLOR_BGR2HSV)
    
    # Define range for brown stones (H: 10-25, S: 40-255, V: 40-150)
    lower_brown = np.array([10, 40, 40])
    upper_brown = np.array([25, 255, 150])
    
    # Create mask for brown pixels
    brown_mask = cv2.inRange(hsv, lower_brown, upper_brown)
    
    # Dilate slightly to capture full stones
    kernel = np.ones((3, 3), np.uint8)
    brown_mask = cv2.dilate(brown_mask, kernel, iterations=1)
    
    # Inpaint to remove stones and fill with surrounding colors
    result = cv2.inpaint(minimap, brown_mask, inpaintRadius=3, flags=cv2.INPAINT_TELEA)
    
    return result, brown_mask


def draw_player_on_minimap(minimap_img, player_pos):
    """Draw player position on minimap."""
    if player_pos is None:
        return minimap_img
    
    img = minimap_img.copy()
    x, y = player_pos
    # Draw cross in blue
    cv2.drawMarker(img, (int(x), int(y)), (255, 0, 0), cv2.MARKER_CROSS, 20, 2)
    cv2.circle(img, (int(x), int(y)), 5, (255, 0, 0), -1)
    return img


def main():
    print("=" * 70)
    print("🔍 STATIC MAP LOCALIZATION TEST")
    print("=" * 70)
    print()
    
    # Create output directory
    output_dir = Path(__file__).parent / "data" / "screenshots" / "outputs" / "localization_test"
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"📁 Debug images will be saved to: {output_dir}")
    print()
    
    # Step 1: Initialize capture
    print("=" * 70)
    print("STEP 1: Screen Capture (with background subtraction)")
    print("=" * 70)
    try:
        capture = WindowsScreenCapture()
        print(f"✓ Found window: {capture.window_title}")
        
        # Verify minimap is NOT displayed before first capture
        print("\n🔍 Verifying minimap is HIDDEN (checking zone OCR)...")
        ui_vision = UIVisionModule(debug=True)  # Enable debug to see OCR results
        
        # Check initial state
        temp_frame = capture.get_frame()
        if temp_frame is not None:
            if ui_vision.is_minimap_visible(temp_frame):
                print(f"   Minimap fullscreen detected, pressing TAB to hide...")
                capture.activate_window()
                time.sleep(0.2)  # Wait for window to be active
                pyautogui.press('tab')
                time.sleep(0.5)  # Wait for animation
                
                # Verify it's hidden
                verify_frame = capture.get_frame()
                still_visible = ui_vision.is_minimap_visible(verify_frame)
                print(f"   After TAB - Minimap visible: {still_visible}")
                
                if still_visible:
                    print("   ⚠️  TAB didn't hide minimap, trying again...")
                    capture.activate_window()
                    time.sleep(0.2)
                    pyautogui.press('tab')
                    time.sleep(0.5)
                
            else:
                print(f"✓ Minimap already hidden")
        
        print("✓ Minimap should now be hidden")
        
        # Capture 1: WITHOUT minimap (background)
        print("\n📸 Capturing frame WITHOUT minimap...")
        frame_without_minimap = capture.get_frame()
        if frame_without_minimap is None:
            print("❌ Failed to capture frame without minimap")
            return
        
        print(f"✓ Captured background frame: {frame_without_minimap.shape}")
        save_debug_image(frame_without_minimap, "01a_frame_without_minimap", output_dir)
        
        # Press TAB to show minimap
        print("\n⌨️  Pressing TAB to show minimap...")
        capture.activate_window()
        time.sleep(0.2)  # Wait for window to be active
        pyautogui.press('tab')
        time.sleep(0.6)  # Wait longer for animation to complete
        
        # Wait and verify minimap is displayed using OCR
        print("🔍 Verifying minimap is displayed (checking zone OCR)...")
        verify_frame = capture.get_frame()
        minimap_visible = ui_vision.is_minimap_visible(verify_frame)
        
        if minimap_visible:
            zone_name = ui_vision.extract_zone(verify_frame)
            print(f"✓ Minimap confirmed visible (Zone: {zone_name})")
        else:
            print(f"⚠️  Minimap not detected, proceeding anyway...")
        
        # Capture 2: WITH minimap (foreground)
        print("📸 Capturing frame WITH minimap...")
        frame_with_minimap = capture.get_frame()
        if frame_with_minimap is None:
            print("❌ Failed to capture frame with minimap")
            return
        
        print(f"✓ Captured frame with minimap: {frame_with_minimap.shape}")
        save_debug_image(frame_with_minimap, "01b_frame_with_minimap", output_dir)
        
        # Compute difference to isolate minimap
        print("\n🔍 Computing difference to isolate minimap...")
        diff = cv2.absdiff(frame_without_minimap, frame_with_minimap)
        save_debug_image(diff, "01b1_diff", output_dir)
        
        diff_cleaned, stones_mask = remove_path_stones(diff)
        save_debug_image(diff_cleaned, "01b2_diff_cleaned", output_dir)
        save_debug_image(stones_mask, "01b2_stones_mask", output_dir)
        
        # Use cleaned difference
        diff = diff_cleaned
        
        diff_gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
        
        # Threshold to get minimap mask
        _, mask = cv2.threshold(diff_gray, 10, 255, cv2.THRESH_BINARY)
        
        # Find minimap region from mask
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if contours:
            # Find largest contour (should be minimap)
            largest_contour = max(contours, key=cv2.contourArea)
            x, y, w, h = cv2.boundingRect(largest_contour)
            
            print(f"✓ Minimap region detected: x={x}, y={y}, w={w}, h={h}")
            
            # Draw bounding box on difference image for visualization
            diff_vis = cv2.cvtColor(diff_gray, cv2.COLOR_GRAY2BGR)
            cv2.rectangle(diff_vis, (x, y), (x+w, y+h), (0, 255, 0), 2)
            save_debug_image(diff_vis, "01c_difference_with_bbox", output_dir)
        else:
            print("⚠️  No contours found in difference, using default extraction")
        
        save_debug_image(diff, "01d_difference", output_dir)
        
        # Use original frame with minimap for subsequent processing
        frame = frame_with_minimap
        
    except Exception as e:
        print(f"❌ Failed to initialize capture: {e}")
        import traceback
        traceback.print_exc()
        return
    print()
    
    # Step 2: Extract minimap
    print("=" * 70)
    print("STEP 2: Minimap Extraction (using difference)")
    print("=" * 70)
    
    # Extract minimap from the frame with minimap visible
    extractor = MinimapExtractor(fullscreen_mode=True)
    
    # Extract minimap
    minimap = extractor.extract(frame_with_minimap)
    if minimap is None:
        print("❌ Failed to extract minimap")
        return
    
    print(f"✓ Extracted minimap: {minimap.shape}")
    save_debug_image(minimap, "02_minimap_extracted", output_dir)
    
    # Remove path stones from minimap
    print("🔍 Removing brown path stones from minimap...")
    minimap_clean, stones_mask = remove_path_stones(minimap)
    save_debug_image(minimap_clean, "02c_minimap_without_stones", output_dir)
    save_debug_image(stones_mask, "02c_stones_mask", output_dir)
    
    # Use cleaned minimap for processing
    minimap = minimap_clean
    
    # Also extract the same region from background for comparison
    minimap_background = extractor.extract(frame_without_minimap)
    if minimap_background is not None:
        save_debug_image(minimap_background, "02a_minimap_background", output_dir)
        
        # Show the difference in minimap region only
        minimap_diff = cv2.absdiff(minimap_background, minimap)
        save_debug_image(minimap_diff, "02b_minimap_difference", output_dir)
    
    print()
    
    # Step 3: Process minimap (background subtraction)
    print("=" * 70)
    print("STEP 3: Minimap Processing (Background Subtraction)")
    print("=" * 70)
    processor = MinimapProcessor(
        grid_size=64,
        wall_threshold=49,
        use_background_subtraction=True
    )
    
    # Process minimap
    grid_result = processor.process(minimap)
    
    # Get processed grid
    grid = grid_result.grid
    print(f"✓ Grid: {grid.shape}, Free cells: {np.sum(grid == 128)}, Walls: {np.sum(grid == 255)}")
    
    # Visualize grid
    grid_vis = np.zeros((grid.shape[0], grid.shape[1], 3), dtype=np.uint8)
    grid_vis[grid == 128] = [0, 255, 0]  # Free = green
    grid_vis[grid == 255] = [128, 128, 128]  # Wall = gray
    
    # Scale up for visibility
    scale = 8
    grid_vis = cv2.resize(grid_vis, (grid.shape[1] * scale, grid.shape[0] * scale), 
                          interpolation=cv2.INTER_NEAREST)
    save_debug_image(grid_vis, "03_minimap_grid", output_dir)
    
    # Save background subtraction result
    # The background subtraction is internal to process(), visualize the processed grid instead
    save_debug_image(grid, "03b_processed_binary", output_dir)
    print()
    
    # Step 4: Locate player (blue cross)
    print("=" * 70)
    print("STEP 4: Player Localization (Blue Cross Detection)")
    print("=" * 70)
    locator = PlayerLocator()
    
    player_pos = locator.detect_player_cross(minimap)
    if player_pos is not None:
        print(f"✓ Player found at: {player_pos}")
        
        # Draw player on minimap
        minimap_with_player = draw_player_on_minimap(minimap, player_pos)
        save_debug_image(minimap_with_player, "04_player_located", output_dir)
        
        # Also show on grid
        # Convert minimap coords to grid coords
        grid_x = int(player_pos[0] * grid.shape[1] / minimap.shape[1])
        grid_y = int(player_pos[1] * grid.shape[0] / minimap.shape[0])
        
        grid_vis_with_player = grid_vis.copy()
        cv2.drawMarker(grid_vis_with_player, (grid_x * scale, grid_y * scale), 
                      (255, 0, 0), cv2.MARKER_CROSS, 40, 3)
        save_debug_image(grid_vis_with_player, "04b_player_on_grid", output_dir)
    else:
        print("⚠️  Player not found (no blue cross detected), using center")
        player_pos = (minimap.shape[1] // 2, minimap.shape[0] // 2)
        
        minimap_with_player = draw_player_on_minimap(minimap, player_pos)
        save_debug_image(minimap_with_player, "04_player_center_fallback", output_dir)
    print()
    
    # Step 5: Zone detection for context
    print("=" * 70)
    print("STEP 5: Zone Detection")
    print("=" * 70)
    ui_vision = UIVisionModule()
    zone_name = ui_vision.extract_zone(frame)
    
    if not zone_name:
        zone_name = 'UNKNOWN'
    print(f"✓ Current zone: {zone_name}")
    print()
    
    # Step 6: Static map localization
    print("=" * 70)
    print("STEP 6: Static Map Localization")
    print("=" * 70)
    
    # Load static map for current zone
    static_map_path = load_zone_static_map(zone_name)
    
    if static_map_path is None:
        print(f"⚠️  No static map found for zone: {zone_name}")
        print("   Available maps should be in: data/maps/")
        return
    
    print(f"✓ Loading map: {static_map_path}")
    static_localizer = StaticMapLocalizer(static_map_path=static_map_path)
    
    # Load annotations (POIs)
    annotations_path = static_map_path.parent / f"{static_map_path.stem}_annotations.json"
    pois = []
    
    if annotations_path.exists():
        import json
        with open(annotations_path, 'r') as f:
            annotations = json.load(f)
            pois = annotations.get('pois', [])
        print(f"✓ Loaded {len(pois)} POIs from annotations")
    else:
        print(f"⚠️  No annotations found: {annotations_path}")
    
    # Save static map reference
    save_debug_image(static_localizer.static_map, "05_static_map_reference", output_dir)
    
    # Attempt localization with multi-scale
    print("\n🔍 Attempting multi-scale localization...")
    
    localization = static_localizer.localize(
        minimap,
        use_edges=True,
        multi_scale=True
    )
    
    if localization.found:
        print(f"\n✓ LOCALIZATION SUCCESS!")
        print(f"  Position: {localization.position}")
        print(f"  Confidence: {localization.confidence:.3f}")
        print(f"  Quality: {localization.match_quality}")
        
        # Draw localization on static map
        static_vis = static_localizer.static_map.copy()
        player_x, player_y = localization.position
        
        # Draw player position
        cv2.circle(static_vis, (player_x, player_y), 8, (0, 255, 0), -1)  # Green
        cv2.putText(static_vis, "PLAYER", (player_x + 15, player_y), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        # Draw all POIs
        for poi in pois:
            poi_x, poi_y = poi['position']
            if poi['type'] == 'exit':
                color = (0, 0, 255)  # Red
            elif poi['type'] == 'waypoint':
                color = (255, 255, 0)  # Cyan
            elif poi['type'] == 'npc':
                color = (255, 0, 255)  # Magenta
            else:
                color = (128, 128, 128)  # Gray
            
            cv2.circle(static_vis, (poi_x, poi_y), 6, color, -1)
            cv2.putText(static_vis, poi['name'], (poi_x + 10, poi_y), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
        
        # Find closest exit
        closest_exit = None
        min_dist = float('inf')
        for poi in pois:
            if poi['type'] == 'exit':
                poi_x, poi_y = poi['position']
                dist = np.sqrt((poi_x - player_x)**2 + (poi_y - player_y)**2)
                if dist < min_dist:
                    min_dist = dist
                    closest_exit = poi
        
        if closest_exit:
            exit_x, exit_y = closest_exit['position']
            
            # Draw arrow to exit
            cv2.arrowedLine(static_vis, (player_x, player_y), (exit_x, exit_y), 
                          (0, 255, 255), 2, tipLength=0.1)
            
            # Calculate direction
            angle, distance, dx, dy = static_localizer.get_direction_to_target(
                (player_x, player_y),
                (exit_x, exit_y)
            )
            
            print(f"\n📍 Closest Exit: {closest_exit['name']}")
            print(f"  Position: ({exit_x}, {exit_y})")
            print(f"  Direction: {angle:.1f}°")
            print(f"  Distance: {distance:.1f}px")
            
            # Add info text on image
            info_text = [
                f"Player: ({player_x}, {player_y})",
                f"Exit: {closest_exit['name']} ({exit_x}, {exit_y})",
                f"Direction: {angle:.1f}deg",
                f"Distance: {distance:.1f}px",
                f"Confidence: {localization.confidence:.3f}"
            ]
            
            y_offset = 20
            for i, text in enumerate(info_text):
                cv2.putText(static_vis, text, (10, y_offset + i * 20),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            # Calculate click coordinates (map coords to screen)
            # This is simplified - in real bot this would use game coordinate system
            print(f"\n🎯 Click Calculation:")
            print(f"  Map direction vector: dx={dx:.1f}, dy={dy:.1f}")
            print(f"  Normalized angle: {angle:.1f}°")
            
        save_debug_image(static_vis, "06_localization_result", output_dir)
        
    else:
        print(f"\n❌ LOCALIZATION FAILED")
        print(f"  Confidence: {localization.confidence:.3f}")
        print(f"  Tried scales: {[0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0]}")
    
    print()
    print("=" * 70)
    print("✅ TEST COMPLETE")
    print("=" * 70)
    print(f"\n📁 All debug images saved to: {output_dir}")
    print()


if __name__ == "__main__":
    main()
