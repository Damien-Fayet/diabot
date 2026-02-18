"""
Simplified static map localization test using phase correlation.

Supports two modes:
  1. Live mode (default): Capture from game window
  2. Static mode (--static): Use pre-captured screenshots from inputs/

Usage:
  python test_localization_simple.py               # Live capture
  python test_localization_simple.py --static      # Use static images
  python test_localization_simple.py --use-ecc     # Force ECC instead of phase correlation
"""

import argparse
import time
import json
from pathlib import Path
from typing import Optional, Tuple
import cv2
import numpy as np
import pyautogui

from src.diabot.navigation.ecc_static_localizer import ECCStaticMapLocalizer
from src.diabot.navigation.static_map_localizer import load_zone_static_map
from diabot.core.implementations import WindowsScreenCapture
from diabot.vision.ui_vision import UIVisionModule


def visualize_minimap_extraction(frame_with: np.ndarray, frame_without: np.ndarray, output_dir: Path):
    """Visualize the extracted minimap with player position at adjusted center."""
    from src.diabot.navigation.minimap_edge_extractor import MinimapEdgeExtractor
    
    print("\n[DEBUG] Extracting minimap for visualization...")
    
    # Create extractor
    extractor = MinimapEdgeExtractor(debug=False, output_dir=output_dir)
    
    # Extract minimap difference
    minimap_diff = extractor.extract_difference(frame_with, frame_without)
    
    if minimap_diff is None:
        print("[!] Could not extract minimap")
        return None, None
    
    print(f"  Minimap shape: {minimap_diff.shape}")
    
    # Calculate player position: geometric center with Y adjustment
    center_x = minimap_diff.shape[1] // 2
    center_y = minimap_diff.shape[0] // 2 - 10  # -10px Y adjustment for better accuracy
    player_pos = (center_x, center_y)
    detection_method = "geometric center (Y-10px)"
    
    print(f"  ✓ Player position: ({center_x}, {center_y}) using {detection_method}")
    
    # Create visualization
    # Convert grayscale to BGR for colored annotations
    if len(minimap_diff.shape) == 2:
        minimap_vis = cv2.cvtColor(minimap_diff, cv2.COLOR_GRAY2BGR)
    else:
        minimap_vis = minimap_diff.copy()
    
    # Draw player marker at center
    # Large red crosshair
    cv2.line(minimap_vis, (center_x - 40, center_y), (center_x + 40, center_y), (0, 0, 255), 3)
    cv2.line(minimap_vis, (center_x, center_y - 40), (center_x, center_y + 40), (0, 0, 255), 3)
    
    # White outline
    cv2.line(minimap_vis, (center_x - 40, center_y), (center_x + 40, center_y), (255, 255, 255), 1)
    cv2.line(minimap_vis, (center_x, center_y - 40), (center_x, center_y + 40), (255, 255, 255), 1)
    
    # Circle around center
    cv2.circle(minimap_vis, (center_x, center_y), 50, (0, 0, 255), 2)
    cv2.circle(minimap_vis, (center_x, center_y), 50, (255, 255, 255), 1)
    
    # Add text label
    cv2.putText(
        minimap_vis,
        "PLAYER (center of minimap)",
        (center_x - 120, center_y - 70),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 0, 255),
        2
    )
    
    # Add info text
    info_text = [
        f"Minimap size: {minimap_diff.shape[1]}x{minimap_diff.shape[0]}px",
        f"Player position: ({center_x}, {center_y})",
        f"Detection: {detection_method}",
        "This is the minimap BEFORE alignment"
    ]
    
    y_offset = 30
    for text in info_text:
        # Background
        (text_w, text_h), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
        cv2.rectangle(
            minimap_vis,
            (10, y_offset - text_h - 5),
            (20 + text_w, y_offset + 5),
            (0, 0, 0),
            -1
        )
        # Text
        cv2.putText(
            minimap_vis,
            text,
            (15, y_offset),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 255),
            1
        )
        y_offset += 30
    
    # Save
    output_path = output_dir / "00_minimap_with_player_position.png"
    cv2.imwrite(str(output_path), minimap_vis)
    print(f"  ✓ Minimap visualization saved: {output_path}")
    
    return minimap_vis, player_pos


def load_static_images():
    """Load pre-captured images from inputs folder."""
    inputs_dir = Path(__file__).parent / "data" / "screenshots" / "inputs"
    
    # Try to find most recent images
    without_images = sorted(inputs_dir.glob("*_without_minimap*.png"))
    with_images = sorted(inputs_dir.glob("*_with_minimap*.png"))
    
    if not without_images or not with_images:
        print(f"[!] No input images found in {inputs_dir}")
        print("    Expected files: *_without_minimap*.png and *_with_minimap*.png")
        return None, None
    
    # Use most recent files
    frame_without = cv2.imread(str(without_images[-1]))
    frame_with = cv2.imread(str(with_images[-1]))
    
    if frame_without is None or frame_with is None:
        print("[!] Failed to load input images")
        return None, None
    
    print(f"✓ Loaded: {without_images[-1].name}")
    print(f"  Shape: {frame_without.shape}")
    print(f"✓ Loaded: {with_images[-1].name}")
    print(f"  Shape: {frame_with.shape}")
    
    return frame_without, frame_with


def ensure_minimap_hidden(capture, ui_vision, max_attempts=3):
    """Ensure minimap is hidden before capturing background."""
    for attempt in range(max_attempts):
        frame = capture.get_frame()
        minimap_visible = ui_vision.is_minimap_visible(frame)
        
        if not minimap_visible:
            print("✓ Minimap already hidden")
            return True
        
        print(f"  Attempt {attempt+1}/{max_attempts}: Minimap detected, pressing TAB...")
        capture.activate_window()
        time.sleep(0.2)
        pyautogui.press('tab')
        time.sleep(0.3)
    
    print("[!] Warning: Could not confirm minimap is hidden")
    return False


def load_poi_annotations(map_path: Path):
    """Load POI annotations for a static map."""
    annotations_path = map_path.parent / f"{map_path.stem}_annotations.json"
    
    if not annotations_path.exists():
        return None
    
    try:
        with open(annotations_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return data.get('pois', [])
    except Exception as e:
        print(f"[!] Could not load annotations: {e}")
        return None


def draw_poi_vectors(result_map: np.ndarray, player_pos: tuple, pois: list):
    """Draw vectors from player to all POI."""
    if not pois:
        return result_map
    
    # POI colors by type (BGR format for OpenCV)
    poi_colors = {
        'waypoint': (0, 215, 255),   # Gold
        'exit': (0, 255, 0),         # Green
        'stash': (0, 140, 255),      # Orange
        'npc': (219, 112, 147),      # Purple
        'portal': (255, 255, 0),     # Cyan
        'quest': (255, 0, 255),      # Magenta
        'default': (200, 200, 200)   # Gray
    }
    
    player_x, player_y = player_pos
    
    print(f"\n📍 POI Detected ({len(pois)} total):")
    
    for poi in pois:
        poi_type = poi.get('type', 'unknown')
        poi_name = poi.get('name', poi_type)
        poi_pos = poi.get('position', [])
        
        if len(poi_pos) != 2:
            continue
        
        poi_x, poi_y = poi_pos
        
        # Calculate distance and direction
        dx = poi_x - player_x
        dy = poi_y - player_y
        distance = np.sqrt(dx**2 + dy**2)
        
        # Get color for this POI type
        color = poi_colors.get(poi_type, poi_colors['default'])
        
        # Draw arrow from player to POI
        cv2.arrowedLine(
            result_map,
            (player_x, player_y),
            (poi_x, poi_y),
            color,
            2,
            tipLength=0.02
        )
        
        # Draw POI marker
        cv2.circle(result_map, (poi_x, poi_y), 8, color, -1)
        cv2.circle(result_map, (poi_x, poi_y), 8, (255, 255, 255), 1)
        
        # Draw label
        label = f"{poi_name} ({distance:.0f}px)"
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5
        thickness = 1
        
        # Get text size for background
        (text_w, text_h), baseline = cv2.getTextSize(label, font, font_scale, thickness)
        
        # Draw background rectangle
        padding = 3
        cv2.rectangle(
            result_map,
            (poi_x + 12, poi_y - text_h - padding),
            (poi_x + 12 + text_w + padding*2, poi_y + padding),
            (0, 0, 0),
            -1
        )
        
        # Draw text
        cv2.putText(
            result_map,
            label,
            (poi_x + 12 + padding, poi_y),
            font,
            font_scale,
            color,
            thickness
        )
        
        print(f"   {poi_type:12s} | {poi_name:20s} | {distance:6.0f}px | direction: ({dx:+5.0f}, {dy:+5.0f})")
    
    # Draw legend in top-right corner
    legend_x = result_map.shape[1] - 280
    legend_y = 30
    legend_height = 30 + (len(poi_colors) - 1) * 25
    
    # Draw legend background
    cv2.rectangle(
        result_map,
        (legend_x - 10, legend_y - 10),
        (legend_x + 250, legend_y + legend_height),
        (0, 0, 0),
        -1
    )
    cv2.rectangle(
        result_map,
        (legend_x - 10, legend_y - 10),
        (legend_x + 250, legend_y + legend_height),
        (255, 255, 255),
        1
    )
    
    # Draw legend title
    cv2.putText(
        result_map,
        "POI Legend:",
        (legend_x, legend_y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        2
    )
    
    # Draw legend items
    y_offset = legend_y + 25
    for poi_type, color in poi_colors.items():
        if poi_type == 'default':
            continue
        
        # Draw colored circle
        cv2.circle(result_map, (legend_x + 10, y_offset), 6, color, -1)
        cv2.circle(result_map, (legend_x + 10, y_offset), 6, (255, 255, 255), 1)
        
        # Draw type label
        cv2.putText(
            result_map,
            poi_type.capitalize(),
            (legend_x + 25, y_offset + 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1
        )
        
        y_offset += 25
    
    return result_map


def capture_live_frames(output_dir):
    """Capture frames from live game window."""
    print("\n[*] Initializing live capture...")
    
    try:
        capture = WindowsScreenCapture()
        print(f"✓ Found window: {capture.window_title}")
    except Exception as e:
        print(f"[!] Failed to initialize capture: {e}")
        return None, None
    
    ui_vision = UIVisionModule(debug=False)
    
    print("\n[*] Ensuring minimap is hidden...")
    ensure_minimap_hidden(capture, ui_vision)
    
    print("[*] Capturing background (without minimap)...")
    time.sleep(0.5)
    frame_without = capture.get_frame()
    
    # Save background
    output_path = output_dir / "01_frame_without_minimap.png"
    cv2.imwrite(str(output_path), frame_without)
    print(f"✓ Saved: {output_path}")
    
    print("\n[*] Press TAB to show minimap...")
    capture.activate_window()
    time.sleep(0.2)
    pyautogui.press('tab')
    time.sleep(0.5)
    
    print("[*] Capturing with minimap...")
    frame_with = capture.get_frame()
    
    # Verify minimap is visible
    minimap_visible = ui_vision.is_minimap_visible(frame_with)
    if not minimap_visible:
        print("[!] Warning: Minimap not detected in capture")
    else:
        print("✓ Minimap detected")
    
    # Save
    output_path = output_dir / "02_frame_with_minimap.png"
    cv2.imwrite(str(output_path), frame_with)
    print(f"✓ Saved: {output_path}")
    
    return frame_without, frame_with


def main():
    parser = argparse.ArgumentParser(
        description="Test phase correlation localization (live or static)"
    )
    parser.add_argument(
        "--static",
        action="store_true",
        help="Use static images from inputs/ folder instead of live capture"
    )
    parser.add_argument(
        "--zone",
        type=str,
        default="Rogue Encampment",
        help="Zone name for static map lookup (default: Rogue Encampment)"
    )
    parser.add_argument(
        "--static-map",
        type=str,
        default=None,
        help="Path to static reference map (overrides --zone auto-detection)"
    )
    parser.add_argument(
        "--use-ecc",
        action="store_true",
        help="Use ECC instead of phase correlation (slower)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/screenshots/outputs/localization_simple",
        help="Output directory for debug images"
    )
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("="*70)
    print("SIMPLIFIED STATIC MAP LOCALIZATION TEST")
    print("="*70)
    print(f"Mode: {'Static Images (from inputs/)' if args.static else 'Live Capture'}")
    print(f"Method: {'ECC (multi-scale)' if args.use_ecc else 'Phase Correlation (fast)'}")
    print(f"Output: {output_dir}")
    print("="*70)
    
    # Acquire frames
    print("\n[1/4] Acquiring frames...")
    
    if args.static:
        # Load static images from inputs folder
        frame_without, frame_with = load_static_images()
        if frame_without is None or frame_with is None:
            return
    else:
        # Live capture mode
        frame_without, frame_with = capture_live_frames(output_dir)
        if frame_without is None or frame_with is None:
            return
    
    print(f"\n✓ Frames ready: {frame_with.shape}")
    
    # Visualize minimap extraction with player detection
    minimap_vis, detected_player_pos = visualize_minimap_extraction(frame_with, frame_without, output_dir)
    
    # Detect zone or use specified one
    print("\n[2/4] Detecting zone...")
    zone_name = args.zone
    
    if not args.static and not args.static_map:
        # Try to detect zone from frame if in live mode
        ui_vision = UIVisionModule(debug=False)
        detected_zone = ui_vision.extract_zone(frame_with)
        if detected_zone:
            zone_name = detected_zone
            print(f"✓ Detected zone: {zone_name}")
        else:
            print(f"⚠ Could not detect zone, using default: {zone_name}")
    else:
        print(f"✓ Using zone: {zone_name}")
    
    # Load static map
    print("\n[3/4] Loading static map...")
    
    if args.static_map:
        # User specified a map path
        map_path = Path(args.static_map)
        print(f"  Using specified map: {map_path}")
    else:
        # Auto-detect map from zone name
        map_path = load_zone_static_map(zone_name)
        if map_path is None:
            print(f"[!] No static map found for zone: {zone_name}")
            print("    Available maps in data/maps/minimap_images/:")
            maps_dir = Path("data/maps/minimap_images")
            if maps_dir.exists():
                for f in sorted(maps_dir.glob("*.png")):
                    print(f"      - {f.name}")
            return
        print(f"✓ Found map: {map_path}")
    
    # Initialize localizer
    print("\n[4/4] Initializing localizer...")
    localizer = ECCStaticMapLocalizer(
        static_map_path=map_path,
        debug=True,
        output_dir=output_dir
    )
    
    if localizer.static_map is None:
        print(f"[!] Could not load static map: {map_path}")
        return
    
    # Localize
    print("\nRunning localization...")
    player_pos, confidence = localizer.localize(
        frame_with,
        frame_without,
        use_phase_correlation=not args.use_ecc,
        use_ecc_fallback=False,  # No fallback for clean testing
        use_oriented_filter=True
    )
    
    # Results
    print("\n" + "="*70)
    print("RESULTS")
    print("="*70)
    
    if player_pos is not None:
        print(f"✅ SUCCESS")
        print(f"   Player position: ({player_pos[0]}, {player_pos[1]})")
        print(f"   Confidence: {confidence:.3f}")
        
        # Quality interpretation
        if confidence >= 0.70:
            quality = "Excellent"
        elif confidence >= 0.50:
            quality = "Good"
        elif confidence >= 0.30:
            quality = "Moderate"
        else:
            quality = "Low"
        print(f"   Quality: {quality}")
        
        # Load POI annotations
        pois = load_poi_annotations(map_path)
        
        # Visualize on static map
        result_map = localizer.static_map.copy()
        
        # Draw player position
        cv2.circle(result_map, player_pos, 15, (0, 0, 255), -1)
        cv2.circle(result_map, player_pos, 15, (255, 255, 255), 2)
        cv2.line(
            result_map,
            (player_pos[0]-30, player_pos[1]),
            (player_pos[0]+30, player_pos[1]),
            (0, 0, 255),
            2
        )
        cv2.line(
            result_map,
            (player_pos[0], player_pos[1]-30),
            (player_pos[0], player_pos[1]+30),
            (0, 0, 255),
            2
        )
        
        # Draw POI vectors if available
        if pois:
            result_map = draw_poi_vectors(result_map, player_pos, pois)
        else:
            print(f"\n⚠ No POI annotations found for this map")
        
        output_path = output_dir / "player_position_on_map.png"
        cv2.imwrite(str(output_path), result_map)
        print(f"\n   Visualization saved: {output_path}")
    else:
        print(f"❌ FAILED")
        print(f"   Confidence: {confidence:.3f}")
    
    print("\n" + "="*70)


if __name__ == "__main__":
    main()
