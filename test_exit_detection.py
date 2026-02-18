#!/usr/bin/env python3
"""
Test script for exit detection on minimap using distance transform.

This script detects narrow passages (exits/doorways) on the minimap by:
1. Cleaning artifacts (small components)
2. Computing distance transform on navigable space
3. Detecting narrow passages (low distance) connected to large rooms (high distance)

Usage:
  python test_exit_detection.py           # Live capture mode
  python test_exit_detection.py --static   # Use static images from inputs/
"""
import sys
from pathlib import Path
import cv2
import numpy as np
from datetime import datetime
import time
import pyautogui
import argparse

# Setup path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from diabot.core.implementations import WindowsScreenCapture
from diabot.navigation import ECCStaticMapLocalizer
from diabot.vision.ui_vision import UIVisionModule


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def save_debug_image(img, name, output_dir):
    """Save a debug image with timestamp."""
    if img is None or img.size == 0:
        print(f"[!] Cannot save {name}: image is empty")
        return None
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{name}_{timestamp}.png"
    filepath = output_dir / filename
    cv2.imwrite(str(filepath), img)
    print(f"OK Saved: {filepath}")
    return filepath


# ============================================================================
# FRAME CAPTURE & SETUP
# ============================================================================

def load_static_images(output_dir):
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
    
    print(f"OK Loaded: {without_images[-1].name}")
    print(f"   Shape: {frame_without.shape}")
    print(f"OK Loaded: {with_images[-1].name}")
    print(f"   Shape: {frame_with.shape}")
    
    # Save copies to output for comparison
    save_debug_image(frame_without, "01_frame_without_minimap", output_dir)
    save_debug_image(frame_with, "02_frame_with_minimap", output_dir)
    
    return frame_without, frame_with


def ensure_minimap_hidden(capture, ui_vision, max_attempts=3):
    """Ensure minimap is hidden before capturing background."""
    for attempt in range(max_attempts):
        frame = capture.get_frame()
        minimap_visible = ui_vision.is_minimap_visible(frame)
        
        if not minimap_visible:
            print("OK Minimap already hidden")
            return True
        
        print("   Minimap detected, pressing TAB to hide...")
        capture.activate_window()
        time.sleep(0.2)
        pyautogui.press('tab')
        time.sleep(0.3)
        
        frame_after = capture.get_frame()
        minimap_visible_after = ui_vision.is_minimap_visible(frame_after)
        print(f"   After TAB - Minimap visible: {minimap_visible_after}")
        
        if not minimap_visible_after:
            return True
        
        if attempt < max_attempts - 1:
            capture.activate_window()
            time.sleep(0.2)
            print("   [!] TAB didn't hide minimap, trying again...")
            time.sleep(0.5)
    
    print("[!] Warning: Could not confirm minimap is hidden")
    return False


def capture_background_frame(capture, output_dir):
    """Capture frame without minimap."""
    print("\n[Camera] Capturing frame WITHOUT minimap...")
    time.sleep(0.2)
    frame = capture.get_frame()
    print(f"OK Captured background frame: {frame.shape}")
    save_debug_image(frame, "01_frame_without_minimap", output_dir)
    return frame


def show_minimap_and_capture(capture, ui_vision, output_dir):
    """Show minimap and capture frame."""
    print("\n[Keyboard] Pressing TAB to show minimap...")
    capture.activate_window()
    time.sleep(0.2)
    pyautogui.press('tab')
    time.sleep(0.3)
    
    print("[*] Verifying minimap is displayed...")
    time.sleep(0.2)
    
    frame = capture.get_frame()
    minimap_visible = ui_vision.is_minimap_visible(frame)
    
    if not minimap_visible:
        print(f"[!] Minimap not detected, proceeding anyway...")
    else:
        zone_name = ui_vision.extract_zone(frame)
        print(f"OK Minimap confirmed visible (Zone: {zone_name})")
    
    print("\n[Camera] Capturing frame WITH minimap...")
    frame = capture.get_frame()
    print(f"OK Captured frame with minimap: {frame.shape}")
    save_debug_image(frame, "02_frame_with_minimap", output_dir)
    return frame


# ============================================================================
# EXIT DETECTION USING DISTANCE TRANSFORM
# ============================================================================

def detect_exits_on_minimap(
    frame_with_minimap,
    frame_without_minimap,
    output_dir,
    min_component_area=150,
    narrow_threshold=8,
    visualization_scale=3
):
    """
    Detect exits/doorways on the minimap using distance transform method.
    
    Args:
        frame_with_minimap: Frame with minimap visible
        frame_without_minimap: Frame without minimap (background)
        output_dir: Directory for debug images
        min_component_area: Minimum area to keep when cleaning artifacts (pixels)
        narrow_threshold: Distance threshold to identify narrow passages (pixels)
        visualization_scale: Scale factor for visualization images
        
    Returns:
        List of exit candidates (x, y, score) or empty list
    """
    print("\n" + "="*70)
    print("EXIT DETECTION: Distance Transform Method")
    print("="*70)
    
    # Create temporary localizer to extract minimap edges
    localizer = ECCStaticMapLocalizer(debug=False, output_dir=output_dir)
    
    # ========================================================================
    # STEP 1: Extract minimap edges
    # ========================================================================
    print("\n[STEP 1] Extracting minimap edges...")
    
    minimap_edges = localizer.extract_minimap_edges_canny(
        frame_with_minimap,
        frame_without_minimap,
        use_oriented_filter=True,
        canny_low=50,
        canny_high=150
    )
    
    if minimap_edges is None:
        print("[!] Failed to extract minimap edges")
        return []
    
    save_debug_image(minimap_edges, "exit_01_minimap_edges", output_dir)
    print(f"   Minimap edges: {minimap_edges.shape[1]}×{minimap_edges.shape[0]}")
    
    # ========================================================================
    # STEP 2: Clean artifacts (remove small components)
    # ========================================================================
    print("\n[STEP 2] Cleaning artifacts (removing small components)...")
    
    # Connected components analysis
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        minimap_edges, 
        connectivity=8
    )
    
    print(f"   Found {num_labels - 1} components before filtering")
    
    # Create clean binary with only large components
    clean = np.zeros_like(minimap_edges)
    kept_count = 0
    
    for i in range(1, num_labels):  # Skip background (label 0)
        area = stats[i, cv2.CC_STAT_AREA]
        if area > min_component_area:
            clean[labels == i] = 255
            kept_count += 1
    
    print(f"   Kept {kept_count} large components (area > {min_component_area} px)")
    save_debug_image(clean, "exit_02_cleaned_walls", output_dir)
    
    # ========================================================================
    # STEP 3: Invert to get navigable free space
    # ========================================================================
    print("\n[STEP 3] Computing navigable free space...")
    
    free_space = cv2.bitwise_not(clean)
    save_debug_image(free_space, "exit_03_free_space", output_dir)
    print(f"   Free space computed (white = navigable)")
    
    # ========================================================================
    # STEP 4: Distance transform (MAGIC! 🔑)
    # ========================================================================
    print("\n[STEP 4] Computing distance transform...")
    
    dist = cv2.distanceTransform(free_space, cv2.DIST_L2, 5)
    
    # Normalize for visualization
    dist_normalized = cv2.normalize(dist, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    dist_colored = cv2.applyColorMap(dist_normalized, cv2.COLORMAP_JET)
    
    save_debug_image(dist_colored, "exit_04_distance_transform", output_dir)
    print(f"   Distance transform computed")
    print(f"   Max distance: {dist.max():.2f} px")
    print(f"   Mean distance: {dist.mean():.2f} px")
    
    # ========================================================================
    # STEP 5: Detect narrow passages (potential exits)
    # ========================================================================
    print("\n[STEP 5] Detecting narrow passages...")
    
    # Narrow passages: low distance but not zero (connected to free space)
    narrow = ((dist < narrow_threshold) & (dist > 0.5)).astype(np.uint8) * 255
    
    print(f"   Narrow threshold: < {narrow_threshold} px")
    save_debug_image(narrow, "exit_05_narrow_passages", output_dir)
    
    # Count narrow pixels
    narrow_pixels = np.count_nonzero(narrow)
    print(f"   Found {narrow_pixels} narrow passage pixels")
    
    # ========================================================================
    # STEP 6: Find narrow passage candidates connected to large rooms
    # ========================================================================
    print("\n[STEP 6] Finding exit candidates (narrow → large rooms)...")
    
    # Dilate narrow passages to find connection regions
    kernel = np.ones((5, 5), np.uint8)
    narrow_dilated = cv2.dilate(narrow, kernel, iterations=2)
    
    # Find regions where narrow passages connect to large rooms (high distance)
    large_room_threshold = narrow_threshold * 2  # Large rooms have high distance
    large_rooms = (dist > large_room_threshold).astype(np.uint8) * 255
    
    # Exit candidates: narrow passages near large rooms
    exit_candidates_mask = cv2.bitwise_and(narrow_dilated, large_rooms)
    
    save_debug_image(large_rooms, "exit_06_large_rooms", output_dir)
    save_debug_image(exit_candidates_mask, "exit_07_exit_candidates", output_dir)
    
    # Find contours of exit candidates
    contours, _ = cv2.findContours(exit_candidates_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    print(f"   Found {len(contours)} exit candidate regions")
    
    # Extract exit positions and scores
    exits = []
    for contour in contours:
        # Get centroid
        M = cv2.moments(contour)
        if M["m00"] > 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            
            # Score based on area and distance gradient
            area = cv2.contourArea(contour)
            
            # Check distance gradient (should connect narrow → wide)
            # Sample distance at contour points
            mask = np.zeros_like(dist)
            cv2.drawContours(mask, [contour], -1, 1, -1)
            distances = dist[mask > 0]
            
            if len(distances) > 0:
                dist_std = distances.std()  # High std = good gradient
                score = area * dist_std
                exits.append((cx, cy, score, area, dist_std))
    
    # Sort by score (best candidates first)
    exits.sort(key=lambda x: x[2], reverse=True)
    
    print(f"   Exit candidates sorted by score:")
    for i, (x, y, score, area, std) in enumerate(exits[:10]):  # Top 10
        print(f"      {i+1}. Position ({x}, {y}), Score: {score:.1f}, Area: {area:.0f}, Gradient: {std:.2f}")
    
    # ========================================================================
    # STEP 7: Visualization with annotations
    # ========================================================================
    print("\n[STEP 7] Creating annotated visualizations...")
    
    # Create upscaled visualization for better visibility
    h, w = minimap_edges.shape
    viz_h, viz_w = h * visualization_scale, w * visualization_scale
    
    # Base: distance transform overlay on minimap
    minimap_rgb = cv2.cvtColor(minimap_edges, cv2.COLOR_GRAY2BGR)
    dist_overlay = cv2.addWeighted(minimap_rgb, 0.4, dist_colored, 0.6, 0)
    dist_overlay_scaled = cv2.resize(dist_overlay, (viz_w, viz_h), interpolation=cv2.INTER_NEAREST)
    
    # Draw narrow passages in yellow
    narrow_rgb = cv2.cvtColor(narrow, cv2.COLOR_GRAY2BGR)
    narrow_yellow = np.zeros_like(narrow_rgb)
    narrow_yellow[narrow > 0] = [0, 255, 255]  # Yellow
    narrow_scaled = cv2.resize(narrow_yellow, (viz_w, viz_h), interpolation=cv2.INTER_NEAREST)
    
    viz_combined = cv2.addWeighted(dist_overlay_scaled, 0.7, narrow_scaled, 0.3, 0)
    
    # Draw exit candidates with numbered markers
    for i, (x, y, score, area, std) in enumerate(exits[:10]):  # Top 10 exits
        # Scale coordinates
        x_scaled = int(x * visualization_scale)
        y_scaled = int(y * visualization_scale)
        
        # Color by rank (green = best, red = worst)
        color_ratio = i / max(len(exits[:10]) - 1, 1)
        color = (0, int(255 * (1 - color_ratio)), int(255 * color_ratio))  # Green → Red
        
        # Draw marker
        cv2.circle(viz_combined, (x_scaled, y_scaled), 8, color, -1)
        cv2.circle(viz_combined, (x_scaled, y_scaled), 10, (255, 255, 255), 2)
        
        # Draw number
        cv2.putText(viz_combined, str(i+1), (x_scaled - 6, y_scaled + 5),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        
        # Draw score
        cv2.putText(viz_combined, f"{score:.0f}", (x_scaled + 15, y_scaled),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
    
    # Add legend
    cv2.putText(viz_combined, "Exit Detection (Distance Transform)", (10, 25),
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(viz_combined, f"Found {len(exits)} candidates", (10, 50),
               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    cv2.putText(viz_combined, "Yellow = narrow passages", (10, 70),
               cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)
    cv2.putText(viz_combined, "Green = best exits", (10, 90),
               cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
    
    save_debug_image(viz_combined, "exit_08_final_annotated", output_dir)
    
    # Also create a simple overview showing distance transform levels
    overview = np.hstack([
        cv2.resize(cv2.cvtColor(minimap_edges, cv2.COLOR_GRAY2BGR), (viz_w, viz_h), interpolation=cv2.INTER_NEAREST),
        cv2.resize(dist_colored, (viz_w, viz_h), interpolation=cv2.INTER_NEAREST),
        viz_combined
    ])
    
    cv2.putText(overview, "Original edges", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    cv2.putText(overview, "Distance transform", (viz_w + 10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    cv2.putText(overview, "Exit candidates", (viz_w * 2 + 10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    save_debug_image(overview, "exit_09_overview_comparison", output_dir)
    
    print(f"   ✅ Visualizations saved")
    
    # Return exits (without area and std, just position and score)
    return [(x, y, score) for x, y, score, _, _ in exits]


# ============================================================================
# MAIN
# ============================================================================

def main():
    """Main test function."""
    # Parse arguments
    parser = argparse.ArgumentParser(description='Test exit detection on minimap')
    parser.add_argument('--static', action='store_true', 
                       help='Use static images from inputs/ folder instead of live capture')
    parser.add_argument('--min-area', type=int, default=150,
                       help='Minimum component area to keep (default: 150)')
    parser.add_argument('--narrow-threshold', type=int, default=8,
                       help='Distance threshold for narrow passages (default: 8)')
    parser.add_argument('--viz-scale', type=int, default=3,
                       help='Visualization scale factor (default: 3)')
    args = parser.parse_args()
    
    print("=" * 70)
    print("EXIT DETECTION TEST (Distance Transform Method)")
    if args.static:
        print("MODE: Static Images (from inputs/)")
    else:
        print("MODE: Live Capture")
    print("=" * 70)
    print()
    
    # Create output directory (clean if exists)
    output_dir = Path(__file__).parent / "data" / "screenshots" / "outputs" / "exit_detection_test"
    if output_dir.exists():
        import shutil
        print(f"[*] Cleaning output directory: {output_dir}")
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"[*] Debug images will be saved to: {output_dir}")
    print()
    
    # Display parameters
    print("PARAMETERS:")
    print(f"  Min component area: {args.min_area} px²")
    print(f"  Narrow threshold: {args.narrow_threshold} px")
    print(f"  Visualization scale: {args.viz_scale}×")
    print()
    
    try:
        print("=" * 70)
        print("STEP 1: Frame Acquisition")
        print("=" * 70)
        
        if args.static:
            # Load static images from inputs folder
            frame_without_minimap, frame_with_minimap = load_static_images(output_dir)
            if frame_without_minimap is None or frame_with_minimap is None:
                return
        else:
            # Live capture mode
            capture = WindowsScreenCapture()
            print(f"OK Found window: {capture.window_title}")
            
            ui_vision = UIVisionModule(debug=True)
            print("\n[*] Verifying minimap is HIDDEN...")
            
            # Ensure minimap is hidden
            ensure_minimap_hidden(capture, ui_vision)
            print("OK Minimap should now be hidden")
            
            # Capture background
            frame_without_minimap = capture_background_frame(capture, output_dir)
            if frame_without_minimap is None:
                return
            
            # Show minimap and capture
            frame_with_minimap = show_minimap_and_capture(capture, ui_vision, output_dir)
            if frame_with_minimap is None:
                return
        
        # ================================================================
        # EXIT DETECTION
        # ================================================================
        print("\n" + "="*70)
        print("STEP 2: Exit Detection")
        print("="*70)
        
        exits = detect_exits_on_minimap(
            frame_with_minimap,
            frame_without_minimap,
            output_dir,
            min_component_area=args.min_area,
            narrow_threshold=args.narrow_threshold,
            visualization_scale=args.viz_scale
        )
        
        # ================================================================
        # RESULTS
        # ================================================================
        print()
        print("=" * 70)
        print("FINAL RESULTS")
        print("=" * 70)
        print()
        
        if exits:
            print(f"✅ Found {len(exits)} exit candidates:")
            print(f"{'Rank':<6} {'Position':<20} {'Score':<10}")
            print("-" * 70)
            for i, (x, y, score) in enumerate(exits[:10], 1):
                print(f"{i:<6} ({x:3d}, {y:3d}){'':>10} {score:>8.1f}")
            
            if len(exits) > 10:
                print(f"... and {len(exits) - 10} more")
            
            print()
            print("💡 Best exit candidate (highest score):")
            x, y, score = exits[0]
            print(f"   Position: ({x}, {y})")
            print(f"   Score: {score:.1f}")
        else:
            print("❌ No exits detected")
            print("   Try adjusting --narrow-threshold or --min-area parameters")
        
        print()
        print("=" * 70)
        print("DEBUG IMAGES SAVED TO:")
        print(f"  {output_dir}")
        print("=" * 70)
        
    except Exception as e:
        print(f"[X] Failed: {e}")
        import traceback
        traceback.print_exc()
        return


if __name__ == "__main__":
    main()
