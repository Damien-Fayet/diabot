#!/usr/bin/env python3
"""
Test script for static map localization debugging.
Refactored version with clear function separation for each step.
"""
import sys
from pathlib import Path
import cv2
import numpy as np
from datetime import datetime
import time
import pyautogui
import json

# Setup path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from diabot.core.implementations import WindowsScreenCapture
from diabot.navigation.minimap_extractor import MinimapExtractor
from diabot.navigation.minimap_processor import MinimapProcessor
from diabot.navigation.player_locator import PlayerLocator
from diabot.navigation.static_map_localizer import StaticMapLocalizer, load_zone_static_map
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
    print(f"✓ Saved: {filepath}")
    return filepath


def draw_player_on_minimap(minimap_img, player_pos):
    """Draw player position on minimap."""
    if player_pos is None:
        return minimap_img
    
    img = minimap_img.copy()
    x, y = player_pos
    cv2.drawMarker(img, (int(x), int(y)), (255, 0, 0), cv2.MARKER_CROSS, 20, 2)
    cv2.circle(img, (int(x), int(y)), 5, (255, 0, 0), -1)
    return img


# ============================================================================
# STEP 1: SCREEN CAPTURE
# ============================================================================

def ensure_minimap_hidden(capture, ui_vision):
    """Ensure minimap is hidden before background capture."""
    temp_frame = capture.get_frame()
    if temp_frame is None:
        return False
    
    if ui_vision.is_minimap_visible(temp_frame):
        print(f"   Minimap detected, pressing TAB to hide...")
        capture.activate_window()
        time.sleep(0.2)
        pyautogui.press('tab')
        time.sleep(0.5)
        
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
        
        return True
    else:
        print(f"✓ Minimap already hidden")
        return True


def capture_background_frame(capture, output_dir):
    """Capture frame without minimap."""
    print("\n📸 Capturing frame WITHOUT minimap...")
    frame = capture.get_frame()
    if frame is None:
        print("❌ Failed to capture frame without minimap")
        return None
    
    print(f"✓ Captured background frame: {frame.shape}")
    save_debug_image(frame, "01_frame_without_minimap", output_dir)
    return frame


def show_minimap_and_capture(capture, ui_vision, output_dir):
    """Press TAB to show minimap and capture frame."""
    print("\n⌨️  Pressing TAB to show minimap...")
    capture.activate_window()
    time.sleep(0.2)
    pyautogui.press('tab')
    time.sleep(0.6)
    
    # Verify minimap is displayed
    print("🔍 Verifying minimap is displayed...")
    verify_frame = capture.get_frame()
    minimap_visible = ui_vision.is_minimap_visible(verify_frame)
    
    if minimap_visible:
        zone_name = ui_vision.extract_zone(verify_frame)
        print(f"✓ Minimap confirmed visible (Zone: {zone_name})")
    else:
        print(f"⚠️  Minimap not detected, proceeding anyway...")
    
    # Capture frame with minimap
    print("📸 Capturing frame WITH minimap...")
    frame = capture.get_frame()
    if frame is None:
        print("❌ Failed to capture frame with minimap")
        return None
    
    print(f"✓ Captured frame with minimap: {frame.shape}")
    save_debug_image(frame, "02_frame_with_minimap", output_dir)
    return frame


def compute_minimap_difference(frame_without, frame_with, output_dir, 
                              min_brightness=20, noise_kernel_size=3,
                              threshold_value=10, dilate_iterations=1):
    """Compute difference between frames to isolate minimap region.
    
    Args:
        min_brightness: Seuil minimum de luminosité pour garder un pixel (défaut: 15)
        noise_kernel_size: Taille du kernel pour éliminer le bruit (défaut: 3, 0=désactivé)
        threshold_value: Seuil pour binariser la différence (défaut: 10)
        dilate_iterations: Nombre d'itérations de dilatation pour connecter les régions (défaut: 3)
    """
    print("\n🔍 Computing difference to isolate minimap...")
    print(f"   Parameters: min_brightness={min_brightness}, noise_kernel={noise_kernel_size}")
    print(f"               threshold={threshold_value}, dilate={dilate_iterations}")
    
    # Directional difference: keep only pixels that appear in frame_with (minimap)
    # cv2.subtract handles underflow properly (clips to 0)
    diff = cv2.subtract(frame_with, frame_without)
    
    # NOTE: Ne pas appliquer remove_minimap_background() sur la différence
    # car cela crée des doublons des éléments animés (PNJs, monstres)
    save_debug_image(diff, "03_diff_raw", output_dir)
    
    # Convert to grayscale for preprocessing
    diff_gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
    
    # STEP 1: Filter dark pixels (likely noise or shadows)
    print(f"🔍 Filtering pixels darker than {min_brightness}...")
    _, dark_mask = cv2.threshold(diff_gray, min_brightness, 255, cv2.THRESH_BINARY)
    diff_gray_filtered = cv2.bitwise_and(diff_gray, diff_gray, mask=dark_mask)
    save_debug_image(diff_gray_filtered, "04_diff_brightness_filtered", output_dir)
    
    # STEP 2: Remove small artifacts
    if noise_kernel_size > 0:
        print(f"[*] Removing small artifacts (kernel={noise_kernel_size}x{noise_kernel_size})...")
        # kernel = np.ones((noise_kernel_size, noise_kernel_size), np.uint8)
        kernel = np.ones((2, 2), np.uint8)
        # Opening = erosion puis dilation (enlève petits objets)
        diff_gray_clean = cv2.morphologyEx(diff_gray_filtered, cv2.MORPH_OPEN, kernel)
        # diff_gray_clean = diff_gray_filtered
        save_debug_image(diff_gray_clean, "04b_diff_grey_cleaned used by mask", output_dir)
    else:
        diff_gray_clean = diff_gray_filtered
    
    # STEP 3: Threshold to get minimap mask
    _, mask = cv2.threshold(diff_gray_clean, threshold_value, 255, cv2.THRESH_BINARY)
    save_debug_image(mask, "05_mask_thresholded", output_dir)
    
    # STEP 4: Squelettisation to get clean contour
    print("🔍 Applying skeletonization to refine mask...")
    try:
        # Try using cv2.ximgproc if available (requires opencv-contrib-python)
        skeleton = cv2.ximgproc.thinning(mask, thinningType=cv2.ximgproc.THINNING_ZHANGSUEN)
    except AttributeError:
        # Fallback: manual thinning using morphological operations
        print("   ⚠️  cv2.ximgproc not available, using morphological thinning")
        kernel = np.ones((3, 3), np.uint8)
        skeleton = cv2.morphologyEx(mask, cv2.MORPH_GRADIENT, kernel)
    
    save_debug_image(skeleton, "05b_skeleton", output_dir)
    
    # Extract skeleton points
    skeleton_points = np.column_stack(np.where(skeleton > 0))
    print(f"   Extracted {len(skeleton_points)} skeleton points")
    
    # Create improved mask using convex hull of skeleton points
    if len(skeleton_points) > 3:
        # Swap columns for cv2 (needs x,y not y,x)
        skeleton_points_xy = skeleton_points[:, [1, 0]]
        hull = cv2.convexHull(skeleton_points_xy)
        
        # Create refined mask from convex hull
        mask_refined = np.zeros_like(mask)
        cv2.drawContours(mask_refined, [hull], -1, 255, -1)
        save_debug_image(mask_refined, "05c_mask_refined_hull", output_dir)
        
        # Visualize hull overlay
        hull_vis = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        cv2.drawContours(hull_vis, [hull], -1, (0, 255, 0), 2)
        save_debug_image(hull_vis, "05d_hull_overlay", output_dir)
        
        print(f"   ✓ Created convex hull with {len(hull)} points")
        mask = mask_refined  # Use refined mask
    else:
        print(f"   ⚠️  Not enough skeleton points, using original mask")
    
    # Dilate mask to connect nearby regions
    # if dilate_iterations > 0:
    #     kernel = np.ones((5, 5), np.uint8)
    #     mask = cv2.dilate(mask, kernel, iterations=dilate_iterations)
    #     save_debug_image(mask, "01c2_mask_after_dilate", output_dir)
    
    # Clean isolated pixels from grayscale difference before final save
    # print("🔍 Cleaning isolated pixels from difference...")
    kernel_clean = np.ones((1, 1), np.uint8)
    diff_gray_final = cv2.morphologyEx(diff_gray_clean, cv2.MORPH_OPEN, kernel_clean, iterations=2)
    save_debug_image(diff_gray_final, "06_diff_cleaned", output_dir)
    
    # Find contours and visualize
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if contours:
        print(f"🔍 Found {len(contours)} contours")
        
        # Visualize ALL contours with their areas
        contours_vis = cv2.cvtColor(diff_gray_final, cv2.COLOR_GRAY2BGR)
        for i, contour in enumerate(contours):
            area = cv2.contourArea(contour)
            if area > 1000:  # Only show significant contours
                color = (0, 255, 0) if i == 0 else (0, 165, 255)  # Green for largest, orange for others
                cv2.drawContours(contours_vis, [contour], -1, color, 2)
                x, y, w, h = cv2.boundingRect(contour)
                cv2.putText(contours_vis, f"#{i+1}: {area:.0f}px", (x, y-5),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
                print(f"   Contour #{i+1}: area={area:.0f}px, bbox=({x},{y},{w},{h})")
        
        save_debug_image(contours_vis, "07_all_contours", output_dir)
        
        # Get largest contour
        largest_contour = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(largest_contour)
        print(f"✓ Largest contour selected: x={x}, y={y}, w={w}, h={h}")
        
        # Draw bounding box of largest
        diff_vis = cv2.cvtColor(diff_gray_final, cv2.COLOR_GRAY2BGR)
        cv2.rectangle(diff_vis, (x, y), (x+w, y+h), (0, 255, 0), 2)
        save_debug_image(diff_vis, "08_largest_contour_bbox", output_dir)
    
    print("   Skeleton extraction complete, ready for matching")
    return skeleton, hull if len(skeleton_points) > 3 else None, mask


def remove_minimap_background(frame):
    """Remove pixels close to minimap background color #877657."""
    target_color = np.array([87, 118, 135], dtype=np.uint8)  # BGR
    tolerance = 30
    lower_bound = np.clip(target_color - tolerance, 0, 255)
    upper_bound = np.clip(target_color + tolerance, 0, 255)
    
    mask = cv2.inRange(frame, lower_bound, upper_bound)
    mask_inv = cv2.bitwise_not(mask)
    result = cv2.bitwise_and(frame, frame, mask=mask_inv)
    return result


# ============================================================================
# STEP 2: SKELETON-BASED LOCALIZATION
# ============================================================================

def localize_player_with_skeleton(frame_with_minimap, skeleton, hull, zone_name, output_dir):
    """Localize player using skeleton matching with static map."""
    print("\n" + "=" * 70)
    print("STEP 2: Skeleton-Based Localization")
    print("=" * 70)
    
    # Load static map
    static_map_path = load_zone_static_map(zone_name)
    if static_map_path is None:
        print(f"⚠️  No static map found for zone: {zone_name}")
        return None, None
    
    print(f"✓ Loading static map: {static_map_path}")
    static_map = cv2.imread(str(static_map_path))
    save_debug_image(static_map, "06_static_map_reference", output_dir)
    
    # Extract minimap region using hull
    print("🔍 Extracting minimap region from hull...")
    x, y, w, h = cv2.boundingRect(hull)
    minimap = frame_with_minimap[y:y+h, x:x+w].copy()
    print(f"   Minimap size: {w}x{h}")
    save_debug_image(minimap, "07_minimap_extracted", output_dir)
    
    # Convert minimap to grayscale and extract edges
    print("🔍 Extracting edges from minimap...")
    minimap_gray = cv2.cvtColor(minimap, cv2.COLOR_BGR2GRAY)
    minimap_edges = cv2.Canny(minimap_gray, 50, 150)
    save_debug_image(minimap_edges, "08_minimap_edges", output_dir)
    
    # Convert static map to grayscale and extract edges
    print("🔍 Extracting edges from static map...")
    static_gray = cv2.cvtColor(static_map, cv2.COLOR_BGR2GRAY)
    static_edges = cv2.Canny(static_gray, 50, 150)
    save_debug_image(static_edges, "09_static_edges", output_dir)
    
    # Try multi-scale template matching
    print("\n🔍 Attempting multi-scale edge matching...")
    best_match = None
    best_confidence = 0
    best_scale = 1.0
    
    scales = [0.5, 0.75, 1.0, 1.25, 1.5, 2.0]
    
    for scale in scales:
        # Resize minimap edges
        scaled_w = int(w * scale)
        scaled_h = int(h * scale)
        
        if scaled_w > static_edges.shape[1] or scaled_h > static_edges.shape[0]:
            continue
        
        minimap_scaled = cv2.resize(minimap_edges, (scaled_w, scaled_h))
        
        # Template matching
        result = cv2.matchTemplate(static_edges, minimap_scaled, cv2.TM_CCOEFF_NORMED)
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
        
        print(f"   Scale {scale:.2f}: confidence={max_val:.3f} at ({max_loc[0]}, {max_loc[1]})")
        
        if max_val > best_confidence:
            best_confidence = max_val
            best_match = max_loc
            best_scale = scale
    
    if best_confidence > 0.3:  # Lower threshold for skeleton matching
        print(f"\n✓ MATCH FOUND!")
        print(f"  Position: {best_match}")
        print(f"  Confidence: {best_confidence:.3f}")
        print(f"  Scale: {best_scale:.2f}")
        
        # Calculate player position (center of minimap match)
        scaled_w = int(w * best_scale)
        scaled_h = int(h * best_scale)
        player_x = best_match[0] + scaled_w // 2
        player_y = best_match[1] + scaled_h // 2
        
        # Visualize result
        result_vis = static_map.copy()
        cv2.rectangle(result_vis, best_match, 
                     (best_match[0] + scaled_w, best_match[1] + scaled_h), 
                     (0, 255, 0), 3)
        cv2.circle(result_vis, (player_x, player_y), 10, (255, 0, 0), -1)
        cv2.putText(result_vis, f"PLAYER ({player_x}, {player_y})", 
                   (player_x + 15, player_y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
        cv2.putText(result_vis, f"Confidence: {best_confidence:.3f}", 
                   (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        
        save_debug_image(result_vis, "10_localization_result", output_dir)
        
        return (player_x, player_y), best_confidence
    else:
        print(f"\n❌ LOCALIZATION FAILED")
        print(f"  Best confidence: {best_confidence:.3f} (threshold: 0.3)")
        return None, best_confidence


def load_pois(static_map_path):
    """Load POIs from annotations file."""
    annotations_path = static_map_path.parent / f"{static_map_path.stem}_annotations.json"
    pois = []
    
    if annotations_path.exists():
        with open(annotations_path, 'r') as f:
            annotations = json.load(f)
            pois = annotations.get('pois', [])
        print(f"✓ Loaded {len(pois)} POIs from annotations")
    else:
        print(f"⚠️  No annotations found: {annotations_path}")
    
    return pois


# ============================================================================
# REMOVED: Old extraction and processing functions
# ============================================================================


# ============================================================================
# MAIN FUNCTION
# ============================================================================

def main():
    """Main test function."""
    print("=" * 70)
    print("STATIC MAP LOCALIZATION TEST")
    print("=" * 70)
    print()
    
    # Create output directory
    output_dir = Path(__file__).parent / "data" / "screenshots" / "outputs" / "localization_test"
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"[*] Debug images will be saved to: {output_dir}")
    print()
    
    # Initialize
    try:
        print("=" * 70)
        print("STEP 1: Screen Capture")
        print("=" * 70)
        
        capture = WindowsScreenCapture()
        print(f"✓ Found window: {capture.window_title}")
        
        ui_vision = UIVisionModule(debug=True)
        print("\n[*] Verifying minimap is HIDDEN...")
        
        # Ensure minimap is hidden
        ensure_minimap_hidden(capture, ui_vision)
        print("✓ Minimap should now be hidden")
        
        # Capture background
        frame_without_minimap = capture_background_frame(capture, output_dir)
        if frame_without_minimap is None:
            return
        
        # Show minimap and capture
        frame_with_minimap = show_minimap_and_capture(capture, ui_vision, output_dir)
        if frame_with_minimap is None:
            return
        
        # Detect zone first
        ui_vision_zone = UIVisionModule(debug=False)
        zone_name = ui_vision_zone.extract_zone(frame_with_minimap)
        if not zone_name:
            zone_name = 'UNKNOWN'
        print(f"\n✓ Current zone: {zone_name}")
        
        # Compute difference and extract skeleton
        skeleton, hull, mask = compute_minimap_difference(
            frame_without_minimap, 
            frame_with_minimap, 
            output_dir,
            min_brightness=15,
            noise_kernel_size=3,
            threshold_value=10,
            dilate_iterations=3
        )
        
        if hull is None:
            print("[X] Failed to extract skeleton hull")
            return
    
    except Exception as e:
        print(f"[X] Failed: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Localize player using skeleton matching
    player_pos, confidence = localize_player_with_skeleton(
        frame_with_minimap,
        skeleton,
        hull,
        zone_name,
        output_dir
    )
    
    # Done
    print()
    print("=" * 70)
    print("[OK] TEST COMPLETE")
    print("=" * 70)
    print(f"\n[*] All debug images saved to: {output_dir}")
    if player_pos:
        print(f"[+] Player position: {player_pos}")
        print(f"    Confidence: {confidence:.3f}")
    print()


if __name__ == "__main__":
    main()
