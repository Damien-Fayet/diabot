#!/usr/bin/env python3
"""
Test script for static map localization - Isometric structure-based approach.
Uses oriented filtering to detect isometric game structures (fences, walls, paths).
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
from diabot.navigation.static_map_localizer import load_zone_static_map
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
# STEP 1: FRAME CAPTURE (same as v2)
# ============================================================================

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
    print("\n[Camera icon] Capturing frame WITHOUT minimap...")
    time.sleep(0.2)
    frame = capture.get_frame()
    print(f"OK Captured background frame: {frame.shape}")
    save_debug_image(frame, "01_frame_without_minimap", output_dir)
    return frame


def show_minimap_and_capture(capture, ui_vision, output_dir):
    """Show minimap and capture frame."""
    print("\n[Keyboard icon] Pressing TAB to show minimap...")
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
    
    print("\n[Camera icon] Capturing frame WITH minimap...")
    frame = capture.get_frame()
    print(f"OK Captured frame with minimap: {frame.shape}")
    save_debug_image(frame, "02_frame_with_minimap", output_dir)
    return frame


# ============================================================================
# STEP 2: DIFFERENCE & BASIC EXTRACTION
# ============================================================================

def compute_minimap_difference(frame_without, frame_with, output_dir):
    """Compute difference to isolate minimap region."""
    print("\n" + "="*70)
    print("STEP 2: Difference Computation")
    print("="*70)
    
    # Directional difference
    diff = cv2.subtract(frame_with, frame_without)
    save_debug_image(diff, "03_diff_raw", output_dir)
    
    print(f"OK Difference computed: {diff.shape}")
    # Convert to grayscale for preprocessing
    diff_gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
    # STEP 1: Filter dark pixels (likely noise or shadows)
    min_brightness = 20
    print(f"🔍 Filtering pixels darker than {min_brightness}...")
    _, dark_mask = cv2.threshold(diff_gray, min_brightness, 255, cv2.THRESH_BINARY)
    diff_gray_filtered = cv2.bitwise_and(diff_gray, diff_gray, mask=dark_mask)
    save_debug_image(diff_gray_filtered, "04_diff_brightness_filtered", output_dir)
    
    # STEP 2: Remove small artifacts
    noise_kernel_size = 3
    if noise_kernel_size > 0:
        print(f"[*] Removing small artifacts (kernel={noise_kernel_size}x{noise_kernel_size})...")
        # kernel = np.ones((noise_kernel_size, noise_kernel_size), np.uint8)
        kernel = np.ones((1, 1), np.uint8)
        # Opening = erosion puis dilation (enlève petits objets)
        diff_gray_clean = cv2.morphologyEx(diff_gray_filtered, cv2.MORPH_OPEN, kernel)
        # diff_gray_clean = diff_gray_filtered
        save_debug_image(diff_gray_clean, "04b_diff_grey_cleaned used by mask", output_dir)
    else:
        diff_gray_clean = diff_gray_filtered
    return diff_gray_clean


# ============================================================================
# STEP 3: ISOMETRIC-ORIENTED FILTERING
# ============================================================================

def create_gabor_kernels(angles=[0, 30, 60, 90, 120, 150]):
    """Create Gabor filters for isometric angles."""
    kernels = []
    for theta in angles:
        theta_rad = theta * np.pi / 180
        kernel = cv2.getGaborKernel(
            ksize=(21, 21),
            sigma=5.0,
            theta=theta_rad,
            lambd=10.0,
            gamma=0.5,
            psi=0
        )
        kernels.append(kernel)
    return kernels


def apply_oriented_filtering(image, output_dir):
    """Apply Gabor filtering for isometric angles."""
    print("\n" + "="*70)
    print("STEP 3: Oriented Filtering (Isometric Angles)")
    print("="*70)
    
    # Convert to grayscale if needed
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()
    
    # Create Gabor kernels for isometric angles (0°, 30°, 60°, 90°, 120°, 150°)
    # angles = [0, 30, 60, 90, 120, 150]
    angles = [ 60, 120]
    print(f"   Creating Gabor filters for angles: {angles}")
    kernels = create_gabor_kernels(angles)
    
    # Apply each filter and combine
    filtered_images = []
    for i, (kernel, angle) in enumerate(zip(kernels, angles)):
        filtered = cv2.filter2D(gray, cv2.CV_32F, kernel)
        filtered = np.abs(filtered)
        filtered_images.append(filtered)
        
        # Save individual angle
        filtered_vis = cv2.normalize(filtered, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        save_debug_image(filtered_vis, f"08a_gabor_{angle:03d}deg", output_dir)
    
    # Combine all orientations (max response)
    combined = np.maximum.reduce(filtered_images)
    combined_vis = cv2.normalize(combined, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    save_debug_image(combined_vis, "08b_gabor_combined", output_dir)
    
    print("OK Oriented filtering complete")
    return combined_vis


# ============================================================================
# STEP 4: ADAPTIVE THRESHOLDING
# ============================================================================

def apply_adaptive_threshold(image, output_dir):
    """Apply adaptive thresholding for stable edges."""
    print("\n" + "="*70)
    print("STEP 4: Adaptive Thresholding")
    print("="*70)
    
    # Gaussian blur to reduce texture noise
    blurred = cv2.GaussianBlur(image, (5, 5), 0)
    save_debug_image(blurred, "09a_blurred", output_dir)
    
    # Adaptive threshold
    # C parameter: higher = stricter (fewer white pixels)
    adaptive = cv2.adaptiveThreshold(
        blurred,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,  # INV = lines become white on black background
        blockSize=15,  # Larger block for more context
        C=6  # Higher C = stricter threshold
    )
    save_debug_image(adaptive, "09b_adaptive_threshold", output_dir)
    
    print("OK Adaptive thresholding complete")
    return adaptive


# ============================================================================
# STEP 5: ORIENTED MORPHOLOGICAL CLOSING
# ============================================================================

def create_oriented_kernels():
    """Create morphological kernels that follow true isometric wall angles (60° and 120°)."""
    kernels = {}
    size = 21  # Larger size for better angle precision
    
    # For 60°: tan(60°) ≈ 1.732 ≈ 3/2 ratio
    # This means: for every 2 pixels RIGHT (j+2), go 3 pixels UP (i-3)
    kernel_60 = np.zeros((size, size), dtype=np.uint8)
    for step in range(size):
        j = step * 2 // 3  # Horizontal position
        i = size - 1 - step  # Vertical position (going up)
        if 0 <= i < size and 0 <= j < size:
            kernel_60[i, j] = 1
            # Thicken the line slightly
            if i + 1 < size:
                kernel_60[i + 1, j] = 1
    kernels['isometric_60'] = kernel_60
    
    # For 120°: tan(120°) ≈ -1.732 ≈ -3/2 ratio  
    # This means: for every 2 pixels RIGHT (j+2), go 3 pixels DOWN (i+3)
    kernel_120 = np.zeros((size, size), dtype=np.uint8)
    for step in range(size):
        j = step * 2 // 3  # Horizontal position
        i = step  # Vertical position (going down)
        if 0 <= i < size and 0 <= j < size:
            kernel_120[i, j] = 1
            # Thicken the line slightly
            if i - 1 >= 0:
                kernel_120[i - 1, j] = 1
    kernels['isometric_120'] = kernel_120
    
    return kernels


def apply_oriented_closing(image, output_dir):
    """Apply morphological closing with oriented kernels."""
    print("\n" + "="*70)
    print("STEP 5: Oriented Morphological Closing")
    print("="*70)
    
    kernels = create_oriented_kernels()
    
    # Apply closing with each oriented kernel
    closed_images = []
    for name, kernel in kernels.items():
        closed = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)
        closed_images.append(closed)
        save_debug_image(closed, f"10a_closed_{name}", output_dir)
        print(f"   Applied {name} kernel")
    
    # Combine all closings (union = max)
    combined = np.maximum.reduce(closed_images)
    save_debug_image(combined, "10b_closed_combined", output_dir)
    
    print("OK Oriented closing complete")
    return combined


# ============================================================================
# STEP 6: PROBABILISTIC HOUGH TRANSFORM
# ============================================================================

def detect_lines_hough(image, output_dir):
    """Detect lines using probabilistic Hough transform."""
    print("\n" + "="*70)
    print("STEP 6: Probabilistic Hough Line Detection")
    print("="*70)
    
    # Hough line detection
    lines = cv2.HoughLinesP(
        image,
        rho=1,
        theta=np.pi/60,
        threshold=50,
        minLineLength=50,
        maxLineGap=20
    )
    
    if lines is None:
        print("[!] No lines detected")
        return None, None
    
    print(f"OK Detected {len(lines)} line segments")
    
    # Visualize detected lines
    vis = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    for line in lines:
        x1, y1, x2, y2 = line[0]
        cv2.line(vis, (x1, y1), (x2, y2), (0, 255, 0), 2)
    
    save_debug_image(vis, "11a_hough_lines_raw", output_dir)
    
    return lines, vis


# ============================================================================
# STEP 7: LINE FUSION & CLEANUP
# ============================================================================

def calculate_line_angle(x1, y1, x2, y2):
    """Calculate angle of line in degrees."""
    angle = np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi
    # Normalize to [0, 180)
    if angle < 0:
        angle += 180
    return angle


def are_lines_collinear(line1, line2, angle_threshold=10, distance_threshold=30):
    """Check if two lines are collinear and close."""
    x1, y1, x2, y2 = line1
    x3, y3, x4, y4 = line2
    
    # Check angle similarity
    angle1 = calculate_line_angle(x1, y1, x2, y2)
    angle2 = calculate_line_angle(x3, y3, x4, y4)
    angle_diff = min(abs(angle1 - angle2), 180 - abs(angle1 - angle2))
    
    if angle_diff > angle_threshold:
        return False
    
    # Check distance between line segments
    # Distance from point to line
    def point_to_line_dist(px, py, x1, y1, x2, y2):
        num = abs((y2-y1)*px - (x2-x1)*py + x2*y1 - y2*x1)
        den = np.sqrt((y2-y1)**2 + (x2-x1)**2)
        return num / (den + 1e-6)
    
    dist1 = point_to_line_dist(x3, y3, x1, y1, x2, y2)
    dist2 = point_to_line_dist(x4, y4, x1, y1, x2, y2)
    
    return max(dist1, dist2) < distance_threshold


def merge_collinear_lines(lines):
    """Merge collinear and close lines."""
    if lines is None or len(lines) == 0:
        return None
    
    merged = []
    used = [False] * len(lines)
    
    for i in range(len(lines)):
        if used[i]:
            continue
        
        x1, y1, x2, y2 = lines[i][0]
        group = [(x1, y1), (x2, y2)]
        used[i] = True
        
        # Find all collinear lines
        for j in range(i+1, len(lines)):
            if used[j]:
                continue
            
            if are_lines_collinear(lines[i][0], lines[j][0]):
                x3, y3, x4, y4 = lines[j][0]
                group.extend([(x3, y3), (x4, y4)])
                used[j] = True
        
        # Find extremities of the group
        if len(group) > 0:
            points = np.array(group)
            # Fit line to all points
            vx, vy, x, y = cv2.fitLine(points, cv2.DIST_L2, 0, 0.01, 0.01)
            
            # Extract scalar values from arrays
            vx, vy, x, y = float(vx[0]), float(vy[0]), float(x[0]), float(y[0])
            
            # Project all points on the line and find extremities
            projections = []
            for px, py in points:
                t = (px - x) * vx + (py - y) * vy
                projections.append(t)
            
            t_min, t_max = min(projections), max(projections)
            
            # Calculate extremity points
            x1_new = int(x + t_min * vx)
            y1_new = int(y + t_min * vy)
            x2_new = int(x + t_max * vx)
            y2_new = int(y + t_max * vy)
            
            merged.append([[x1_new, y1_new, x2_new, y2_new]])
    
    return np.array(merged)


def fuse_and_clean_lines(lines, image_shape, output_dir):
    """Fuse collinear lines and clean up."""
    print("\n" + "="*70)
    print("STEP 7: Line Fusion & Cleanup")
    print("="*70)
    
    if lines is None:
        return None
    
    print(f"   Input: {len(lines)} line segments")
    
    # Merge collinear lines
    merged_lines = merge_collinear_lines(lines)
    
    if merged_lines is None:
        print("[!] No lines after merging")
        return None
    
    print(f"OK Output: {len(merged_lines)} merged lines")
    
    # Visualize merged lines
    vis = np.zeros((*image_shape[:2], 3), dtype=np.uint8)
    
    # Group by angle and color-code
    angles = []
    for line in merged_lines:
        x1, y1, x2, y2 = line[0]
        angle = calculate_line_angle(x1, y1, x2, y2)
        angles.append(angle)
    
    angles = np.array(angles)
    
    # Color by orientation
    for i, line in enumerate(merged_lines):
        x1, y1, x2, y2 = line[0]
        angle = angles[i]
        
        # Color-code by angle range
        if angle < 15 or angle > 165:  # Horizontal
            color = (0, 0, 255)  # Red
        elif 75 <= angle <= 105:  # Vertical
            color = (255, 0, 0)  # Blue
        elif 30 <= angle <= 60:  # Diagonal 1
            color = (0, 255, 0)  # Green
        elif 120 <= angle <= 150:  # Diagonal 2
            color = (0, 255, 255)  # Yellow
        else:
            color = (128, 128, 128)  # Gray
        
        cv2.line(vis, (x1, y1), (x2, y2), color, 2)
    
    save_debug_image(vis, "11b_hough_lines_merged", output_dir)
    
    return merged_lines


# ============================================================================
# STEP 8: MATCH WITH STATIC MAP
# ============================================================================

def create_line_image(lines, shape):
    """Create binary image from line segments."""
    img = np.zeros(shape[:2], dtype=np.uint8)
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(img, (x1, y1), (x2, y2), 255, 2)
    return img


def match_with_static_map(minimap_lines, zone_name, image_shape, output_dir):
    """Match minimap line structure with static map."""
    print("\n" + "="*70)
    print("STEP 8: Match with Static Map")
    print("="*70)
    
    # Load static map
    static_map_path = load_zone_static_map(zone_name)
    if static_map_path is None:
        print(f"[!] No static map found for zone: {zone_name}")
        return None, 0.0
    
    # Read the image
    static_map = cv2.imread(str(static_map_path))
    if static_map is None:
        print(f"[!] Failed to load static map: {static_map_path}")
        return None, 0.0
    
    save_debug_image(static_map, "12_static_map_reference", output_dir)
    
    # Process static map with same pipeline (simplified)
    static_gray = cv2.cvtColor(static_map, cv2.COLOR_BGR2GRAY)
    
    # Apply edge detection
    static_edges = cv2.Canny(static_gray, 50, 150)
    save_debug_image(static_edges, "13_static_edges", output_dir)
    
    # Detect lines in static map
    static_lines = cv2.HoughLinesP(
        static_edges,
        rho=1,
        theta=np.pi/180,
        threshold=50,
        minLineLength=30,
        maxLineGap=20
    )
    
    if static_lines is None:
        print("[!] No lines detected in static map")
        return None, 0.0
    
    print(f"OK Detected {len(static_lines)} lines in static map")
    
    # Create line images for template matching
    minimap_line_img = create_line_image(minimap_lines, image_shape)
    save_debug_image(minimap_line_img, "14_minimap_line_structure", output_dir)
    
    static_line_img = create_line_image(static_lines, static_map.shape)
    save_debug_image(static_line_img, "15_static_line_structure", output_dir)
    
    # Multi-scale template matching
    print("\n   Attempting multi-scale matching...")
    scales = [0.5, 0.75, 1.0, 1.25, 1.5]
    best_match = None
    best_confidence = 0.0
    
    for scale in scales:
        if scale != 1.0:
            scaled = cv2.resize(minimap_line_img, None, fx=scale, fy=scale)
        else:
            scaled = minimap_line_img
        
        if scaled.shape[0] > static_line_img.shape[0] or scaled.shape[1] > static_line_img.shape[1]:
            continue
        
        result = cv2.matchTemplate(static_line_img, scaled, cv2.TM_CCOEFF_NORMED)
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
        
        print(f"      Scale {scale:.2f}: confidence={max_val:.3f}")
        
        if max_val > best_confidence:
            best_confidence = max_val
            best_match = (max_loc, scale)
    
    # Visualize result
    if best_match and best_confidence > 0.3:
        pos, scale = best_match
        h, w = int(image_shape[0] * scale), int(image_shape[1] * scale)
        
        result_vis = static_map.copy()
        cv2.rectangle(result_vis, pos, (pos[0] + w, pos[1] + h), (0, 255, 0), 3)
        
        # Player position (center of matched region)
        player_x = pos[0] + w // 2
        player_y = pos[1] + h // 2
        cv2.circle(result_vis, (player_x, player_y), 10, (0, 0, 255), -1)
        
        save_debug_image(result_vis, "16_localization_result", output_dir)
        
        print(f"\nOK LOCALIZATION SUCCESS")
        print(f"   Position: ({player_x}, {player_y})")
        print(f"   Confidence: {best_confidence:.3f}")
        print(f"   Scale: {scale:.2f}")
        
        return (player_x, player_y), best_confidence
    else:
        print(f"\n[X] LOCALIZATION FAILED")
        print(f"   Best confidence: {best_confidence:.3f} (threshold: 0.3)")
        return None, best_confidence


# ============================================================================
# MAIN FUNCTION
# ============================================================================

def main():
    """Main test function."""
    print("=" * 70)
    print("STATIC MAP LOCALIZATION TEST v3 - Isometric Structure Detection")
    print("=" * 70)
    print()
    
    # Create output directory
    output_dir = Path(__file__).parent / "data" / "screenshots" / "outputs" / "localization_test_v3"
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"[*] Debug images will be saved to: {output_dir}")
    print()
    
    # Initialize
    try:
        print("=" * 70)
        print("STEP 1: Screen Capture")
        print("=" * 70)
        
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
        
        # Detect zone first
        ui_vision_zone = UIVisionModule(debug=False)
        zone_name = ui_vision_zone.extract_zone(frame_with_minimap)
        if not zone_name:
            zone_name = 'UNKNOWN'
        print(f"\nOK Current zone: {zone_name}")
        
        # Compute difference
        diff_image = compute_minimap_difference(
            frame_without_minimap,
            frame_with_minimap,
            output_dir
        )
        
        if diff_image is None:
            print("[X] Failed to compute difference")
            return
        
        # Apply isometric-oriented processing on full diff image
        oriented = apply_oriented_filtering(diff_image, output_dir)
        adaptive = apply_adaptive_threshold(oriented, output_dir)
        closed = apply_oriented_closing(adaptive, output_dir)
        lines, _ = detect_lines_hough(closed, output_dir)
        merged_lines = fuse_and_clean_lines(lines, diff_image.shape, output_dir)
        
        # Match with static map
        player_pos, confidence = match_with_static_map(
            merged_lines,
            zone_name,
            diff_image.shape,
            output_dir
        )
        
    except Exception as e:
        print(f"[X] Failed: {e}")
        import traceback
        traceback.print_exc()
        return
    
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
