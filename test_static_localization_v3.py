#!/usr/bin/env python3
"""
Test script for static map localization - Isometric structure-based approach.
Uses oriented filtering to detect isometric game structures (fences, walls, paths).

Usage:
  python test_static_localization_v3.py           # Live capture mode
  python test_static_localization_v3.py --static   # Use static images from inputs/
"""
import sys
from pathlib import Path
import cv2
import numpy as np
from datetime import datetime
import time
import pyautogui
import json
import argparse

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
    
    # STEP 3: Remove bright crosses (NPCs) by thresholding very bright pixels
    print(f"[*] Removing bright NPC crosses...")
    
    # Debug: show actual brightness distribution
    print(f"   Brightness range in cleaned diff: [{diff_gray_clean.min()}, {diff_gray_clean.max()}]")
    
    # Threshold to identify bright NPC crosses
    npc_threshold = 150  # Threshold for NPC crosses
    _, npc_mask = cv2.threshold(diff_gray_clean, npc_threshold, 255, cv2.THRESH_BINARY)
    
    # Debug: count how many pixels are above threshold
    npc_pixels = np.count_nonzero(npc_mask)
    print(f"   NPC mask - pixels >= {npc_threshold}: {npc_pixels}")
    
    # Save NPC mask for debugging
    save_debug_image(npc_mask, "04c_npc_detected_mask", output_dir)
    
    # Directly subtract NPC pixels from the image (set them to black)
    diff_gray_no_npc = cv2.bitwise_and(diff_gray_clean, cv2.bitwise_not(npc_mask))
    print(f"   NPCs removed from image")
    
    save_debug_image(diff_gray_no_npc, "04d_diff_no_npc", output_dir)
    
    return diff_gray_no_npc


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
    
    _, white_mask = cv2.threshold(image, 40, 255, cv2.THRESH_BINARY)
    save_debug_image(white_mask, "09_thresh", output_dir)
    # Gaussian blur to reduce texture noise
    blurred = cv2.GaussianBlur(white_mask, (5, 5), 0)
    save_debug_image(blurred, "09a_blurred", output_dir)
    
    # Adaptive threshold
    # C parameter: higher = stricter (fewer white pixels)
    # adaptive = cv2.adaptiveThreshold(
    #     blurred,
    #     255,
    #     cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
    #     cv2.THRESH_BINARY_INV,  # INV = lines become white on black background
    #     blockSize=15,  # Larger block for more context
    #     C=8  # Higher C = stricter threshold
    # )
    # save_debug_image(adaptive, "09b_adaptive_threshold", output_dir)
    
    # Thicken parallel lines: dilate to merge nearby edges
    print("   Thickening parallel lines with morphological dilation...")
    kernel_dilate = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    adaptive_thick = cv2.dilate(blurred, kernel_dilate, iterations=1)
    save_debug_image(adaptive_thick, "09c_adaptive_thickened", output_dir)
    
    # Close gaps with morphological closing
    # print("   Closing gaps between lines...")
    # kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    # adaptive_closed = cv2.morphologyEx(adaptive_thick, cv2.MORPH_CLOSE, kernel_close, iterations=1)
    # save_debug_image(adaptive_closed, "09d_adaptive_closed", output_dir)
    
    print("OK Adaptive thresholding complete")
    return blurred, adaptive_thick


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
        j = step * 4 // 3  # Horizontal position
        i = size - 1 - step  # Vertical position (going up)
        if 0 <= i < size and 0 <= j < size:
            kernel_60[i, j] = 1
            # Thicken the line slightly
            if i + 1 < size:
                kernel_60[i + 1, j] = 1
    kernels['isometric_60'] = kernel_60
    
    # For 120°: tan(120°) ≈ -1.732 ≈ -3/2 ratio  
    # This means: for every 4 pixels RIGHT (j+4), go 3 pixels DOWN (i+3)
    kernel_120 = np.zeros((size, size), dtype=np.uint8)
    for step in range(size):
        j = step * 4 // 3  # Horizontal position
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


def prepare_edges_for_ecc(img, blur_kernel=5):
    """Prepare edge image for ECC alignment.
    
    Args:
        img: Input image (can be binary or grayscale)
        blur_kernel: Gaussian blur kernel size
        
    Returns:
        Normalized float32 image (0-1) with smoothed edges
    """
    # Ensure grayscale
    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img.copy()
    
    # Smooth the edges
    smoothed = cv2.GaussianBlur(gray, (blur_kernel, blur_kernel), 1.0)
    
    # Normalize to [0, 1] float32
    normalized = smoothed.astype(np.float32) / 255.0
    
    return normalized


def build_image_pyramid(img, levels=4):
    """Build a Gaussian pyramid for multi-scale alignment.
    
    Args:
        img: Input image (float32)
        levels: Number of pyramid levels
        
    Returns:
        List of pyramid levels (coarse to fine)
    """
    pyramid = [img]
    current = img.copy()
    
    for i in range(levels - 1):
        current = cv2.pyrDown(current)
        pyramid.append(current)
    
    return pyramid[::-1]  # Return coarse-to-fine


def align_with_ecc(query_img, ref_img, motion_type='AFFINE', output_dir=None):
    """Align query_img to ref_img using ECC with multi-scale pyramid.
    
    Args:
        query_img: Query image (minimap edges, float32)
        ref_img: Reference image (static map edges, float32)
        motion_type: 'AFFINE' or 'HOMOGRAPHY'
        output_dir: Directory to save debug images
        
    Returns:
        (warp_matrix, cc_score) where warp_matrix aligns query to reference
    """
    print("\n   Building image pyramids...")
    
    # Build pyramids
    query_pyramid = build_image_pyramid(query_img, levels=4)
    ref_pyramid = build_image_pyramid(ref_img, levels=4)
    
    print(f"   Pyramid levels: {len(query_pyramid)}")
    
    # Set motion type
    if motion_type == 'AFFINE':
        warp_mode = cv2.MOTION_AFFINE
        warp_matrix = np.eye(2, 3, dtype=np.float32)
    else:  # HOMOGRAPHY
        warp_mode = cv2.MOTION_HOMOGRAPHY
        warp_matrix = np.eye(3, 3, dtype=np.float32)
    
    # Multi-scale ECC alignment (coarse to fine)
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 5000, 1e-6)
    
    for level in range(len(query_pyramid)):
        print(f"   Level {level+1}/{len(query_pyramid)} (scale {2**(len(query_pyramid)-level-1)})...")
        
        query_level = query_pyramid[level]
        ref_level = ref_pyramid[level]
        
        print(f"      Shapes - Query: {query_level.shape}, Ref: {ref_level.shape}")
        print(f"      Query range: [{query_level.min():.3f}, {query_level.max():.3f}]")
        print(f"      Ref range: [{ref_level.min():.3f}, {ref_level.max():.3f}]")
        
        try:
            cc, warp_matrix = cv2.findTransformECC(
                query_level,
                ref_level,
                warp_matrix,
                warp_mode,
                criteria
            )
            print(f"      ✓ ECC score: {cc:.4f}")
        except cv2.error as e:
            print(f"      [!] ECC failed: {str(e)[:100]}")
            # Continue anyway - keep previous warp_matrix
            continue
        
        # Scale warp_matrix for next level (only if it converged)
        if level < len(query_pyramid) - 1:
            if motion_type == 'AFFINE':
                warp_matrix[0, 2] *= 2
                warp_matrix[1, 2] *= 2
            else:
                warp_matrix[0, 2] *= 2
                warp_matrix[1, 2] *= 2
                warp_matrix[2, 0] *= 0.5
                warp_matrix[2, 1] *= 0.5
    
    return warp_matrix, cc if 'cc' in locals() else 0.0


def match_with_static_map_ecc(adaptive_img, zone_name, image_shape, output_dir):
    """Match adaptive image edges with static map using ECC alignment.
    
    Args:
        adaptive_img: Binary edge image from minimap difference
        zone_name: Name of the zone to find static map
        image_shape: Shape of the original image (H, W, C)
        output_dir: Directory to save debug images
        
    Returns:
        (player_x, player_y), confidence or None, 0.0
    """
    print("\n" + "="*70)
    print("STEP 9: Match with Static Map (ECC Multi-Scale)")
    print("="*70)
    
    # Load static map
    static_map_path = load_zone_static_map(zone_name)
    if static_map_path is None:
        print(f"[!] No static map found for zone: {zone_name}")
        return None, 0.0
    
    # Read static map
    static_map = cv2.imread(str(static_map_path))
    if static_map is None:
        print(f"[!] Failed to load static map: {static_map_path}")
        return None, 0.0
    
    save_debug_image(static_map, "12_static_map_reference", output_dir)
    print(f"OK Loaded static map: {static_map.shape}")
    
    # Convert static map to edges
    static_gray = cv2.cvtColor(static_map, cv2.COLOR_BGR2GRAY)
    static_edges = cv2.Canny(static_gray, 50, 150)
    save_debug_image(static_edges, "13_static_edges", output_dir)
    print(f"OK Extracted edges from static map: {static_edges.shape}")
    print(f"   Static edges - Non-zero: {np.count_nonzero(static_edges)}")
    
    # Prepare edges for ECC (normalize, smooth)
    print("\n   Preparing images for ECC alignment...")
    print(f"   Input adaptive_img: shape={adaptive_img.shape}, min={adaptive_img.min()}, max={adaptive_img.max()}, non-zero={np.count_nonzero(adaptive_img)}")
    
    minimap_edges = cv2.Canny(adaptive_img, 50, 150)
    save_debug_image(minimap_edges, "13_minimap_canny_edges", output_dir)
    print(f"   Minimap Canny edges: {minimap_edges.shape}, non-zero={np.count_nonzero(minimap_edges)}")
    
    query_ecc = prepare_edges_for_ecc(minimap_edges, blur_kernel=5)
    ref_ecc = prepare_edges_for_ecc(static_edges, blur_kernel=5)
    
    print(f"   Query ECC shape: {query_ecc.shape}, Reference ECC shape: {ref_ecc.shape}")
    
    # CRITICAL: Resize query to match reference size (ECC needs compatible dimensions)
    if query_ecc.shape != ref_ecc.shape:
        print(f"   Resizing query from {query_ecc.shape} to {ref_ecc.shape}")
        query_ecc = cv2.resize(query_ecc, (ref_ecc.shape[1], ref_ecc.shape[0]))
    
    # Save preprocessed images to verify content
    query_vis = (query_ecc * 255).astype(np.uint8)
    ref_vis = (ref_ecc * 255).astype(np.uint8)
    save_debug_image(query_vis, "13a_minimap_edges_normalized", output_dir)
    save_debug_image(ref_vis, "13b_static_edges_normalized", output_dir)
    print(f"   Minimap edges - Min: {query_ecc.min():.3f}, Max: {query_ecc.max():.3f}, Non-zero: {np.count_nonzero(query_ecc > 0)}")
    print(f"   Static edges - Min: {ref_ecc.min():.3f}, Max: {ref_ecc.max():.3f}, Non-zero: {np.count_nonzero(ref_ecc > 0)}")
    
    # Determine motion model - start with AFFINE (more stable than HOMOGRAPHY)
    motion_type = 'HOMOGRAPHY'  # Try homography first for better accuracy, fallback to affine if it fails
    print(f"   Using {motion_type} motion model")
    
    # Perform multi-scale ECC alignment
    warp_matrix, cc_score = align_with_ecc(query_ecc, ref_ecc, motion_type, output_dir)
    
    print(f"\nOK ECC alignment complete (score: {cc_score:.4f})")
    
    # Debug: print transformation matrix
    print(f"\n   Transformation matrix ({motion_type}):")
    print(f"   {warp_matrix}")
    
    # Check if warp_matrix is identity (no transformation found)
    if motion_type == 'HOMOGRAPHY':
        identity = np.eye(3, dtype=np.float32)
        if np.allclose(warp_matrix, identity):
            print("   [!] WARNING: Warp matrix is identity (no transformation found)")
    
    # Project player position (center of adaptive_img) to static map
    h_orig, w_orig = adaptive_img.shape[0], adaptive_img.shape[1]
    h_resized, w_resized = query_ecc.shape[0], query_ecc.shape[1]
    
    # Player center in ORIGINAL image coordinates
    player_x_orig = w_orig / 2
    player_y_orig = h_orig / 2
    
    # CRITICAL: Scale player position to RESIZED image coordinates (what ECC was trained on)
    player_x_resized = player_x_orig * w_resized / w_orig
    player_y_resized = player_y_orig * h_resized / h_orig
    
    player_center = np.array([player_x_resized, player_y_resized, 1], dtype=np.float32)
    
    print(f"   Original minimap shape: {adaptive_img.shape}")
    print(f"   Player position in original coords: ({player_x_orig:.1f}, {player_y_orig:.1f})")
    print(f"   Resized for ECC to: ({query_ecc.shape[1]}, {query_ecc.shape[0]})")
    print(f"   Player position in resized coords: ({player_x_resized:.1f}, {player_y_resized:.1f})")
    print(f"   Static map size: {static_edges.shape}")
    
    # Apply transformation
    if motion_type == 'AFFINE':
        projected = warp_matrix @ player_center
        player_x = int(projected[0])
        player_y = int(projected[1])
    else:  # HOMOGRAPHY
        projected = warp_matrix @ player_center
        player_x = int(projected[0] / (projected[2] + 1e-6))
        player_y = int(projected[1] / (projected[2] + 1e-6))
    
    print(f"   Projected on static map: ({player_x}, {player_y})")
    print(f"   ECC score (confidence): {cc_score:.4f}")
    
    # Visualize: warp minimap edges directly (debug)
    print("\n   Warping minimap edges to reference space...")
    minimap_edges_resized = cv2.resize(minimap_edges, (ref_ecc.shape[1], ref_ecc.shape[0]))
    
    if motion_type == 'AFFINE':
        warped = cv2.warpAffine(
            minimap_edges_resized,
            warp_matrix,
            (static_edges.shape[1], static_edges.shape[0]),
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0
        )
    else:
        warped = cv2.warpPerspective(
            minimap_edges_resized,
            warp_matrix,
            (static_edges.shape[1], static_edges.shape[0]),
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0
        )
    
    print(f"   Warped shape: {warped.shape}, min: {warped.min()}, max: {warped.max()}, non-zero: {np.count_nonzero(warped)}")
    
    # Save warped as-is (may be mostly black if transform is large)
    save_debug_image(warped, "14_warped_minimap_edges", output_dir)
    
    # Also warp the original adaptive_img for comparison
    adaptive_resized = cv2.resize(adaptive_img, (ref_ecc.shape[1], ref_ecc.shape[0]))
    if motion_type == 'AFFINE':
        warped_adaptive = cv2.warpAffine(
            adaptive_resized,
            warp_matrix,
            (static_edges.shape[1], static_edges.shape[0]),
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0
        )
    else:
        warped_adaptive = cv2.warpPerspective(
            adaptive_resized,
            warp_matrix,
            (static_edges.shape[1], static_edges.shape[0]),
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0
        )
    save_debug_image(warped_adaptive, "14a_warped_adaptive_img", output_dir)
    print(f"   Warped adaptive_img - non-zero: {np.count_nonzero(warped_adaptive)}")
    
    # Overlay: blend warped with static map
    warped_vis = warped.astype(np.uint8)
    overlay = static_map.copy().astype(float)
    
    # Find non-zero pixels in warped
    mask = warped_vis > 0
    if np.any(mask):
        overlay[mask] = [0, 255, 0]  # Green where warped query exists
        print(f"   [+] Found {np.count_nonzero(mask)} overlapping pixels")
    else:
        print("   [!] Warped image is completely black - no visible overlap")
    
    overlay = overlay.astype(np.uint8)
    save_debug_image(overlay, "14b_alignment_overlay", output_dir)
    
    # Visualize result on static map
    result_vis = static_map.copy()
    
    # Draw player position
    cv2.circle(result_vis, (player_x, player_y), 15, (0, 0, 255), -1)
    cv2.circle(result_vis, (player_x, player_y), 15, (255, 255, 255), 2)
    
    # Add crosshair
    cv2.line(result_vis, (player_x - 30, player_y), (player_x + 30, player_y), (0, 0, 255), 2)
    cv2.line(result_vis, (player_x, player_y - 30), (player_x, player_y + 30), (0, 0, 255), 2)
    
    # Add confidence text
    cv2.putText(
        result_vis,
        f"ECC: {cc_score:.3f}",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        (0, 255, 0),
        2
    )
    
    save_debug_image(result_vis, "15_localization_result", output_dir)
    
    return (player_x, player_y), cc_score


# ============================================================================
# MAIN FUNCTION
# ============================================================================

def main():
    """Main test function."""
    # Parse arguments
    parser = argparse.ArgumentParser(description='Test static map localization v3')
    parser.add_argument('--static', action='store_true', 
                       help='Use static images from inputs/ folder instead of live capture')
    args = parser.parse_args()
    
    print("=" * 70)
    print("STATIC MAP LOCALIZATION TEST v3 - Isometric Structure Detection")
    if args.static:
        print("MODE: Static Images (from inputs/)")
    else:
        print("MODE: Live Capture")
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
        adaptive, adaptive_closed = apply_adaptive_threshold(oriented, output_dir)
        closed = apply_oriented_closing(adaptive_closed, output_dir)
        # lines, _ = detect_lines_hough(closed, output_dir)
        lines, _ = detect_lines_hough(adaptive_closed, output_dir)
        merged_lines = fuse_and_clean_lines(lines, diff_image.shape, output_dir)
        
        # Draw red dot at center of diff_image (player position in screenshot)
        # print("\n" + "="*70)
        # print("STEP 8: Player Position Marker")
        # print("="*70)
        # center_h, center_w = adaptive.shape[0] // 2, adaptive.shape[1] // 2
        # diff_with_marker = adaptive.copy()
        # #cv2.circle(diff_with_marker, (center_w, center_h), 15, (0, 0, 255), -1)  # Red circle
        # cv2.circle(diff_with_marker, (center_w, center_h), 15, (255, 255, 255), 2)  # White outline
        # save_debug_image(diff_with_marker, "08_diff_with_player_marker", output_dir)
        # print(f"OK Player marker drawn at center: ({center_w}, {center_h})")
        
        # Match with static map
        player_pos, confidence = match_with_static_map_ecc(
            adaptive_closed,
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
