#!/usr/bin/env python3
"""
Test script for static map localization - Using ECC with Phase Correlation.

Uses modular ECC system with Phase Correlation (FFT) on processed minimap images
(after Gabor filters, before Canny) for robust initialization before edge-based alignment.

Key improvements:
- Phase correlation applied to PROCESSED minimap (Gabor-filtered, no Canny)
- More accurate initial offset detection on structural features
- ECC alignment on Canny edges for fine-tuning

Usage:
  python test_static_localization_v3_modular.py           # Live capture mode
  python test_static_localization_v3_modular.py --static   # Use static images from inputs/
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
from diabot.navigation import (
    ECCStaticMapLocalizer,
    load_zone_static_map,
)
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
# LOCALIZATION USING NEW MODULES
# ============================================================================

def localize_with_ecc(
    frame_with_minimap,
    frame_without_minimap,
    zone_name,
    output_dir
):
    """
    Localize player using ECC alignment with Phase Correlation initialization.
    
    Pipeline:
    1. Extract edges (both Canny and non-Canny versions)
    0. Phase Correlation (FFT) on PROCESSED minimap (before Canny) for robust initial offset
    2. Rescale edges to match static map
    3. Multi-scale ECC alignment for fine-tuning
    
    Args:
        frame_with_minimap: Frame with minimap visible
        frame_without_minimap: Frame without minimap (background)
        zone_name: Current zone name (for static map lookup)
        output_dir: Directory for debug images
        
    Returns:
        (player_position, confidence) or (None, 0.0)
    """
    print("\n" + "="*70)
    print("LOCALIZATION: ECC with Phase Correlation")
    print("="*70)
    
    # Create localizer
    localizer = ECCStaticMapLocalizer(
        debug=False,  # Disable internal debug to control output
        output_dir=output_dir
    )
    
    # Load static map
    print(f"\n[*] Loading static map for zone: {zone_name}")
    map_path = load_zone_static_map(zone_name)
    
    if map_path is None:
        print(f"[!] No static map found for zone: {zone_name}")
        return None, 0.0
    
    if not localizer.load_static_map(map_path):
        print(f"[!] Failed to load static map")
        return None, 0.0
    
    print(f"[+] Loaded static map from: {map_path}")
    
    # ========================================================================
    # STEP 1: Extract edges (both versions: non-Canny and Canny)
    # ========================================================================
    print("\n[STEP 1] Extracting edges...")
    
    # Get minimap edges - both versions
    minimap_edges_no_canny, minimap_edges_canny = localizer.extract_minimap_edges_canny(
        frame_with_minimap,
        frame_without_minimap,
        use_oriented_filter=True,
        canny_low=50,
        canny_high=150,
        return_both=True
    )
    
    if minimap_edges_canny is None:
        print("[!] Failed to extract minimap edges")
        return None, 0.0
    
    save_debug_image(minimap_edges_no_canny, "ecc_01a_minimap_edges_no_canny", output_dir)
    save_debug_image(minimap_edges_canny, "ecc_01b_minimap_edges_canny", output_dir)
    print(f"   Minimap edges (no Canny): {minimap_edges_no_canny.shape[1]}×{minimap_edges_no_canny.shape[0]}")
    print(f"   Minimap edges (Canny): {minimap_edges_canny.shape[1]}×{minimap_edges_canny.shape[0]}")
    
    # Get static map Canny edges
    static_map_edges_canny_original = localizer.extract_static_map_edges_canny(
        white_threshold=120,
        canny_low=50,
        canny_high=150
    )
    save_debug_image(static_map_edges_canny_original, "ecc_02_static_map_edges_original", output_dir)
    print(f"   Static edges (original): {static_map_edges_canny_original.shape[1]}×{static_map_edges_canny_original.shape[0]}")
    
    # ========================================================================
    # STEP 0: Phase Correlation on processed minimap (BEFORE Canny)
    # ========================================================================
    print("\n[STEP 0] Phase Correlation (FFT) on processed minimap (before Canny)...")
    print("   Working on PROCESSED minimap (Gabor filters) BEFORE Canny for better accuracy")
    
    # Initialize phase correlation results
    phase_offset_x_raw = 0.0
    phase_offset_y_raw = 0.0
    response_raw = 0.0
    use_phase_init = False
    
    if minimap_edges_no_canny is not None and minimap_edges_no_canny.size > 0:
        
        # Get static map in grayscale
        if localizer.static_map is not None:
            if len(localizer.static_map.shape) == 3:
                static_gray = cv2.cvtColor(localizer.static_map, cv2.COLOR_BGR2GRAY)
            else:
                static_gray = localizer.static_map
            
            save_debug_image(minimap_edges_no_canny, "ecc_00a_minimap_processed_no_canny", output_dir)
            save_debug_image(static_gray, "ecc_00b_static_original_gray", output_dir)
            
            print(f"   Minimap processed (no Canny): {minimap_edges_no_canny.shape[1]}×{minimap_edges_no_canny.shape[0]}")
            print(f"   Static original: {static_gray.shape[1]}×{static_gray.shape[0]}")
            
            # Apply Hann window to reduce edge effects in FFT
            hann_window = cv2.createHanningWindow(
                (minimap_edges_no_canny.shape[1], minimap_edges_no_canny.shape[0]),
                cv2.CV_32F
            )
            
            minimap_float = minimap_edges_no_canny.astype(np.float32)
            static_float = static_gray.astype(np.float32)
            
            minimap_windowed = minimap_float * hann_window
            static_windowed = static_float * hann_window
            
            # Phase correlation using FFT
            try:
                shift, response_raw = cv2.phaseCorrelate(minimap_windowed, static_windowed)
                
                phase_offset_x_raw = shift[0]
                phase_offset_y_raw = shift[1]
                
                print(f"\n   📊 Phase Correlation (on PROCESSED minimap - no Canny):")
                print(f"      Translation detected: X={phase_offset_x_raw:+.2f} px, Y={phase_offset_y_raw:+.2f} px")
                print(f"      Response (confidence): {response_raw:.4f}")
                
                # Visualize the phase correlation result
                viz_phase = cv2.cvtColor(static_gray, cv2.COLOR_GRAY2BGR)
                
                # Draw the shifted minimap position
                h, w = minimap_edges_no_canny.shape
                center_x = w // 2
                center_y = h // 2
                
                # New position after phase correlation shift
                new_center_x = int(center_x - phase_offset_x_raw)
                new_center_y = int(center_y - phase_offset_y_raw)
                
                # Draw bounding box at detected position
                top_left = (new_center_x - w//2, new_center_y - h//2)
                bottom_right = (new_center_x + w//2, new_center_y + h//2)
                
                # Ensure coordinates are in bounds
                if (0 <= top_left[0] < viz_phase.shape[1] and 
                    0 <= top_left[1] < viz_phase.shape[0] and
                    0 <= bottom_right[0] < viz_phase.shape[1] and 
                    0 <= bottom_right[1] < viz_phase.shape[0]):
                    
                    cv2.rectangle(viz_phase, top_left, bottom_right, (255, 0, 255), 2)
                    cv2.circle(viz_phase, (new_center_x, new_center_y), 5, (255, 0, 255), -1)
                    
                    cv2.putText(viz_phase, f"Phase Correlation (Processed minimap): shift=({phase_offset_x_raw:.1f}, {phase_offset_y_raw:.1f})", 
                               (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 2)
                    cv2.putText(viz_phase, f"Response: {response_raw:.4f}", 
                               (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)
                
                save_debug_image(viz_phase, "ecc_00d_phase_correlation_processed", output_dir)
                
                # Create overlay with phase-corrected alignment
                shift_matrix = np.array([
                    [1.0, 0.0, -phase_offset_x_raw],
                    [0.0, 1.0, -phase_offset_y_raw]
                ], dtype=np.float32)
                
                minimap_shifted = cv2.warpAffine(
                    minimap_edges_no_canny,
                    shift_matrix,
                    (static_gray.shape[1], static_gray.shape[0])
                )
                
                # Create colored overlay
                overlay_phase = np.zeros((static_gray.shape[0], static_gray.shape[1], 3), dtype=np.uint8)
                overlay_phase[:, :, 1] = minimap_shifted  # Green = phase-corrected minimap
                overlay_phase[:, :, 2] = static_gray  # Red = static
                
                # Calculate overlap metrics
                minimap_pixels = np.count_nonzero(minimap_shifted)
                static_pixels = np.count_nonzero(static_gray)
                overlap_pixels = np.count_nonzero(cv2.bitwise_and(minimap_shifted, static_gray))
                overlap_ratio_phase = (overlap_pixels / max(minimap_pixels, static_pixels)) * 100 if max(minimap_pixels, static_pixels) > 0 else 0
                
                cv2.putText(overlay_phase, f"After Phase Correlation (Processed): {overlap_ratio_phase:.1f}% overlap", 
                           (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
                cv2.putText(overlay_phase, f"Response: {response_raw:.4f}", 
                           (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 1)
                
                save_debug_image(overlay_phase, "ecc_00e_overlay_after_phase_corr_processed", output_dir)
                
                print(f"      Overlap after phase correction: {overlap_ratio_phase:.2f}%")
                
                # Decide if we use this initialization
                if response_raw > 0.1:
                    print(f"      ✅ Phase correlation result seems reliable (response > 0.1)")
                    print(f"      💡 Will use this offset to initialize ECC")
                    use_phase_init = True
                else:
                    print(f"      ⚠️  Phase correlation low confidence (response={response_raw:.4f})")
                    print(f"      Will use identity matrix for ECC initialization")
                    use_phase_init = False
                    
            except cv2.error as e:
                print(f"      ❌ Phase correlation failed: {e}")
                print(f"      Will use identity matrix for ECC initialization")
                use_phase_init = False
        else:
            print("   [!] Static map not loaded, skipping phase correlation")
    else:
        print("   [!] Could not extract processed minimap (no Canny), skipping phase correlation")
    
    print(f"   ✅ Phase correlation on processed minimap completed")
    
    # ========================================================================
    # STEP 2: Rescale minimap to match static map size
    # ========================================================================
    print("\n[STEP 2] Rescaling minimap to match static map size...")
    
    target_size = static_map_edges_canny_original.shape  # Utiliser taille de la carte statique
    minimap_edges_rescaled = cv2.resize(
        minimap_edges_canny, 
        (target_size[1], target_size[0]),
        interpolation=cv2.INTER_LINEAR
    )
    
    # Maintenant minimap et static ont la MÊME taille → ECC devrait mieux marcher
    save_debug_image(minimap_edges_rescaled, "ecc_03_minimap_edges_rescaled", output_dir)
    print(f"   Minimap edges (rescaled): {minimap_edges_rescaled.shape[1]}×{minimap_edges_rescaled.shape[0]}")
    print(f"   ✅ Sizes now match!")
    
    # Comparison visualization
    viz_height = target_size[0]
    minimap_rgb = cv2.cvtColor(minimap_edges_canny, cv2.COLOR_GRAY2BGR)
    static_rgb = cv2.cvtColor(static_map_edges_canny_original, cv2.COLOR_GRAY2BGR)
    
    composite = np.ones((viz_height, minimap_rgb.shape[1] + static_rgb.shape[1] + 20, 3), dtype=np.uint8) * 255
    composite[0:minimap_rgb.shape[0], 0:minimap_rgb.shape[1]] = minimap_rgb
    composite[0:static_rgb.shape[0], minimap_rgb.shape[1]+20:] = static_rgb
    
    cv2.putText(composite, f"Minimap (original): {minimap_edges_canny.shape[1]}×{minimap_edges_canny.shape[0]}", 
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(composite, f"Static map: {static_map_edges_canny_original.shape[1]}×{static_map_edges_canny_original.shape[0]}", 
                (minimap_rgb.shape[1] + 30, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    
    save_debug_image(composite, "ecc_04_comparison_same_size", output_dir)
    
    # ========================================================================
    # STEP 2b: SUPERPOSITION BEFORE ALIGNMENT (diagnostic)
    # ========================================================================
    print("\n[STEP 2b] Creating overlay BEFORE alignment (diagnostic)...")
    
    # Convert to RGB for overlay
    minimap_rescaled_rgb = cv2.cvtColor(minimap_edges_rescaled, cv2.COLOR_GRAY2BGR)
    static_rgb = cv2.cvtColor(static_map_edges_canny_original, cv2.COLOR_GRAY2BGR)
    
    # Create colored overlay: minimap in GREEN, static in RED
    overlay_before = np.zeros_like(static_rgb)
    overlay_before[:, :, 0] = static_map_edges_canny_original  # Blue channel = 0
    overlay_before[:, :, 1] = minimap_edges_rescaled  # Green channel = minimap
    overlay_before[:, :, 2] = static_map_edges_canny_original  # Red channel = static
    
    # Where both overlap = should be yellow (G+R), green = minimap only, red = static only
    save_debug_image(overlay_before, "ecc_04b_overlay_BEFORE_alignment", output_dir)
    
    # Also create a blend overlay (50/50)
    overlay_blend_before = cv2.addWeighted(minimap_rescaled_rgb, 0.5, static_rgb, 0.5, 0)
    cv2.putText(overlay_blend_before, "BEFORE alignment - 50% blend", (10, 25),
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
    cv2.putText(overlay_blend_before, "Green=Minimap, Red=Static, Yellow=Match", (10, 50),
               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    save_debug_image(overlay_blend_before, "ecc_04c_blend_BEFORE_alignment", output_dir)
    
    # ========================================================================
    # STEP 2c: COMPUTE SIMILARITY METRICS (critical diagnostic)
    # ========================================================================
    print(f"\n   📊 SIMILARITY ANALYSIS (BEFORE alignment):")
    
    # Count non-zero pixels
    minimap_pixels = np.count_nonzero(minimap_edges_rescaled)
    static_pixels = np.count_nonzero(static_map_edges_canny_original)
    overlap_pixels = np.count_nonzero(
        cv2.bitwise_and(minimap_edges_rescaled, static_map_edges_canny_original)
    )
    
    total_size = minimap_edges_rescaled.size
    minimap_density = (minimap_pixels / total_size) * 100
    static_density = (static_pixels / total_size) * 100
    overlap_ratio = (overlap_pixels / max(minimap_pixels, static_pixels)) * 100 if max(minimap_pixels, static_pixels) > 0 else 0
    
    print(f"      Minimap edge pixels: {minimap_pixels:,} ({minimap_density:.2f}% density)")
    print(f"      Static edge pixels: {static_pixels:,} ({static_density:.2f}% density)")
    print(f"      Overlapping pixels: {overlap_pixels:,} ({overlap_ratio:.2f}% overlap)")
    
    if overlap_ratio < 5:
        print(f"      ❌ CRITICAL: < 5% overlap - images don't match at all!")
        print(f"         Possible causes:")
        print(f"         - Wrong zone (minimap from different area)")
        print(f"         - Scale mismatch (despite rescaling)")
        print(f"         - Rotation difference")
        print(f"         - Very different content")
    elif overlap_ratio < 15:
        print(f"      ⚠️  WARNING: {overlap_ratio:.1f}% overlap - poor initial alignment")
    else:
        print(f"      ✅ OK: {overlap_ratio:.1f}% overlap - reasonable starting point")
    
    # Try template matching to find best initial offset
    print(f"\n   🔍 SEARCHING for best initial alignment with template matching...")
    
    # Initialize variables
    template_offset_x = 0
    template_offset_y = 0
    max_val = 0.0
    
    try:
        result = cv2.matchTemplate(
            static_map_edges_canny_original.astype(np.float32),
            minimap_edges_rescaled.astype(np.float32),
            cv2.TM_CCORR_NORMED
        )
        _, max_val, _, max_loc = cv2.minMaxLoc(result)
        
        print(f"      Best match score: {max_val:.4f}")
        print(f"      Best match position: ({max_loc[0]}, {max_loc[1]})")
        
        # Calculate offset from center (store for later use in ECC initialization)
        center_x = static_map_edges_canny_original.shape[1] // 2
        center_y = static_map_edges_canny_original.shape[0] // 2
        template_offset_x = max_loc[0] - (center_x - minimap_edges_rescaled.shape[1] // 2)
        template_offset_y = max_loc[1] - (center_y - minimap_edges_rescaled.shape[0] // 2)
        
        print(f"      Suggested offset from center: X={template_offset_x:+d} px, Y={template_offset_y:+d} px")
        
        if abs(template_offset_x) > 50 or abs(template_offset_y) > 50:
            print(f"      ⚠️  Large offset needed ({abs(template_offset_x)}×{abs(template_offset_y)}) - ECC may struggle")
            print(f"      💡 Will initialize ECC with this offset instead of identity matrix")
        
        # Create visualization with template matching result
        viz_match = static_rgb.copy()
        cv2.rectangle(viz_match, max_loc, 
                     (max_loc[0] + minimap_edges_rescaled.shape[1], 
                      max_loc[1] + minimap_edges_rescaled.shape[0]),
                     (0, 255, 0), 2)
        cv2.putText(viz_match, f"Best match: score={max_val:.3f}", (10, 25),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        save_debug_image(viz_match, "ecc_04d_template_matching_result", output_dir)
        
    except Exception as e:
        print(f"      ❌ Template matching failed: {e}")
        print(f"      Using center position (no offset)")
    
    print(f"   ✅ Template matching completed")
    
    # ========================================================================
    # STEP 3: Prepare images for ECC (normalize, blur)
    # ========================================================================
    print("\n[STEP 3] Preparing images for ECC alignment...")
    
    query_ecc = localizer.matcher.aligner.prepare_for_ecc(minimap_edges_rescaled, blur_kernel=5)
    ref_ecc = localizer.matcher.aligner.prepare_for_ecc(static_map_edges_canny_original, blur_kernel=5)
    
    # Visualize prepared images
    query_viz = (query_ecc * 255).astype(np.uint8)
    ref_viz = (ref_ecc * 255).astype(np.uint8)
    save_debug_image(query_viz, "ecc_05_query_prepared", output_dir)
    save_debug_image(ref_viz, "ecc_06_ref_prepared", output_dir)
    print(f"   Prepared for ECC (float32, normalized, blurred)")
    
    # ========================================================================
    # STEP 4: Build pyramids and visualize
    # ========================================================================
    print("\n[STEP 4] Building image pyramids...")
    
    pyramid_levels = 4
    query_pyramid = localizer.matcher.aligner.build_pyramid(query_ecc, pyramid_levels)
    ref_pyramid = localizer.matcher.aligner.build_pyramid(ref_ecc, pyramid_levels)
    
    print(f"   Pyramid levels: {pyramid_levels}")
    for i, (qp, rp) in enumerate(zip(query_pyramid, ref_pyramid)):
        print(f"   Level {i}: Query {qp.shape[1]}×{qp.shape[0]}, Ref {rp.shape[1]}×{rp.shape[0]}")
    
    # Visualize all pyramid levels
    pyramid_viz = []
    max_width = 0
    
    # First pass: create visualizations and find max width
    for level_idx, (query_level, ref_level) in enumerate(zip(query_pyramid, ref_pyramid)):
        q_viz = (query_level * 255).astype(np.uint8)
        r_viz = (ref_level * 255).astype(np.uint8)
        
        q_rgb = cv2.cvtColor(q_viz, cv2.COLOR_GRAY2BGR)
        r_rgb = cv2.cvtColor(r_viz, cv2.COLOR_GRAY2BGR)
        
        # Side by side
        h = max(q_rgb.shape[0], r_rgb.shape[0])
        side_by_side = np.ones((h, q_rgb.shape[1] + r_rgb.shape[1] + 20, 3), dtype=np.uint8) * 240
        side_by_side[0:q_rgb.shape[0], 0:q_rgb.shape[1]] = q_rgb
        side_by_side[0:r_rgb.shape[0], q_rgb.shape[1]+20:] = r_rgb
        
        cv2.putText(side_by_side, f"Level {level_idx} - Query {q_viz.shape[1]}×{q_viz.shape[0]}", 
                    (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        cv2.putText(side_by_side, f"Ref {r_viz.shape[1]}×{r_viz.shape[0]}", 
                    (q_rgb.shape[1] + 30, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        
        pyramid_viz.append(side_by_side)
        max_width = max(max_width, side_by_side.shape[1])
    
    # Second pass: resize all to max width and stack
    pyramid_viz_resized = []
    for viz in pyramid_viz:
        if viz.shape[1] < max_width:
            # Pad with white background
            padded = np.ones((viz.shape[0], max_width, 3), dtype=np.uint8) * 240
            padded[0:viz.shape[0], 0:viz.shape[1]] = viz
            pyramid_viz_resized.append(padded)
        else:
            pyramid_viz_resized.append(viz)
    
    # Stack all levels vertically
    pyramid_composite = np.vstack(pyramid_viz_resized)
    save_debug_image(pyramid_composite, "ecc_07_pyramid_all_levels", output_dir)
    print(f"   ✅ Pyramid visualization saved")
    
    # ========================================================================
    # STEP 5: Run ECC alignment with detailed tracking
    # ========================================================================
    print("\n[STEP 5] Running multi-scale ECC alignment...")
    
    # Use TRANSLATION mode: only X,Y displacement (no rotation, no scale, no perspective)
    # This is appropriate since the minimap and static map have the same size and orientation
    motion_type = 'TRANSLATION'
    warp_mode = cv2.MOTION_TRANSLATION
    
    # Choose best initialization method based on previous analysis
    print(f"   Choosing initialization method:")
    
    init_method = "Identity Matrix"  # Default
    
    if use_phase_init and response_raw > 0.1:
        # Phase correlation (on PROCESSED images) gave good result - use it
        warp_matrix = np.array([
            [1.0, 0.0, -phase_offset_x_raw],
            [0.0, 1.0, -phase_offset_y_raw]
        ], dtype=np.float32)
        print(f"   ✅ Using PHASE CORRELATION initialization (from PROCESSED minimap)")
        print(f"      Initial offset: X={-phase_offset_x_raw:+.2f}, Y={-phase_offset_y_raw:+.2f}")
        print(f"      Response: {response_raw:.4f}")
        init_method = "Phase Correlation (FFT on PROCESSED minimap)"
    elif overlap_ratio < 10 and (abs(template_offset_x) < 100 and abs(template_offset_y) < 100):
        # Poor overlap but template matching found reasonable offset
        warp_matrix = np.array([
            [1.0, 0.0, float(template_offset_x)],
            [0.0, 1.0, float(template_offset_y)]
        ], dtype=np.float32)
        print(f"   ✅ Using TEMPLATE MATCHING initialization")
        print(f"      Initial offset: X={template_offset_x:+d}, Y={template_offset_y:+d}")
        init_method = "Template Matching"
    else:
        # Good overlap or no hint - use identity
        warp_matrix = np.eye(2, 3, dtype=np.float32)
        print(f"   ✅ Using IDENTITY initialization (no offset)")
        init_method = "Identity Matrix"
    
    max_iterations = 5000
    epsilon = 1e-6
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, max_iterations, epsilon)
    
    alignment_results = []
    
    for level in range(len(query_pyramid)):
        query_level = query_pyramid[level]
        ref_level = ref_pyramid[level]
        scale = 2 ** (len(query_pyramid) - level - 1)
        
        print(f"\n   Level {level+1}/{len(query_pyramid)} (scale {scale}×)")
        print(f"      Query: {query_level.shape[1]}×{query_level.shape[0]}, Ref: {ref_level.shape[1]}×{ref_level.shape[0]}")
        
        try:
            cc_score, warp_matrix = cv2.findTransformECC(
                query_level,
                ref_level,
                warp_matrix,
                warp_mode,
                criteria
            )
            print(f"      ✅ ECC score: {cc_score:.4f}")
            print(f"         Translation: X={warp_matrix[0, 2]:.2f}, Y={warp_matrix[1, 2]:.2f}")
            
            # Warp query to reference using affine transformation
            warped = cv2.warpAffine(
                query_level,
                warp_matrix,
                (ref_level.shape[1], ref_level.shape[0])
            )
            
            # Store result
            alignment_results.append({
                'level': level,
                'score': cc_score,
                'warp_matrix': warp_matrix.copy(),
                'query': query_level,
                'ref': ref_level,
                'warped': warped
            })
            
        except cv2.error as e:
            print(f"      ❌ ECC failed: {str(e)[:80]}")
            alignment_results.append({
                'level': level,
                'score': 0.0,
                'failed': True
            })
            continue
        
        # Scale translation offsets for next pyramid level (double for higher resolution)
        if level < len(query_pyramid) - 1:
            warp_matrix[0, 2] *= 2  # Scale X translation
            warp_matrix[1, 2] *= 2  # Scale Y translation
    
    # ========================================================================
    # STEP 6: Visualize alignment at each pyramid level
    # ========================================================================
    print("\n[STEP 6] Creating alignment visualizations...")
    
    for result in alignment_results:
        if result.get('failed', False):
            continue
        
        level = result['level']
        warped = result['warped']
        ref = result['ref']
        score = result['score']
        
        # Convert to uint8 for visualization
        warped_viz = (warped * 255).astype(np.uint8)
        ref_viz = (ref * 255).astype(np.uint8)
        
        # Side by side
        warped_rgb = cv2.cvtColor(warped_viz, cv2.COLOR_GRAY2BGR)
        ref_rgb = cv2.cvtColor(ref_viz, cv2.COLOR_GRAY2BGR)
        
        side_by_side = np.hstack([warped_rgb, np.ones((warped_rgb.shape[0], 20, 3), dtype=np.uint8)*255, ref_rgb])
        cv2.putText(side_by_side, f"Level {level} - Warped Query", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.putText(side_by_side, f"Reference (score: {score:.4f})", (warped_rgb.shape[1] + 30, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        # Overlay (blend)
        overlay = cv2.addWeighted(warped_rgb, 0.5, ref_rgb, 0.5, 0)
        cv2.putText(overlay, f"Level {level} Overlay (score: {score:.4f})", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        
        # Pad overlay to match side_by_side width
        if overlay.shape[1] < side_by_side.shape[1]:
            padded_overlay = np.ones((overlay.shape[0], side_by_side.shape[1], 3), dtype=np.uint8) * 240
            padded_overlay[0:overlay.shape[0], 0:overlay.shape[1]] = overlay
            overlay = padded_overlay
        
        # Stack vertically
        viz = np.vstack([side_by_side, overlay])
        save_debug_image(viz, f"ecc_08_alignment_level_{level}", output_dir)
    
    print(f"   ✅ Saved {len([r for r in alignment_results if not r.get('failed', False)])} alignment visualizations")
    
    # ========================================================================
    # STEP 6b: Create AFTER alignment overlay comparison
    # ========================================================================
    print("\n[STEP 6b] Creating overlay AFTER alignment (comparison)...")
    
    if alignment_results and not alignment_results[-1].get('failed', False):
        final_result = alignment_results[-1]
        warped_final = final_result['warped']
        ref_final = final_result['ref']
        final_score = final_result['score']
        
        # Convert to uint8 RGB
        warped_viz = (warped_final * 255).astype(np.uint8)
        ref_viz = (ref_final * 255).astype(np.uint8)
        
        # Create colored overlay: warped minimap in GREEN, static in RED
        overlay_after = np.zeros((ref_viz.shape[0], ref_viz.shape[1], 3), dtype=np.uint8)
        overlay_after[:, :, 1] = warped_viz  # Green channel = warped minimap
        overlay_after[:, :, 2] = ref_viz  # Red channel = static
        
        # Calculate overlap for visualization
        warped_pixels_viz = np.count_nonzero(warped_viz)
        ref_pixels_viz = np.count_nonzero(ref_viz)
        overlap_pixels_viz = np.count_nonzero(cv2.bitwise_and(warped_viz, ref_viz))
        overlap_ratio_viz = (overlap_pixels_viz / max(warped_pixels_viz, ref_pixels_viz)) * 100 if max(warped_pixels_viz, ref_pixels_viz) > 0 else 0
        
        cv2.putText(overlay_after, f"Overlap: {overlap_ratio_viz:.1f}%", 
                   (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
        cv2.putText(overlay_after, f"ECC Score: {final_score:.3f}", 
                   (10, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        
        save_debug_image(overlay_after, "ecc_08b_overlay_AFTER_alignment", output_dir)
        
        # Also create blend
        warped_rgb = cv2.cvtColor(warped_viz, cv2.COLOR_GRAY2BGR)
        ref_rgb = cv2.cvtColor(ref_viz, cv2.COLOR_GRAY2BGR)
        overlay_blend_after = cv2.addWeighted(warped_rgb, 0.5, ref_rgb, 0.5, 0)
        cv2.putText(overlay_blend_after, f"AFTER alignment - Score: {final_score:.4f}", (10, 25),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        cv2.putText(overlay_blend_after, "Green=Warped Minimap, Red=Static, Yellow=Match", (10, 50),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        save_debug_image(overlay_blend_after, "ecc_08c_blend_AFTER_alignment", output_dir)
        
        print(f"   ✅ AFTER alignment overlay saved")
        print(f"      Compare with BEFORE overlay (04b/04c) to see improvement")
    
    # ========================================================================
    # STEP 7: Final alignment on full resolution
    # ========================================================================
    print("\n[STEP 7] Computing final player position...")
    
    if alignment_results and not alignment_results[-1].get('failed', False):
        final_warp = alignment_results[-1]['warp_matrix']
        final_score = alignment_results[-1]['score']
        warped_final = alignment_results[-1]['warped']
        ref_final = alignment_results[-1]['ref']
        
        # Analyze translation offset
        tx = final_warp[0, 2]
        ty = final_warp[1, 2]
        total_offset = np.sqrt(tx**2 + ty**2)
        
        # Calculate overlap ratio after alignment (more intuitive metric)
        warped_final_uint8 = (warped_final * 255).astype(np.uint8)
        ref_final_uint8 = (ref_final * 255).astype(np.uint8)
        
        # Threshold to binary (handle floating point precision)
        _, warped_binary = cv2.threshold(warped_final_uint8, 10, 255, cv2.THRESH_BINARY)
        _, ref_binary = cv2.threshold(ref_final_uint8, 10, 255, cv2.THRESH_BINARY)
        
        # Count overlapping pixels
        warped_pixels_final = np.count_nonzero(warped_binary)
        ref_pixels_final = np.count_nonzero(ref_binary)
        overlap_pixels_final = np.count_nonzero(cv2.bitwise_and(warped_binary, ref_binary))
        
        # Calculate overlap ratio (0-100%)
        if max(warped_pixels_final, ref_pixels_final) > 0:
            overlap_ratio_final = (overlap_pixels_final / max(warped_pixels_final, ref_pixels_final)) * 100
        else:
            overlap_ratio_final = 0.0
        
        # Alternative confidence based on overlap
        confidence_human = min(overlap_ratio_final / 100.0, 1.0)  # Normalize to 0-1
        
        print(f"\n   📊 ALIGNMENT ANALYSIS:")
        print(f"      Initialization method: {init_method}")
        print(f"      Translation X: {tx:+.2f} px")
        print(f"      Translation Y: {ty:+.2f} px")
        print(f"      Total offset: {total_offset:.2f} px")
        print(f"      ECC Score (correlation): {final_score:.4f}")
        print(f"      Overlap After Alignment: {overlap_ratio_final:.1f}%")
        print(f"      Confidence (overlap-based): {confidence_human:.3f}")
        
        # Interpret overlap quality
        if overlap_ratio_final >= 70:
            print(f"      ✅ Excellent overlap (≥70%) - Very high quality alignment")
        elif overlap_ratio_final >= 50:
            print(f"      ✅ Good overlap (≥50%) - Reliable alignment")
        elif overlap_ratio_final >= 30:
            print(f"      ⚠️  Moderate overlap (≥30%) - Acceptable but verify visually")
        else:
            print(f"      ❌ Low overlap (<30%) - Alignment may be incorrect")
        
        # Interpret the offset
        if total_offset < 5:
            print(f"      ✅ Very small offset - images were already well aligned")
        elif total_offset < 20:
            print(f"      ✅ Small offset - reasonable alignment")
        elif total_offset < 50:
            print(f"      ⚠️  Moderate offset - check overlay images")
        else:
            print(f"      ❌ Large offset ({total_offset:.0f} px) - alignment may be incorrect!")
            print(f"         Check BEFORE/AFTER overlay images to diagnose")
        
        # Project player position (center of rescaled minimap)
        # Since minimap_edges_rescaled has same size as static map, no scale_factor needed
        center = np.array([minimap_edges_rescaled.shape[1] / 2, minimap_edges_rescaled.shape[0] / 2], dtype=np.float32)
        
        # Apply affine transformation (2x3 matrix)
        # For translation: [x', y'] = [x, y] + [tx, ty]
        projected = cv2.transform(np.array([[center]]), final_warp)[0][0]
        
        # Player position in static map coordinates (already at correct scale)
        player_pos = (int(projected[0]), int(projected[1]))
        
        # Use overlap-based confidence (more intuitive than ECC score)
        # Combine both metrics: 70% overlap + 30% ECC score
        confidence = (0.7 * confidence_human) + (0.3 * final_score)
        
        print(f"\n   Player position (static map coords): {player_pos}")
        print(f"   Final Confidence: {confidence:.4f}")
        print(f"   Translation applied: X={tx:.2f}, Y={ty:.2f}")
        
        # Visualize on original static map
        if localizer.static_map is not None:
            result_viz = localizer.static_map.copy()
            cv2.circle(result_viz, player_pos, 8, (0, 0, 255), -1)
            cv2.circle(result_viz, player_pos, 10, (255, 255, 255), 2)
            cv2.line(result_viz, (player_pos[0]-20, player_pos[1]), (player_pos[0]+20, player_pos[1]), (0, 0, 255), 2)
            cv2.line(result_viz, (player_pos[0], player_pos[1]-20), (player_pos[0], player_pos[1]+20), (0, 0, 255), 2)
            cv2.putText(result_viz, f"ECC: {player_pos} (conf: {confidence:.3f})", 
                       (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            save_debug_image(result_viz, "ecc_09_final_position", output_dir)
        
        player_pos = player_pos
        confidence = confidence
    else:
        print("   ❌ All pyramid levels failed")
        player_pos = None
        confidence = 0.0
    
    if player_pos:
        print(f"\n[SUCCESS] Player localized with ECC!")
        print(f"         Position: {player_pos}")
        print(f"         Confidence: {confidence:.3f}")
        
        # Visualize position on static map (using base class method)
        print(f"\n[*] Visualizing position on static map...")
        viz_path = localizer.visualize_player_position(
            player_pos,
            confidence,
            method_name="ECC (Translation)",
            output_filename="ecc_10_player_position_on_map.png"
        )
        if viz_path:
            print(f"[+] Saved visualization: {viz_path.name}")
    else:
        print(f"\n[FAILED] ECC could not localize player")
    
    return player_pos, confidence


def main():
    """Main test function."""
    # Parse arguments
    parser = argparse.ArgumentParser(description='Test static map localization v3 (Modular)')
    parser.add_argument('--static', action='store_true', 
                       help='Use static images from inputs/ folder instead of live capture')
    args = parser.parse_args()
    
    print("=" * 70)
    print("STATIC MAP LOCALIZATION TEST v3 (MODULAR SYSTEM)")
    if args.static:
        print("MODE: Static Images (from inputs/)")
    else:
        print("MODE: Live Capture")
    print("=" * 70)
    print()
    
    # Create output directory (clean if exists)
    output_dir = Path(__file__).parent / "data" / "screenshots" / "outputs" / "localization_test_v3_modular"
    if output_dir.exists():
        import shutil
        print(f"[*] Cleaning output directory: {output_dir}")
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"[*] Debug images will be saved to: {output_dir}")
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
        
        # Detect zone first
        ui_vision_zone = UIVisionModule(debug=False)
        zone_name = ui_vision_zone.extract_zone(frame_with_minimap)
        if not zone_name:
            zone_name = 'Rogue Encampment'  # Default zone
        print(f"\nOK Current zone: {zone_name}")

    
        # ================================================================
        # MAIN LOCALIZATION: ECC Method
        # ================================================================
        print("\n" + "="*70)
        print("STEP 2: ECC Localization")
        print("="*70)
        
        player_pos_ecc, confidence_ecc = localize_with_ecc(
            frame_with_minimap,
            frame_without_minimap,
            zone_name,
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
    print("FINAL RESULTS")
    print("=" * 70)
    print()
    
    if player_pos_ecc:
        print(f"✅ SUCCESS: Player localized with ECC")
        print(f"   Position: {player_pos_ecc}")
        print(f"   Confidence: {confidence_ecc:.3f}")
    else:
        print(f"❌ FAILED: ECC could not localize player")
        print(f"   Confidence: {confidence_ecc:.3f}")
        print(f"\n💡 Check diagnostic analysis and debug images for potential issues")
    
    print()
    print("=" * 70)
    print("DEBUG IMAGES SAVED TO:")
    print(f"  {output_dir}")
    print("=" * 70)


if __name__ == "__main__":
    main()
