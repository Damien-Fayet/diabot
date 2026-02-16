#!/usr/bin/env python3
"""
Test script for static map localization - Using modular ECC system.

Refactored to use new production modules:
- image_preprocessing: Filters and preprocessing
- minimap_edge_extractor: Minimap edge extraction
- ecc_localizer: ECC alignment engine
- ecc_static_localizer: High-level unified interface

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
import argparse

# Setup path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from diabot.core.implementations import WindowsScreenCapture
from diabot.navigation import (
    ECCStaticMapLocalizer,
    RANSACStaticMapLocalizer,
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
    Localize player using modular ECC system with detailed visualizations.
    
    Args:
        frame_with_minimap: Frame with minimap visible
        frame_without_minimap: Frame without minimap (background)
        zone_name: Current zone name (for static map lookup)
        output_dir: Directory for debug images
        
    Returns:
        (player_position, confidence) or (None, 0.0)
    """
    print("\n" + "="*70)
    print("LOCALIZATION: ECC Method (with upscaling)")
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
    # STEP 1: Extract edges
    # ========================================================================
    print("\n[STEP 1] Extracting edges...")
    
    # Get minimap edges
    minimap_edges_canny = localizer.extract_minimap_edges_canny(
        frame_with_minimap,
        frame_without_minimap,
        use_oriented_filter=True,
        canny_low=50,
        canny_high=150
    )
    
    if minimap_edges_canny is None:
        print("[!] Failed to extract minimap edges")
        return None, 0.0
    
    save_debug_image(minimap_edges_canny, "ecc_01_minimap_edges_canny", output_dir)
    print(f"   Minimap edges: {minimap_edges_canny.shape[1]}×{minimap_edges_canny.shape[0]}")
    
    # Get static map Canny edges
    static_map_edges_canny_original = localizer.extract_static_map_edges_canny(
        white_threshold=120,
        canny_low=50,
        canny_high=150
    )
    save_debug_image(static_map_edges_canny_original, "ecc_02_static_map_edges_original", output_dir)
    print(f"   Static edges (original): {static_map_edges_canny_original.shape[1]}×{static_map_edges_canny_original.shape[0]}")
    
    # ========================================================================
    # STEP 2: Upscale static map to match minimap size
    # ========================================================================
    print("\n[STEP 2] Upscaling static map to match minimap...")
    
    target_shape = minimap_edges_canny.shape
    static_map_edges_upscaled = cv2.resize(
        static_map_edges_canny_original,
        (target_shape[1], target_shape[0]),
        interpolation=cv2.INTER_LINEAR
    )
    
    save_debug_image(static_map_edges_upscaled, "ecc_03_static_map_edges_upscaled", output_dir)
    print(f"   Static edges (upscaled): {static_map_edges_upscaled.shape[1]}×{static_map_edges_upscaled.shape[0]}")
    print(f"   ✅ Sizes now match!")
    
    # Comparison visualization
    viz_height = target_shape[0]
    minimap_rgb = cv2.cvtColor(minimap_edges_canny, cv2.COLOR_GRAY2BGR)
    static_rgb = cv2.cvtColor(static_map_edges_upscaled, cv2.COLOR_GRAY2BGR)
    
    composite = np.ones((viz_height, minimap_rgb.shape[1] + static_rgb.shape[1] + 20, 3), dtype=np.uint8) * 255
    composite[0:minimap_rgb.shape[0], 0:minimap_rgb.shape[1]] = minimap_rgb
    composite[0:static_rgb.shape[0], minimap_rgb.shape[1]+20:] = static_rgb
    
    cv2.putText(composite, f"Minimap: {minimap_edges_canny.shape[1]}×{minimap_edges_canny.shape[0]}", 
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(composite, f"Static (upscaled): {static_map_edges_upscaled.shape[1]}×{static_map_edges_upscaled.shape[0]}", 
                (minimap_rgb.shape[1] + 30, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    
    save_debug_image(composite, "ecc_04_comparison_same_size", output_dir)
    
    # ========================================================================
    # STEP 3: Prepare images for ECC (normalize, blur)
    # ========================================================================
    print("\n[STEP 3] Preparing images for ECC alignment...")
    
    query_ecc = localizer.matcher.aligner.prepare_for_ecc(minimap_edges_canny, blur_kernel=5)
    ref_ecc = localizer.matcher.aligner.prepare_for_ecc(static_map_edges_upscaled, blur_kernel=5)
    
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
    
    motion_type = 'HOMOGRAPHY'
    warp_mode = cv2.MOTION_HOMOGRAPHY
    warp_matrix = np.eye(3, 3, dtype=np.float32)
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
            
            # Warp query to reference
            warped = cv2.warpPerspective(
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
        
        # Scale warp matrix for next level
        if level < len(query_pyramid) - 1:
            warp_matrix[0, 2] *= 2
            warp_matrix[1, 2] *= 2
            warp_matrix[2, 0] *= 0.5
            warp_matrix[2, 1] *= 0.5
    
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
    # STEP 7: Final alignment on full resolution
    # ========================================================================
    print("\n[STEP 7] Computing final player position...")
    
    if alignment_results and not alignment_results[-1].get('failed', False):
        final_warp = alignment_results[-1]['warp_matrix']
        final_score = alignment_results[-1]['score']
        
        # Project player position (center of minimap)
        center = (minimap_edges_canny.shape[1] / 2, minimap_edges_canny.shape[0] / 2)
        
        # Account for upscaling: need to project back to original static map coordinates
        scale_factor = static_map_edges_canny_original.shape[1] / static_map_edges_upscaled.shape[1]
        
        # Project through warp matrix
        point_3d = np.array([center[0], center[1], 1], dtype=np.float32)
        projected = final_warp @ point_3d
        proj_x = projected[0] / (projected[2] + 1e-6)
        proj_y = projected[1] / (projected[2] + 1e-6)
        
        # Scale back to original static map size
        player_pos = (int(proj_x * scale_factor), int(proj_y * scale_factor))
        confidence = final_score
        
        print(f"   Player position (original static map coords): {player_pos}")
        print(f"   Confidence: {confidence:.4f}")
        
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
            method_name="ECC (with upscaling)",
            output_filename="ecc_10_player_position_on_map.png"
        )
        if viz_path:
            print(f"[+] Saved visualization: {viz_path.name}")
    else:
        print(f"\n[FAILED] ECC could not localize player")
    
    return player_pos, confidence


def localize_with_ransac(
    frame_with_minimap,
    frame_without_minimap,
    zone_name,
    output_dir
):
    """
    Localize player using RANSAC-based feature matching.
    
    Args:
        frame_with_minimap: Frame with minimap visible
        frame_without_minimap: Frame without minimap (background)
        zone_name: Current zone name (for static map lookup)
        output_dir: Directory for debug images
        
    Returns:
        (player_position, confidence) or (None, 0.0)
    """
    print("\n" + "="*70)
    print("LOCALIZATION: RANSAC Method (Feature Matching)")
    print("="*70)
    
    # Create localizer
    localizer = RANSACStaticMapLocalizer(
        debug=True,
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
    
    # Run full localization pipeline
    print("\n[*] Running RANSAC localization pipeline...")
    
    # Extract and visualize edges using localizer's methods
    minimap_edges = localizer.extract_minimap_edges(
        frame_with_minimap,
        frame_without_minimap,
        use_oriented_filter=True
    )
    
    if minimap_edges is not None and minimap_edges.size > 0:
        # Get Canny edges from localizer
        minimap_edges_canny = localizer.extract_minimap_edges_canny(
            frame_with_minimap,
            frame_without_minimap,
            use_oriented_filter=True,
            canny_low=50,
            canny_high=150
        )
        save_debug_image(minimap_edges_canny, "ransac_01_minimap_edges_canny", output_dir)
        
        # Get static map Canny edges from localizer
        static_map_edges_canny = localizer.extract_static_map_edges_canny(
            white_threshold=120,
            canny_low=50,
            canny_high=150
        )
        save_debug_image(static_map_edges_canny, "ransac_02_static_map_edges_canny", output_dir)
        
        # Create side-by-side visualization with edges
        if minimap_edges_canny is not None and static_map_edges_canny is not None:
            viz_height = max(minimap_edges_canny.shape[0], static_map_edges_canny.shape[0])
            
            # Convert to RGB for visualization
            minimap_rgb = cv2.cvtColor(minimap_edges_canny, cv2.COLOR_GRAY2BGR)
            static_rgb = cv2.cvtColor(static_map_edges_canny, cv2.COLOR_GRAY2BGR)
            
            # Resize to same height
            scale_minimap = viz_height / minimap_rgb.shape[0]
            minimap_resized = cv2.resize(minimap_rgb, None, fx=scale_minimap, fy=scale_minimap, interpolation=cv2.INTER_LINEAR)
            
            scale_static = viz_height / static_rgb.shape[0]
            static_resized = cv2.resize(static_rgb, None, fx=scale_static, fy=scale_static, interpolation=cv2.INTER_LINEAR)
            
            # Create composite
            composite = np.ones((viz_height, minimap_resized.shape[1] + static_resized.shape[1] + 20, 3), dtype=np.uint8) * 255
            composite[0:minimap_resized.shape[0], 0:minimap_resized.shape[1]] = minimap_resized
            composite[0:static_resized.shape[0], minimap_resized.shape[1]+20:] = static_resized
            
            # Add labels
            cv2.putText(composite, "Minimap Edges (Canny)", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            cv2.putText(composite, "Static Map (White+Canny)", (minimap_resized.shape[1] + 30, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            
            save_debug_image(composite, "ransac_03_comparison_edges", output_dir)
            print(f"[+] Saved comparison visualization with edges")
        
        # ================================================================
        # VISUALIZE SIFT KEYPOINTS AND MATCHES
        # ================================================================
        print(f"\n[*] SIFT Keypoint Detection and Matching...")
        
        if localizer.sift is not None:
            # Detect SIFT keypoints
            kp_minimap, des_minimap = localizer.sift.detectAndCompute(minimap_edges_canny, None)
            kp_static, des_static = localizer.sift.detectAndCompute(static_map_edges_canny, None)
            
            print(f"    Minimap keypoints: {len(kp_minimap)}")
            print(f"    Static map keypoints: {len(kp_static)}")
            
            # Visualize minimap keypoints
            if len(kp_minimap) > 0:
                minimap_kp_viz = cv2.drawKeypoints(
                    minimap_edges_canny,
                    kp_minimap,
                    None,
                    flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
                )
                save_debug_image(minimap_kp_viz, "ransac_05_minimap_sift_keypoints", output_dir)
                print(f"[+] Saved minimap SIFT keypoints visualization")
            
            # Visualize static map keypoints
            if len(kp_static) > 0:
                static_kp_viz = cv2.drawKeypoints(
                    static_map_edges_canny,
                    kp_static,
                    None,
                    flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
                )
                save_debug_image(static_kp_viz, "ransac_06_static_sift_keypoints", output_dir)
                print(f"[+] Saved static map SIFT keypoints visualization")
            
            # Feature matching
            if des_minimap is not None and des_static is not None and len(kp_minimap) > 0 and len(kp_static) > 0:
                print(f"\n    Feature matching with FLANN...")
                
                FLANN_INDEX_KDTREE = 1
                index_params = {"algorithm": FLANN_INDEX_KDTREE, "trees": 5}  # type: ignore
                search_params = {"checks": 50}  # type: ignore
                flann = cv2.FlannBasedMatcher(index_params, search_params)  # type: ignore
                
                matches = flann.knnMatch(des_minimap, des_static, k=2)
                
                # Apply Lowe's ratio test
                good_matches = []
                for match_pair in matches:
                    if len(match_pair) == 2:
                        m, n = match_pair
                        if m.distance < localizer.sift_ratio_threshold * n.distance:
                            good_matches.append(m)
                
                print(f"    Total matches: {len(matches)}")
                print(f"    Good matches (Lowe ratio {localizer.sift_ratio_threshold}): {len(good_matches)}")
                
                # Visualize good matches
                if len(good_matches) > 0:
                    # Convert edge images to RGB for match visualization
                    minimap_for_matches = cv2.cvtColor(minimap_edges_canny, cv2.COLOR_GRAY2BGR)
                    static_for_matches = cv2.cvtColor(static_map_edges_canny, cv2.COLOR_GRAY2BGR)
                    
                    match_img = cv2.drawMatches(
                        minimap_for_matches, kp_minimap,
                        static_for_matches, kp_static,
                        good_matches[:30],  # Show first 30 matches
                        None,
                        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
                        matchColor=(0, 255, 0),
                        singlePointColor=(255, 0, 0)
                    )
                    
                    # Add text info
                    cv2.putText(match_img, f"Good matches: {len(good_matches)}/{len(matches)}", (10, 30), 
                               cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    cv2.putText(match_img, f"(Showing first 30)", (10, 60), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (100, 100, 100), 1)
                    
                    save_debug_image(match_img, "ransac_07_feature_matches", output_dir)
                    print(f"[+] Saved feature matches visualization")

    
    # Use RANSAC to localize
    player_pos, confidence = localizer.localize(
        frame_with_minimap,
        frame_without_minimap
    )
    
    if player_pos:
        print(f"\n[SUCCESS] Player localized with RANSAC!")
        print(f"         Position: {player_pos}")
        print(f"         Confidence: {confidence:.3f}")
        
        # Visualize position on static map
        print(f"\n[*] Visualizing position on static map...")
        viz_path = localizer.visualize_player_position(
            player_pos,
            confidence,
            method_name="RANSAC",
            output_filename="ransac_04_player_position_on_map.png"
        )
        if viz_path:
            print(f"[+] Saved visualization: {viz_path.name}")
    else:
        print(f"\n[FAILED] RANSAC could not localize player")
    
    return player_pos, confidence


# ============================================================================
# DIAGNOSTIC ANALYSIS SUITE
# ============================================================================

def run_diagnostic_analysis(
    frame_with_minimap: np.ndarray,
    frame_without_minimap: np.ndarray,
    zone_name: str,
    output_dir: Path
):
    """
    Run comprehensive diagnostic analysis to identify improvement opportunities.
    
    Analyzes:
    1. Image dimensions and aspect ratios
    2. Pyramid analysis for ECC algorithm
    3. Edge detection quality
    4. Recommendations for improvement
    
    Args:
        frame_with_minimap: Frame with minimap visible
        frame_without_minimap: Frame without minimap (background)
        zone_name: Current zone name
        output_dir: Directory for outputs
    """
    print("\n" + "="*70)
    print("DIAGNOSTIC ANALYSIS: Image Dimensions & Algorithm Compatibility")
    print("="*70)
    
    # Create localizer to extract edges
    localizer = ECCStaticMapLocalizer(debug=False, output_dir=output_dir)
    map_path = load_zone_static_map(zone_name)
    if not localizer.load_static_map(map_path):
        print("[!] Could not load static map for analysis")
        return
    
    # Extract edges
    minimap_edges = localizer.extract_minimap_edges_canny(
        frame_with_minimap,
        frame_without_minimap,
        canny_low=50,
        canny_high=150
    )
    static_edges = localizer.extract_static_map_edges_canny(
        white_threshold=120,
        canny_low=50,
        canny_high=150
    )
    
    if minimap_edges is None or static_edges is None:
        print("[!] Could not extract edges for analysis")
        return
    
    print("\n[1] IMAGE DIMENSIONS")
    print("-" * 70)
    print(f"Minimap edges: {minimap_edges.shape[1]:4d} × {minimap_edges.shape[0]:4d} px")
    print(f"Static map edges: {static_edges.shape[1]:4d} × {static_edges.shape[0]:4d} px")
    
    width_ratio = minimap_edges.shape[1] / static_edges.shape[1]
    height_ratio = minimap_edges.shape[0] / static_edges.shape[0]
    area_ratio = (minimap_edges.shape[0] * minimap_edges.shape[1]) / (static_edges.shape[0] * static_edges.shape[1])
    
    print(f"\nSize ratios (minimap / static):")
    print(f"  Width:  {width_ratio:.2f}×")
    print(f"  Height: {height_ratio:.2f}×")
    print(f"  Area:   {area_ratio:.2f}×")
    
    avg_ratio = (width_ratio + height_ratio) / 2
    if avg_ratio > 2.0:
        print(f"\n⚠️  WARNING: Size mismatch {avg_ratio:.2f}× is > 2.0×")
        print("   This can cause ECC algorithm to fail at pyramid initialization!")
    
    # Analyze pyramid levels
    print("\n[2] ECC PYRAMID ANALYSIS (Multi-scale Alignment)")
    print("-" * 70)
    
    # Build pyramids like ECC does
    minimap_pyr = [minimap_edges]
    static_pyr = [static_edges]
    
    for level in range(3):
        minimap_pyr.append(cv2.resize(minimap_pyr[-1], (minimap_pyr[-1].shape[1]//2, minimap_pyr[-1].shape[0]//2)))
        static_pyr.append(cv2.resize(static_pyr[-1], (static_pyr[-1].shape[1]//2, static_pyr[-1].shape[0]//2)))
    
    print(f"{'Level':<6} {'Minimap':<15} {'Static':<15} {'Ratio':<8} {'Status':<20}")
    print("-" * 70)
    
    min_viable_size = 30
    for level, (m, s) in enumerate(zip(minimap_pyr, static_pyr)):
        ratio = m.shape[1] / s.shape[1] if s.shape[1] > 0 else 0
        status = "✅ OK" if ratio < 2.0 and m.shape[1] > min_viable_size else "❌ PROBLEM"
        print(f"{level:<6} {m.shape[1]:3d}×{m.shape[0]:3d}      {s.shape[1]:3d}×{s.shape[0]:3d}      {ratio:5.2f}×   {status:<20}")
    
    print(f"\n💡 ECC works best when all pyramid levels have ratio < 2.0×")
    if avg_ratio > 2.0:
        print(f"   Current ratio {avg_ratio:.2f}× → Consider RESCALING before alignment")
    
    # Edge coverage analysis
    print("\n[3] EDGE CONTENT ANALYSIS")
    print("-" * 70)
    
    minimap_coverage = np.count_nonzero(minimap_edges) / minimap_edges.size * 100
    static_coverage = np.count_nonzero(static_edges) / static_edges.size * 100
    
    print(f"Minimap edge coverage: {minimap_coverage:5.2f}%")
    print(f"Static map edge coverage: {static_coverage:5.2f}%")
    
    if minimap_coverage < 2.0 or static_coverage < 2.0:
        print(f"⚠️  WARNING: Low edge coverage detected")
    
    # Recommendations
    print("\n[4] IDENTIFIED PROBLEMS & SOLUTIONS")
    print("-" * 70)
    
    problems = []
    solutions = []
    
    if avg_ratio > 2.0:
        problems.append(f"Size mismatch: {avg_ratio:.2f}× → ECC pyramid fails")
        solutions.append("▶ SOLUTION A: Rescale minimap to match static map size before alignment")
        solutions.append(f"  Test: Resize {minimap_edges.shape} → {static_edges.shape}")
    
    if minimap_coverage < 2.0 or static_coverage < 2.0:
        problems.append(f"Low edge coverage → weak feature for alignment")
        solutions.append("▶ SOLUTION B: Improve edge detection parameters (Canny thresholds, Gabor angles)")
    
    if len(problems) == 0:
        problems.append("No critical issues detected")
        solutions.append("✅ Images appear compatible with ECC algorithm")
    
    for prob in problems:
        print(f"❌ {prob}")
    
    print()
    for sol in solutions:
        print(f"{sol}")
    
    # Create diagnostic visualization
    print("\n[5] DIAGNOSTIC VISUALIZATION")
    print("-" * 70)
    
    # Create composite showing dimensions
    minimap_display = cv2.cvtColor(minimap_edges, cv2.COLOR_GRAY2BGR)
    static_display = cv2.cvtColor(static_edges, cv2.COLOR_GRAY2BGR)
    
    # Resize for display (same height)
    display_height = 400
    scale_mini = display_height / minimap_display.shape[0]
    scale_static = display_height / static_display.shape[0]
    
    minimap_display = cv2.resize(minimap_display, None, fx=scale_mini, fy=scale_mini, interpolation=cv2.INTER_LINEAR)
    static_display = cv2.resize(static_display, None, fx=scale_static, fy=scale_static, interpolation=cv2.INTER_LINEAR)
    
    # Create composite with proper sizing
    total_width = minimap_display.shape[1] + static_display.shape[1] + 60
    canvas_height = max(minimap_display.shape[0], static_display.shape[0]) + 100
    composite = np.ones((canvas_height, total_width, 3), dtype=np.uint8) * 240
    
    # Place images
    composite[0:minimap_display.shape[0], 20:20+minimap_display.shape[1]] = minimap_display
    static_x_start = minimap_display.shape[1] + 40
    static_x_end = static_x_start + static_display.shape[1]
    composite[0:static_display.shape[0], static_x_start:static_x_end] = static_display
    
    # Add labels with dimensions
    cv2.putText(composite, f"Minimap: {minimap_edges.shape[1]}×{minimap_edges.shape[0]}", 
                (30, minimap_display.shape[0] + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    cv2.putText(composite, f"Static: {static_edges.shape[1]}×{static_edges.shape[0]}", 
                (static_x_start + 10, static_display.shape[0] + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    
    ratio_str = f"Ratio: {width_ratio:.2f}× W × {height_ratio:.2f}× H"
    if avg_ratio > 2.0:
        cv2.putText(composite, f"⚠️ {ratio_str} - SIZE MISMATCH!", 
                    (30, minimap_display.shape[0] + 70), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
    else:
        cv2.putText(composite, f"✅ {ratio_str} - Compatible", 
                    (30, minimap_display.shape[0] + 70), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 200, 0), 2)
    
    save_debug_image(composite, "diagnostic_01_image_dimensions", output_dir)
    
    print(f"[+] Saved diagnostic visualization")
    print()

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
    
    # Create output directory
    output_dir = Path(__file__).parent / "data" / "screenshots" / "outputs" / "localization_test_v3_modular"
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
        # DIAGNOSTIC ANALYSIS (before comparing methods)
        # ================================================================
        print("\n" + "="*70)
        print("STEP 2: Diagnostic Analysis")
        print("="*70)
        
        run_diagnostic_analysis(
            frame_with_minimap,
            frame_without_minimap,
            zone_name,
            output_dir
        )
        
        # ================================================================
        # MAIN LOCALIZATION: Compare ECC vs RANSAC
        # ================================================================
        print("\n" + "="*70)
        print("STEP 3: Localization Methods Comparison")
        print("="*70)
        
        player_pos_ecc, confidence_ecc = localize_with_ecc(
            frame_with_minimap,
            frame_without_minimap,
            zone_name,
            output_dir
        )
        
        player_pos_ransac, confidence_ransac = localize_with_ransac(
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
    print("FINAL RESULTS & RECOMMENDATIONS")
    print("=" * 70)
    print()
    
    # Display comparison
    print("LOCALIZATION RESULTS")
    print("-" * 70)
    print(f"{'Method':<20} {'Position':<25} {'Confidence':<15}")
    print("-" * 70)
    
    if player_pos_ecc:
        print(f"{'ECC':<20} {str(player_pos_ecc):<25} {confidence_ecc:.3f}")
    else:
        print(f"{'ECC':<20} {'FAILED':<25} {confidence_ecc:.3f}")
    
    if player_pos_ransac:
        print(f"{'RANSAC':<20} {str(player_pos_ransac):<25} {confidence_ransac:.3f}")
    else:
        print(f"{'RANSAC':<20} {'FAILED':<25} {confidence_ransac:.3f}")
    
    print("-" * 70)
    
    # Determine winner and provide insights
    print("\nALGORITHM ANALYSIS")
    print("-" * 70)
    
    if confidence_ecc == 0.0 and confidence_ransac > 0:
        print(f"⚠️  ECC FAILED (confidence 0.0) - Likely cause: IMAGE SIZE MISMATCH")
        print(f"    RANSAC SUCCEEDED (confidence {confidence_ransac:.3f})")
        print(f"\n💡 RECOMMENDATION: Rescale minimap to match static map dimensions")
        print(f"    before running ECC alignment. See diagnostic output above.")
        print(f"    Alternative: Continue using RANSAC (feature-based, more tolerant)")
    elif player_pos_ecc and player_pos_ransac:
        # Both succeeded - compare confidence
        if confidence_ecc > confidence_ransac:
            print(f"✅ WINNER: ECC (confidence {confidence_ecc:.3f} > {confidence_ransac:.3f})")
        elif confidence_ransac > confidence_ecc:
            print(f"✅ WINNER: RANSAC (confidence {confidence_ransac:.3f} > {confidence_ecc:.3f})")
        else:
            print(f"🤝 TIE: Both methods equal (confidence {confidence_ecc:.3f})")
    elif player_pos_ecc:
        print(f"✅ WINNER: ECC (RANSAC failed)")
    elif player_pos_ransac:
        print(f"✅ WINNER: RANSAC (ECC failed)")
    else:
        print(f"❌ BOTH METHODS FAILED - See diagnostic analysis for potential issues")
    
    print()
    print("=" * 70)
    print("DEBUG IMAGES SAVED TO:")
    print(f"  {output_dir}")
    print("=" * 70)


if __name__ == "__main__":
    main()
