#!/usr/bin/env python3
"""Debug script to visualize SIFT keypoints for diagnostic."""

import sys
from pathlib import Path
import cv2
import numpy as np
import base64
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent / "src"))

from diabot.navigation import RANSACStaticMapLocalizer, load_zone_static_map


def visualize_keypoints(image, keypoints, title="Keypoints"):
    """Draw keypoints on image."""
    img_kp = cv2.drawKeypoints(
        image,
        keypoints,
        None,
        flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
    )
    return img_kp


def main():
    """Main diagnostic function."""
    output_dir = Path(__file__).parent / "data" / "screenshots" / "outputs" / "sift_debug"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 70)
    print("SIFT KEYPOINTS DIAGNOSTIC")
    print("=" * 70)
    
    # Load test images
    inputs_dir = Path(__file__).parent / "data" / "screenshots" / "inputs"
    
    without_images = sorted(inputs_dir.glob("*_without_minimap*.png"))
    with_images = sorted(inputs_dir.glob("*_with_minimap*.png"))
    
    if not without_images or not with_images:
        print(f"[!] No test images found in {inputs_dir}")
        return
    
    frame_without = cv2.imread(str(without_images[-1]))
    frame_with = cv2.imread(str(with_images[-1]))
    
    if frame_without is None or frame_with is None:
        print("[!] Failed to load test images")
        return
    
    print(f"[+] Loaded test images")
    print(f"    Frame size: {frame_with.shape}")
    
    # Create RANSAC localizer
    localizer = RANSACStaticMapLocalizer(debug=True, output_dir=output_dir)
    
    # Load static map
    zone_name = 'Rogue Encampment'
    map_path = load_zone_static_map(zone_name)
    
    if map_path is None:
        print(f"[!] No static map for {zone_name}")
        return
    
    if not localizer.load_static_map(map_path):
        print(f"[!] Failed to load static map")
        return
    
    print(f"[+] Loaded static map: {localizer.static_map.shape}")
    
    # Extract minimap edges
    print("\n[*] Extracting minimap edges...")
    minimap_edges = localizer.extract_minimap_edges(
        frame_with,
        frame_without,
        use_oriented_filter=True
    )
    
    if minimap_edges is None:
        print("[!] Failed to extract minimap edges")
        return
    
    print(f"[+] Minimap edges: {minimap_edges.shape}, non-zero: {np.count_nonzero(minimap_edges)}")
    
    # Apply Canny
    minimap_edges_canny = cv2.Canny(minimap_edges, 50, 150)
    print(f"[+] After Canny: non-zero: {np.count_nonzero(minimap_edges_canny)}")
    
    # Extract static map edges
    print("\n[*] Extracting static map edges...")
    static_map_edges_canny = localizer.extract_static_map_edges_canny(
        white_threshold=120,
        canny_low=50,
        canny_high=150
    )
    
    if static_map_edges_canny is None:
        print("[!] Failed to extract static map edges")
        return
    
    print(f"[+] Static map edges: {static_map_edges_canny.shape}, non-zero: {np.count_nonzero(static_map_edges_canny)}")
    
    # Detect SIFT keypoints
    print("\n[*] Detecting SIFT keypoints...")
    
    if localizer.sift is None:
        print("[!] SIFT not available")
        return
    
    kp_minimap, des_minimap = localizer.sift.detectAndCompute(minimap_edges_canny, None)
    kp_static, des_static = localizer.sift.detectAndCompute(static_map_edges_canny, None)
    
    print(f"[+] Minimap keypoints: {len(kp_minimap)}")
    print(f"[+] Static map keypoints: {len(kp_static)}")
    
    # Visualize minimap keypoints
    print("\n[*] Visualizing minimap keypoints...")
    minimap_rgb = cv2.cvtColor(minimap_edges_canny, cv2.COLOR_GRAY2BGR)
    minimap_kp_viz = visualize_keypoints(minimap_rgb, kp_minimap, "Minimap Keypoints")
    cv2.imwrite(str(output_dir / "01_minimap_keypoints.png"), minimap_kp_viz)
    print(f"[+] Saved: 01_minimap_keypoints.png")
    
    # Visualize static map keypoints
    print("\n[*] Visualizing static map keypoints...")
    static_rgb = cv2.cvtColor(static_map_edges_canny, cv2.COLOR_GRAY2BGR)
    static_kp_viz = visualize_keypoints(static_rgb, kp_static, "Static Map Keypoints")
    cv2.imwrite(str(output_dir / "02_static_keypoints.png"), static_kp_viz)
    print(f"[+] Saved: 02_static_keypoints.png")
    
    # Create side-by-side comparison
    print("\n[*] Creating side-by-side comparison...")
    
    # Resize to same height for display
    h1, w1 = minimap_kp_viz.shape[:2]
    h2, w2 = static_kp_viz.shape[:2]
    max_h = max(h1, h2)
    
    scale1 = max_h / h1
    scale2 = max_h / h2
    
    minimap_resized = cv2.resize(minimap_kp_viz, None, fx=scale1, fy=scale1, interpolation=cv2.INTER_LINEAR)
    static_resized = cv2.resize(static_kp_viz, None, fx=scale2, fy=scale2, interpolation=cv2.INTER_LINEAR)
    
    # Create composite
    composite = np.ones((max_h, minimap_resized.shape[1] + static_resized.shape[1] + 20, 3), dtype=np.uint8) * 255
    composite[0:minimap_resized.shape[0], 0:minimap_resized.shape[1]] = minimap_resized
    composite[0:static_resized.shape[0], minimap_resized.shape[1]+20:] = static_resized
    
    cv2.putText(composite, f"Minimap ({len(kp_minimap)} kpts)", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    cv2.putText(composite, f"Static ({len(kp_static)} kpts)", (minimap_resized.shape[1] + 30, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    
    cv2.imwrite(str(output_dir / "03_keypoints_comparison.png"), composite)
    print(f"[+] Saved: 03_keypoints_comparison.png")
    
    # Feature matching
    if des_minimap is not None and des_static is not None:
        print("\n[*] Feature matching...")
        
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
                if m.distance < 0.7 * n.distance:
                    good_matches.append(m)
        
        print(f"[+] Total matches: {len(matches)}")
        print(f"[+] Good matches (Lowe ratio test): {len(good_matches)}")
        
        # Draw matches
        if len(good_matches) > 0:
            print("\n[*] Visualizing matches...")
            
            match_img = cv2.drawMatches(
                minimap_rgb, kp_minimap,
                static_rgb, kp_static,
                good_matches[:50],  # Show first 50 matches
                None,
                flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
            )
            
            cv2.imwrite(str(output_dir / "04_feature_matches.png"), match_img)
            print(f"[+] Saved: 04_feature_matches.png (showing first 50 matches)")
    
    # Stats
    print("\n" + "=" * 70)
    print("DIAGNOSTIC SUMMARY")
    print("=" * 70)
    print(f"Minimap edges coverage: {np.count_nonzero(minimap_edges_canny) / minimap_edges_canny.size * 100:.2f}%")
    print(f"Static map edges coverage: {np.count_nonzero(static_map_edges_canny) / static_map_edges_canny.size * 100:.2f}%")
    print(f"Minimap SIFT keypoints: {len(kp_minimap)}")
    print(f"Static map SIFT keypoints: {len(kp_static)}")
    
    if des_minimap is not None and des_static is not None:
        print(f"Feature matches (raw): {len(matches)}")
        print(f"Feature matches (Lowe ratio): {len(good_matches)}")
    
    print(f"\nDebug images saved to: {output_dir}")


if __name__ == "__main__":
    main()
