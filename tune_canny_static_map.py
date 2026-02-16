#!/usr/bin/env python3
"""Find optimal Canny thresholds for static map edge detection."""

import sys
from pathlib import Path
import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).parent / "src"))

from diabot.navigation import load_zone_static_map


def test_canny_thresholds():
    """Test different Canny thresholds."""
    
    output_dir = Path(__file__).parent / "data" / "screenshots" / "outputs" / "canny_tuning"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load static map
    zone_name = 'Rogue Encampment'
    map_path = load_zone_static_map(zone_name)
    
    if map_path is None:
        print(f"[!] No static map for {zone_name}")
        return
    
    static_map = cv2.imread(str(map_path), cv2.IMREAD_GRAYSCALE)
    if static_map is None:
        print(f"[!] Failed to load static map")
        return
    
    print(f"Static map: {static_map.shape}")
    print("\n" + "="*70)
    print("TESTING CANNY THRESHOLDS")
    print("="*70)
    
    # Different threshold combinations
    thresholds = [
        (30, 100),   # Very permissive
        (50, 150),   # Current
        (50, 100),   # Lower high threshold
        (40, 120),   # Middle ground
    ]
    
    # Filter to white pixels first
    ret, static_map_white = cv2.threshold(static_map, 120, 255, cv2.THRESH_BINARY)
    
    results = []
    
    for low, high in thresholds:
        edges = cv2.Canny(static_map_white, low, high)
        non_zero = np.count_nonzero(edges)
        coverage = non_zero / edges.size * 100
        
        # Detect SIFT keypoints
        sift = cv2.SIFT_create()
        kp, des = sift.detectAndCompute(edges, None)
        
        results.append({
            'thresholds': (low, high),
            'edges_pixels': non_zero,
            'coverage': coverage,
            'keypoints': len(kp),
            'descriptors': des is not None,
        })
        
        print(f"Canny ({low:3d}, {high:3d}): {non_zero:6d} pixels ({coverage:5.2f}%), {len(kp):4d} keypoints")
        
        # Save edge visualization
        edges_rgb = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
        cv2.putText(edges_rgb, f"Canny({low},{high}) - {len(kp)} kpts", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        cv2.imwrite(str(output_dir / f"canny_{low:03d}_{high:03d}.png"), edges_rgb)
    
    print("\n" + "="*70)
    print("RECOMMENDATION")
    print("="*70)
    
    # Find best (highest keypoints)
    best = max(results, key=lambda x: x['keypoints'])
    print(f"Best for keypoints: Canny({best['thresholds'][0]}, {best['thresholds'][1]}) → {best['keypoints']} kpts")
    
    # Find best coverage
    best_cov = max(results, key=lambda x: x['coverage'])
    print(f"Best for coverage: Canny({best_cov['thresholds'][0]}, {best_cov['thresholds'][1]}) → {best_cov['coverage']:.2f}%")
    
    print(f"\nDebug images saved to: {output_dir}")


if __name__ == "__main__":
    test_canny_thresholds()
