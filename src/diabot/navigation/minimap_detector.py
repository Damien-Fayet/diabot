"""
Minimap POI detection using color-based analysis.

Detects:
- Waypoints (blue/cyan)
- Player position (yellow/white)
- Monsters (red)
- Exits/Portals (bright colors)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import cv2
import numpy as np


@dataclass
class MinimapPOI:
    """Detected POI on minimap."""
    position: Tuple[int, int]  # (x, y) on minimap
    poi_type: str  # 'waypoint', 'player', 'monster', 'exit'
    confidence: float  # 0.0 to 1.0
    color: Tuple[int, int, int] = None  # BGR color for debugging


class MinimapPOIDetector:
    """
    Detect Points of Interest on Diablo 2 minimap using color analysis.
    """

    def __init__(self, debug: bool = False):
        """
        Initialize minimap POI detector.

        Args:
            debug: Enable debug output and visualization
        """
        self.debug = debug
        
        # HSV color ranges for detection
        # Format: (lower_bound, upper_bound) in HSV
        self.color_ranges = {
            'waypoint': [
                # Blue waypoint markers
                (np.array([100, 100, 100]), np.array([130, 255, 255])),
                # Cyan variant
                (np.array([85, 100, 100]), np.array([100, 255, 255])),
            ],
            'player': [
                # Blue cross (player marker on minimap)
                (np.array([105, 120, 120]), np.array([130, 255, 255])),
                # Cyan/blue variant
                (np.array([90, 100, 120]), np.array([110, 255, 255])),
            ],
            'monster': [
                # Red monsters
                (np.array([0, 150, 150]), np.array([10, 255, 255])),
                # Red variant 2
                (np.array([170, 150, 150]), np.array([180, 255, 255])),
            ],
            'exit': [
                # Bright red/orange exits
                (np.array([0, 200, 200]), np.array([15, 255, 255])),
            ],
        }
        
        # Morphological kernel for noise reduction
        self.kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))

    def detect(self, minimap: np.ndarray) -> List[MinimapPOI]:
        """
        Detect all POIs on minimap.

        Args:
            minimap: Minimap image (BGR numpy array)

        Returns:
            List of detected POIs
        """
        if minimap is None or minimap.size == 0:
            return []
        
        pois = []
        
        # Convert to HSV for color detection
        hsv = cv2.cvtColor(minimap, cv2.COLOR_BGR2HSV)
        
        # Detect each type of POI
        for poi_type, ranges in self.color_ranges.items():
            type_pois = self._detect_poi_type(minimap, hsv, poi_type, ranges)
            pois.extend(type_pois)
        
        if self.debug and pois:
            print(f"[MINIMAP] Detected {len(pois)} POIs")
        
        return pois

    def _detect_poi_type(
        self,
        minimap: np.ndarray,
        hsv: np.ndarray,
        poi_type: str,
        color_ranges: List[Tuple[np.ndarray, np.ndarray]]
    ) -> List[MinimapPOI]:
        """
        Detect POIs of a specific type using color ranges.

        Args:
            minimap: Original minimap image
            hsv: HSV version of minimap
            poi_type: Type of POI to detect
            color_ranges: List of (lower, upper) HSV bounds

        Returns:
            List of detected POIs
        """
        # Combine all color range masks
        combined_mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
        
        for lower, upper in color_ranges:
            mask = cv2.inRange(hsv, lower, upper)
            combined_mask = cv2.bitwise_or(combined_mask, mask)
        
        # Morphological operations to reduce noise
        combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_OPEN, self.kernel)
        combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, self.kernel)
        
        # Find contours
        contours, _ = cv2.findContours(
            combined_mask,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )
        
        pois = []
        
        for contour in contours:
            area = cv2.contourArea(contour)
            
            # Filter by size (too small = noise, too large = not a POI)
            if area < 5 or area > 500:
                continue
            
            # Get centroid
            M = cv2.moments(contour)
            if M["m00"] == 0:
                continue
            
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            
            # Calculate confidence based on area and circularity
            perimeter = cv2.arcLength(contour, True)
            if perimeter == 0:
                continue
            
            circularity = 4 * np.pi * area / (perimeter * perimeter)
            confidence = min(1.0, circularity * (area / 100.0))
            
            # Get average color for debugging
            mask_single = np.zeros_like(combined_mask)
            cv2.drawContours(mask_single, [contour], -1, 255, -1)
            mean_color = cv2.mean(minimap, mask=mask_single)[:3]
            
            poi = MinimapPOI(
                position=(cx, cy),
                poi_type=poi_type,
                confidence=confidence,
                color=tuple(int(c) for c in mean_color),
            )
            
            pois.append(poi)
        
        return pois

    def visualize_detections(
        self,
        minimap: np.ndarray,
        pois: List[MinimapPOI],
        save_path: str = None
    ) -> np.ndarray:
        """
        Visualize detected POIs on minimap.

        Args:
            minimap: Original minimap image
            pois: List of detected POIs
            save_path: Optional path to save visualization

        Returns:
            Annotated minimap image
        """
        output = minimap.copy()
        
        # Color map for visualization
        colors = {
            'waypoint': (255, 255, 0),    # Cyan
            'player': (0, 255, 255),      # Yellow
            'monster': (0, 0, 255),       # Red
            'exit': (0, 165, 255),        # Orange
        }
        
        for poi in pois:
            color = colors.get(poi.poi_type, (255, 255, 255))
            x, y = poi.position
            
            # Draw circle
            cv2.circle(output, (x, y), 5, color, 2)
            
            # Draw type label
            label = f"{poi.poi_type[:3]}"
            cv2.putText(
                output,
                label,
                (x + 8, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,
                color,
                1,
            )
        
        if save_path:
            cv2.imwrite(save_path, output)
        
        return output


# Demo code for testing
if __name__ == "__main__":
    from pathlib import Path
    
    # Test on saved minimap
    minimap_path = Path("data/screenshots/outputs/live_capture/minimap_extracted.png")
    
    if minimap_path.exists():
        minimap = cv2.imread(str(minimap_path))
        
        detector = MinimapPOIDetector(debug=True)
        pois = detector.detect(minimap)
        
        print(f"\nOK Detected {len(pois)} POIs on minimap")
        for poi in pois:
            print(f"  - {poi.poi_type}: {poi.position} ({poi.confidence:.2f})")
        
        # Visualize
        vis = detector.visualize_detections(minimap, pois)
        output_path = minimap_path.parent / "minimap_pois_detected.png"
        cv2.imwrite(str(output_path), vis)
        print(f"\nOK Saved visualization to {output_path}")
    
    else:
        print(f"WARNING No minimap found at {minimap_path}")
