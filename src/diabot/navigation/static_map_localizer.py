"""
Static map localization using template matching.

For offline/solo play where maps are deterministic, this module uses
pre-generated map images to localize the player by matching the current
minimap against the static reference map.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np


@dataclass
class LocalizationResult:
    """Result of map localization."""
    found: bool
    position: Optional[Tuple[int, int]]  # (x, y) on static map
    confidence: float  # Match confidence 0.0-1.0
    match_quality: str  # "excellent", "good", "poor", "failed"


class StaticMapLocalizer:
    """
    Localize player position on a static pre-generated map.
    
    Uses template matching to find where the current minimap view
    appears on the reference map image.
    """
    
    def __init__(
        self,
        static_map_path: Optional[Path] = None,
        debug: bool = False
    ):
        """
        Initialize static map localizer.
        
        Args:
            static_map_path: Path to static reference map image
            debug: Enable debug output and visualizations
        """
        self.debug = debug
        self.static_map_path = static_map_path
        self.static_map: Optional[np.ndarray] = None
        self.static_map_gray: Optional[np.ndarray] = None
        
        # Load static map if path provided
        if static_map_path:
            self.load_static_map(static_map_path)
        
        # Confidence thresholds for match quality
        self.thresholds = {
            'excellent': 0.75,  # Very confident match
            'good': 0.60,       # Acceptable match
            'poor': 0.45,       # Uncertain match
        }
        
        if self.debug:
            print(f"[StaticMapLocalizer] Initialized")
    
    def load_static_map(self, map_path: Path) -> bool:
        """
        Load static reference map from file.
        
        Args:
            map_path: Path to map image (PNG)
            
        Returns:
            True if loaded successfully
        """
        map_path = Path(map_path)
        
        if not map_path.exists():
            if self.debug:
                print(f"[StaticMapLocalizer] ❌ Map not found: {map_path}")
            return False
        
        # Load map
        self.static_map = cv2.imread(str(map_path))
        
        if self.static_map is None:
            if self.debug:
                print(f"[StaticMapLocalizer] ❌ Failed to load: {map_path}")
            return False
        
        # Convert to grayscale for matching
        self.static_map_gray = cv2.cvtColor(self.static_map, cv2.COLOR_BGR2GRAY)
        
        self.static_map_path = map_path
        
        if self.debug:
            h, w = self.static_map.shape[:2]
            print(f"[StaticMapLocalizer] ✓ Loaded map: {map_path.name} ({w}x{h})")
        
        return True
    
    def localize(
        self,
        minimap_grid: np.ndarray,
        method: int = cv2.TM_CCOEFF_NORMED,
        use_edges: bool = True,
        multi_scale: bool = True
    ) -> LocalizationResult:
        """
        Find player position on static map using current minimap.
        
        Args:
            minimap_grid: Current minimap observation (64x64 or similar)
            method: OpenCV template matching method
            use_edges: Use edge detection for better structural matching
            multi_scale: Try multiple scales to find best match
            
        Returns:
            LocalizationResult with position and confidence
        """
        if self.static_map_gray is None:
            if self.debug:
                print("[StaticMapLocalizer] ❌ No static map loaded")
            return LocalizationResult(
                found=False,
                position=None,
                confidence=0.0,
                match_quality="failed"
            )
        
        # Prepare minimap template
        if len(minimap_grid.shape) == 3:
            template_gray = cv2.cvtColor(minimap_grid, cv2.COLOR_BGR2GRAY)
        else:
            template_gray = minimap_grid
        
        # Ensure template is 8-bit
        if template_gray.dtype != np.uint8:
            template_gray = (template_gray * 255).astype(np.uint8)
        
        # Prepare static map
        map_h, map_w = self.static_map_gray.shape
        
        # Apply edge detection if requested
        if use_edges:
            map_img = cv2.Canny(self.static_map_gray, 50, 150)
            template_img = cv2.Canny(template_gray, 50, 150)
            
            # Dilate edges slightly to make them more robust
            kernel = np.ones((3, 3), np.uint8)
            map_img = cv2.dilate(map_img, kernel, iterations=1)
            template_img = cv2.dilate(template_img, kernel, iterations=1)
        else:
            map_img = self.static_map_gray
            template_img = template_gray
        
        # Adjust thresholds for edge-based matching (typically lower)
        if use_edges:
            thresholds = {
                'excellent': 0.50,
                'good': 0.35,
                'poor': 0.25,
            }
        else:
            thresholds = self.thresholds
        
        # Multi-scale matching to handle zoom differences
        best_result = None
        best_confidence = -1.0
        best_scale = 1.0
        
        if multi_scale:
            # Test multiple scales (minimap might be different size than expected)
            scales = [0.25, 0.5, 0.75, 1.0, 1.5, 2.0]
        else:
            scales = [1.0]
        
        for scale in scales:
            # Resize template
            new_w = int(template_img.shape[1] * scale)
            new_h = int(template_img.shape[0] * scale)
            
            # Skip if too large or too small
            if new_w > map_w or new_h > map_h:
                continue
            if new_w < 10 or new_h < 10:
                continue
            
            scaled_template = cv2.resize(template_img, (new_w, new_h))
            
            # Match
            result = cv2.matchTemplate(map_img, scaled_template, method)
            min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
            
            # Get confidence
            if method in [cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED]:
                confidence = 1.0 - min_val
                match_loc = min_loc
            else:
                confidence = max_val
                match_loc = max_loc
            
            # Track best match
            if confidence > best_confidence:
                best_confidence = confidence
                best_scale = scale
                center_x = match_loc[0] + new_w // 2
                center_y = match_loc[1] + new_h // 2
                best_result = {
                    'position': (center_x, center_y),
                    'template_size': (new_w, new_h)
                }
        
        # No valid match found
        if best_result is None:
            if self.debug:
                print("[StaticMapLocalizer] ❌ No valid scale found")
            return LocalizationResult(
                found=False,
                position=None,
                confidence=0.0,
                match_quality="failed"
            )
        
        # Determine match quality
        if best_confidence >= thresholds['excellent']:
            quality = "excellent"
        elif best_confidence >= thresholds['good']:
            quality = "good"
        elif best_confidence >= thresholds['poor']:
            quality = "poor"
        else:
            quality = "failed"
        
        if self.debug:
            px, py = best_result['position']
            print(f"[StaticMapLocalizer] Match: ({px}, {py}) "
                  f"confidence={best_confidence:.3f} quality={quality} "
                  f"scale={best_scale:.2f}x (edges={use_edges})")
        
        found = best_confidence >= thresholds['poor']
        
        return LocalizationResult(
            found=found,
            position=best_result['position'] if found else None,
            confidence=best_confidence,
            match_quality=quality
        )
    
    def visualize_match(
        self,
        minimap_grid: np.ndarray,
        result: LocalizationResult,
        output_path: Optional[Path] = None
    ) -> np.ndarray:
        """
        Create visualization of localization result.
        
        Args:
            minimap_grid: Minimap used for matching
            result: Localization result
            output_path: Optional path to save visualization
            
        Returns:
            Visualization image
        """
        if self.static_map is None or not result.found:
            return np.zeros((100, 100, 3), dtype=np.uint8)
        
        # Create copy of static map
        vis = self.static_map.copy()
        
        # Draw player position
        if result.position:
            px, py = result.position
            
            # Draw crosshair at player position
            color = (0, 255, 0) if result.match_quality == "excellent" else \
                    (0, 200, 200) if result.match_quality == "good" else \
                    (0, 100, 255)
            
            cv2.drawMarker(vis, (px, py), color, 
                          markerType=cv2.MARKER_CROSS, 
                          markerSize=20, thickness=2)
            
            # Draw circle around position
            cv2.circle(vis, (px, py), 30, color, 2)
            
            # Draw minimap footprint (rectangle)
            template_h, template_w = minimap_grid.shape[:2]
            top_left = (px - template_w // 2, py - template_h // 2)
            bottom_right = (px + template_w // 2, py + template_h // 2)
            cv2.rectangle(vis, top_left, bottom_right, color, 2)
            
            # Add text
            text = f"Pos: ({px}, {py})"
            cv2.putText(vis, text, (px + 35, py - 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            
            conf_text = f"Conf: {result.confidence:.2f} ({result.match_quality})"
            cv2.putText(vis, conf_text, (px + 35, py),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        # Save if requested
        if output_path:
            cv2.imwrite(str(output_path), vis)
            if self.debug:
                print(f"[StaticMapLocalizer] Saved visualization: {output_path}")
        
        return vis
    
    def get_direction_to_target(
        self,
        player_pos: Tuple[int, int],
        target_pos: Tuple[int, int]
    ) -> Tuple[float, float, float]:
        """
        Calculate direction vector from player to target.
        
        Args:
            player_pos: Player position (x, y) on map
            target_pos: Target position (x, y) on map
            
        Returns:
            (angle_degrees, distance, normalized_dx, normalized_dy)
        """
        px, py = player_pos
        tx, ty = target_pos
        
        # Calculate vector
        dx = tx - px
        dy = ty - py
        
        # Calculate distance
        distance = np.sqrt(dx**2 + dy**2)
        
        # Calculate angle (in degrees, 0 = right, 90 = down)
        angle = np.degrees(np.arctan2(dy, dx))
        
        # Normalize direction
        if distance > 0:
            ndx = dx / distance
            ndy = dy / distance
        else:
            ndx = 0.0
            ndy = 0.0
        
        return angle, distance, ndx, ndy
    
    def mark_poi_on_map(
        self,
        poi_name: str,
        position: Tuple[int, int],
        color: Tuple[int, int, int] = (0, 255, 255)
    ):
        """
        Permanently mark a POI on the static map (for visualization).
        
        Args:
            poi_name: Name of POI
            position: (x, y) position on map
            color: BGR color for marker
        """
        if self.static_map is None:
            return
        
        x, y = position
        
        # Draw marker
        cv2.drawMarker(self.static_map, (x, y), color,
                      markerType=cv2.MARKER_DIAMOND,
                      markerSize=15, thickness=2)
        
        # Add label
        cv2.putText(self.static_map, poi_name, (x + 20, y - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
        
        if self.debug:
            print(f"[StaticMapLocalizer] Marked POI: {poi_name} at ({x}, {y})")


def load_zone_static_map(zone_name: str, maps_dir: Path = None) -> Optional[Path]:
    """
    Load static map for a zone by name.
    
    Args:
        zone_name: Zone name (e.g., "ROGUE ENCAMPMENT", "Blood Moor")
        maps_dir: Directory containing static maps
        
    Returns:
        Path to map file if found, None otherwise
    """
    if maps_dir is None:
        maps_dir = Path("data/maps/minimap_images")
    
    maps_dir = Path(maps_dir)
    
    # Try exact match first
    zone_file = maps_dir / f"{zone_name}.png"
    if zone_file.exists():
        return zone_file
    
    # Try with act prefix (A1-, A2-, etc.)
    for act in ["A1", "A2", "A3", "A4", "A5"]:
        zone_file = maps_dir / f"{act}-{zone_name}.png"
        if zone_file.exists():
            return zone_file
    
    # Try case-insensitive search
    if maps_dir.exists():
        for file in maps_dir.glob("*.png"):
            if zone_name.lower() in file.stem.lower():
                return file
    
    return None
