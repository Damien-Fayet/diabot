"""
Environment Vision Module - Detects and analyzes environment elements only.

Handles:
- Enemies (monsters)
- Items (ground loot)
- Obstacles (walls, pillars)
- Doors
- Traps
- Player position

Separated from UI vision for clarity and maintainability.
"""

from dataclasses import dataclass, field
from typing import List, Tuple, Optional
import cv2
import numpy as np
import time

from .screen_regions import ENVIRONMENT_REGIONS
from .template_detector import TemplateDetector, TemplateMatch


@dataclass
class ROICandidate:
    """Region of Interest candidate for further analysis."""
    
    roi_type: str  # "static" (NPC, object) or "moving" (enemy, player)
    bbox: Tuple[int, int, int, int]  # (x, y, w, h) in playfield coordinates
    mask: np.ndarray  # Binary mask of the object
    confidence: float  # 0.0-1.0 confidence this is an object vs background
    features: dict = field(default_factory=dict)  # Additional features (area, aspect_ratio, etc.)


@dataclass
class TemplateObjectInfo:
    """Information about a detected template object (NPC, waypoint, etc.)."""
    
    object_type: str        # "npc", "waypoint", "quest", etc. (from template name)
    template_name: str      # Full template name (e.g., "a1_kashya_1")
    position: Tuple[int, int]  # (x, y) center
    confidence: float       # 0.0-1.0 match confidence


@dataclass
class EnemyInfo:
    """Information about a detected enemy."""
    
    enemy_type: str         # "small", "large", "champion", etc.
    position: Tuple[int, int]  # (x, y) center
    bbox: Tuple[int, int, int, int]  # (x, y, w, h)
    confidence: float       # 0.0-1.0
    color_hsv: Tuple[int, int, int] = None  # Dominant color


@dataclass
class EnvironmentState:
    """Current state of environment elements."""
    
    roi_candidates: List[ROICandidate] = field(default_factory=list)  # Pre-selected regions
    enemies: List[EnemyInfo] = field(default_factory=list)
    items: List[str] = field(default_factory=list)  # ["item", "item"]
    obstacles: List[dict] = field(default_factory=list)
    doors: List[dict] = field(default_factory=list)
    template_objects: List[TemplateObjectInfo] = field(default_factory=list)  # NPCs, waypoints, quests
    player_position: Tuple[int, int] = (512, 384)  # Estimated center
    debug_timings: dict = field(default_factory=dict)  # Performance timings


class EnvironmentVisionModule:
    """
    Pre-processes game environment to extract regions of interest (ROIs).
    
    Responsibilities:
    - Background removal/segmentation
    - Foreground object extraction (NPCs, enemies, items)
    - ROI candidate generation for precise detection
    
    Does NOT perform precise identification - that's handled by ObjectDetector.
    Focuses on separating background from foreground elements.
    """
    
    def __init__(self, debug: bool = False):
        """
        Initialize environment vision module.
        
        Args:
            debug: Enable debug output
        """
        self.debug = debug
        self.template_detector: Optional[TemplateDetector] = None
        try:
            self.template_detector = TemplateDetector(debug=debug)
        except Exception as e:
            if debug:
                print(f"⚠️  Failed to initialize template detector: {e}")
        
        # Background frame for temporal differencing
        self.background_frame: Optional[np.ndarray] = None
        self.background_update_alpha = 0.05  # Running average weight
    
    def analyze(self, frame: np.ndarray, current_zone: str = "", 
                detect_templates: bool = False, detect_enemies: bool = False,
                extract_rois: bool = True, debug_output_dir: str = None) -> EnvironmentState:
        """
        Analyze environment from frame.
        
        Args:
            frame: BGR image from game
            current_zone: Current zone name (e.g., "BLOOD MOOR")
            detect_templates: Enable template-based object detection (expensive)
            detect_enemies: Enable contour-based enemy detection (expensive, disabled by default)
            extract_rois: Extract ROI candidates for downstream processing
            
        Returns:
            EnvironmentState with detected environment info
        """
        if self.debug:
            print("\n[ENVIRONMENT VISION TIMING]")
        
        timings = {}
        
        # Extract ROI candidates (background removal)
        roi_candidates = []
        if extract_rois:
            t0 = time.time()
            playfield = ENVIRONMENT_REGIONS['playfield'].extract_from_frame(frame)
            if playfield.size > 0:
                roi_candidates = self._extract_roi_candidates(playfield, debug_output_dir=debug_output_dir)
            timings['roi_extraction'] = time.time() - t0
            if self.debug:
                print(f"  ROI extraction: {timings['roi_extraction']*1000:.1f}ms ({len(roi_candidates)} candidates)")
        
        # Detect enemies (disabled by default)
        enemies = []
        if detect_enemies:
            t0 = time.time()
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            playfield = ENVIRONMENT_REGIONS['playfield'].extract_from_frame(hsv)
            if playfield.size > 0:
                enemies = self._detect_enemies(playfield, frame)
            timings['enemies'] = time.time() - t0
            if self.debug:
                print(f"  Enemies: {timings['enemies']*1000:.1f}ms ({len(enemies)} found)")
        
        # Detect templates (on demand)
        template_objects = []
        if detect_templates:
            t0 = time.time()
            template_objects = self._detect_template_objects(frame, current_zone)
            timings['templates'] = time.time() - t0
            if self.debug:
                print(f"  Templates: {timings['templates']*1000:.1f}ms ({len(template_objects)} found)")
        
        # Estimate player position
        t0 = time.time()
        player_pos = self._estimate_player_position(frame)
        timings['player_est'] = time.time() - t0
        if self.debug:
            print(f"  Player estimate: {timings['player_est']*1000:.1f}ms")
        
        state = EnvironmentState(
            roi_candidates=roi_candidates,
            enemies=enemies,
            items=[],  # Items intentionally disabled
            template_objects=template_objects,
            player_position=player_pos,
        )
        state.debug_timings = timings
        
        return state
    
    def _detect_enemies(self, playfield_hsv: np.ndarray, original_frame: np.ndarray) -> List[EnemyInfo]:
        """
        Detect enemies using edge/contour detection only.
        
        Since graphics settings enhance character outlines, we detect enemies
        by finding strong contours with appropriate size and shape.
        
        Args:
            playfield_hsv: HSV playfield region (unused, kept for interface compatibility)
            original_frame: BGR full frame
            
        Returns:
            List of detected enemies
        """
        enemies: List[EnemyInfo] = []

        # Extract playfield from BGR frame
        playfield_bgr = ENVIRONMENT_REGIONS['playfield'].extract_from_frame(original_frame)

        play_h, play_w = playfield_bgr.shape[:2]
        if play_h == 0 or play_w == 0:
            return enemies

        playfield_area = play_h * play_w

        # Convert to grayscale for edge detection
        gray = cv2.cvtColor(playfield_bgr, cv2.COLOR_BGR2GRAY)

        # Boost local contrast to emphasize silhouettes and shadows
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)
        
        # Apply Gaussian blur to reduce noise after enhancement
        blurred = cv2.GaussianBlur(enhanced, (5, 5), 0)
        
        # Edge detection with Canny (enhanced outlines)
        edges = cv2.Canny(blurred, 100, 200)
        
        # Dilate to connect nearby edges (character outlines)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        edges_dilated = cv2.dilate(edges, kernel, iterations=2)
        
        # Close small gaps
        edges_closed = cv2.morphologyEx(edges_dilated, cv2.MORPH_CLOSE, kernel, iterations=1)
        
        # Detect shadows (dark areas) - characters have prominent shadows
        _, shadow_mask = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        shadow_mask = cv2.morphologyEx(shadow_mask, cv2.MORPH_OPEN, kernel, iterations=1)
        
        # Find contours
        contours, _ = cv2.findContours(edges_closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Size filters - increased minimum to filter out tiny false positives
        min_area = 500      # Minimum character size (filter very small blobs)
        max_area = playfield_area * 0.08  # Maximum (avoid large environment objects)

        for contour in contours:
            area = cv2.contourArea(contour)
            if area < min_area or area > max_area:
                continue

            x, y, w, h = cv2.boundingRect(contour)
            if h == 0 or w == 0:
                continue

            # Aspect ratio filter (characters are roughly vertical)
            aspect_ratio = w / h
            if not 0.2 < aspect_ratio < 4.0:
                continue  # Too flat or too tall

            # Calculate edge density in bounding box
            edges_crop = edges_closed[y:y+h, x:x+w]
            edge_pixels = cv2.countNonZero(edges_crop)
            bbox_area = w * h
            edge_density = edge_pixels / bbox_area
            
            # Characters should have moderate edge density (not solid, not empty)
            if edge_density < 0.05 or edge_density > 0.8:
                continue

            # Calculate solidity (contour area vs bounding box area)
            solidity = area / bbox_area
            
            # Check for shadow presence below the character
            shadow_y1 = min(play_h, y + h)
            shadow_y2 = min(play_h, y + h + int(0.3 * h))  # Check 30% below
            if shadow_y2 > shadow_y1:
                shadow_crop = shadow_mask[shadow_y1:shadow_y2, x:x+w]
                shadow_area = (shadow_y2 - shadow_y1) * w
                shadow_ratio = cv2.countNonZero(shadow_crop) / shadow_area if shadow_area > 0 else 0
            else:
                shadow_ratio = 0

            # Classify by size
            enemy_type = "large_enemy" if area > 1000 else "small_enemy"
            
            # Confidence based on edge density, solidity, and shadow presence
            confidence = float(min(1.0, (edge_density * 1.5 + solidity + shadow_ratio * 2.0) / 3.5))

            enemies.append(EnemyInfo(
                enemy_type=enemy_type,
                position=(x + w // 2, y + h // 2),
                bbox=(x, y, w, h),
                confidence=confidence,
            ))

        # Sort by confidence and return
        enemies.sort(key=lambda e: e.confidence, reverse=True)
        return enemies
    
    def _extract_roi_candidates(self, playfield_bgr: np.ndarray, debug_output_dir: str = None) -> List[ROICandidate]:
        """
        Extract regions of interest using multiple detection methods:
        1. Canny + morphologie
        2. Laplacian / Sobel
        3. Background suppression (temporal differencing)
        
        Args:
            playfield_bgr: BGR playfield region
            debug_output_dir: If set, saves intermediate images here
            
        Returns:
            List of ROI candidates (bounding boxes + masks)
        """
        candidates = []
        
        h, w = playfield_bgr.shape[:2]
        if h == 0 or w == 0:
            return candidates
        
        # Convert to grayscale
        gray = cv2.cvtColor(playfield_bgr, cv2.COLOR_BGR2GRAY)
        
        if debug_output_dir:
            from pathlib import Path
            debug_path = Path(debug_output_dir)
            debug_path.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(debug_path / "01_gray.png"), gray)
        
        # Method 1: Canny + morphology
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edges_canny = cv2.Canny(blurred, 50, 150)
        
        if debug_output_dir:
            cv2.imwrite(str(debug_path / "02_canny_edges.png"), edges_canny)
        
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        edges_dilated = cv2.dilate(edges_canny, kernel, iterations=3)
        edges_closed = cv2.morphologyEx(edges_dilated, cv2.MORPH_CLOSE, kernel, iterations=2)
        
        if debug_output_dir:
            cv2.imwrite(str(debug_path / "03_canny_morphed.png"), edges_closed)
        
        # Method 2: Laplacian (edge detection via 2nd derivative)
        laplacian = cv2.Laplacian(gray, cv2.CV_64F, ksize=3)
        laplacian = np.uint8(np.absolute(laplacian))
        _, laplacian_thresh = cv2.threshold(laplacian, 20, 255, cv2.THRESH_BINARY)
        
        if debug_output_dir:
            cv2.imwrite(str(debug_path / "04_laplacian.png"), laplacian_thresh)
        
        # Method 3: Sobel (gradient magnitude)
        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        sobel_mag = np.sqrt(sobelx**2 + sobely**2)
        sobel_mag = np.uint8(sobel_mag / sobel_mag.max() * 255)
        _, sobel_thresh = cv2.threshold(sobel_mag, 40, 255, cv2.THRESH_BINARY)
        
        if debug_output_dir:
            cv2.imwrite(str(debug_path / "05_sobel.png"), sobel_thresh)
        
        # Method 4: Temporal differencing (background subtraction)
        fg_mask = np.zeros_like(gray)
        if self.background_frame is not None and self.background_frame.shape == gray.shape:
            # Compute absolute difference
            diff = cv2.absdiff(gray, self.background_frame)
            _, fg_mask = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)
            
            # Clean up noise
            fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, np.ones((3,3), np.uint8))
            
            if debug_output_dir:
                cv2.imwrite(str(debug_path / "06_temporal_diff.png"), fg_mask)
        else:
            if debug_output_dir:
                print("  [DEBUG] No background frame yet for temporal differencing")
        
        # Update background (running average)
        if self.background_frame is None:
            self.background_frame = gray.astype(np.float32)
        else:
            cv2.accumulateWeighted(gray, self.background_frame, self.background_update_alpha)
        
        # Combine all methods
        combined = cv2.bitwise_or(edges_closed, laplacian_thresh)
        combined = cv2.bitwise_or(combined, sobel_thresh)
        if fg_mask.max() > 0:
            combined = cv2.bitwise_or(combined, fg_mask)
        
        if debug_output_dir:
            cv2.imwrite(str(debug_path / "07_combined_all_methods.png"), combined)
        
        # Final morphology to group nearby regions
        combined = cv2.dilate(combined, kernel, iterations=2)
        combined = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, kernel, iterations=2)
        
        if debug_output_dir:
            cv2.imwrite(str(debug_path / "08_final_mask.png"), combined)
        
        # Find contours
        contours, _ = cv2.findContours(combined, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if debug_output_dir:
            contour_vis = cv2.cvtColor(combined, cv2.COLOR_GRAY2BGR)
            cv2.drawContours(contour_vis, contours, -1, (0, 255, 0), 2)
            cv2.imwrite(str(debug_path / "09_all_contours.png"), contour_vis)
            print(f"  [DEBUG] Found {len(contours)} contours before filtering")
        
        # Filter by size
        min_area = 400  # Further lowered to catch fragments
        max_area = h * w * 0.15  # Allow larger regions
        
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < min_area or area > max_area:
                if debug_output_dir and area > 0:
                    print(f"    [FILTER] Contour area={int(area)} (min={min_area}, max={int(max_area)}) - rejected by size")
                continue
            
            x, y, w_box, h_box = cv2.boundingRect(contour)
            
            if w_box == 0 or h_box == 0:
                continue
            
            # Create mask for this ROI
            mask = np.zeros((h, w), dtype=np.uint8)
            cv2.drawContours(mask, [contour], -1, 255, -1)
            
            # Calculate features
            aspect_ratio = w_box / h_box
            solidity = area / (w_box * h_box)
            
            # Compute edge density within bbox
            roi_edges = edges_closed[y:y+h_box, x:x+w_box]
            edge_density = cv2.countNonZero(roi_edges) / (w_box * h_box)
            
            if debug_output_dir:
                print(f"    [CANDIDATE] area={int(area)}, aspect={aspect_ratio:.2f}, solid={solidity:.2f}, edge_dens={edge_density:.2f}")
            
            # Classify as static (NPC, object) or moving (enemy)
            # Static objects tend to have cleaner edges and higher solidity
            if solidity > 0.55 and edge_density > 0.12:
                roi_type = "static"
                confidence = min(1.0, solidity * 1.2 + edge_density * 0.8)
                if debug_output_dir:
                    print(f"      -> STATIC conf={confidence:.2f}")
            elif 0.25 < aspect_ratio < 3.0 and edge_density > 0.08:
                roi_type = "moving"
                confidence = min(1.0, edge_density * 2.0 + solidity * 0.5)
                if debug_output_dir:
                    print(f"      -> MOVING conf={confidence:.2f}")
            else:
                if debug_output_dir:
                    print(f"      -> REJECTED by classification")
                continue  # Skip ambiguous regions
            
            candidates.append(ROICandidate(
                roi_type=roi_type,
                bbox=(x, y, w_box, h_box),
                mask=mask[y:y+h_box, x:x+w_box],
                confidence=confidence,
                features={
                    'area': area,
                    'aspect_ratio': aspect_ratio,
                    'solidity': solidity,
                    'edge_density': edge_density,
                }
            ))
        
        # Sort by confidence
        candidates.sort(key=lambda c: c.confidence, reverse=True)
        
        return candidates
    
    def _detect_items(self, playfield_hsv: np.ndarray) -> List[str]:
        """Item detection disabled (placeholder)."""
        return []
    
    def _detect_template_objects(self, frame: np.ndarray, current_zone: str) -> List[TemplateObjectInfo]:
        """
        Detect objects by template matching (NPCs, waypoints, quests, etc.).
        
        Args:
            frame: BGR full frame
            current_zone: Current zone name (e.g., "BLOOD MOOR")
            
        Returns:
            List of detected template objects
        """
        if self.template_detector is None:
            return []
        
        try:
            # Perform template detection
            template_matches = self.template_detector.detect(frame, current_zone, threshold=0.7)
            
            # Convert to TemplateObjectInfo
            objects = []
            for match in template_matches:
                # Parse object type from template name
                # Pattern: "{act}_{name}_{id}" e.g., "a1_kashya_1" -> "kashya"
                parts = match.template_name.split('_')
                if len(parts) >= 2:
                    object_type = parts[1]
                else:
                    object_type = "unknown"
                
                objects.append(TemplateObjectInfo(
                    object_type=object_type,
                    template_name=match.template_name,
                    position=match.location,
                    confidence=match.confidence,
                ))
            
            return objects
        except Exception as e:
            if self.debug:
                print(f"⚠️  Template detection error: {e}")
            return []
    
    def _estimate_player_position(self, frame: np.ndarray) -> Tuple[int, int]:
        """
        Estimate player position.
        
        In Diablo 2 isometric view, player is typically near center-bottom
        of playfield.
        
        Returns:
            (x, y) estimated position
        """
        h, w = frame.shape[:2]
        
        # Playfield is roughly center of screen
        # Player is typically slightly above center due to isometric view
        player_x = w // 2
        player_y = int(h * 0.5)  # Slightly above center
        
        return (player_x, player_y)
