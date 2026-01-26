"""
Object Detector - Detects game objects using template matching.

Handles:
- Waypoints (active/inactive)
- Quest markers/bubbles
- NPCs
- Shrines
- Chests
"""

from dataclasses import dataclass
from typing import List, Tuple, Optional
from pathlib import Path
import cv2
import numpy as np


@dataclass
class DetectedObject:
    """A detected game object."""
    
    object_type: str        # "waypoint_active", "waypoint_inactive", "quest_marker", etc.
    confidence: float       # 0.0-1.0
    position: Tuple[int, int]  # (x, y) center position in frame
    bbox: Tuple[int, int, int, int]  # (x, y, width, height)


class ObjectDetector:
    """
    Detects game objects using template matching.
    
    Templates are loaded from data/screenshots/inputs/ or a templates directory.
    """
    
    def __init__(self, templates_dir: str = "data/screenshots/inputs", debug: bool = False):
        """
        Initialize object detector.
        
        Args:
            templates_dir: Directory containing template images
            debug: Enable debug output
        """
        self.debug = debug
        self.templates_dir = Path(templates_dir)
        self.templates = {}
        
        # Load templates
        self._load_templates()
    
    def _load_templates(self):
        """Load all template images from disk."""
        template_files = {
            'waypoint_active': 'waypoint_active.png',
            'waypoint_inactive': 'waypoint_inactive.png',
            # Add more templates as needed
        }
        
        for object_type, filename in template_files.items():
            template_path = self.templates_dir / filename
            
            if template_path.exists():
                template = cv2.imread(str(template_path))
                if template is not None:
                    self.templates[object_type] = template
                    if self.debug:
                        print(f"✓ Loaded template: {object_type} ({template.shape})")
                else:
                    if self.debug:
                        print(f"⚠️  Failed to load: {template_path}")
            else:
                if self.debug:
                    print(f"⚠️  Template not found: {template_path}")
    
    def detect_objects(self, frame: np.ndarray, 
                      object_types: Optional[List[str]] = None,
                      confidence_threshold: float = 0.7,
                      max_detections: int = 10,
                      roi_candidates: Optional[List] = None) -> List[DetectedObject]:
        """
        Detect objects in frame using template matching.
        
        Args:
            frame: BGR image to search in
            object_types: List of object types to search for (None = all)
            confidence_threshold: Minimum confidence (0.0-1.0)
            max_detections: Maximum number of detections per object type
            roi_candidates: Optional list of ROICandidate to search within (speeds up detection)
            
        Returns:
            List of detected objects
        """
        if frame.size == 0:
            return []
        
        # Determine which templates to use
        if object_types is None:
            search_templates = self.templates
        else:
            search_templates = {k: v for k, v in self.templates.items() 
                              if k in object_types}
        
        all_detections = []
        
        for object_type, template in search_templates.items():
            detections = self._match_template(
                frame, template, object_type, 
                confidence_threshold, max_detections
            )
            all_detections.extend(detections)
        
        return all_detections
    
    def _match_template(self, frame: np.ndarray, template: np.ndarray,
                       object_type: str, threshold: float, 
                       max_detections: int) -> List[DetectedObject]:
        """
        Match a single template in the frame.
        
        Args:
            frame: BGR image to search in
            template: BGR template to search for
            object_type: Type of object being detected
            threshold: Minimum confidence
            max_detections: Maximum detections
            
        Returns:
            List of detected objects
        """
        # Convert to grayscale for matching
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        template_gray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
        
        # Get template dimensions
        h, w = template_gray.shape
        
        # Perform template matching
        # Using TM_CCOEFF_NORMED (normalized cross-correlation)
        result = cv2.matchTemplate(frame_gray, template_gray, cv2.TM_CCOEFF_NORMED)
        
        # Find all matches above threshold
        locations = np.where(result >= threshold)
        
        detections = []
        
        # Group nearby detections (non-maximum suppression)
        matches = list(zip(*locations[::-1]))  # Switch x and y
        
        if not matches:
            return []
        
        # Sort by confidence
        matches_with_conf = [(pt, result[pt[1], pt[0]]) for pt in matches]
        matches_with_conf.sort(key=lambda x: x[1], reverse=True)
        
        # Non-maximum suppression: keep detections that are far apart
        min_distance = min(w, h) * 0.5  # Minimum distance between detections
        
        for (x, y), confidence in matches_with_conf:
            if len(detections) >= max_detections:
                break
            
            # Check if this detection is far enough from existing ones
            too_close = False
            for existing in detections:
                ex, ey = existing.position
                distance = np.sqrt((x - ex) ** 2 + (y - ey) ** 2)
                if distance < min_distance:
                    too_close = True
                    break
            
            if not too_close:
                # Calculate center position
                center_x = x + w // 2
                center_y = y + h // 2
                
                detection = DetectedObject(
                    object_type=object_type,
                    confidence=float(confidence),
                    position=(center_x, center_y),
                    bbox=(x, y, w, h)
                )
                detections.append(detection)
                
                if self.debug:
                    print(f"  → {object_type} at ({center_x}, {center_y}) confidence={confidence:.2f}")
        
        return detections
    
    def detect_waypoints(self, frame: np.ndarray, 
                        confidence_threshold: float = 0.7) -> List[DetectedObject]:
        """
        Detect waypoints (both active and inactive).
        
        Args:
            frame: BGR image
            confidence_threshold: Minimum confidence
            
        Returns:
            List of detected waypoints
        """
        return self.detect_objects(
            frame,
            object_types=['waypoint_active', 'waypoint_inactive'],
            confidence_threshold=confidence_threshold,
            max_detections=5  # Usually only 1-2 waypoints visible
        )
    
    def detect_quest_markers(self, frame: np.ndarray,
                           confidence_threshold: float = 0.6) -> List[DetectedObject]:
        """
        Detect quest markers/bubbles.
        
        Args:
            frame: BGR image
            confidence_threshold: Minimum confidence
            
        Returns:
            List of detected quest markers
        """
        # Quest markers need their templates to be added
        return self.detect_objects(
            frame,
            object_types=['quest_marker'],  # Will return empty until template added
            confidence_threshold=confidence_threshold,
            max_detections=10
        )
    
    def add_template(self, object_type: str, template: np.ndarray):
        """
        Add a template at runtime.
        
        Args:
            object_type: Type identifier
            template: BGR image template
        """
        self.templates[object_type] = template
        if self.debug:
            print(f"✓ Added template: {object_type} ({template.shape})")
