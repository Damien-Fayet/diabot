"""YOLO-based object detection for Diablo 2 bot.

Provides a typed wrapper around YOLO inference for detecting NPCs, quests, waypoints, etc.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import cv2
import numpy as np
from ultralytics import YOLO


@dataclass
class Detection:
    """Single YOLO detection result."""
    class_name: str
    class_id: int
    confidence: float
    bbox: tuple[int, int, int, int]  # (x1, y1, x2, y2)
    center: tuple[float, float]  # (cx, cy)
    
    def width(self) -> int:
        """Bounding box width."""
        return self.bbox[2] - self.bbox[0]
    
    def height(self) -> int:
        """Bounding box height."""
        return self.bbox[3] - self.bbox[1]


class YOLODetector:
    """YOLO object detector for game vision.
    
    Wraps ultralytics YOLO model and provides typed detection results.
    """
    
    def __init__(self, model_path: Path | str, confidence_threshold: float = 0.35, debug: bool = False):
        """Initialize YOLO detector.
        
        Args:
            model_path: Path to YOLO model (.pt file)
            confidence_threshold: Min confidence for detections
            debug: Enable debug logging
        """
        self.model_path = Path(model_path)
        self.confidence_threshold = confidence_threshold
        self.debug = debug
        
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model not found: {model_path}")
        
        self.model = YOLO(str(self.model_path))
        self.class_names = self.model.names  # Dict[int, str]
        
        if debug:
            print(f"✓ YOLO loaded: {self.model_path}")
            print(f"  Classes: {list(self.class_names.values())}")
    
    def detect(self, frame: np.ndarray) -> List[Detection]:
        """Run detection on frame.
        
        Args:
            frame: BGR image array
            
        Returns:
            List of detections sorted by confidence (descending)
        """
        results = self.model.predict(source=frame, conf=self.confidence_threshold, verbose=False)
        
        detections = []
        for box in results[0].boxes:
            cls_id = int(box.cls.item())
            class_name = self.class_names.get(cls_id, f"unknown_{cls_id}")
            confidence = float(box.conf.item())
            
            # Bounding box in (x1, y1, x2, y2) format
            xyxy = box.xyxy.cpu().numpy().astype(int).tolist()[0]
            x1, y1, x2, y2 = xyxy
            
            # Center point
            cx = (x1 + x2) / 2.0
            cy = (y1 + y2) / 2.0
            
            detection = Detection(
                class_name=class_name,
                class_id=cls_id,
                confidence=confidence,
                bbox=(x1, y1, x2, y2),
                center=(cx, cy),
            )
            detections.append(detection)
        
        # Sort by confidence descending
        detections.sort(key=lambda d: d.confidence, reverse=True)
        
        return detections
    
    def filter_by_class(self, detections: List[Detection], class_names: List[str]) -> List[Detection]:
        """Filter detections by class names.
        
        Args:
            detections: List of detections
            class_names: Class names to keep
            
        Returns:
            Filtered detections
        """
        return [d for d in detections if d.class_name in class_names]
    
    def find_nearest(self, detections: List[Detection], 
                    reference_point: tuple[float, float]) -> Optional[Detection]:
        """Find detection nearest to reference point.
        
        Args:
            detections: List of detections
            reference_point: (x, y) reference point
            
        Returns:
            Nearest detection or None
        """
        if not detections:
            return None
        
        ref_x, ref_y = reference_point
        nearest = None
        min_dist = float('inf')
        
        for detection in detections:
            cx, cy = detection.center
            dist = ((cx - ref_x) ** 2 + (cy - ref_y) ** 2) ** 0.5
            if dist < min_dist:
                min_dist = dist
                nearest = detection
        
        return nearest
    
    def draw_detections(self, frame: np.ndarray, detections: List[Detection],
                       color: tuple = (0, 255, 0), thickness: int = 2) -> np.ndarray:
        """Draw detections on frame.
        
        Args:
            frame: Input frame
            detections: List of detections
            color: BGR color for boxes
            thickness: Line thickness
            
        Returns:
            Frame with drawn detections
        """
        output = frame.copy()
        
        for detection in detections:
            x1, y1, x2, y2 = detection.bbox
            
            # Draw bounding box
            cv2.rectangle(output, (x1, y1), (x2, y2), color, thickness)
            
            # Draw label
            label = f"{detection.class_name} {detection.confidence:.2f}"
            label_size, baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(output, (x1, y1 - label_size[1] - baseline),
                         (x1 + label_size[0], y1), color, -1)
            cv2.putText(output, label, (x1, y1 - baseline),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        return output
