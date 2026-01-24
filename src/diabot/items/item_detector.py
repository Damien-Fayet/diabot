"""Item detection module for finding items on ground."""

from dataclasses import dataclass
from typing import List, Optional, Tuple
import cv2
import numpy as np


class ItemQuality:
    """Item quality constants matching Diablo 2 color coding."""
    
    UNIQUE = "unique"        # Gold
    SET = "set"              # Green  
    RARE = "rare"            # Yellow
    MAGIC = "magic"           # Blue
    NORMAL = "normal"         # White
    SUPERIOR = "superior"     # White with +
    
    ALL = [UNIQUE, SET, RARE, MAGIC, NORMAL, SUPERIOR]


@dataclass
class DetectedItem:
    """Represents a detected item on screen."""
    
    quality: str              # unique, set, rare, magic, normal
    position: Tuple[int, int] # (x, y) center of item
    bbox: Tuple[int, int, int, int]  # (x1, y1, x2, y2) bounding box
    brightness: float         # Estimated brightness (0.0-1.0)
    color_hsv: Tuple[int, int, int]  # Dominant HSV color
    confidence: float         # Detection confidence (0.0-1.0)
    name: Optional[str] = None  # Item name if recognized


class ItemDetector:
    """
    Detects items on ground using color-based segmentation.
    
    Diablo 2 item color coding:
    - Unique: Gold (yellow-gold in HSV)
    - Set: Green (bright green)
    - Rare: Yellow (pure yellow)
    - Magic: Blue (bright blue)
    - Normal: White (low saturation)
    - Superior: Indicated by "+", appears as white
    """
    
    def __init__(self):
        """Initialize item detector with HSV color ranges."""
        # HSV ranges for item qualities
        # Format: (lower_HSV, upper_HSV)
        
        # Unique items: Gold color
        self.unique_range = (
            np.array([15, 100, 100]),    # Lower bound
            np.array([35, 255, 255]),    # Upper bound
        )
        
        # Set items: Green color
        self.set_range = (
            np.array([60, 100, 100]),
            np.array([90, 255, 255]),
        )
        
        # Rare items: Yellow color
        self.rare_range = (
            np.array([20, 100, 80]),
            np.array([40, 255, 255]),
        )
        
        # Magic items: Blue color
        self.magic_range = (
            np.array([100, 80, 80]),
            np.array([130, 255, 255]),
        )
        
        # Normal items: Low saturation (white/gray)
        self.normal_range = (
            np.array([0, 0, 60]),
            np.array([180, 50, 255]),
        )
        
        # Minimum item size (pixels)
        self.min_item_size = 5
        self.max_item_size = 100
    
    def detect_items(
        self,
        frame: np.ndarray,
        qualities: Optional[List[str]] = None,
    ) -> List[DetectedItem]:
        """
        Detect items in frame by color.
        
        Args:
            frame: Image frame (BGR format from OpenCV)
            qualities: Which qualities to detect (None = all)
            
        Returns:
            List of detected items
        """
        if qualities is None:
            qualities = ItemQuality.ALL
        
        # Convert to HSV
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        items = []
        
        # Detect each quality type
        for quality in qualities:
            quality_items = self._detect_quality(frame, hsv, quality)
            items.extend(quality_items)
        
        # Sort by confidence (highest first)
        items.sort(key=lambda x: x.confidence, reverse=True)
        
        return items
    
    def _detect_quality(
        self,
        frame: np.ndarray,
        hsv: np.ndarray,
        quality: str,
    ) -> List[DetectedItem]:
        """
        Detect items of specific quality.
        
        Args:
            frame: Original frame (BGR)
            hsv: HSV version of frame
            quality: Quality type (unique, set, etc.)
            
        Returns:
            List of detected items of this quality
        """
        # Get color range for quality
        if quality == ItemQuality.UNIQUE:
            lower, upper = self.unique_range
        elif quality == ItemQuality.SET:
            lower, upper = self.set_range
        elif quality == ItemQuality.RARE:
            lower, upper = self.rare_range
        elif quality == ItemQuality.MAGIC:
            lower, upper = self.magic_range
        elif quality == ItemQuality.NORMAL:
            lower, upper = self.normal_range
        else:
            return []
        
        # Create mask for this color range
        mask = cv2.inRange(hsv, lower, upper)
        
        # Morphological operations to clean up
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        
        # Find contours
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        items = []
        
        for contour in contours:
            # Filter by size
            area = cv2.contourArea(contour)
            if area < self.min_item_size ** 2:
                continue
            
            if area > self.max_item_size ** 2:
                continue
            
            # Get bounding box
            x, y, w, h = cv2.boundingRect(contour)
            
            # Filter by aspect ratio (items shouldn't be too stretched)
            if w > 0 and h > 0:
                aspect_ratio = w / h
                if aspect_ratio < 0.3 or aspect_ratio > 3.0:
                    continue
            
            # Calculate confidence based on contour shape
            contour_area = cv2.contourArea(contour)
            hull_area = cv2.contourArea(cv2.convexHull(contour))
            if hull_area > 0:
                confidence = contour_area / hull_area
            else:
                confidence = 0.5
            
            confidence = min(1.0, confidence * 0.9)  # Scale confidence
            
            # Get dominant color at center
            cx = x + w // 2
            cy = y + h // 2
            center_color = hsv[cy, cx]
            
            # Calculate brightness
            brightness = float(frame[cy, cx].max()) / 255.0
            
            # Create detected item
            item = DetectedItem(
                quality=quality,
                position=(cx, cy),
                bbox=(x, y, x + w, y + h),
                brightness=brightness,
                color_hsv=tuple(center_color),
                confidence=confidence,
            )
            
            items.append(item)
        
        return items
    
    def get_item_color_rgb(self, quality: str) -> Tuple[int, int, int]:
        """
        Get approximate RGB color for item quality for drawing.
        
        Args:
            quality: Item quality type
            
        Returns:
            (B, G, R) tuple for OpenCV drawing
        """
        colors = {
            ItemQuality.UNIQUE: (0, 215, 255),    # Gold (BGR)
            ItemQuality.SET: (0, 255, 0),         # Green
            ItemQuality.RARE: (0, 255, 255),      # Yellow
            ItemQuality.MAGIC: (255, 0, 0),       # Blue
            ItemQuality.NORMAL: (200, 200, 200),  # White/Gray
            ItemQuality.SUPERIOR: (150, 150, 150), # Darker gray
        }
        return colors.get(quality, (128, 128, 128))
    
    def draw_items_on_frame(
        self,
        frame: np.ndarray,
        items: List[DetectedItem],
        show_quality: bool = True,
        show_confidence: bool = False,
    ) -> np.ndarray:
        """
        Draw detected items on frame for visualization.
        
        Args:
            frame: Image frame
            items: List of detected items
            show_quality: Show item quality label
            show_confidence: Show confidence value
            
        Returns:
            Frame with drawn items
        """
        output = frame.copy()
        
        for item in items:
            # Get color for quality
            color = self.get_item_color_rgb(item.quality)
            
            # Draw bounding box
            x1, y1, x2, y2 = item.bbox
            cv2.rectangle(output, (x1, y1), (x2, y2), color, 2)
            
            # Draw center point
            cx, cy = item.position
            cv2.circle(output, (cx, cy), 3, color, -1)
            
            # Draw label
            label = item.quality.upper()
            if show_confidence:
                label += f" {item.confidence:.0%}"
            
            # Position label above box
            cv2.putText(
                output,
                label,
                (x1, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,
                color,
                1,
            )
        
        return output
    
    def filter_by_quality(
        self,
        items: List[DetectedItem],
        quality: str,
    ) -> List[DetectedItem]:
        """Filter items by specific quality."""
        return [item for item in items if item.quality == quality]
    
    def filter_by_confidence(
        self,
        items: List[DetectedItem],
        min_confidence: float,
    ) -> List[DetectedItem]:
        """Filter items by minimum confidence."""
        return [item for item in items if item.confidence >= min_confidence]
