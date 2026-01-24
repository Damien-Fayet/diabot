#!/usr/bin/env python3
"""
Test object detection (waypoints, quest markers, etc.)
"""

import cv2
import numpy as np
from pathlib import Path
from src.diabot.vision.object_detector import ObjectDetector


def test_object_detection(image_path: str, output_path: str = None):
    """Test object detection and visualize results."""
    img = cv2.imread(image_path)
    if img is None:
        print(f"❌ Cannot load: {image_path}")
        return
    
    height, width = img.shape[:2]
    print(f"Image size: {width}x{height}\n")
    
    # Create detector
    print("Initializing object detector...")
    detector = ObjectDetector(debug=True)
    
    print("\n" + "=" * 80)
    print("DETECTING OBJECTS")
    print("=" * 80)
    
    # Detect all objects
    detections = detector.detect_objects(img, confidence_threshold=0.6)
    
    print(f"\nFound {len(detections)} objects:\n")
    
    # Draw detections on image
    vis = img.copy()
    
    # Colors for different object types
    colors = {
        'waypoint_active': (0, 255, 0),      # Green
        'waypoint_inactive': (0, 165, 255),  # Orange
        'quest_marker': (0, 255, 255),       # Yellow
        'default': (255, 0, 255)             # Magenta
    }
    
    for detection in detections:
        x, y, w, h = detection.bbox
        cx, cy = detection.position
        
        # Choose color
        color = colors.get(detection.object_type, colors['default'])
        
        # Draw bounding box
        cv2.rectangle(vis, (x, y), (x + w, y + h), color, 2)
        
        # Draw center point
        cv2.circle(vis, (cx, cy), 5, color, -1)
        
        # Draw label with confidence
        label = f"{detection.object_type}: {detection.confidence:.2f}"
        
        # Background for text
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5
        thickness = 1
        (label_w, label_h), baseline = cv2.getTextSize(label, font, font_scale, thickness)
        
        # Position label above bbox
        label_x = x
        label_y = y - 5
        if label_y < label_h + 5:
            label_y = y + h + label_h + 5
        
        # Draw label background
        cv2.rectangle(vis, 
                     (label_x, label_y - label_h - 3),
                     (label_x + label_w + 4, label_y + 3),
                     color, -1)
        
        # Draw label text
        cv2.putText(vis, label, (label_x + 2, label_y),
                   font, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)
        
        print(f"  {detection.object_type}:")
        print(f"    Position: ({cx}, {cy})")
        print(f"    Confidence: {detection.confidence:.2%}")
        print(f"    BBox: x={x}, y={y}, w={w}, h={h}")
        print()
    
    # Add summary panel
    panel_height = 100
    panel = np.zeros((panel_height, width, 3), dtype=np.uint8)
    
    cv2.putText(panel, "OBJECT DETECTION", (10, 30),
               cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2, cv2.LINE_AA)
    
    cv2.putText(panel, f"Total detections: {len(detections)}", (10, 60),
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1, cv2.LINE_AA)
    
    # Count by type
    type_counts = {}
    for det in detections:
        type_counts[det.object_type] = type_counts.get(det.object_type, 0) + 1
    
    y_pos = 80
    for obj_type, count in type_counts.items():
        color = colors.get(obj_type, colors['default'])
        cv2.putText(panel, f"{obj_type}: {count}", (10, y_pos),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
        y_pos += 20
    
    # Combine image and panel
    result = np.vstack([vis, panel])
    
    # Save output
    if output_path is None:
        output_path = "data/screenshots/outputs/object_detection.png"
    
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(output_path, result)
    
    print("=" * 80)
    print(f"✓ Saved visualization to: {output_path}")
    print("=" * 80)
    
    return detections


if __name__ == "__main__":
    import sys
    
    image_path = "data/screenshots/inputs/game_a1.png"
    if len(sys.argv) > 1:
        image_path = sys.argv[1]
    
    print(f"\nTesting object detection on: {image_path}\n")
    test_object_detection(image_path)
