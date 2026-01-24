#!/usr/bin/env python3
"""
Extract templates from reference images for object detection.
Allows user to select regions interactively.
"""

import cv2
import numpy as np
from pathlib import Path


class TemplateExtractor:
    """Interactive template extraction tool."""
    
    def __init__(self):
        self.image = None
        self.clone = None
        self.ref_points = []
        self.cropping = False
        self.window_name = "Template Extractor"
    
    def click_and_crop(self, event, x, y, flags, param):
        """Mouse callback for selecting region."""
        if event == cv2.EVENT_LBUTTONDOWN:
            self.ref_points = [(x, y)]
            self.cropping = True
        
        elif event == cv2.EVENT_LBUTTONUP:
            self.ref_points.append((x, y))
            self.cropping = False
            
            # Draw rectangle
            cv2.rectangle(self.image, self.ref_points[0], self.ref_points[1], (0, 255, 0), 2)
            cv2.imshow(self.window_name, self.image)
    
    def extract(self, image_path: str, output_name: str, output_dir: str = "data/templates"):
        """
        Extract template from image.
        
        Args:
            image_path: Path to source image
            output_name: Name for template (e.g., "waypoint_active")
            output_dir: Directory to save template
        """
        self.image = cv2.imread(image_path)
        if self.image is None:
            print(f"❌ Cannot load: {image_path}")
            return
        
        self.clone = self.image.copy()
        
        cv2.namedWindow(self.window_name)
        cv2.setMouseCallback(self.window_name, self.click_and_crop)
        
        print(f"\n{'=' * 80}")
        print(f"EXTRACTING TEMPLATE: {output_name}")
        print(f"{'=' * 80}")
        print("\nInstructions:")
        print("  1. Click and drag to select the object region")
        print("  2. Press 's' to save the template")
        print("  3. Press 'r' to reset selection")
        print("  4. Press 'q' to quit without saving")
        print()
        
        while True:
            cv2.imshow(self.window_name, self.image)
            key = cv2.waitKey(1) & 0xFF
            
            # Reset
            if key == ord("r"):
                self.image = self.clone.copy()
                self.ref_points = []
                print("  → Selection reset")
            
            # Save
            elif key == ord("s"):
                if len(self.ref_points) == 2:
                    # Extract crop
                    x1, y1 = self.ref_points[0]
                    x2, y2 = self.ref_points[1]
                    
                    # Ensure correct order
                    x1, x2 = min(x1, x2), max(x1, x2)
                    y1, y2 = min(y1, y2), max(y1, y2)
                    
                    crop = self.clone[y1:y2, x1:x2]
                    
                    if crop.size > 0:
                        # Save template
                        output_path = Path(output_dir)
                        output_path.mkdir(parents=True, exist_ok=True)
                        
                        template_file = output_path / f"{output_name}.png"
                        cv2.imwrite(str(template_file), crop)
                        
                        print(f"✓ Saved template: {template_file}")
                        print(f"  Size: {crop.shape[1]}x{crop.shape[0]} pixels")
                        
                        # Show preview
                        preview = cv2.resize(crop, None, fx=2, fy=2, interpolation=cv2.INTER_NEAREST)
                        cv2.imshow("Template Preview", preview)
                        cv2.waitKey(2000)
                        break
                    else:
                        print("  ⚠️  Invalid selection (empty crop)")
                else:
                    print("  ⚠️  No region selected")
            
            # Quit
            elif key == ord("q"):
                print("  → Cancelled")
                break
        
        cv2.destroyAllWindows()


def auto_extract_from_coords(image_path: str, coords: tuple, output_name: str, 
                            output_dir: str = "data/templates"):
    """
    Extract template from specific coordinates without GUI.
    
    Args:
        image_path: Path to source image  
        coords: (x, y, width, height) of region to extract
        output_name: Name for template
        output_dir: Directory to save template
    """
    img = cv2.imread(image_path)
    if img is None:
        print(f"❌ Cannot load: {image_path}")
        return
    
    x, y, w, h = coords
    crop = img[y:y+h, x:x+w]
    
    if crop.size == 0:
        print(f"❌ Invalid coordinates")
        return
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    template_file = output_path / f"{output_name}.png"
    cv2.imwrite(str(template_file), crop)
    
    print(f"✓ Extracted template: {template_file}")
    print(f"  Size: {crop.shape[1]}x{crop.shape[0]} pixels")


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 3:
        print("Usage:")
        print("  Interactive: python extract_template.py <image> <template_name>")
        print("  Auto: python extract_template.py <image> <template_name> <x> <y> <w> <h>")
        print()
        print("Examples:")
        print("  python extract_template.py data/screenshots/inputs/waypoint_active.png waypoint_active")
        print("  python extract_template.py game.png chest 100 200 50 50")
        sys.exit(1)
    
    image_path = sys.argv[1]
    template_name = sys.argv[2]
    
    if len(sys.argv) == 7:
        # Auto mode with coordinates
        x, y, w, h = map(int, sys.argv[3:7])
        auto_extract_from_coords(image_path, (x, y, w, h), template_name)
    else:
        # Interactive mode
        extractor = TemplateExtractor()
        extractor.extract(image_path, template_name)
