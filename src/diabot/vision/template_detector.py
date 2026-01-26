"""
Template-based object detection for Diablo 2.

Matches templates from data/templates/ directory using pattern matching.
Templates are organized by category:
- data/templates/npcs/ - Non-player characters
- data/templates/waypoints/ - Waypoint portals
- data/templates/quests/ - Quest markers

Template naming: "{zone/act_prefix}_{name}_{id}.png"
- all_* : Valid everywhere
- a1_* : Act 1 only
- a2_* : Act 2 only, etc.
"""

import cv2
import numpy as np
from pathlib import Path
from dataclasses import dataclass
from typing import List, Dict, Optional


@dataclass
class TemplateMatch:
    """Represents a matched template in the frame."""
    template_name: str
    location: tuple  # (x, y) - center of match
    confidence: float  # 0.0-1.0
    bbox: tuple  # (x1, y1, x2, y2)
    category: str = "unknown"  # template category (npcs, waypoints, quests)


class TemplateDetector:
    """Detect objects by matching templates."""
    
    # Template categories with descriptions
    TEMPLATE_CATEGORIES = {
        'npcs': 'Non-player characters (merchants, guides)',
        'waypoints': 'Waypoint portals',
        'quests': 'Quest markers',
    }
    
    # Comprehensive zone-to-act mapping for all Diablo 2 zones
    ZONE_TO_ACT = {
        # Act 1
        "ROGUE ENCAMPMENT": "a1",
        "BLOOD MOOR": "a1",
        "COLD PLAINS": "a1",
        "STONY FIELD": "a1",
        "DARK WOOD": "a1",
        "BLACK MARSH": "a1",
        "TAMOE HIGHLAND": "a1",
        "THE PIT": "a1",
        "ANCIENT CRYPT": "a1",
        "INNER CLOISTER": "a1",
        "CATHEDRAL": "a1",
        "BARRACKS": "a1",
        "TRISTRAM": "a1",
        "MAUSOLEUM": "a1",
        "UNDERGROUND PASSAGE": "a1",
        "TOWER": "a1",
        "OUTER CLOISTER": "a1",
        
        # Act 2
        "LUTHIEN ENCAMPMENT": "a2",
        "ROCKY WASTE": "a2",
        "DESERT": "a2",
        "SEWERS": "a2",
        "ARANOCH": "a2",
        "LOOT STORAGE": "a2",
        "MAGGOT LAIR": "a2",
        "HALLS OF THE DEAD": "a2",
        "CLAW VIPER TEMPLE": "a2",
        "TOMB": "a2",
        "DURIEL'S LAIR": "a2",
        "STONY TOMB": "a2",
        "ANCIENT TUNNELS": "a2",
        
        # Act 3
        "THE ENTRYWAY": "a3",
        "SNAKE CATACOMBS": "a3",
        "DURANCE OF HATE": "a3",
        "THE PANDEMONIUM FORTRESS": "a3",
        "OUTER SANCTUM": "a3",
        "FLAYER JUNGLE": "a3",
        "LOWER KURAST": "a3",
        "KURAST BAZAAR": "a3",
        "UPPER KURAST": "a3",
        "TRAVINCAL": "a3",
        
        # Act 4
        "THE PANDEMONIUM FORTRESS": "a4",
        "OUTER SANCTUM": "a4",
        "HALLS OF VAUGHT": "a4",
        "THE CHAOS SANCTUARY": "a4",
        "DIABLO'S LAIR": "a4",
        
        # Act 5
        "THE WORLDSTONE KEEP": "a5",
        "FRIGID HIGHLANDS": "a5",
        "ARREAT PLATEAU": "a5",
        "WORLDSTONE KEEP": "a5",
        "THRONE OF DESTRUCTION": "a5",
        "THE WORLDSTONE CHAMBER": "a5",
        "THE CRYSTALLINE PASSAGE": "a5",
        "THE FROZEN TUNDRA": "a5",
    }
    
    def __init__(self, templates_dir: Path = None, debug: bool = False):
        """
        Initialize template detector.
        
        Args:
            templates_dir: Path to templates directory
            debug: Enable debug output
        """
        self.debug = debug
        if templates_dir is None:
            templates_dir = Path(__file__).parent.parent.parent.parent / "data" / "templates"
        
        self.templates_dir = Path(templates_dir)
        self.templates_by_category: Dict[str, Dict[str, np.ndarray]] = {}  # category -> {name -> image}
        self.template_category_by_name: Dict[str, str] = {}
        self._load_templates()
    
    def _load_templates(self):
        """Load all templates from directory, organized by category."""
        if not self.templates_dir.exists():
            if self.debug:
                print(f"⚠️  Templates directory not found: {self.templates_dir}")
            return
        
        # Try to load from subdirectories (npcs/, waypoints/, quests/)
        for category in self.TEMPLATE_CATEGORIES.keys():
            category_dir = self.templates_dir / category
            if category_dir.exists():
                self.templates_by_category[category] = {}
                template_files = sorted(category_dir.glob("*.png"))
                for template_path in template_files:
                    try:
                        img = cv2.imread(str(template_path))
                        if img is None:
                            continue
                        self.templates_by_category[category][template_path.stem] = img
                        self.template_category_by_name[template_path.stem] = category
                    except Exception as e:
                        if self.debug:
                            print(f"⚠️  Failed to load template {template_path.name}: {e}")
                
                if self.debug and self.templates_by_category[category]:
                    print(f"✓ Loaded {len(self.templates_by_category[category])} {category} templates")
        
        # Fallback: if no category subdirs, load all PNG from root
        if not self.templates_by_category:
            if self.debug:
                print(f"⚠️  No category subdirectories found, loading from root...")
            self.templates_by_category['all'] = {}
            template_files = sorted(self.templates_dir.glob("*.png"))
            for template_path in template_files:
                try:
                    img = cv2.imread(str(template_path))
                    if img is None:
                        continue
                    self.templates_by_category['all'][template_path.stem] = img
                    self.template_category_by_name[template_path.stem] = 'all'
                except Exception as e:
                    if self.debug:
                        print(f"⚠️  Failed to load template {template_path.name}: {e}")
            
            if self.debug:
                print(f"✓ Loaded {len(self.templates_by_category['all'])} templates")
    
    def detect(self, frame: np.ndarray, current_zone: str = "", 
               categories: List[str] = None, threshold: float = 0.7,
               category_thresholds: Dict[str, float] = None) -> List[TemplateMatch]:
        """
        Detect objects in frame by matching templates.
        
        Args:
            frame: Input BGR image
            current_zone: Current zone name (e.g., "BLOOD MOOR")
            categories: Template categories to search ('npcs', 'waypoints', 'quests')
                       If None, searches all categories. If empty list, searches nothing.
            threshold: Default confidence threshold (0.0-1.0)
            category_thresholds: Optional dict of category-specific thresholds
                                e.g., {'quests': 0.80, 'waypoints': 0.70}
            
        Returns:
            List of TemplateMatch objects
        """
        if categories is None:
            categories = list(self.TEMPLATE_CATEGORIES.keys())
        
        if not categories:
            return []
        
        matches = []
        
        if self.debug:
            print(f"\n🔍 Template detection for zone: {current_zone}")
            print(f"   Categories: {categories}")
        
        # Collect all applicable templates from requested categories
        all_templates = {}
        category_by_template: Dict[str, str] = {}
        for category in categories:
            if category in self.templates_by_category:
                templates = self._get_applicable_templates(current_zone, 
                                                          self.templates_by_category[category])
                all_templates.update(templates)
                category_by_template.update({name: category for name in templates.keys()})
        
        if self.debug:
            print(f"   Checking {len(all_templates)} templates")
        
        for template_name, template_img in all_templates.items():
            try:
                cat = category_by_template.get(template_name, self.template_category_by_name.get(template_name, "unknown"))
                cat_threshold = category_thresholds.get(cat, threshold) if category_thresholds else threshold
                frame_matches = self._match_template(
                    frame, template_img, template_name, cat_threshold, min_size=35,
                    category=cat
                )
                matches.extend(frame_matches)
                
            except Exception as e:
                if self.debug:
                    print(f"  ⚠️  Error matching {template_name}: {e}")
        
        # Sort by confidence descending
        matches.sort(key=lambda m: m.confidence, reverse=True)
        
        if self.debug:
            print(f"   Found {len(matches)} matches")
            for match in matches[:5]:  # Show top 5
                print(f"     - {match.template_name}: {match.confidence:.2f} @ {match.location}")
        
        return matches
    
    def _get_applicable_templates(self, current_zone: str, 
                                 templates_dict: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """
        Get applicable templates for current zone.
        
        Includes:
        - all_* templates (everywhere)
        - Zone-specific templates (if zone name matches)
        - Act-specific templates (if act matches)
        
        Args:
            current_zone: Current zone (e.g., "BLOOD MOOR", "ROGUE ENCAMPMENT")
            templates_dict: Dict of template names to images to filter
            
        Returns:
            Dict of template_name -> template_image
        """
        applicable = {}
        current_zone_upper = current_zone.upper()
        
        # Determine act from zone using comprehensive mapping
        current_act = self.ZONE_TO_ACT.get(current_zone_upper, "")
        
        for template_name, template_img in templates_dict.items():
            # Parse template name: "{zone/act}_{name}_{id}.png" -> stem is prefix
            parts = template_name.split('_')
            if len(parts) < 2:
                continue
            
            prefix = parts[0]  # "all", "a1", "a2", "blood", etc.
            
            # Always include "all_*" templates
            if prefix == "all":
                applicable[template_name] = template_img
            
            # Include act-specific templates if act matches
            elif prefix.startswith("a") and len(prefix) == 2:  # "a1", "a2", etc.
                if prefix == current_act:
                    applicable[template_name] = template_img
            
            # Include zone-specific templates (exact match on first part)
            elif prefix.lower() in current_zone_upper.lower():
                applicable[template_name] = template_img
        
        return applicable
    
    def _match_template(self, frame: np.ndarray, template: np.ndarray, 
                       template_name: str, threshold: float, min_size: int = 30,
                       category: str = "unknown") -> List[TemplateMatch]:
        """
        Match a single template in frame.
        
        Args:
            frame: Input BGR image
            template: Template to match
            template_name: Name of template (for reporting)
            threshold: Confidence threshold
            min_size: Minimum width/height in pixels for valid match
            
        Returns:
            List of TemplateMatch objects found
        """
        matches = []
        
        # Handle different template sizes
        frame_h, frame_w = frame.shape[:2]
        template_h, template_w = template.shape[:2]
        
        # Skip if template is larger than frame
        if template_h > frame_h or template_w > frame_w:
            return matches
        
        # Try multi-scale template matching
        for scale in [1.0, 0.9, 0.8, 1.1, 1.2]:  # Try different scales
            scaled_template = cv2.resize(
                template, 
                (int(template_w * scale), int(template_h * scale))
            )
            
            scaled_h, scaled_w = scaled_template.shape[:2]
            
            # Skip if scaled template is too small
            if scaled_w < min_size or scaled_h < min_size:
                continue
            
            # Perform template matching with category-specific methods
            if 'quest' in template_name.lower():
                # For quest markers, try both normalized methods (stricter)
                result1 = cv2.matchTemplate(frame, scaled_template, cv2.TM_CCOEFF_NORMED)
                result2 = cv2.matchTemplate(frame, scaled_template, cv2.TM_CCORR_NORMED)
                result = np.minimum(result1, result2)
            elif category == 'npcs':
                # For NPCs, combine shape matching with color histogram matching
                # This helps with clothing colors
                result_shape = cv2.matchTemplate(frame, scaled_template, cv2.TM_CCOEFF_NORMED)
                result_color = self._match_template_with_histogram(frame, scaled_template)
                # Weighted combination: 60% shape, 40% color
                result = result_shape * 0.6 + result_color * 0.4
            else:
                result = cv2.matchTemplate(frame, scaled_template, cv2.TM_CCOEFF_NORMED)
            
            # Find matches above threshold
            loc = np.where(result >= threshold)
            
            for pt in zip(*loc[::-1]):  # (x, y)
                x, y = pt
                h, w = scaled_h, scaled_w
                
                # Skip if too close to another match
                is_duplicate = False
                for existing_match in matches:
                    ex, ey = existing_match.location
                    if abs(ex - x) < w and abs(ey - y) < h:
                        is_duplicate = True
                        break
                
                if not is_duplicate:
                    confidence = float(result[y, x])
                    matches.append(TemplateMatch(
                        template_name=template_name,
                        location=(x + w // 2, y + h // 2),  # Center
                        confidence=confidence,
                        bbox=(x, y, x + w, y + h),
                        category=category,
                    ))
        
        return matches
    
    def _match_template_with_histogram(self, frame: np.ndarray, template: np.ndarray) -> np.ndarray:
        """
        Match template using color histogram comparison (HSV space).
        Useful for NPCs with distinctive clothing colors.
        
        Args:
            frame: BGR frame to search in
            template: BGR template to match
            
        Returns:
            Result map with similarity scores (0.0-1.0)
        """
        template_h, template_w = template.shape[:2]
        frame_h, frame_w = frame.shape[:2]
        
        # Convert to HSV for better color matching
        template_hsv = cv2.cvtColor(template, cv2.COLOR_BGR2HSV)
        frame_hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # Compute histogram for template (H and S channels, ignore V for lighting invariance)
        hist_template = cv2.calcHist([template_hsv], [0, 1], None, [50, 60], [0, 180, 0, 256])
        cv2.normalize(hist_template, hist_template, 0, 1, cv2.NORM_MINMAX)
        
        # Prepare result map
        result_h = frame_h - template_h + 1
        result_w = frame_w - template_w + 1
        
        if result_h <= 0 or result_w <= 0:
            return np.zeros((1, 1), dtype=np.float32)
        
        result = np.zeros((result_h, result_w), dtype=np.float32)
        
        # Slide window and compare histograms
        # Only sample every 4 pixels to avoid excessive computation
        step = 4
        for y in range(0, result_h, step):
            for x in range(0, result_w, step):
                roi = frame_hsv[y:y+template_h, x:x+template_w]
                hist_roi = cv2.calcHist([roi], [0, 1], None, [50, 60], [0, 180, 0, 256])
                cv2.normalize(hist_roi, hist_roi, 0, 1, cv2.NORM_MINMAX)
                
                # Compare histograms using correlation
                similarity = cv2.compareHist(hist_template, hist_roi, cv2.HISTCMP_CORREL)
                result[y:min(y+step, result_h), x:min(x+step, result_w)] = max(0, similarity)
        
        return result


def main():
    """Test template detector."""
    detector = TemplateDetector(debug=True)
    
    # Try to load a test frame
    test_frame_path = Path(__file__).parent.parent.parent.parent / "data" / "screenshots" / "outputs" / "live_capture" / "live_capture_raw.jpg"
    
    if test_frame_path.exists():
        frame = cv2.imread(str(test_frame_path))
        
        # Test detection with only waypoints
        matches = detector.detect(frame, current_zone="BLOOD MOOR", categories=['waypoints'])
        
        print(f"\n✓ Found {len(matches)} waypoint matches")
        for match in matches[:10]:
            print(f"  - {match.template_name}: {match.confidence:.2f} @ {match.location}")
    else:
        print(f"No test frame found at {test_frame_path}")


if __name__ == '__main__':
    main()
