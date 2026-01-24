"""
UI Vision Module - Detects and analyzes UI elements only.

Handles:
- Health bar (OCR on text region)
- Mana bar (OCR on text region)
- Potion indicators
- Buff/debuff icons
- Spell cooldowns

Uses OCR to read numeric values from text regions above HP/Mana orbs.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional
import cv2
import numpy as np
import re

try:
    import pytesseract
    TESSERACT_AVAILABLE = True
except ImportError:
    TESSERACT_AVAILABLE = False

from .screen_regions import UI_REGIONS


@dataclass
class UIState:
    """Current state of UI elements."""
    
    hp_ratio: float = 0.5          # 0.0 = dead, 1.0 = full health
    mana_ratio: float = 0.5        # 0.0 = empty, 1.0 = full mana
    zone_name: str = ""            # Current game zone/area name
    potions_available: Dict[str, int] = field(default_factory=lambda: {
        'health': 0,
        'mana': 0,
        'rejuvenation': 0
    })
    buffs: List[str] = field(default_factory=list)
    debuffs: List[str] = field(default_factory=list)
    is_dead: bool = False


class UIVisionModule:
    """
    Detects and analyzes game UI elements.
    
    Separate from environment vision for better reliability.
    """
    
    def __init__(self, debug: bool = False):
        """
        Initialize UI vision module.
        
        Args:
            debug: Enable debug output
        """
        self.debug = debug
        
        # HSV ranges for UI elements
        # These are specific to UI areas, different from playfield
        self.health_bar_range = (
            np.array([0, 80, 100]),      # Low saturation red
            np.array([10, 255, 255]),
        )
        
        self.mana_bar_range = (
            np.array([100, 80, 100]),    # Low saturation blue
            np.array([130, 255, 255]),
        )
    
    def analyze(self, frame: np.ndarray) -> UIState:
        """
        Analyze UI from frame using OCR.
        
        Args:
            frame: BGR image from game
            
        Returns:
            UIState with detected UI info
        """
        # Extract text regions (BGR format for OCR)
        lifebar_region = UI_REGIONS['lifebar_ui'].extract_from_frame(frame)
        manabar_region = UI_REGIONS['manabar_ui'].extract_from_frame(frame)
        zone_region = UI_REGIONS['zone_ui'].extract_from_frame(frame)
        
        # Read values via OCR
        hp_ratio = self._read_bar_via_ocr(lifebar_region, "HP")
        mana_ratio = self._read_bar_via_ocr(manabar_region, "Mana")
        zone_name = self._read_zone_via_ocr(zone_region)
        
        # Detect potions (simplified - not yet implemented)
        potions = {'health': 0, 'mana': 0, 'rejuvenation': 0}
        
        return UIState(
            hp_ratio=hp_ratio,
            mana_ratio=mana_ratio,
            zone_name=zone_name,
            potions_available=potions,
        )
    
    def _read_bar_via_ocr(self, region: np.ndarray, bar_name: str) -> float:
        """
        Read HP/Mana value from text region using OCR.
        
        Expected format: "120/150" or just "120"
        
        Args:
            region: BGR image crop of text region
            bar_name: "HP" or "Mana" for debug messages
            
        Returns:
            Ratio (0.0-1.0), or 0.5 as fallback
        """
        if region.size == 0:
            if self.debug:
                print(f"⚠️  {bar_name}: Empty region")
            return 0.5
        
        if not TESSERACT_AVAILABLE:
            if self.debug:
                print(f"⚠️  {bar_name}: Tesseract not available, using fallback")
            return 0.5
        
        # Preprocess for OCR
        # Convert to grayscale
        gray = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)
        
        # Resize to improve OCR accuracy (make text larger)
        scale_factor = 3
        h, w = gray.shape
        resized = cv2.resize(gray, (w * scale_factor, h * scale_factor), 
                            interpolation=cv2.INTER_CUBIC)
        
        # Threshold to get clean text
        # Text is typically white/bright on dark background
        _, binary = cv2.threshold(resized, 150, 255, cv2.THRESH_BINARY)
        
        # Optional: morphological operations to clean up
        kernel = np.ones((2, 2), np.uint8)
        cleaned = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        
        # OCR configuration for numeric text
        # --psm 7: single text line
        # --oem 3: default OCR engine mode
        # -c tessedit_char_whitelist: only allow digits and slash
        config = '--psm 7 --oem 3 -c tessedit_char_whitelist=0123456789/'
        
        try:
            text = pytesseract.image_to_string(cleaned, config=config).strip()
            
            if self.debug:
                print(f"📖 {bar_name} OCR: '{text}'")
            
            # Parse text: expect "current/max" or just "current"
            ratio = self._parse_bar_text(text, bar_name)
            return ratio
            
        except Exception as e:
            if self.debug:
                print(f"❌ {bar_name} OCR failed: {e}")
            return 0.5
    
    def _parse_bar_text(self, text: str, bar_name: str) -> float:
        """
        Parse OCR text to extract ratio.
        
        Formats supported:
        - "120/150" → 0.8
        - "120" → assume mid-range, 0.5
        - "120 / 150" → 0.8 (with spaces)
        
        Args:
            text: OCR result
            bar_name: For debug messages
            
        Returns:
            Ratio (0.0-1.0)
        """
        # Remove spaces
        text = text.replace(" ", "")
        
        # Try to match "number/number"
        match = re.search(r'(\d+)/(\d+)', text)
        if match:
            current = int(match.group(1))
            maximum = int(match.group(2))
            
            if maximum > 0:
                ratio = current / maximum
                if self.debug:
                    print(f"  → {bar_name}: {current}/{maximum} = {ratio:.2%}")
                return max(0.0, min(1.0, ratio))
        
        # Try to match just a single number
        match = re.search(r'(\d+)', text)
        if match:
            value = int(match.group(1))
            if self.debug:
                print(f"  → {bar_name}: Got single value {value}, assuming mid-range")
            # Without max, assume some reasonable range
            # Could use heuristics based on typical HP/Mana values
            return 0.5
        
        # Failed to parse
        if self.debug:
            print(f"  → {bar_name}: Could not parse '{text}', using fallback")
        return 0.5
    
    def _read_zone_via_ocr(self, region: np.ndarray) -> str:
        """
        Read zone/area name from text region using OCR.
        
        Args:
            region: BGR image crop of zone text region
            
        Returns:
            Zone name as string, or empty string if failed
        """
        if region.size == 0:
            if self.debug:
                print(f"⚠️  Zone: Empty region")
            return ""
        
        if not TESSERACT_AVAILABLE:
            if self.debug:
                print(f"⚠️  Zone: Tesseract not available")
            return ""
        
        # Preprocess for OCR
        gray = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)
        
        # Resize to improve OCR accuracy
        scale_factor = 2
        h, w = gray.shape
        resized = cv2.resize(gray, (w * scale_factor, h * scale_factor), 
                            interpolation=cv2.INTER_CUBIC)
        
        # Threshold to get clean text
        # Zone text is typically white/bright
        _, binary = cv2.threshold(resized, 150, 255, cv2.THRESH_BINARY)
        
        # Clean up noise
        kernel = np.ones((2, 2), np.uint8)
        cleaned = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        
        # OCR configuration for text (allow letters, spaces, punctuation)
        # --psm 7: single text line
        config = '--psm 7 --oem 3'
        
        try:
            text = pytesseract.image_to_string(cleaned, config=config).strip()
            
            # Clean up: remove extra whitespace
            text = ' '.join(text.split())
            
            if self.debug:
                print(f"📍 Zone OCR: '{text}'")
            
            return text
            
        except Exception as e:
            if self.debug:
                print(f"❌ Zone OCR failed: {e}")
            return ""
    
    # Legacy color-based detection methods (deprecated, kept for reference)
    # Now using OCR-based detection instead
        """
        Detect health bar from UI region.
        
        Returns:
            Ratio of health (0.0-1.0)
        """
        if ui_region.size == 0:
            return 0.5  # Default middle

        # Slight blur to reduce noise
        blurred = cv2.GaussianBlur(ui_region, (3, 3), 0)

        # Red mask (covering both low and high hue reds)
        mask1 = cv2.inRange(blurred, np.array([0, 70, 80]), np.array([10, 255, 255]))
        mask2 = cv2.inRange(blurred, np.array([170, 70, 80]), np.array([180, 255, 255]))
        mask = cv2.bitwise_or(mask1, mask2)

        red_pixels = cv2.countNonZero(mask)
        total_pixels = ui_region.shape[0] * ui_region.shape[1]
        if total_pixels == 0:
            return 0.5

        # Take the largest red contour to avoid noise
        contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        largest_area = 0
        for c in contours:
            area = cv2.contourArea(c)
            if area > largest_area:
                largest_area = area

        # Combine overall coverage and largest blob area
        coverage_ratio = red_pixels / total_pixels
        blob_ratio = largest_area / total_pixels

        # Assume full bubble occupies ~20% of the region; normalize to that
        normalized = max(coverage_ratio, blob_ratio) / 0.20

        hp_ratio = float(max(0.0, min(1.0, normalized)))

        # Fallback: if low ratio, try broader top-left UI (bubble may be outside lifebar region)
        if hp_ratio < 0.6 and ui_wide_region.size > 0:
            blurred_wide = cv2.GaussianBlur(ui_wide_region, (3, 3), 0)
            mask1 = cv2.inRange(blurred_wide, np.array([0, 70, 80]), np.array([10, 255, 255]))
            mask2 = cv2.inRange(blurred_wide, np.array([170, 70, 80]), np.array([180, 255, 255]))
            mask_wide = cv2.bitwise_or(mask1, mask2)
            coverage_wide = cv2.countNonZero(mask_wide) / (ui_wide_region.shape[0] * ui_wide_region.shape[1])
            blob_wide = 0
            contours, _ = cv2.findContours(mask_wide, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            for c in contours:
                blob_wide = max(blob_wide, cv2.contourArea(c))
            blob_wide_ratio = blob_wide / (ui_wide_region.shape[0] * ui_wide_region.shape[1])
            fallback_ratio = max(coverage_wide, blob_wide_ratio) / 0.15  # bubble bigger proportion here
            hp_ratio = max(hp_ratio, float(max(0.0, min(1.0, fallback_ratio))))

        # Global fallback: search the full frame for the largest red blob (bubble)
        if hp_ratio < 0.6 and full_hsv is not None and full_hsv.size > 0:
            blurred_full = cv2.GaussianBlur(full_hsv, (3, 3), 0)
            mask1_f = cv2.inRange(blurred_full, np.array([0, 70, 80]), np.array([10, 255, 255]))
            mask2_f = cv2.inRange(blurred_full, np.array([170, 70, 80]), np.array([180, 255, 255]))
            mask_full = cv2.bitwise_or(mask1_f, mask2_f)
            contours, _ = cv2.findContours(mask_full, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            largest_area = 0
            for c in contours:
                largest_area = max(largest_area, cv2.contourArea(c))
            full_pixels = full_hsv.shape[0] * full_hsv.shape[1]
            if full_pixels > 0:
                # Assume bubble occupies ~0.5% of full frame when full
                ratio_full = (largest_area / full_pixels) / 0.005
                hp_ratio = max(hp_ratio, float(max(0.0, min(1.0, ratio_full))))

        return hp_ratio
    
    def _detect_mana_bar(self, ui_region: np.ndarray, ui_wide_region: np.ndarray, full_hsv: np.ndarray) -> float:
        """
        Detect mana bar from UI region.
        
        Returns:
            Ratio of mana (0.0-1.0)
        """
        if ui_region.size == 0:
            return 0.5

        blurred = cv2.GaussianBlur(ui_region, (3, 3), 0)

        # Blue mask
        mask = cv2.inRange(blurred,
                          np.array([95, 70, 80]),
                          np.array([135, 255, 255]))

        blue_pixels = cv2.countNonZero(mask)
        total_pixels = ui_region.shape[0] * ui_region.shape[1]
        if total_pixels == 0:
            return 0.5

        contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        largest_area = 0
        for c in contours:
            area = cv2.contourArea(c)
            if area > largest_area:
                largest_area = area

        coverage_ratio = blue_pixels / total_pixels
        blob_ratio = largest_area / total_pixels

        normalized = max(coverage_ratio, blob_ratio) / 0.20

        mana_ratio = float(max(0.0, min(1.0, normalized)))

        if mana_ratio < 0.6 and ui_wide_region.size > 0:
            blurred_wide = cv2.GaussianBlur(ui_wide_region, (3, 3), 0)
            mask_wide = cv2.inRange(blurred_wide,
                                    np.array([95, 70, 80]),
                                    np.array([135, 255, 255]))
            coverage_wide = cv2.countNonZero(mask_wide) / (ui_wide_region.shape[0] * ui_wide_region.shape[1])
            blob_wide = 0
            contours, _ = cv2.findContours(mask_wide, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            for c in contours:
                blob_wide = max(blob_wide, cv2.contourArea(c))
            blob_wide_ratio = blob_wide / (ui_wide_region.shape[0] * ui_wide_region.shape[1])
            fallback_ratio = max(coverage_wide, blob_wide_ratio) / 0.15
            mana_ratio = max(mana_ratio, float(max(0.0, min(1.0, fallback_ratio))))

        if mana_ratio < 0.6 and full_hsv is not None and full_hsv.size > 0:
            blurred_full = cv2.GaussianBlur(full_hsv, (3, 3), 0)
            mask_full = cv2.inRange(blurred_full,
                                    np.array([95, 70, 80]),
                                    np.array([135, 255, 255]))
            contours, _ = cv2.findContours(mask_full, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            largest_area = 0
            for c in contours:
                largest_area = max(largest_area, cv2.contourArea(c))
            full_pixels = full_hsv.shape[0] * full_hsv.shape[1]
            if full_pixels > 0:
                ratio_full = (largest_area / full_pixels) / 0.005
                mana_ratio = max(mana_ratio, float(max(0.0, min(1.0, ratio_full))))

        return mana_ratio
    
    
    # End of legacy color-based detection methods (commented out above)
    
    def _detect_potions(self, bottom_ui_region: np.ndarray) -> Dict[str, int]:
        """
        Detect potion indicators.
        
        Returns:
            Dict with potion counts
        """
        # Simplified: just count potion-like colors
        # Full implementation would detect actual potion icons
        
        return {
            'health': 0,
            'mana': 0,
            'rejuvenation': 0,
        }
