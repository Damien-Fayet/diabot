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
import shutil
import json
from pathlib import Path

try:
    import pytesseract
    TESSERACT_AVAILABLE = True
except ImportError:
    TESSERACT_AVAILABLE = False

# Resolve tesseract executable if available in PATH or common install dir.
TESSERACT_CMD = shutil.which("tesseract") or r"C:\\Program Files\\Tesseract-OCR\\tesseract.exe"
if TESSERACT_AVAILABLE and TESSERACT_CMD:
    pytesseract.pytesseract.tesseract_cmd = TESSERACT_CMD

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
        
        # Load zones database for OCR correction
        self._load_zones_database()
        
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
    
    def _load_zones_database(self):
        """Load Diablo 2 zones database for OCR correction."""
        zones_path = Path(__file__).parent.parent.parent.parent / "data" / "zones_database.json"
        
        try:
            with open(zones_path, 'r', encoding='utf-8') as f:
                zones_data = json.load(f)
                self.known_zones = zones_data.get('zones', [])
                self.ocr_corrections = zones_data.get('ocr_corrections', {})
                self.fuzzy_threshold = zones_data.get('fuzzy_match_threshold', 0.6)
                
                if self.debug:
                    print(f"✓ Loaded {len(self.known_zones)} zones from database")
        except Exception as e:
            if self.debug:
                print(f"⚠️  Could not load zones database: {e}")
            # Fallback to basic list
            self.known_zones = ["ROGUE ENCAMPMENT", "BLOOD MOOR", "COLD PLAINS"]
            self.ocr_corrections = {"REOVE": "ROGUE", "MOON": "ROGUE"}
            self.fuzzy_threshold = 0.6
    
    def analyze(self, frame: np.ndarray) -> UIState:
        """
        Analyze UI from frame using OCR.
        
        Args:
            frame: BGR image from game
            
        Returns:
            UIState with detected UI info
        """
        hp_ratio = self.extract_hp(frame)
        mana_ratio = self.extract_mana(frame)
        zone_name = self.extract_zone(frame)
        
        # Detect potions (simplified - not yet implemented)
        potions = {'health': 0, 'mana': 0, 'rejuvenation': 0}
        
        return UIState(
            hp_ratio=hp_ratio,
            mana_ratio=mana_ratio,
            zone_name=zone_name,
            potions_available=potions,
        )
    
    def extract_hp(self, frame: np.ndarray) -> float:
        """
        Extract HP ratio from frame using OCR.
        
        Args:
            frame: BGR image from game
            
        Returns:
            HP ratio (0.0-1.0)
        """
        lifebar_region = UI_REGIONS['lifebar_ui'].extract_from_frame(frame)
        hp_ratio = self._read_bar_via_ocr(lifebar_region, "HP")
        
        # Try alternative preprocessing if primary fails
        if hp_ratio == 0.5:  # fallback value means OCR failed
            hp_ratio_alt = self._read_bar_via_ocr_alternative(lifebar_region, "HP")
            if hp_ratio_alt != 0.5:
                hp_ratio = hp_ratio_alt
        
        return hp_ratio
    
    def extract_mana(self, frame: np.ndarray) -> float:
        """
        Extract Mana ratio from frame using OCR.
        
        Args:
            frame: BGR image from game
            
        Returns:
            Mana ratio (0.0-1.0)
        """
        manabar_region = UI_REGIONS['manabar_ui'].extract_from_frame(frame)
        mana_ratio = self._read_bar_via_ocr(manabar_region, "Mana")
        
        # Try alternative preprocessing if primary fails
        if mana_ratio == 0.5:
            mana_ratio_alt = self._read_bar_via_ocr_alternative(manabar_region, "Mana")
            if mana_ratio_alt != 0.5:
                mana_ratio = mana_ratio_alt
        
        return mana_ratio
    
    def extract_zone(self, frame: np.ndarray) -> str:
        """
        Extract zone name from frame using OCR.
        
        Args:
            frame: BGR image from game
            
        Returns:
            Zone name string (e.g. "ROGUE ENCAMPMENT")
        """
        zone_region = UI_REGIONS['zone_ui'].extract_from_frame(frame)
        zone_name = self._read_zone_via_ocr(zone_region)
        return zone_name
    
    def is_minimap_visible(self, frame: np.ndarray) -> bool:
        """
        Check if fullscreen minimap is currently visible.
        
        Uses zone OCR to determine minimap state:
        - If zone name matches a known zone after fuzzy matching: minimap is visible
        - Otherwise: minimap is hidden
        
        Args:
            frame: BGR image from game
            
        Returns:
            True if fullscreen minimap is visible, False otherwise
        """
        zone_name = self.extract_zone(frame)
        
        # Minimap is visible only if the zone name matches a known zone
        # This filters out OCR artifacts like single letters or random characters
        if zone_name and zone_name in self.known_zones:
            return True
        return False
    
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
        
        # Resize first to improve OCR accuracy (make text larger)
        scale_factor = 4
        h, w = gray.shape
        resized = cv2.resize(gray, (w * scale_factor, h * scale_factor), 
                            interpolation=cv2.INTER_CUBIC)
        
        # Apply bilateral filter for edge-preserving smoothing (better than aggressive denoise)
        smoothed = cv2.bilateralFilter(resized, 9, 75, 75)
        
        # Enhance contrast with CLAHE
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(smoothed)
        
        # Simple binary threshold works better for Diablo 2 crisp text
        # Otsu finds optimal threshold automatically
        _, binary = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Light morphological cleanup
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
        cleaned = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=1)
        
        # Debug: save preprocessed image
        if self.debug:
            debug_path = f"data/screenshots/outputs/diagnostic/ocr_{bar_name}_processed.png"
            import os
            os.makedirs(os.path.dirname(debug_path), exist_ok=True)
            cv2.imwrite(debug_path, cleaned)
        
        # OCR configuration for numeric text (Diablo 2 uses specific font)
        # Try multiple PSM modes in priority order:
        # PSM 8: single word (good for short numeric strings)
        # PSM 7: single line
        # PSM 13: raw line
        # PSM 6: uniform block (default)
        configs = [
            ('--psm 8 --oem 1', 'PSM8+OEM1'),
            ('--psm 8 --oem 1 -c tessedit_char_whitelist=" 0123456789/"', 'PSM8+whitelist'),
            ('--psm 7 --oem 1', 'PSM7+OEM1'),
            ('--psm 7 --oem 1 -c tessedit_char_whitelist=" 0123456789/"', 'PSM7+whitelist'),
            ('--psm 6 --oem 1', 'PSM6+OEM1'),
            ('--psm 11 --oem 1', 'PSM11+OEM1'),  # Sparse text
            ('--psm 13 --oem 1', 'PSM13+OEM1'),  # Raw line
        ]
        
        text = ""
        for config, label in configs:
            try:
                result = pytesseract.image_to_string(cleaned, config=config).strip()
                # if self.debug:
                #     print(f"  [{label}] → '{result}'")
                if result:  # First non-empty result wins
                    text = result
                    # if self.debug:
                    #     print(f"📖 {bar_name} OCR success with {label}: '{text}'")
                    break
            except Exception as e:
                # if self.debug:
                #     print(f"  [{label}] → ERROR: {e}")
                pass
        
        if not text and self.debug:
            print(f"📖 {bar_name} OCR: '' (all configs failed)")
        
        # Parse text: expect "current/max" or just "current"
        ratio = self._parse_bar_text(text, bar_name)
        return ratio
    
    def _read_bar_via_ocr_alternative(self, region: np.ndarray, bar_name: str) -> float:
        """Alternative OCR approach with inverted threshold for white text on dark background.
        
        Args:
            region: BGR image patch
            bar_name: "HP" or "Mana" for debug
            
        Returns:
            Ratio (0.0-1.0), or 0.5 as fallback
        """
        if region.size == 0 or not TESSERACT_AVAILABLE:
            return 0.5
        
        gray = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)
        
        # Larger scale for small text
        scale_factor = 6
        h, w = gray.shape
        resized = cv2.resize(gray, (w * scale_factor, h * scale_factor), 
                            interpolation=cv2.INTER_CUBIC)
        
        # Try inverted: black text on white background
        _, binary = cv2.threshold(resized, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        
        # Save alternative preprocessing
        if self.debug:
            debug_path = f"data/screenshots/outputs/diagnostic/ocr_{bar_name}_alt_processed.png"
            import os
            os.makedirs(os.path.dirname(debug_path), exist_ok=True)
            cv2.imwrite(debug_path, binary)
        
        config = '--psm 8 --oem 1 -c tessedit_char_whitelist=" 0123456789/"'
        try:
            text = pytesseract.image_to_string(binary, config=config).strip()
            if self.debug:
                print(f"📖 {bar_name} OCR (alternative inverted): '{text}'")
            if text:
                return self._parse_bar_text(text, bar_name)
        except Exception as e:
            if self.debug:
                print(f"❌ {bar_name} alternative OCR failed: {e}")
        
        return 0.5
    
    def _parse_bar_text(self, text: str, bar_name: str) -> float:
        """
        Parse OCR text to extract ratio.
        
        Formats supported:
        - "120/150" → 0.8
        - "120" → assume mid-range, 0.5
        - "120 / 150" → 0.8 (with spaces)
        - "Life: 32/40" → 0.8 (with label prefix)
        - "Lire: 32/40" → 0.8 (OCR misread of Life)
        - "Mana: 35 / 35" → 1.0 (with label prefix and spaces)
        
        Args:
            text: OCR result
            bar_name: For debug messages
            
        Returns:
            Ratio (0.0-1.0)
        """
        # Remove common label prefixes (case insensitive)
        text = re.sub(r'^(life|lire|mana|hp|mp):\s*', '', text, flags=re.IGNORECASE)
        
        # Remove spaces
        text = text.replace(" ", "")
        
        # Try to match "number/number"
        match = re.search(r'(\d+)/(\d+)', text)
        if match:
            current = int(match.group(1))
            maximum = int(match.group(2))
            
            if maximum > 0:
                ratio = current / maximum
                # if self.debug:
                #     print(f"  → {bar_name}: {current}/{maximum} = {ratio:.2%}")
                return max(0.0, min(1.0, ratio))
        
        # Try to match just a single number
        match = re.search(r'(\d+)', text)
        if match:
            value = int(match.group(1))
            # if self.debug:
            #     print(f"  → {bar_name}: Got single value {value}, assuming mid-range")
            # Without max, assume some reasonable range
            # Could use heuristics based on typical HP/Mana values
            return 0.5
        
        # Failed to parse
        # if self.debug:
        #     print(f"  → {bar_name}: Could not parse '{text}', using fallback")
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
        
        import os
        debug_dir = "data/screenshots/outputs/diagnostic"
        os.makedirs(debug_dir, exist_ok=True)
        
        # Debug: save original zone region
        if self.debug:
            debug_path = f"{debug_dir}/ocr_Zone_0_original.png"
            cv2.imwrite(debug_path, region)
            print(f"🔍 [Zone OCR] Processing region: {region.shape}")
        
        # Try color-based filtering first (Diablo 2 zone text is gold/yellow)
        hsv = cv2.cvtColor(region, cv2.COLOR_BGR2HSV)
        
        # Gold/yellow range in HSV
        # H: 15-40 (yellow/gold), S: 50-255 (saturated), V: 100-255 (bright)
        lower_gold = np.array([15, 50, 100])
        upper_gold = np.array([40, 255, 255])
        
        mask = cv2.inRange(hsv, lower_gold, upper_gold)
        
        # Debug: save color mask
        if self.debug:
            debug_path = f"{debug_dir}/ocr_Zone_1_color_mask.png"
            cv2.imwrite(debug_path, mask)
        
        # SIMPLIFIED: Use color mask directly for OCR, skip grayscale and binary
        # The mask is already binary (0 or 255) and isolates gold text
        
        # If no gold text found, OCR will likely fail (which is expected)
        # Light morphological cleanup
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
        cleaned = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)
        
        # Debug: save final preprocessed zone image
        if self.debug:
            debug_path = f"{debug_dir}/ocr_Zone_2_cleaned.png"
            cv2.imwrite(debug_path, cleaned)
        
        # OCR configuration - PSM 7 (single text line) works best for zone names
        # tessedit_write_images=true saves Tesseract's internal processing steps
        config = '--psm 7 --oem 1 -c tessedit_write_images=true'
        
        text = ""
        try:
            result = pytesseract.image_to_string(cleaned, config=config).strip()
            if self.debug:
                print(f"  [PSM7] → '{result}'")
            if result:
                # Clean up: remove extra whitespace, remove common OCR artifacts
                text = ' '.join(result.split())
                # Remove noise: keep only letters, spaces, and common punctuation
                text = re.sub(r'[^A-Za-z\s\'-]', '', text)
                text = text.strip()
                if self.debug:
                    print(f"📍 Zone OCR cleaned text: '{text}'")
        except Exception as e:
            if self.debug:
                print(f"  [PSM7] → ERROR: {e}")
        
        if not text and self.debug:
            print(f"📍 Zone OCR failed")
        
        # Apply fuzzy matching for known zone names
        if text:
            text = self._correct_zone_name(text)
        
        if self.debug:
            print(f"📍 Zone OCR final result: '{text}'\n")
        
        return text
    
    def _correct_zone_name(self, ocr_text: str) -> str:
        """
        Correct common OCR mistakes for Diablo 2 zone names.
        Uses fuzzy matching to map to known zone names from database.
        
        Args:
            ocr_text: Raw OCR output
            
        Returns:
            Corrected zone name
        """
        # Convert to uppercase for comparison
        ocr_text_upper = ocr_text.upper()
        
        # Apply direct corrections from database
        for mistake, correction in self.ocr_corrections.items():
            ocr_text_upper = ocr_text_upper.replace(mistake, correction)
        
        # Additional contextual corrections specific to zone text
        # These are more aggressive than database corrections
        contextual_fixes = {
            "BLEEP": "BLOOD",
            "BLEOD": "BLOOD",
            "PLOOD": "BLOOD",
            "BLEED": "BLOOD",
            "BLEER": "BLOOD",
            "MEER": "MOOR",
            "MOZE": "MOOR",
            "PLAIR": "PLAIN",
            "FLAIN": "PLAIN",
            "STONY FEILD": "STONY FIELD",
            "TAME HIGHLAND": "TAMOE HIGHLAND",
            "TAMBOE": "TAMOE",
            "TAROE": "TAMOE",
        }
        
        for mistake, correction in contextual_fixes.items():
            if mistake in ocr_text_upper:
                ocr_text_upper = ocr_text_upper.replace(mistake, correction)
                if self.debug:
                    print(f"  🔧 Context fix: {mistake} → {correction}")
        
        # Try to match with known zones using fuzzy similarity
        from difflib import get_close_matches
        matches = get_close_matches(
            ocr_text_upper, 
            self.known_zones, 
            n=1, 
            cutoff=self.fuzzy_threshold
        )
        
        if matches:
            if self.debug:
                print(f"  🔧 Fuzzy match: '{ocr_text}' → '{matches[0]}'")
            return matches[0]
        
        # No match found, return cleaned text
        return ocr_text_upper
    
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
