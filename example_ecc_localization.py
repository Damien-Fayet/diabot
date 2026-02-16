#!/usr/bin/env python3
"""
Example: Using ECC Static Map Localizer with the Diabot pipeline.

This demonstrates how to integrate the advanced ECC localization
into the bot's main navigation loop.
"""

import sys
from pathlib import Path

# Setup path
sys.path.insert(0, str(Path(__file__).parent / "src"))

import cv2
import numpy as np
from diabot.navigation import ECCStaticMapLocalizer, load_zone_static_map
from diabot.core.implementations import ScreenshotFileSource, WindowsScreenCapture
from diabot.vision.ui_vision import UIVisionModule


def example_1_static_image():
    """Example 1: Process static screenshot files."""
    print("\n" + "="*70)
    print("EXAMPLE 1: Static Screenshot Localization")
    print("="*70)
    
    # Create localizer
    localizer = ECCStaticMapLocalizer(
        debug=True,
        output_dir=Path("data/screenshots/outputs/ecc_localization")
    )
    
    # Load static map for current zone (Rogue Encampment)
    zone_name = "Rogue Encampment"
    map_path = load_zone_static_map(zone_name)
    
    if map_path:
        localizer.load_static_map(map_path)
        print(f"[+] Loaded static map for {zone_name}")
    
    # Load test screenshots
    inputs_dir = Path("data/screenshots/inputs")
    without = sorted(inputs_dir.glob("*_without_minimap*.png"))[-1]
    with_img = sorted(inputs_dir.glob("*_with_minimap*.png"))[-1]
    
    print(f"[*] Using screenshots:")
    print(f"    Without minimap: {without.name}")
    print(f"    With minimap: {with_img.name}")
    
    frame_without = cv2.imread(str(without))
    frame_with = cv2.imread(str(with_img))
    
    # Localize
    player_pos, confidence = localizer.localize(
        frame_with,
        frame_without,
        motion_type='HOMOGRAPHY',
        use_oriented_filter=True
    )
    
    if player_pos:
        print(f"\n[SUCCESS] Player localized!")
        print(f"           Position: {player_pos}")
        print(f"           Confidence: {confidence:.3f}")
    else:
        print(f"\n[FAILED] Could not localize player")


def example_2_live_capture():
    """Example 2: Live capture with minimap toggle."""
    print("\n" + "="*70)
    print("EXAMPLE 2: Live Capture Localization")
    print("="*70)
    print("[!] This requires Diablo 2 Resurrected running on Windows")
    print("[!] The bot will toggle minimap visibility automatically")
    
    try:
        from diabot.core.implementations import WindowsScreenCapture
    except ImportError:
        print("[!] pyautogui not available, skipping live demo")
        return
    
    import time
    import pyautogui
    
    # Setup
    capture = WindowsScreenCapture()
    ui_vision = UIVisionModule(debug=False)
    localizer = ECCStaticMapLocalizer(
        debug=True,
        output_dir=Path("data/screenshots/outputs/ecc_localization_live")
    )
    
    try:
        # Hide minimap and capture background
        print("\n[*] Hiding minimap...")
        capture.activate_window()
        time.sleep(0.2)
        pyautogui.press('tab')
        time.sleep(0.5)
        
        frame_without = capture.get_frame()
        print(f"[+] Captured background: {frame_without.shape}")
        
        # Show minimap and capture
        print("\n[*] Showing minimap...")
        pyautogui.press('tab')
        time.sleep(0.5)
        
        frame_with = capture.get_frame()
        print(f"[+] Captured with minimap: {frame_with.shape}")
        
        # Detect zone
        zone = ui_vision.extract_zone(frame_with)
        print(f"[+] Current zone: {zone}")
        
        # Load static map
        map_path = load_zone_static_map(zone)
        if map_path:
            localizer.load_static_map(map_path)
            print(f"[+] Loaded static map")
        
        # Localize
        print("\n[*] Localizing player...")
        player_pos, confidence = localizer.localize(
            frame_with,
            frame_without,
            motion_type='HOMOGRAPHY'
        )
        
        if player_pos:
            print(f"\n[SUCCESS] Player position: {player_pos}")
            print(f"          Confidence: {confidence:.3f}")
        else:
            print(f"\n[FAILED] Could not localize")
    
    finally:
        # Show minimap again
        print("\n[*] Restoring minimap...")
        capture.activate_window()
        time.sleep(0.2)
        pyautogui.press('tab')


def example_3_integration_with_bot():
    """Example 3: Integration pattern for main bot class."""
    print("\n" + "="*70)
    print("EXAMPLE 3: Integration Pattern")
    print("="*70)
    
    code = '''
# In DiabotRunner or similar:

class DiabotRunner:
    def __init__(self, ...):
        # ... existing init code ...
        
        # Initialize ECC localizer for advanced positioning
        self.ecc_localizer = ECCStaticMapLocalizer(
            debug=debug,
            output_dir=Path("data/debug/localization")
        )
    
    def on_zone_change(self, zone_name: str):
        """Handle zone transitions."""
        # Load static map for new zone
        map_path = load_zone_static_map(zone_name)
        if map_path:
            self.ecc_localizer.load_static_map(map_path)
            print(f"[+] Loaded map for {zone_name}")
    
    def update_position(self):
        """Call from main loop to update player position."""
        # Optionally use ECC for robust positioning
        # (more expensive than minimap-only, so use sparingly)
        
        frame = self.image_source.get_frame()
        
        # Try ECC localization if we haven't moved for a while
        if self.should_use_ecc_localization():
            # Capture background and foreground
            # This would require frame buffering
            player_pos, confidence = self.ecc_localizer.localize(
                frame_with_minimap,
                frame_without_minimap
            )
            
            if confidence > 0.5:  # Good confidence
                self.player_position = player_pos
    
    def should_use_ecc_localization(self) -> bool:
        """Determine when to use expensive ECC alignment."""
        # Use when:
        # - Every N frames (e.g., every 100 frames)
        # - When minimap movement uncertainty is high
        # - After zone transitions
        return self.frame_count % 100 == 0
'''
    
    print(code)


def example_4_pipeline_components():
    """Example 4: Using individual pipeline components."""
    print("\n" + "="*70)
    print("EXAMPLE 4: Using Components Separately")
    print("="*70)
    
    from diabot.navigation import (
        MinimapEdgeExtractor,
        ECCAligner,
        StaticMapMatcher,
        OrientedFilterBank,
        OrientedMorphology
    )
    
    code = '''
# Component-based usage for more control:

class MyCustomLocalizer:
    def __init__(self):
        # Extractor: Get edges from minimap difference
        self.extractor = MinimapEdgeExtractor(debug=True)
        
        # Aligner: Multi-scale ECC alignment
        self.aligner = ECCAligner(debug=True)
        
        # Matcher: Connect to static map
        self.matcher = StaticMapMatcher(
            static_map_path="path/to/map.png",
            debug=True
        )
        
        # Filters: Custom preprocessing
        self.filter_bank = OrientedFilterBank(angles=[60, 120])
        self.morph = OrientedMorphology()
    
    def custom_pipeline(self, frame_with, frame_without):
        # Step 1: Extract minimap difference
        diff = self.extractor.extract_difference(frame_with, frame_without)
        
        # Step 2: Custom preprocessing
        processed = self.extractor.process_difference(diff)
        
        # Step 3: Apply oriented filters
        oriented = self.filter_bank.apply(processed)
        
        # Step 4: Apply morphological closing
        closed = self.morph.apply_oriented_closing(oriented)
        
        # Step 5: Extract final edges
        edges = cv2.Canny(closed, 50, 150)
        
        # Step 6: Align with static map
        player_pos, confidence = self.matcher.match(edges)
        
        return player_pos, confidence
'''
    
    print(code)


def main():
    """Run all examples."""
    print("\n" + "="*70)
    print("ECC STATIC MAP LOCALIZER - INTEGRATION EXAMPLES")
    print("="*70)
    
    # Try example 1 (static images)
    try:
        example_1_static_image()
    except Exception as e:
        print(f"\n[!] Example 1 failed: {e}")
    
    # Show other examples
    example_2_live_capture()
    example_3_integration_with_bot()
    example_4_pipeline_components()
    
    print("\n" + "="*70)
    print("Examples complete!")
    print("="*70)


if __name__ == "__main__":
    main()
