#!/usr/bin/env python3
"""
Simple continuous test for minimap visibility detection.
Displays True/False in real-time to verify is_minimap_visible() works correctly.
"""
import sys
from pathlib import Path
import time

# Setup path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from diabot.core.implementations import WindowsScreenCapture
from diabot.vision.ui_vision import UIVisionModule


def main():
    print("=" * 70)
    print("🔍 MINIMAP VISIBILITY TEST (Press Ctrl+C to stop)")
    print("=" * 70)
    print()
    print("This script continuously checks if the fullscreen minimap is visible.")
    print("Press TAB in game to toggle minimap and watch the output change.\n")
    
    try:
        # Initialize
        capture = WindowsScreenCapture()
        print(f"✓ Found window: {capture.window_title}\n")
        
        ui_vision = UIVisionModule(debug=True)  # Enable debug for OCR details
        
        print("Starting continuous monitoring...")
        print("-" * 70)
        
        frame_count = 0
        last_state = None
        
        while True:
            frame = capture.get_frame()
            if frame is None:
                print("⚠️  Failed to capture frame")
                time.sleep(1)
                continue
            
            # Check minimap visibility (extract zone once and reuse)
            zone_name = ui_vision.extract_zone(frame)
            is_visible = zone_name in ui_vision.known_zones if zone_name else False
            
            print("🟢 VISIBLE" if is_visible else "🔴 HIDDEN")
            
            frame_count += 1
            time.sleep(0.5)  # Check twice per second
            
    except KeyboardInterrupt:
        print("\n\n" + "=" * 70)
        print("✅ Test stopped by user")
        print("=" * 70)
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
