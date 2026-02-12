"""
Test OCR on all preprocessing steps for zone detection.

This script captures the current game frame and tests OCR on each
preprocessing variant to find which works best.
"""

from src.diabot.core.implementations import WindowsScreenCapture
from src.diabot.vision.ui_vision import UIVisionModule


def main():
    print("="*80)
    print("🧪 ZONE OCR MULTI-STEP TESTING")
    print("="*80)
    print("\nThis will test OCR on multiple preprocessing variants:")
    print("  0. Original image")
    print("  1. Color mask (gold text only)")
    print("  2. Cleaned color mask (morphology)")
    print("  3. Grayscale")
    print("  4. Binary (Otsu)")
    print("  5. Binary inverted")
    print("  6. Adaptive threshold")
    print("\nCapturing game screen...")
    
    # Capture frame
    try:
        capture = WindowsScreenCapture(window_title="Diablo II: Resurrected")
        frame = capture.get_frame()
        if frame is None:
            print("❌ Failed to capture frame")
            return
        print(f"✓ Captured frame: {frame.shape}\n")
    except Exception as e:
        print(f"❌ Capture error: {e}")
        return
    
    # Create UI vision module with debug enabled
    ui_vision = UIVisionModule(debug=True)
    
    # Test zone OCR with all preprocessing steps
    print("Testing zone OCR with all preprocessing steps...\n")
    zone_name = ui_vision.extract_zone(frame, test_all_steps=True)
    
    print("\n" + "="*80)
    print(f"🎯 FINAL RESULT: '{zone_name}'")
    print("="*80)
    print(f"\nAll intermediate images saved to: data/screenshots/outputs/diagnostic/")
    print("Check the images to see which preprocessing works best!")


if __name__ == '__main__':
    main()
