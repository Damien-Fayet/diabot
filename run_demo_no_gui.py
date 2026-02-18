"""Run demo without GUI for CI/testing."""
import sys
sys.argv = ['demo_minimap_slam.py', '--no-viz']

# Monkey patch cv2.imshow and cv2.waitKey to avoid GUI
import cv2
original_imshow = cv2.imshow
original_waitKey = cv2.waitKey

def fake_imshow(window_name, image):
    print(f"[Display] Would show window: {window_name}, image shape: {image.shape}")
    return None

def fake_waitKey(delay):
    return -1  # No key pressed

cv2.imshow = fake_imshow
cv2.waitKey = fake_waitKey

# Now run the demo
from demo_minimap_slam import demo_movement_sequence

try:
    print("\n" + "="*70)
    print("RUNNING SLAM DEMO (NO GUI)")
    print("="*70)
    demo_movement_sequence()
    print("\n✓ Demo completed successfully!")
except Exception as e:
    print(f"\n❌ Demo failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
