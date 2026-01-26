"""
Analyze minimap to find optimal wall detection threshold.
"""
import cv2
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from diabot.core.implementations import WindowsScreenCapture
from diabot.navigation.minimap_extractor import MinimapExtractor

def analyze_minimap():
    """Capture and analyze minimap to find optimal threshold."""
    print("=" * 60)
    print("MINIMAP THRESHOLD ANALYZER")
    print("=" * 60)
    
    # Capture current screen
    capture = WindowsScreenCapture(window_title="Diablo II: Resurrected")
    frame = capture.get_frame()
    
    if frame is None:
        print("ERROR: Could not capture frame")
        return
    
    print(f"✓ Captured frame: {frame.shape}")
    
    # Extract minimap
    extractor = MinimapExtractor()
    minimap = extractor.extract(frame)
    
    if minimap is None:
        print("ERROR: Could not extract minimap")
        return
    
    print(f"✓ Extracted minimap: {minimap.shape}")
    
    # Convert to grayscale
    gray = cv2.cvtColor(minimap, cv2.COLOR_BGR2GRAY)
    
    # Apply bilateral filter (same as processor)
    filtered = cv2.bilateralFilter(gray, 5, 50, 50)
    
    # Analyze pixel value distribution
    print("\n" + "=" * 60)
    print("PIXEL VALUE ANALYSIS")
    print("=" * 60)
    
    hist, bins = np.histogram(filtered.flatten(), bins=256, range=(0, 256))
    
    print(f"Min pixel value: {filtered.min()}")
    print(f"Max pixel value: {filtered.max()}")
    print(f"Mean pixel value: {filtered.mean():.1f}")
    print(f"Median pixel value: {np.median(filtered):.1f}")
    print(f"Std deviation: {filtered.std():.1f}")
    
    # Find percentiles
    p10 = np.percentile(filtered, 10)
    p25 = np.percentile(filtered, 25)
    p50 = np.percentile(filtered, 50)
    p75 = np.percentile(filtered, 75)
    p90 = np.percentile(filtered, 90)
    
    print(f"\nPercentiles:")
    print(f"  10%: {p10:.1f}")
    print(f"  25%: {p25:.1f}")
    print(f"  50%: {p50:.1f}")
    print(f"  75%: {p75:.1f}")
    print(f"  90%: {p90:.1f}")
    
    # Test multiple thresholds
    print("\n" + "=" * 60)
    print("TESTING THRESHOLDS")
    print("=" * 60)
    
    thresholds = [30, 40, 50, 60, 70, 80, 100]
    
    fig, axes = plt.subplots(3, 3, figsize=(18, 18))
    axes = axes.flatten()
    
    # Show original
    axes[0].imshow(cv2.cvtColor(minimap, cv2.COLOR_BGR2RGB))
    axes[0].set_title("Original Minimap", fontsize=14, fontweight='bold')
    axes[0].axis('off')
    
    # Show filtered grayscale
    axes[1].imshow(filtered, cmap='gray')
    axes[1].set_title(f"Filtered Grayscale\nMean: {filtered.mean():.1f}, Median: {np.median(filtered):.1f}", fontsize=12)
    axes[1].axis('off')
    
    # Test each threshold
    for idx, thresh in enumerate(thresholds, start=2):
        # CORRECTED: D2R walls are BRIGHT, so use THRESH_BINARY (not INV)
        _, binary = cv2.threshold(filtered, thresh, 255, cv2.THRESH_BINARY)
        
        # Count walls vs free
        # After THRESH_BINARY: pixels > thresh = 255 (walls), pixels <= thresh = 0 (free)
        wall_pixels = np.sum(binary > 127)
        free_pixels = np.sum(binary <= 127)
        total = wall_pixels + free_pixels
        wall_pct = (wall_pixels / total) * 100
        free_pct = (free_pixels / total) * 100
        
        # Create colored version: Red=walls, Green=free
        colored = np.zeros((*binary.shape, 3), dtype=np.uint8)
        colored[binary > 127] = [255, 0, 0]  # Red for walls
        colored[binary <= 127] = [0, 255, 0]  # Green for free
        
        axes[idx].imshow(colored)
        axes[idx].set_title(
            f"Threshold: {thresh}\n"
            f"🔴 Walls: {wall_pct:.1f}% | 🟢 Free: {free_pct:.1f}%",
            fontsize=12,
            fontweight='bold' if thresh == int(p50) else 'normal'
        )
        axes[idx].axis('off')
        
        print(f"Threshold {thresh:3d}: Walls={wall_pct:5.1f}% Free={free_pct:5.1f}%")
    
    plt.tight_layout()
    
    # Save analysis
    output_dir = Path(__file__).parent / "data" / "screenshots" / "outputs"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "minimap_threshold_analysis.png"
    plt.savefig(str(output_path), dpi=150, bbox_inches='tight')
    print(f"\n✓ Saved analysis: {output_path}")
    
    # Show plot
    plt.show()
    
    # Recommendation
    print("\n" + "=" * 60)
    print("RECOMMENDATION")
    print("=" * 60)
    
    # Good threshold should give roughly 20-40% walls for town, 40-60% for dungeons
    recommended_thresh = int(p50)  # Start with median
    print(f"Recommended starting threshold: {recommended_thresh}")
    print(f"  - For towns (open areas): try {int(p50) - 20} to {int(p50)}")
    print(f"  - For dungeons (dense): try {int(p50)} to {int(p50) + 20}")
    print("\nLook at the images above to see which threshold works best!")
    print("Aim for: Clear paths visible, walls distinct, ~30-50% free space")

if __name__ == "__main__":
    analyze_minimap()
