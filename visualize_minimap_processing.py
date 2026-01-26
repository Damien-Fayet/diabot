"""
Visualize minimap processing step by step.
Shows what the navigation system sees.
"""

import cv2
import numpy as np
from pathlib import Path

from src.diabot.navigation import (
    MinimapExtractor,
    MinimapProcessor,
    LocalMap
)


def main():
    # Load game screenshot
    image_path = "data/screenshots/inputs/game.png"
    print(f"Loading: {image_path}")
    frame = cv2.imread(image_path)
    
    if frame is None:
        print(f"ERROR: Could not load {image_path}")
        return
    
    print(f"Frame size: {frame.shape[1]}x{frame.shape[0]}")
    
    # Step 1: Extract minimap
    print("\n1. Extracting minimap...")
    extractor = MinimapExtractor(debug=True)
    minimap = extractor.extract(frame)
    print(f"   Minimap size: {minimap.shape[1]}x{minimap.shape[0]}")
    
    # Step 2: Process minimap at different thresholds
    print("\n2. Processing minimap with different wall thresholds...")
    
    thresholds = [60, 80, 100, 120]
    results = []
    
    for threshold in thresholds:
        processor = MinimapProcessor(grid_size=64, wall_threshold=threshold, debug=False)
        grid = processor.process(minimap)
        vis = processor.visualize(grid)
        
        # Count cells
        free_count = np.sum(grid.grid == 128)  # FREE
        wall_count = np.sum(grid.grid == 255)  # WALL
        
        print(f"   Threshold {threshold}: {free_count} free, {wall_count} walls")
        results.append((threshold, vis, free_count, wall_count))
    
    # Step 3: Create composite visualization
    print("\n3. Creating visualization...")
    
    # Resize minimap for display
    minimap_display = cv2.resize(minimap, (256, 256))
    
    # Create composite image
    rows = []
    
    # First row: original minimap and info
    info_img = np.zeros((256, 512, 3), dtype=np.uint8)
    cv2.putText(info_img, "Navigation System Demo", (10, 40), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
    cv2.putText(info_img, "Minimap Processing", (10, 80), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)
    cv2.putText(info_img, "Gray = Free space", (10, 140), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
    cv2.putText(info_img, "Dark = Walls", (10, 170), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (50, 50, 50), 1)
    cv2.putText(info_img, "Green = Player", (10, 200), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    
    row1 = np.hstack([minimap_display, info_img])
    rows.append(row1)
    
    # Second row: processed grids at different thresholds
    row2_images = []
    for threshold, vis, free_count, wall_count in results:
        # Add label
        labeled = vis.copy()
        cv2.putText(labeled, f"Threshold: {threshold}", (5, 15), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)
        cv2.putText(labeled, f"Free: {free_count}", (5, 235), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
        cv2.putText(labeled, f"Walls: {wall_count}", (5, 250), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (50, 50, 50), 1)
        row2_images.append(labeled)
    
    # Add padding to match width
    row2 = np.hstack(row2_images)
    # Pad to match row1 width
    if row2.shape[1] < row1.shape[1]:
        pad_width = row1.shape[1] - row2.shape[1]
        padding = np.zeros((row2.shape[0], pad_width, 3), dtype=np.uint8)
        row2 = np.hstack([row2, padding])
    
    rows.append(row2)
    
    # Combine all rows
    composite = np.vstack(rows)
    
    # Save
    output_path = Path("data/screenshots/outputs/minimap_processing_demo.png")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), composite)
    print(f"\n✓ Saved to: {output_path}")
    
    # Display
    print("\nDisplaying visualization...")
    cv2.imshow("Minimap Processing Demo", composite)
    print("Press any key to close...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    print("\n" + "="*60)
    print("Analysis:")
    print("="*60)
    print("The minimap shows mostly walls (dark areas).")
    print("This is typical for:")
    print("  - Town areas (buildings, walls)")
    print("  - Indoor dungeons")
    print("  - Areas with lots of obstacles")
    print("\nFor better navigation results, try screenshots from:")
    print("  - Open outdoor areas")
    print("  - Wilderness zones")
    print("  - Areas with clear paths")
    print("="*60)


if __name__ == "__main__":
    main()
