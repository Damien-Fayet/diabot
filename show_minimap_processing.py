"""
Visualize minimap processing steps side-by-side.
Shows the 8 steps of minimap analysis from extraction to final grid.
"""
import cv2
import numpy as np
from pathlib import Path


def show_minimap_steps():
    """Display all minimap processing steps in a grid layout."""
    
    steps_dir = Path("data/screenshots/outputs/diagnostic/minimap_steps")
    
    if not steps_dir.exists():
        print(f"❌ Directory not found: {steps_dir}")
        print("Run the bot with --debug to generate diagnostic images")
        return
    
    # Step names and descriptions
    steps = [
        ("step1_extracted.png", "1. Extracted\nMinimap Region"),
        ("step2_grayscale.png", "2. Grayscale\nConversion"),
        ("step3_contrast_enhanced.png", "3. Contrast Enhanced\nWhites→Whiter, Darks→Darker"),
        ("step4_filtered.png", "4. Bilateral Filter\nNoise Reduction"),
        ("step5_binary_threshold.png", "5. Binary Threshold\nWalls vs Free"),
        ("step6_morphology_open.png", "6. Morphology Open\nRemove Small Noise"),
        ("step7_morphology_close.png", "7. Morphology Close\nFill Small Gaps"),
        ("step8_resized_grid.png", "8. Resized Grid\n64×64 Occupancy"),
        ("step9_final_grid_colored.png", "9. Final Grid\nColored Visualization"),
    ]
    
    # Load all images
    images = []
    titles = []
    max_h = 0
    
    for filename, title in steps:
        filepath = steps_dir / filename
        if filepath.exists():
            img = cv2.imread(str(filepath))
            if img is not None:
                # Resize large images for display
                h, w = img.shape[:2]
                if w > 400:
                    scale = 400 / w
                    new_w = 400
                    new_h = int(h * scale)
                    img = cv2.resize(img, (new_w, new_h))
                
                images.append(img)
                titles.append(title)
                max_h = max(max_h, img.shape[0])
            else:
                print(f"⚠️  Failed to load: {filename}")
        else:
            print(f"⚠️  Not found: {filename}")
    
    if not images:
        print("❌ No images found")
        return
    
    print(f"✓ Loaded {len(images)} processing steps")
    
    # Create 3x3 grid layout
    rows = []
    for i in range(0, len(images), 3):
        row_images = images[i:i+3]
        row_titles = titles[i:i+3]
        
        # Normalize heights in this row
        max_row_h = max(img.shape[0] for img in row_images)
        
        # Add title and padding to each image
        row_with_titles = []
        for img, title in zip(row_images, row_titles):
            # Pad image height
            h, w = img.shape[:2]
            if h < max_row_h:
                pad = max_row_h - h
                img = cv2.copyMakeBorder(img, 0, pad, 0, 0, 
                                        cv2.BORDER_CONSTANT, value=(40, 40, 40))
            
            # Add title at the top
            title_height = 80
            title_img = np.zeros((title_height, img.shape[1], 3), dtype=np.uint8)
            title_img[:] = (40, 40, 40)
            
            # Draw title text (multi-line)
            title_lines = title.split('\n')
            y_offset = 20
            for line in title_lines:
                cv2.putText(title_img, line, (10, y_offset), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                y_offset += 20
            
            # Combine title and image
            combined = np.vstack([title_img, img])
            row_with_titles.append(combined)
        
        # Concatenate horizontally with spacing
        spacing = np.zeros((row_with_titles[0].shape[0], 10, 3), dtype=np.uint8)
        spacing[:] = (20, 20, 20)
        
        row_concat = row_with_titles[0]
        for img in row_with_titles[1:]:
            row_concat = np.hstack([row_concat, spacing, img])
        
        rows.append(row_concat)
    
    # Concatenate rows vertically with spacing
    if len(rows) > 1:
        row_spacing = np.zeros((10, rows[0].shape[1], 3), dtype=np.uint8)
        row_spacing[:] = (20, 20, 20)
        
        final = rows[0]
        for row in rows[1:]:
            # Match widths
            if row.shape[1] < final.shape[1]:
                pad_w = final.shape[1] - row.shape[1]
                row = cv2.copyMakeBorder(row, 0, 0, 0, pad_w,
                                        cv2.BORDER_CONSTANT, value=(20, 20, 20))
            final = np.vstack([final, row_spacing, row])
    else:
        final = rows[0]
    
    # Add header
    header_height = 60
    header = np.zeros((header_height, final.shape[1], 3), dtype=np.uint8)
    header[:] = (30, 30, 30)
    cv2.putText(header, "Minimap Processing Pipeline - 9 Steps (Enhanced Contrast)", (20, 35),
               cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 100), 2)
    
    final = np.vstack([header, final])
    
    # Save and display
    output_path = Path("data/screenshots/outputs/minimap_pipeline_visualization.png")
    cv2.imwrite(str(output_path), final)
    print(f"✓ Saved visualization: {output_path}")
    
    # Display
    cv2.imshow("Minimap Processing Steps", final)
    print("\n📊 Minimap Processing Steps:")
    print("=" * 60)
    for i, (_, title) in enumerate(steps[:len(images)], 1):
        print(f"  {i}. {title.replace(chr(10), ' - ')}")
    print("=" * 60)
    print("\nPress any key to close...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    show_minimap_steps()
