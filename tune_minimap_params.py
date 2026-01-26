"""
Interactive minimap parameter tuning tool with HUD crop and HoughLinesP.
"""
import cv2
import numpy as np
from pathlib import Path


class MinimapTuner:
    def __init__(self, minimap_path: str):
        self.original = cv2.imread(minimap_path)
        if self.original is None:
            raise ValueError(f"Could not load image: {minimap_path}")
        
        self.gray = cv2.cvtColor(self.original, cv2.COLOR_BGR2GRAY)
        self.window_name = "Minimap Parameter Tuning"
        
        self.params = {
            'crop_bottom_pct': 15,
            'gamma_x10': 30,  # Default gamma=3.0
            'tophat_kernel': 7,
            'clahe_clip': 30,
            'clahe_grid': 8,
            'threshold': 55,
            'hough_threshold': 50,
            'hough_min_length': 20,
            'hough_max_gap': 10,
            'morph_open': 2,
            'morph_close': 2,
        }
        
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.window_name, 1600, 900)
        self._create_trackbars()
        cv2.waitKey(100)
        self.update()
    
    def _create_trackbars(self):
        cv2.createTrackbar('Crop Bottom % (0-30)', self.window_name, self.params['crop_bottom_pct'], 30, self.on_change)
        cv2.createTrackbar('Gamma x10 (1-30)', self.window_name, self.params['gamma_x10'], 30, self.on_change)
        cv2.createTrackbar('TopHat Kernel (1-15)', self.window_name, self.params['tophat_kernel'], 15, self.on_change)
        cv2.createTrackbar('CLAHE Clip x10 (0-100)', self.window_name, self.params['clahe_clip'], 100, self.on_change)
        cv2.createTrackbar('CLAHE Grid (4/8/16)', self.window_name, 1, 2, self.on_change)
        cv2.createTrackbar('Threshold (0-255)', self.window_name, self.params['threshold'], 255, self.on_change)
        cv2.createTrackbar('Hough Threshold (10-100)', self.window_name, self.params['hough_threshold'], 100, self.on_change)
        cv2.createTrackbar('Hough Min Length (5-50)', self.window_name, self.params['hough_min_length'], 50, self.on_change)
        cv2.createTrackbar('Hough Max Gap (1-20)', self.window_name, self.params['hough_max_gap'], 20, self.on_change)
        cv2.createTrackbar('Morph Open (0-10)', self.window_name, self.params['morph_open'], 10, self.on_change)
        cv2.createTrackbar('Morph Close (0-10)', self.window_name, self.params['morph_close'], 10, self.on_change)
        cv2.createTrackbar('Enable Crop (0/1)', self.window_name, 1, 1, self.on_change)
        cv2.createTrackbar('Enable TopHat (0/1)', self.window_name, 1, 1, self.on_change)
        cv2.createTrackbar('Enable Gamma (0/1)', self.window_name, 1, 1, self.on_change)
        cv2.createTrackbar('Enable CLAHE (0/1)', self.window_name, 1, 1, self.on_change)
        cv2.createTrackbar('Enable Filter (0/1)', self.window_name, 1, 1, self.on_change)
        cv2.createTrackbar('Enable Hough (0/1)', self.window_name, 0, 1, self.on_change)
    
    def on_change(self, value):
        pass
    
    def update(self):
        try:
            crop_bottom_pct = cv2.getTrackbarPos('Crop Bottom % (0-30)', self.window_name)
            gamma_x10 = cv2.getTrackbarPos('Gamma x10 (1-30)', self.window_name)
            gamma = max(0.1, gamma_x10 / 10.0)
            tophat_kernel = cv2.getTrackbarPos('TopHat Kernel (1-15)', self.window_name)
            tophat_kernel = tophat_kernel if tophat_kernel % 2 == 1 else tophat_kernel + 1
            clahe_clip_x10 = cv2.getTrackbarPos('CLAHE Clip x10 (0-100)', self.window_name)
            clahe_clip = clahe_clip_x10 / 10.0
            clahe_grid_idx = cv2.getTrackbarPos('CLAHE Grid (4/8/16)', self.window_name)
            clahe_grid = [4, 8, 16][clahe_grid_idx]
            threshold = cv2.getTrackbarPos('Threshold (0-255)', self.window_name)
            hough_threshold = cv2.getTrackbarPos('Hough Threshold (10-100)', self.window_name)
            hough_min_length = cv2.getTrackbarPos('Hough Min Length (5-50)', self.window_name)
            hough_max_gap = cv2.getTrackbarPos('Hough Max Gap (1-20)', self.window_name)
            morph_open = cv2.getTrackbarPos('Morph Open (0-10)', self.window_name)
            morph_close = cv2.getTrackbarPos('Morph Close (0-10)', self.window_name)
            enable_crop = cv2.getTrackbarPos('Enable Crop (0/1)', self.window_name)
            enable_tophat = cv2.getTrackbarPos('Enable TopHat (0/1)', self.window_name)
            enable_gamma = cv2.getTrackbarPos('Enable Gamma (0/1)', self.window_name)
            enable_clahe = cv2.getTrackbarPos('Enable CLAHE (0/1)', self.window_name)
            enable_filter = cv2.getTrackbarPos('Enable Filter (0/1)', self.window_name)
            enable_hough = cv2.getTrackbarPos('Enable Hough (0/1)', self.window_name)
            
            processed = self.gray.copy()
            
            # Step 0: Crop bottom (HUD removal)
            if enable_crop and crop_bottom_pct > 0:
                h = processed.shape[0]
                crop_pixels = int(h * crop_bottom_pct / 100)
                processed = processed[:-crop_pixels, :] if crop_pixels < h else processed
            step_cropped = processed.copy()
            
            # Step 1: Top Hat
            if enable_tophat and tophat_kernel > 0:
                kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (tophat_kernel, tophat_kernel))
                tophat = cv2.morphologyEx(processed, cv2.MORPH_TOPHAT, kernel)
                processed = cv2.add(processed, tophat)
            step_tophat = processed.copy()
            
            # Step 2: Gamma
            if enable_gamma:
                normalized = processed.astype(np.float32) / 255.0
                processed = np.power(normalized, gamma)
                processed = (processed * 255).astype(np.uint8)
            step_gamma = processed.copy()
            
            # Step 3: CLAHE
            if enable_clahe and clahe_clip > 0:
                clahe = cv2.createCLAHE(clipLimit=clahe_clip, tileGridSize=(clahe_grid, clahe_grid))
                processed = clahe.apply(processed)
            step_clahe = processed.copy()
            
            # Step 4: Bilateral filter
            if enable_filter:
                processed = cv2.bilateralFilter(processed, 5, 50, 50)
            step_filtered = processed.copy()
            
            # Step 5: Binary threshold
            _, binary = cv2.threshold(processed, threshold, 255, cv2.THRESH_BINARY)
            step_binary = binary.copy()
            
            # Step 6: Morphology open
            if morph_open > 0:
                kernel_open = np.ones((morph_open, morph_open), np.uint8)
                binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel_open)
            step_morph_open = binary.copy()
            
            # Step 7: Morphology close
            if morph_close > 0:
                kernel_close = np.ones((morph_close, morph_close), np.uint8)
                binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel_close)
            step_final = binary.copy()
            
            # Step 8: HoughLinesP (visualization)
            step_hough = None
            line_count = 0
            if enable_hough:
                lines = cv2.HoughLinesP(binary, 1, np.pi/180,
                                       threshold=max(10, hough_threshold),
                                       minLineLength=max(5, hough_min_length),
                                       maxLineGap=max(1, hough_max_gap))
                step_hough = cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)
                if lines is not None:
                    line_count = len(lines)
                    for line in lines:
                        x1, y1, x2, y2 = line[0]
                        cv2.line(step_hough, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(step_hough, f"{line_count} lines", (10, 30),
                               cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
            
            # Stats
            total_pixels = binary.size
            wall_pixels = np.count_nonzero(binary)
            free_pixels = total_pixels - wall_pixels
            wall_pct = (wall_pixels / total_pixels) * 100
            free_pct = (free_pixels / total_pixels) * 100
            
            # Display grid
            steps = [
                (step_cropped, f"0. Cropped (-{crop_bottom_pct}%)"),
                (step_tophat, f"1. TopHat k={tophat_kernel}"),
                (step_gamma, f"2. Gamma={gamma:.1f}"),
                (step_clahe, f"3. CLAHE c={clahe_clip:.1f} g={clahe_grid}"),
                (step_filtered, "4. Bilateral Filter"),
                (step_binary, f"5. Threshold={threshold}"),
                (step_morph_open, f"6. Open k={morph_open}"),
                (step_final, f"7. Close k={morph_close}"),
                (step_final, f"8. Final (W:{wall_pct:.0f}% F:{free_pct:.0f}%)"),
            ]
            
            if enable_hough and step_hough is not None:
                steps.append((step_hough, f"9. Hough ({line_count} lines)"))
            
            while len(steps) % 3 != 0:
                steps.append((step_final, ""))
            
            # Render
            display_size = (400, 300)
            rows = []
            for i in range(0, len(steps), 3):
                row_images = []
                for img, title in steps[i:i+3]:
                    resized = cv2.resize(img, display_size)
                    if len(resized.shape) == 2:
                        resized = cv2.cvtColor(resized, cv2.COLOR_GRAY2BGR)
                    cv2.putText(resized, title, (10, 30),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                    row_images.append(resized)
                rows.append(np.hstack(row_images))
            
            display = np.vstack(rows)
            
            # Info panel
            info_height = 140
            info_panel = np.zeros((info_height, display.shape[1], 3), dtype=np.uint8)
            y = 25
            cv2.putText(info_panel, "Parameters:", (10, y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            y += 25
            cv2.putText(info_panel,
                       f"  Crop:{crop_bottom_pct}% | TopHat:{tophat_kernel} | Gamma:{gamma:.2f} | CLAHE:{clahe_clip:.1f}(g{clahe_grid})",
                       (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
            y += 20
            cv2.putText(info_panel,
                       f"  Threshold:{threshold} | Morph:Open={morph_open},Close={morph_close}",
                       (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
            y += 20
            if enable_hough:
                cv2.putText(info_panel,
                           f"  Hough: t={hough_threshold}, minLen={hough_min_length}, maxGap={hough_max_gap} → {line_count} lines",
                           (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 255, 100), 1)
                y += 20
            cv2.putText(info_panel,
                       f"Results: Walls={wall_pct:.1f}% ({wall_pixels}px), Free={free_pct:.1f}% ({free_pixels}px)",
                       (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)
            cv2.putText(info_panel, "Press 'S'=save | 'R'=reset | 'Q'=quit",
                       (display.shape[1] - 400, info_height - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 100, 255), 1)
            
            display = np.vstack([display, info_panel])
            cv2.imshow(self.window_name, display)
        except Exception as e:
            if "trackbar" not in str(e).lower():
                print(f"Error: {e}")
    
    def save_params(self):
        crop_bottom_pct = cv2.getTrackbarPos('Crop Bottom % (0-30)', self.window_name)
        gamma_x10 = cv2.getTrackbarPos('Gamma x10 (1-30)', self.window_name)
        tophat_kernel = cv2.getTrackbarPos('TopHat Kernel (1-15)', self.window_name)
        clahe_clip_x10 = cv2.getTrackbarPos('CLAHE Clip x10 (0-100)', self.window_name)
        clahe_grid_idx = cv2.getTrackbarPos('CLAHE Grid (4/8/16)', self.window_name)
        threshold = cv2.getTrackbarPos('Threshold (0-255)', self.window_name)
        hough_threshold = cv2.getTrackbarPos('Hough Threshold (10-100)', self.window_name)
        hough_min_length = cv2.getTrackbarPos('Hough Min Length (5-50)', self.window_name)
        hough_max_gap = cv2.getTrackbarPos('Hough Max Gap (1-20)', self.window_name)
        morph_open = cv2.getTrackbarPos('Morph Open (0-10)', self.window_name)
        morph_close = cv2.getTrackbarPos('Morph Close (0-10)', self.window_name)
        
        gamma = max(0.1, gamma_x10 / 10.0)
        tophat_kernel = tophat_kernel if tophat_kernel % 2 == 1 else tophat_kernel + 1
        clahe_clip = clahe_clip_x10 / 10.0
        clahe_grid = [4, 8, 16][clahe_grid_idx]
        
        with open("minimap_tuned_params.txt", 'w') as f:
            f.write("# Minimap Processing Parameters\n")
            f.write(f"crop_bottom_percent = {crop_bottom_pct}\n")
            f.write(f"tophat_kernel_size = {tophat_kernel}\n")
            f.write(f"gamma = {gamma:.2f}\n")
            f.write(f"clahe_clip_limit = {clahe_clip:.1f}\n")
            f.write(f"clahe_tile_grid_size = {clahe_grid}\n")
            f.write(f"threshold = {threshold}\n")
            f.write(f"hough_threshold = {hough_threshold}\n")
            f.write(f"hough_min_line_length = {hough_min_length}\n")
            f.write(f"hough_max_line_gap = {hough_max_gap}\n")
            f.write(f"morph_open_kernel = {morph_open}\n")
            f.write(f"morph_close_kernel = {morph_close}\n")
        
        print(f"\n✓ Saved! Crop:{crop_bottom_pct}% TopHat:{tophat_kernel} Gamma:{gamma:.2f} Threshold:{threshold}")
    
    def reset_params(self):
        cv2.setTrackbarPos('Crop Bottom % (0-30)', self.window_name, 15)
        cv2.setTrackbarPos('Gamma x10 (1-30)', self.window_name, 30)
        cv2.setTrackbarPos('TopHat Kernel (1-15)', self.window_name, 7)
        cv2.setTrackbarPos('CLAHE Clip x10 (0-100)', self.window_name, 30)
        cv2.setTrackbarPos('CLAHE Grid (4/8/16)', self.window_name, 1)
        cv2.setTrackbarPos('Threshold (0-255)', self.window_name, 55)
        print("✓ Reset (Gamma=3.0, Crop=15%)")
    
    def run(self):
        print("\n" + "="*70)
        print("MINIMAP TUNING - HUD Crop + Top Hat + Gamma(3.0) + HoughLinesP")
        print("="*70)
        print("Controls: S=save | R=reset | Q=quit")
        print("Target: ~50% walls, ~50% free")
        print("="*70 + "\n")
        
        while True:
            self.update()
            key = cv2.waitKey(50) & 0xFF
            if key == ord('q') or key == 27:
                break
            elif key == ord('s'):
                self.save_params()
            elif key == ord('r'):
                self.reset_params()
        cv2.destroyAllWindows()


def main():
    path = "data/screenshots/outputs/diagnostic/minimap_steps/step1_extracted.png"
    if not Path(path).exists():
        print(f"❌ Not found: {path}")
        print("Run bot with --debug first")
        return
    tuner = MinimapTuner(path)
    tuner.run()


if __name__ == "__main__":
    main()
