"""
Interactive region calibrator for Diablo 2 Resurrected.

Click and drag to draw regions, then save the calibrated coordinates.
"""

import cv2
import argparse
from pathlib import Path
from src.diabot.core.implementations import WindowsScreenCapture


class RegionCalibrator:
    def __init__(self, frame):
        self.frame = frame.copy()
        self.display = frame.copy()
        self.h, self.w = frame.shape[:2]
        self.regions = {}
        self.current_region = None
        self.start_pt = None
        self.drawing = False
        
        print("="*80)
        print("🎯 REGION CALIBRATOR")
        print("="*80)
        print(f"\nFrame size: {self.w}x{self.h}px")
        print("\nInstructions:")
        print("1. Click and drag to draw a region")
        print("2. Press keys to name regions:")
        print("   - 'h' = lifebar_ui (HP)")
        print("   - 'm' = manabar_ui (Mana)")
        print("   - 'z' = zone_ui (Zone)")
        print("   - 'n' = minimap_ui (Minimap)")
        print("   - 'p' = playfield")
        print("   - 'u' = undo last region")
        print("   - 's' = save and show results")
        print("   - 'q' = quit without saving")
        
    def mouse_callback(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.drawing = True
            self.start_pt = (x, y)
            
        elif event == cv2.EVENT_MOUSEMOVE and self.drawing:
            # Update display with preview
            self.display = self.frame.copy()
            cv2.rectangle(self.display, self.start_pt, (x, y), (0, 255, 0), 2)
            cv2.imshow('Calibrator', self.display)
            
        elif event == cv2.EVENT_LBUTTONUP:
            self.drawing = False
            end_pt = (x, y)
            if self.current_region and self.start_pt != end_pt:
                x1, y1 = self.start_pt
                x2, y2 = end_pt
                # Normalize
                x1, x2 = min(x1, x2), max(x1, x2)
                y1, y2 = min(y1, y2), max(y1, y2)
                self.regions[self.current_region] = ((x1, y1), (x2, y2))
                print(f"  ✓ {self.current_region}: ({x1}, {y1}) → ({x2}, {y2})")
                self.current_region = None
                self.redraw()
    
    def redraw(self):
        self.display = self.frame.copy()
        
        colors = {
            'lifebar_ui': (0, 0, 255),      # Red
            'manabar_ui': (255, 0, 0),      # Blue
            'zone_ui': (0, 255, 255),       # Yellow
            'minimap_ui': (255, 255, 0),    # Cyan
            'playfield': (0, 255, 0),       # Green
        }
        
        for region_name, ((x1, y1), (x2, y2)) in self.regions.items():
            color = colors.get(region_name, (255, 255, 255))
            cv2.rectangle(self.display, (x1, y1), (x2, y2), color, 2)
            cv2.putText(self.display, region_name, (x1, y1-5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        
        cv2.imshow('Calibrator', self.display)
    
    def run(self):
        cv2.namedWindow('Calibrator')
        cv2.setMouseCallback('Calibrator', self.mouse_callback)
        
        region_order = ['lifebar_ui', 'manabar_ui', 'zone_ui', 'playfield']
        region_idx = 0
        
        self.redraw()
        
        while True:
            key = cv2.waitKey(0) & 0xFF
            
            if key == ord('h'):
                self.current_region = 'lifebar_ui'
                print("Drawing lifebar_ui (HP) - click and drag...")
                
            elif key == ord('m'):
                self.current_region = 'manabar_ui'
                print("Drawing manabar_ui (Mana) - click and drag...")
                
            elif key == ord('z'):
                self.current_region = 'zone_ui'
                print("Drawing zone_ui (Zone) - click and drag...")
                
            elif key == ord('n'):
                self.current_region = 'minimap_ui'
                print("Drawing minimap_ui (Minimap) - click and drag...")
                
            elif key == ord('p'):
                self.current_region = 'playfield'
                print("Drawing playfield - click and drag...")
                
            elif key == ord('u'):
                if self.regions:
                    removed = self.regions.popitem()
                    print(f"  ↶ Removed {removed[0]}")
                    self.redraw()
                    
            elif key == ord('s'):
                self.save_results()
                break
                
            elif key == ord('q'):
                print("\n✗ Cancelled")
                break
    
    def save_results(self):
        if not self.regions:
            print("No regions defined!")
            return
        
        print("\n" + "="*80)
        print("📊 CALIBRATED REGIONS")
        print("="*80)
        
        output_code = "\n# Updated screen_regions.py values:\n\nUI_REGIONS = {\n"
        
        for region_name in ['lifebar_ui', 'manabar_ui', 'zone_ui', 'minimap_ui']:
            if region_name in self.regions:
                (x1, y1), (x2, y2) = self.regions[region_name]
                x_ratio = x1 / self.w
                y_ratio = y1 / self.h
                w_ratio = (x2 - x1) / self.w
                h_ratio = (y2 - y1) / self.h
                
                print(f"\n{region_name}:")
                print(f"  Pixels: ({x1}, {y1}) → ({x2}, {y2}) = {x2-x1}x{y2-y1}")
                print(f"  Ratios: x={x_ratio:.4f}, y={y_ratio:.4f}, w={w_ratio:.4f}, h={h_ratio:.4f}")
                
                output_code += f"""    '{region_name}': ScreenRegion(
        name='{region_name}',
        x_ratio={x_ratio:.4f},
        y_ratio={y_ratio:.4f},
        w_ratio={w_ratio:.4f},
        h_ratio={h_ratio:.4f},
    ),
"""
        
        output_code += "}\n\nENVIRONMENT_REGIONS = {\n"
        
        if 'playfield' in self.regions:
            (x1, y1), (x2, y2) = self.regions['playfield']
            x_ratio = x1 / self.w
            y_ratio = y1 / self.h
            w_ratio = (x2 - x1) / self.w
            h_ratio = (y2 - y1) / self.h
            
            print(f"\nplayfield:")
            print(f"  Pixels: ({x1}, {y1}) → ({x2}, {y2}) = {x2-x1}x{y2-y1}")
            print(f"  Ratios: x={x_ratio:.4f}, y={y_ratio:.4f}, w={w_ratio:.4f}, h={h_ratio:.4f}")
            
            output_code += f"""    'playfield': ScreenRegion(
        name='playfield',
        x_ratio={x_ratio:.4f},
        y_ratio={y_ratio:.4f},
        w_ratio={w_ratio:.4f},
        h_ratio={h_ratio:.4f},
    ),
"""
        
        output_code += "}\n"
        
        print("\n" + output_code)
        
        # Save to file
        output_path = Path("data/screenshots/outputs/calibration_results.txt")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w') as f:
            f.write(output_code)
        print(f"\n✓ Saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description='Calibrate regions for live capture')
    parser.add_argument(
        '--window-title',
        type=str,
        default='Diablo II: Resurrected',
        help='Window title to capture'
    )
    args = parser.parse_args()
    
    # Capture frame
    try:
        capture = WindowsScreenCapture(window_title=args.window_title)
        frame = capture.get_frame()
    except Exception as e:
        print(f"❌ Failed to capture: {e}")
        return
    
    # Calibrate
    calibrator = RegionCalibrator(frame)
    calibrator.run()


if __name__ == '__main__':
    main()
