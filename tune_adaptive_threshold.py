#!/usr/bin/env python3
"""
Interactive CLI tool to tune apply_adaptive_threshold() parameters.

Loads 08_gabor_combined.png and allows adjusting parameters via command line.
Shows side-by-side comparison of input and output after each change.

Usage:
    python tune_adaptive_threshold.py
    
Commands:
    blur <on/off>              Enable/disable blur
    blur_k <size>              Set blur kernel size (odd)
    threshold <on/off>         Enable/disable adaptive threshold
    threshold_b <size>         Set threshold block size (odd)
    threshold_c <value>        Set threshold C value
    dilate <on/off>            Enable/disable dilation
    dilate_k <size>            Set dilation kernel size (odd)
    dilate_i <iterations>      Set dilation iterations
    close <on/off>             Enable/disable closing
    close_k <size>             Set closing kernel size (odd)
    close_i <iterations>       Set closing iterations
    save                       Save current configuration
    show                       Show current result (without input)
    help                       Show this help
    quit/exit                  Exit
"""
import sys
from pathlib import Path
import cv2
import numpy as np
from datetime import datetime

# Setup path
sys.path.insert(0, str(Path(__file__).parent / "src"))


class AdaptiveThresholdTuner:
    def __init__(self, image_path):
        """Initialize with input image."""
        self.original = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
        if self.original is None:
            raise FileNotFoundError(f"Image not found: {image_path}")
        
        # Resize if too large (for display)
        h, w = self.original.shape
        if h > 600 or w > 800:
            scale = min(600 / h, 800 / w)
            new_h, new_w = int(h * scale), int(w * scale)
            self.original = cv2.resize(self.original, (new_w, new_h))
        
        print(f"[+] Loaded image: {self.original.shape}")
        
        # Parameters
        self.params = {
            'blur_enabled': True,
            'blur_kernel': 5,
            'threshold_enabled': True,
            'threshold_blocksize': 15,
            'threshold_c': 8,
            'dilate_enabled': True,
            'dilate_kernel': 5,
            'dilate_iterations': 1,
            'close_enabled': True,
            'close_kernel': 9,
            'close_iterations': 1,
        }
    
    def ensure_odd(self, val):
        """Ensure value is odd."""
        return val if val % 2 == 1 else val + 1
    
    def process_image(self):
        """Process image with current parameters and return result."""
        result = self.original.copy()
        
        # STEP 1: Gaussian blur
        if self.params['blur_enabled']:
            kernel = self.ensure_odd(self.params['blur_kernel'])
            result = cv2.GaussianBlur(result, (kernel, kernel), 0)
        
        # STEP 2: Adaptive threshold
        if self.params['threshold_enabled']:
            blurred = result if self.params['blur_enabled'] else self.original
            blocksize = self.ensure_odd(self.params['threshold_blocksize'])
            c = self.params['threshold_c']
            result = cv2.adaptiveThreshold(
                blurred,
                255,
                cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY_INV,
                blocksize,
                c
            )
        
        # STEP 3: Dilation
        if self.params['dilate_enabled'] and self.params['threshold_enabled']:
            kernel_size = self.ensure_odd(self.params['dilate_kernel'])
            iterations = self.params['dilate_iterations']
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
            result = cv2.dilate(result, kernel, iterations=iterations)
        
        # STEP 4: Morphological closing
        if self.params['close_enabled'] and self.params['threshold_enabled']:
            kernel_size = self.ensure_odd(self.params['close_kernel'])
            iterations = self.params['close_iterations']
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
            result = cv2.morphologyEx(result, cv2.MORPH_CLOSE, kernel, iterations=iterations)
        
        return result
    
    def display_comparison(self):
        """Show side-by-side comparison of input and output."""
        result = self.process_image()
        
        # Create side-by-side comparison
        h, w = self.original.shape
        comparison = np.zeros((h, w * 2 + 10), dtype=np.uint8)
        comparison[:, :w] = self.original
        comparison[:, w+10:] = result
        
        # Add title text
        cv2.putText(comparison, 'INPUT (08_gabor_combined)', (10, 25),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, 200, 2)
        cv2.putText(comparison, 'OUTPUT (processed)', (w + 20, 25),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, 200, 2)
        
        # Add parameter info
        info_lines = [
            f"Blur: {'ON (k={})'.format(self.ensure_odd(self.params['blur_kernel'])) if self.params['blur_enabled'] else 'OFF'}",
            f"Threshold: {'ON (b={}, c={})'.format(self.ensure_odd(self.params['threshold_blocksize']), self.params['threshold_c']) if self.params['threshold_enabled'] else 'OFF'}",
            f"Dilate: {'ON (k={}, i={})'.format(self.ensure_odd(self.params['dilate_kernel']), self.params['dilate_iterations']) if self.params['dilate_enabled'] else 'OFF'}",
            f"Close: {'ON (k={}, i={})'.format(self.ensure_odd(self.params['close_kernel']), self.params['close_iterations']) if self.params['close_enabled'] else 'OFF'}",
        ]
        
        for i, line in enumerate(info_lines):
            cv2.putText(comparison, line, (10, h - 60 + i * 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, 200, 1)
        
        cv2.imshow('Adaptive Threshold Tuner', comparison)
        cv2.waitKey(1)
    
    def print_status(self):
        """Print current parameters."""
        print("\n" + "="*70)
        print("CURRENT PARAMETERS:")
        print("="*70)
        print(f"Blur:")
        print(f"  enabled: {self.params['blur_enabled']}")
        print(f"  kernel: {self.ensure_odd(self.params['blur_kernel'])}")
        print(f"\nThreshold:")
        print(f"  enabled: {self.params['threshold_enabled']}")
        print(f"  blocksize: {self.ensure_odd(self.params['threshold_blocksize'])}")
        print(f"  C: {self.params['threshold_c']}")
        print(f"\nDilation:")
        print(f"  enabled: {self.params['dilate_enabled']}")
        print(f"  kernel: {self.ensure_odd(self.params['dilate_kernel'])}")
        print(f"  iterations: {self.params['dilate_iterations']}")
        print(f"\nClosing:")
        print(f"  enabled: {self.params['close_enabled']}")
        print(f"  kernel: {self.ensure_odd(self.params['close_kernel'])}")
        print(f"  iterations: {self.params['close_iterations']}")
        print("="*70)
    
    def save_current_config(self):
        """Save current configuration to file."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        config_file = Path(__file__).parent / f"adaptive_threshold_config_{timestamp}.txt"
        
        with open(config_file, 'w') as f:
            f.write("# Adaptive Threshold Configuration\n\n")
            f.write(f"# Generated: {timestamp}\n\n")
            
            f.write("# Gaussian Blur\n")
            f.write(f"blur_enabled = {self.params['blur_enabled']}\n")
            f.write(f"blur_kernel = {self.ensure_odd(self.params['blur_kernel'])}\n\n")
            
            f.write("# Adaptive Threshold\n")
            f.write(f"threshold_enabled = {self.params['threshold_enabled']}\n")
            f.write(f"threshold_blocksize = {self.ensure_odd(self.params['threshold_blocksize'])}\n")
            f.write(f"threshold_c = {self.params['threshold_c']}\n\n")
            
            f.write("# Dilation\n")
            f.write(f"dilate_enabled = {self.params['dilate_enabled']}\n")
            f.write(f"dilate_kernel = {self.ensure_odd(self.params['dilate_kernel'])}\n")
            f.write(f"dilate_iterations = {self.params['dilate_iterations']}\n\n")
            
            f.write("# Morphological Closing\n")
            f.write(f"close_enabled = {self.params['close_enabled']}\n")
            f.write(f"close_kernel = {self.ensure_odd(self.params['close_kernel'])}\n")
            f.write(f"close_iterations = {self.params['close_iterations']}\n")
        
        print(f"\n[+] Config saved to: {config_file}")
        return config_file
    
    def run(self):
        """Run the interactive CLI tuner."""
        print("\n" + "="*70)
        print("ADAPTIVE THRESHOLD TUNER - CLI MODE")
        print("="*70)
        print(__doc__)
        print("="*70 + "\n")
        
        # Show initial status
        self.print_status()
        
        # Initial preview
        self.display_comparison()
        
        # Main loop
        while True:
            try:
                cmd = input("\n> ").strip().lower()
                
                if not cmd:
                    continue
                
                parts = cmd.split()
                command = parts[0]
                arg = parts[1] if len(parts) > 1 else None
                
                # Commands
                if command in ['quit', 'exit', 'q']:
                    print("\n[*] Quitting...")
                    cv2.destroyAllWindows()
                    break
                
                elif command == 'help':
                    print(__doc__)
                
                elif command == 'status':
                    self.print_status()
                
                elif command == 'save':
                    self.save_current_config()
                
                elif command == 'show':
                    self.display_comparison()
                
                # Blur commands
                elif command == 'blur':
                    if arg in ['on', '1', 'true']:
                        self.params['blur_enabled'] = True
                        print("[+] Blur enabled")
                    elif arg in ['off', '0', 'false']:
                        self.params['blur_enabled'] = False
                        print("[+] Blur disabled")
                    else:
                        print("[!] Usage: blur <on/off>")
                        continue
                    self.display_comparison()
                
                elif command == 'blur_k':
                    try:
                        val = int(arg)
                        val = self.ensure_odd(val)
                        if val < 3:
                            val = 3
                        self.params['blur_kernel'] = val
                        print(f"[+] Blur kernel set to {val}")
                        self.display_comparison()
                    except (ValueError, TypeError):
                        print("[!] Usage: blur_k <size_odd>")
                
                # Threshold commands
                elif command == 'threshold':
                    if arg in ['on', '1', 'true']:
                        self.params['threshold_enabled'] = True
                        print("[+] Threshold enabled")
                    elif arg in ['off', '0', 'false']:
                        self.params['threshold_enabled'] = False
                        print("[+] Threshold disabled")
                    else:
                        print("[!] Usage: threshold <on/off>")
                        continue
                    self.display_comparison()
                
                elif command == 'threshold_b':
                    try:
                        val = int(arg)
                        val = self.ensure_odd(val)
                        if val < 3:
                            val = 3
                        self.params['threshold_blocksize'] = val
                        print(f"[+] Threshold blocksize set to {val}")
                        self.display_comparison()
                    except (ValueError, TypeError):
                        print("[!] Usage: threshold_b <size_odd>")
                
                elif command == 'threshold_c':
                    try:
                        val = int(arg)
                        self.params['threshold_c'] = val
                        print(f"[+] Threshold C set to {val}")
                        self.display_comparison()
                    except (ValueError, TypeError):
                        print("[!] Usage: threshold_c <value>")
                
                # Dilation commands
                elif command == 'dilate':
                    if arg in ['on', '1', 'true']:
                        self.params['dilate_enabled'] = True
                        print("[+] Dilation enabled")
                    elif arg in ['off', '0', 'false']:
                        self.params['dilate_enabled'] = False
                        print("[+] Dilation disabled")
                    else:
                        print("[!] Usage: dilate <on/off>")
                        continue
                    self.display_comparison()
                
                elif command == 'dilate_k':
                    try:
                        val = int(arg)
                        val = self.ensure_odd(val)
                        if val < 3:
                            val = 3
                        self.params['dilate_kernel'] = val
                        print(f"[+] Dilation kernel set to {val}")
                        self.display_comparison()
                    except (ValueError, TypeError):
                        print("[!] Usage: dilate_k <size_odd>")
                
                elif command == 'dilate_i':
                    try:
                        val = int(arg)
                        if val < 1:
                            val = 1
                        self.params['dilate_iterations'] = val
                        print(f"[+] Dilation iterations set to {val}")
                        self.display_comparison()
                    except (ValueError, TypeError):
                        print("[!] Usage: dilate_i <iterations>")
                
                # Closing commands
                elif command == 'close':
                    if arg in ['on', '1', 'true']:
                        self.params['close_enabled'] = True
                        print("[+] Closing enabled")
                    elif arg in ['off', '0', 'false']:
                        self.params['close_enabled'] = False
                        print("[+] Closing disabled")
                    else:
                        print("[!] Usage: close <on/off>")
                        continue
                    self.display_comparison()
                
                elif command == 'close_k':
                    try:
                        val = int(arg)
                        val = self.ensure_odd(val)
                        if val < 3:
                            val = 3
                        self.params['close_kernel'] = val
                        print(f"[+] Closing kernel set to {val}")
                        self.display_comparison()
                    except (ValueError, TypeError):
                        print("[!] Usage: close_k <size_odd>")
                
                elif command == 'close_i':
                    try:
                        val = int(arg)
                        if val < 1:
                            val = 1
                        self.params['close_iterations'] = val
                        print(f"[+] Closing iterations set to {val}")
                        self.display_comparison()
                    except (ValueError, TypeError):
                        print("[!] Usage: close_i <iterations>")
                
                else:
                    print(f"[!] Unknown command: {command}")
                    print("    Type 'help' for available commands")
            
            except KeyboardInterrupt:
                print("\n[*] Interrupted")
                cv2.destroyAllWindows()
                break
            except Exception as e:
                print(f"[!] Error: {e}")


def find_gabor_image():
    """Find 08_gabor_combined.png in inputs or outputs."""
    # Try outputs first (latest)
    outputs_dir = Path(__file__).parent / "data" / "screenshots" / "outputs"
    if outputs_dir.exists():
        images = sorted(outputs_dir.glob("**/08_gabor_combined_*.png"), reverse=True)
        if images:
            return images[0]
    
    # Then try inputs
    inputs_dir = Path(__file__).parent / "data" / "screenshots" / "inputs"
    if inputs_dir.exists():
        images = sorted(inputs_dir.glob("*08_gabor_combined*.png"), reverse=True)
        if images:
            return images[0]
    
    return None


def main():
    """Main entry point."""
    # Find image
    image_path = find_gabor_image()
    
    if image_path is None:
        print("[!] Could not find 08_gabor_combined_*.png")
        print("    Make sure test_static_localization_v3.py --static has been run")
        print("    Expected in: data/screenshots/outputs/localization_test_v3/")
        return
    
    print(f"[+] Found image: {image_path}")
    
    # Create and run tuner
    try:
        tuner = AdaptiveThresholdTuner(image_path)
        tuner.run()
    except Exception as e:
        print(f"[!] Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
