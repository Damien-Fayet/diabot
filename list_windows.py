"""
List all open windows to find the correct Diablo 2 window title.
"""

import sys

if sys.platform == 'win32':
    try:
        import win32gui
        
        def callback(hwnd, windows):
            if win32gui.IsWindowVisible(hwnd):
                title = win32gui.GetWindowText(hwnd)
                if title:
                    windows.append((hwnd, title))
            return True
        
        windows = []
        win32gui.EnumWindows(callback, windows)
        
        print("Open windows:")
        for hwnd, title in sorted(windows, key=lambda x: x[1].lower()):
            print(f"  [{hwnd}] {title}")
        
        # Look for Diablo
        print("\nDiablo-related windows:")
        diablo_windows = [(h, t) for h, t in windows if 'diablo' in t.lower() or 'd2' in t.lower()]
        if diablo_windows:
            for hwnd, title in diablo_windows:
                print(f"  [{hwnd}] {title}")
        else:
            print("  None found")
            
    except ImportError:
        print("win32gui not available")
else:
    print("Windows only")
