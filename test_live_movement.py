"""
Test live character movement by clicking random positions in playfield.

Usage:
    python test_live_movement.py --clicks 5
    python test_live_movement.py --clicks 10 --delay 1.5
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

from src.diabot.core.click_executor import ClickActionExecutor
from src.diabot.navigation.click_navigator import ClickNavigator


def main() -> int:
    parser = argparse.ArgumentParser(description="Test live character movement with random clicks")
    parser.add_argument("--clicks", type=int, default=5, help="Number of random clicks to perform")
    parser.add_argument("--delay", type=float, default=2.0, help="Delay between clicks (seconds)")
    parser.add_argument("--debug", action="store_true", help="Enable debug output")
    args = parser.parse_args()

    print("\n" + "=" * 60)
    print("LIVE MOVEMENT TEST")
    print("=" * 60)
    print(f"- Clicks: {args.clicks}")
    print(f"- Delay: {args.delay}s between clicks")
    print("\nMake sure Diablo 2 window is active!")
    print("Starting in 3 seconds...\n")
    
    time.sleep(3)

    try:
        executor = ClickActionExecutor(debug=args.debug)
        
        for i in range(args.clicks):
            print(f"\n[{i+1}/{args.clicks}] Executing random exploration click...")
            success = executor.execute_action("explore_random")
            
            if success:
                print(f"  ✓ Click #{i+1} executed")
            else:
                print(f"  ✗ Click #{i+1} failed")
            
            if i < args.clicks - 1:
                print(f"  Waiting {args.delay}s...")
                time.sleep(args.delay)
        
        print("\n" + "=" * 60)
        print("MOVEMENT TEST COMPLETE")
        print("=" * 60)
        return 0

    except KeyboardInterrupt:
        print("\n\nTest interrupted by user")
        return 1
    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
