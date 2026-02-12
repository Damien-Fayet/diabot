#!/usr/bin/env python3
"""
Main entry point for Diabot - Diablo 2 Bot

Usage:
    python main.py                    # Live mode with debug
    python main.py --no-debug         # Live mode without debug
    python main.py --screenshot PATH  # Test with screenshot
    python main.py --max-frames 100   # Limit execution
"""

import argparse
import sys
import time
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from diabot.main import DiabotRunner


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Diabot - Diablo 2 Bot")
    
    parser.add_argument(
        "--screenshot",
        type=str,
        default=None,
        help="Path to screenshot file for testing (instead of live capture)"
    )
    
    parser.add_argument(
        "--no-debug",
        action="store_true",
        help="Disable debug output"
    )
    
    parser.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="Maximum frames to process (default: unlimited)"
    )
    
    parser.add_argument(
        "--state",
        type=str,
        default="bot_state.json",
        help="Path to bot state file"
    )
    
    return parser.parse_args()


def main():
    """Main entry point."""
    args = parse_args()
    
    print("\n" + "="*70)
    print("🤖 DIABOT - DIABLO 2 RESURRECTED BOT")
    print("="*70)
    
    # Display mode
    if args.screenshot:
        print(f"\n📸 Mode: Screenshot test")
        print(f"   File: {args.screenshot}")
    else:
        print(f"\n🎮 Mode: Live gameplay")
        print(f"   Make sure Diablo 2 Resurrected is running")
    
    print(f"\n⚙️  Settings:")
    print(f"   Debug: {not args.no_debug}")
    print(f"   Max frames: {args.max_frames if args.max_frames else 'unlimited'}")
    print(f"   State file: {args.state}")
    
    print("\n🗺️  Static Map Navigation:")
    print(f"   Rogue Encampment: AUTO-LOADED")
    print(f"   Navigation: Exit-seeking with pre-annotated POIs")
    
    print("\n⌨️  Controls:")
    print(f"   Ctrl+C: Stop bot")
    
    print("\n" + "="*70 + "\n")
    
    try:
        # Initialize bot
        print("[INIT] Creating bot instance...")
        bot = DiabotRunner(
            image_source_path=args.screenshot,
            executor=None,  # Auto-detect Windows executor
            state_file=args.state,
            debug=not args.no_debug
        )
        
        print("[INIT] ✓ Bot initialized successfully\n")
        print("="*70)
        print("🚀 STARTING MAIN LOOP")
        print("="*70 + "\n")
        
        frame_count = 0
        start_time = time.time()
        
        # Main loop
        while True:
            try:
                # Process one frame
                bot.step()
                frame_count += 1
                
                # Check frame limit
                if args.max_frames and frame_count >= args.max_frames:
                    print(f"\n✓ Reached frame limit: {args.max_frames}")
                    break
                
                # Status update every 50 frames
                if frame_count % 50 == 0:
                    elapsed = time.time() - start_time
                    fps = frame_count / elapsed if elapsed > 0 else 0
                    
                    print(f"\n[STATUS] Frames: {frame_count}, FPS: {fps:.1f}, Mode: {bot.navigation_mode}")
                    
                    if bot.static_localizer and bot.current_target_poi:
                        print(f"         Target: {bot.current_target_poi['name']}")
                
                # Small delay in live mode
                if not args.screenshot:
                    time.sleep(0.1)
            
            except KeyboardInterrupt:
                print("\n\n⚠️  Interrupted by user")
                break
            
            except Exception as e:
                print(f"\n❌ Error in main loop: {e}")
                if not args.no_debug:
                    import traceback
                    traceback.print_exc()
                break
        
        # Summary
        elapsed = time.time() - start_time
        print("\n" + "="*70)
        print("✅ BOT STOPPED")
        print("="*70)
        print(f"\n📊 Session Statistics:")
        print(f"   Total frames: {frame_count}")
        print(f"   Duration: {elapsed:.1f}s")
        print(f"   Average FPS: {frame_count/elapsed:.1f}" if elapsed > 0 else "   Average FPS: N/A")
        print(f"   Final mode: {bot.navigation_mode}")
        
        if bot.map_accumulator:
            explored = len(bot.map_accumulator.cells)
            pois = len(bot.map_accumulator.pois)
            print(f"\n🗺️  Map Progress:")
            print(f"   Cells explored: {explored}")
            print(f"   POIs found: {pois}")
        
        print("\n" + "="*70 + "\n")
    
    except Exception as e:
        print(f"\n❌ Fatal error: {e}")
        if not args.no_debug:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
