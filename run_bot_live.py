#!/usr/bin/env python3
"""Launch the Diablo 2 bot in live mode.

Usage:
    python run_bot_live.py --model runs/detect/runs/train/diablo-yolo3/weights/best.pt --fps 5
"""

import argparse
from pathlib import Path

from src.diabot.bot import DiabloBot


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Diablo 2 bot in live mode")
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Path to YOLO model (.pt file)",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=5,
        help="Target FPS",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug logging",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="Max frames to run (None = infinite)",
    )
    parser.add_argument(
        "--progression",
        type=str,
        default="data/game_progression.json",
        help="Path to game_progression.json",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    
    # Create and run bot
    bot = DiabloBot(
        yolo_model_path=args.model,
        progression_file=args.progression,
        debug=args.debug,
    )
    
    bot.run_loop(max_frames=args.max_frames, fps_limit=args.fps)


if __name__ == "__main__":
    main()
