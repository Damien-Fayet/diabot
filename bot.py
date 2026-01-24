#!/usr/bin/env python3
"""
Quick launcher for Diabot main loop.

Usage:
    python bot.py --image path/to/screenshot.jpg
    python bot.py                          # Interactive mode (dev mode)
    python bot.py --max-frames 300 --fps 10
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

from diabot.main import main

if __name__ == "__main__":
    main()
