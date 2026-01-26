"""
Run template matching on a live capture and visualize detected templates (including quests).

Usage examples:
    python show_template_matches.py
    python show_template_matches.py --zone "ROGUE ENCAMPMENT" --threshold 0.72 --show
    python show_template_matches.py --categories npcs,waypoints,quests

Outputs:
    - Saves overlay with bounding boxes to data/screenshots/outputs/diagnostic/template_matches.png
"""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path
from typing import List

import cv2
import matplotlib.pyplot as plt

from src.diabot.vision.template_detector import TemplateDetector


CATEGORY_COLORS = {
    "npcs": (0, 255, 0),       # green
    "waypoints": (255, 0, 0),  # blue
    "quests": (255, 0, 255),   # magenta
    "unknown": (0, 255, 255),  # yellow
}


def draw_matches(frame, matches):
    """Draw bounding boxes and labels for matched templates."""
    overlay = frame.copy()
    for match in matches:
        x1, y1, x2, y2 = match.bbox
        color = CATEGORY_COLORS.get(match.category, CATEGORY_COLORS["unknown"])
        cv2.rectangle(overlay, (x1, y1), (x2, y2), color, 2)
        label = f"{match.template_name} ({match.confidence:.2f})"
        cv2.putText(overlay, label, (x1, max(15, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)
    return overlay


def parse_categories(raw: str) -> List[str]:
    if not raw:
        return []
    return [c.strip().lower() for c in raw.split(",") if c.strip()]


def main() -> int:
    parser = argparse.ArgumentParser(description="Show template matches on a live capture")
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data/screenshots/outputs/live_capture/live_capture_raw.jpg"),
        help="Path to input screenshot (BGR)",
    )
    parser.add_argument("--zone", type=str, default="", help="Current zone name for act/zone filtering")
    parser.add_argument("--threshold", type=float, default=0.70, help="Default matching confidence threshold")
    parser.add_argument("--quest-threshold", type=float, default=0.78, help="Threshold for quest category (higher to reduce false positives)")
    parser.add_argument("--waypoint-threshold", type=float, default=0.70, help="Threshold for waypoint category")
    parser.add_argument("--npc-threshold", type=float, default=0.65, help="Threshold for NPC category (uses color+shape matching)")
    parser.add_argument(
        "--categories",
        type=str,
        default="npcs,waypoints,quests",
        help="Comma-separated template categories to search",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/screenshots/outputs/diagnostic/template_matches.png"),
        help="Where to save the overlay image",
    )
    parser.add_argument("--show", action="store_true", help="Display the overlay with matplotlib")
    parser.add_argument("--debug", action="store_true", help="Enable verbose template detector logs")
    args = parser.parse_args()

    categories = parse_categories(args.categories)

    if not args.input.exists():
        print(f"ERROR: Input screenshot not found: {args.input}")
        return 1

    frame = cv2.imread(str(args.input))
    if frame is None:
        print(f"ERROR: Failed to read input: {args.input}")
        return 1

    detector = TemplateDetector(debug=args.debug)
    
    category_thresholds = {
        'quests': args.quest_threshold,
        'waypoints': args.waypoint_threshold,
        'npcs': args.npc_threshold,
    }
    
    matches = detector.detect(frame, current_zone=args.zone, categories=categories, 
                             threshold=args.threshold, category_thresholds=category_thresholds)

    overlay = draw_matches(frame, matches)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(args.output), overlay)

    counts = Counter(m.category for m in matches)
    print("\nTemplate matches:")
    if matches:
        for match in matches:
            print(f"- {match.category}: {match.template_name} @ {match.location} conf={match.confidence:.2f}")
    else:
        print("- none")

    print("\nSummary:")
    print(f"- Total: {len(matches)} matches")
    print(f"- Thresholds: quests={args.quest_threshold}, waypoints={args.waypoint_threshold}, npcs={args.npc_threshold}")
    if counts:
        print("- By category: " + ", ".join(f"{k}={v}" for k, v in counts.items()))
    print(f"- Saved overlay: {args.output}")

    if args.show:
        rgb = cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)
        plt.figure(figsize=(12, 7))
        plt.imshow(rgb)
        plt.title("Template Matches")
        plt.axis("off")
        plt.tight_layout()
        plt.show()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
