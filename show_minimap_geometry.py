"""
Visualize minimap walls/limits extracted from a screenshot.

Usage:
    python show_minimap_geometry.py --input data/screenshots/outputs/live_capture/minimap_extracted.png

This runs the minimap geometry extractor, overlays walls on the minimap, and
shows the downsampled occupancy grid used for navigation experiments.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

import cv2
import matplotlib.pyplot as plt
import numpy as np

from src.diabot.navigation import MinimapGeometryExtractor


def visualize(
    input_path: Path,
    save_overlay: Optional[Path],
    cell_size: int,
    wall_threshold: float,
) -> int:
    if not input_path.exists():
        print(f"ERROR: Input minimap not found: {input_path}")
        return 1

    minimap = cv2.imread(str(input_path))
    if minimap is None:
        print(f"ERROR: Failed to read minimap image: {input_path}")
        return 1

    extractor = MinimapGeometryExtractor(cell_size=cell_size, wall_threshold=wall_threshold, debug=True)
    geom = extractor.extract(minimap)
    overlay = extractor.overlay(minimap, geom)

    if save_overlay:
        save_overlay.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(save_overlay), overlay)
        print(f"Saved overlay to: {save_overlay}")

    # Build occupancy visualization
    occ = geom.occupancy
    occ_vis = np.where(occ > 0, 0.8, 0.0)

    fig, axes = plt.subplots(1, 3, figsize=(16, 6))
    fig.suptitle("Minimap Geometry Debug", fontsize=14, weight="bold")

    axes[0].imshow(cv2.cvtColor(minimap, cv2.COLOR_BGR2RGB))
    axes[0].set_title("Original Minimap", fontsize=12)
    axes[0].axis("off")

    axes[1].imshow(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB))
    axes[1].set_title("Overlay (walls in red)", fontsize=12)
    axes[1].axis("off")

    axes[2].imshow(occ_vis, cmap="Greys", interpolation="nearest")
    axes[2].set_title("Occupancy Grid (1=wall)", fontsize=12)
    if geom.player_cell:
        px, py = geom.player_cell
        axes[2].scatter([px], [py], c="blue", marker="x", s=80, label="player")
        axes[2].legend(loc="upper right")
    axes[2].invert_yaxis()  # Match image coordinate system

    plt.tight_layout()
    plt.show()

    print("\nGeometry summary:")
    print(f"- Player pixel: {geom.player_px}")
    print(f"- Player cell:  {geom.player_cell}")
    print(f"- Grid size:    {geom.occupancy.shape} (cell={geom.cell_size}px)")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Show minimap geometry overlays")
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data/screenshots/outputs/live_capture/minimap_extracted.png"),
        help="Path to minimap screenshot",
    )
    parser.add_argument(
        "--save-overlay",
        type=Path,
        default=Path("data/maps/minimap_geometry_overlay.png"),
        help="Where to save the overlay image",
    )
    parser.add_argument("--cell-size", type=int, default=4, help="Grid cell size in pixels")
    parser.add_argument("--wall-threshold", type=float, default=0.25, help="Wall fraction threshold per cell")

    args = parser.parse_args()
    return visualize(
        input_path=args.input,
        save_overlay=args.save_overlay,
        cell_size=args.cell_size,
        wall_threshold=args.wall_threshold,
    )


if __name__ == "__main__":
    raise SystemExit(main())
