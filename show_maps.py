"""
Quick map overview - Display all visualizations at once.

Usage: python show_maps.py
"""

import sys
from pathlib import Path

import cv2
import matplotlib.pyplot as plt


def show_all_maps():
    """Display all generated map visualizations."""
    maps_dir = Path("data/maps")
    
    # Find all visualization images
    graph_file = maps_dir / "zone_graph.png"
    zone_vis_files = list(maps_dir.glob("*_visualization.png"))
    
    if not graph_file.exists() and not zone_vis_files:
        print("WARNING: No visualizations found!")
        print("Run: python visualize_maps.py")
        return
    
    # Determine layout
    num_images = (1 if graph_file.exists() else 0) + len(zone_vis_files)
    
    if num_images == 0:
        print("WARNING: No visualizations to display")
        return
    
    # Create figure
    fig = plt.figure(figsize=(16, 10))
    fig.suptitle("Diablo 2 Bot - Map Visualizations", fontsize=16, weight='bold')
    
    plot_idx = 1
    
    # Show zone graph
    if graph_file.exists():
        img = cv2.imread(str(graph_file))
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        ax = plt.subplot(2, 2, plot_idx)
        ax.imshow(img_rgb)
        ax.set_title("Zone Connection Graph", fontsize=12, weight='bold')
        ax.axis('off')
        plot_idx += 1
    
    # Show zone visualizations
    for zone_file in zone_vis_files[:3]:  # Max 3 zones
        img = cv2.imread(str(zone_file))
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        zone_name = zone_file.stem.replace("_visualization", "").replace("_", " ").upper()
        
        ax = plt.subplot(2, 2, plot_idx)
        ax.imshow(img_rgb)
        ax.set_title(f"{zone_name}", fontsize=12, weight='bold')
        ax.axis('off')
        plot_idx += 1
    
    plt.tight_layout()
    plt.show()
    
    print("\nOK All visualizations displayed")
    print(f"   - Zone graph: {graph_file}")
    for zone_file in zone_vis_files:
        print(f"   - Zone: {zone_file.name}")


if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("MAP OVERVIEW - ALL VISUALIZATIONS")
    print("=" * 70 + "\n")
    
    show_all_maps()
