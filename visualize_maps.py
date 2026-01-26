"""
Visualization tool for bot-generated maps.

This script provides:
1. Graph visualization of zone connections
2. Minimap POI display
3. Navigation path preview
4. Zone statistics
"""

import json
from pathlib import Path
from typing import Dict, List

import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.collections import LineCollection


class MapVisualizer:
    """Visualize world maps and navigation data."""
    
    def __init__(self, maps_dir: Path = None):
        """
        Initialize map visualizer.
        
        Args:
            maps_dir: Directory containing map data (default: data/maps/)
        """
        if maps_dir is None:
            maps_dir = Path(__file__).parent / "data" / "maps"
        
        self.maps_dir = Path(maps_dir)
        self.zones_file = self.maps_dir / "zones_maps.json"
        self.minimap_dir = self.maps_dir / "minimap_images"
        
        self.zones = {}
        self._load_maps()
    
    def _load_maps(self):
        """Load zone data from JSON."""
        if not self.zones_file.exists():
            print(f"WARNING: No maps file found at {self.zones_file}")
            return
        
        with open(self.zones_file, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        for zone_data in data.get("zones", []):
            self.zones[zone_data["zone_name"]] = zone_data
        
        print(f"OK Loaded {len(self.zones)} zones")
    
    def visualize_zone_graph(self, save_path: str = None):
        """
        Visualize zones and their connections as a graph.
        
        Args:
            save_path: Optional path to save the figure
        """
        if not self.zones:
            print("WARNING: No zones to visualize")
            return
        
        fig, ax = plt.subplots(figsize=(14, 10))
        
        # Group zones by act
        acts = {}
        for zone_name, zone_data in self.zones.items():
            act = zone_data.get("act", "unknown")
            if act not in acts:
                acts[act] = []
            acts[act].append(zone_name)
        
        # Layout zones by act (horizontal layout)
        positions = {}
        act_colors = {
            'a1': '#FF6B6B',
            'a2': '#4ECDC4',
            'a3': '#45B7D1',
            'a4': '#FFA07A',
            'a5': '#98D8C8',
            'unknown': '#CCCCCC',
        }
        
        act_order = ['a1', 'a2', 'a3', 'a4', 'a5', 'unknown']
        x_offset = 0
        
        for act_idx, act in enumerate(act_order):
            if act not in acts:
                continue
            
            zones_in_act = acts[act]
            y_spacing = 1.0
            
            for i, zone_name in enumerate(zones_in_act):
                x = x_offset + 2.0
                y = i * y_spacing
                positions[zone_name] = (x, y)
            
            x_offset += 4.0
        
        # Draw connections (edges)
        for zone_name, zone_data in self.zones.items():
            if zone_name not in positions:
                continue
            
            x1, y1 = positions[zone_name]
            
            for target_zone in zone_data.get("connections", {}).keys():
                if target_zone in positions:
                    x2, y2 = positions[target_zone]
                    ax.plot([x1, x2], [y1, y2], 'k-', alpha=0.3, linewidth=1.5)
        
        # Draw zones (nodes)
        for zone_name, zone_data in self.zones.items():
            if zone_name not in positions:
                continue
            
            x, y = positions[zone_name]
            act = zone_data.get("act", "unknown")
            color = act_colors.get(act, '#CCCCCC')
            
            # Get POI count
            poi_count = len(zone_data.get("pois", []))
            waypoint_count = sum(1 for poi in zone_data.get("pois", []) 
                                if poi.get("poi_type") == "waypoint")
            
            # Node size based on POI count
            node_size = 300 + poi_count * 50
            
            # Draw node
            circle = plt.Circle((x, y), 0.4, color=color, alpha=0.7, zorder=10)
            ax.add_patch(circle)
            
            # Add waypoint indicator
            if waypoint_count > 0:
                marker = plt.Circle((x, y), 0.15, color='gold', alpha=0.9, zorder=11)
                ax.add_patch(marker)
            
            # Zone label
            label = zone_name
            if len(label) > 20:
                label = label[:17] + "..."
            
            ax.text(x, y - 0.7, label, ha='center', va='top', fontsize=8,
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
            
            # POI count
            if poi_count > 0:
                ax.text(x, y + 0.5, f"{poi_count} POIs", ha='center', va='bottom',
                       fontsize=7, color='darkblue', weight='bold')
        
        # Legend
        legend_elements = [
            mpatches.Patch(color=act_colors['a1'], label='Act 1', alpha=0.7),
            mpatches.Patch(color=act_colors['a2'], label='Act 2', alpha=0.7),
            mpatches.Patch(color=act_colors['a3'], label='Act 3', alpha=0.7),
            mpatches.Patch(color=act_colors['a4'], label='Act 4', alpha=0.7),
            mpatches.Patch(color=act_colors['a5'], label='Act 5', alpha=0.7),
            mpatches.Patch(color='gold', label='Has Waypoint', alpha=0.9),
        ]
        ax.legend(handles=legend_elements, loc='upper right', fontsize=10)
        
        ax.set_title("Diablo 2 Bot - World Map Graph", fontsize=16, weight='bold', pad=20)
        ax.set_xlabel("Acts", fontsize=12)
        ax.set_ylabel("Zones", fontsize=12)
        ax.axis('equal')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"OK Saved graph to {save_path}")
        
        plt.show()
    
    def visualize_zone_minimap(self, zone_name: str, save_path: str = None):
        """
        Visualize a specific zone's minimap with POIs.
        
        Args:
            zone_name: Name of zone to visualize
            save_path: Optional path to save the figure
        """
        if zone_name not in self.zones:
            print(f"WARNING: Zone '{zone_name}' not found")
            return
        
        zone_data = self.zones[zone_name]
        
        # Try to find minimap image
        minimap_hash = zone_data.get("minimap_hash", "")
        minimap_pattern = f"{zone_name.lower().replace(' ', '_')}*.png"
        
        minimap_files = list(self.minimap_dir.glob(minimap_pattern))
        
        if not minimap_files:
            print(f"WARNING: No minimap found for {zone_name}")
            self._visualize_poi_only(zone_data, save_path)
            return
        
        # Load minimap
        minimap_path = minimap_files[0]
        minimap = cv2.imread(str(minimap_path))
        minimap_rgb = cv2.cvtColor(minimap, cv2.COLOR_BGR2RGB)
        
        fig, ax = plt.subplots(figsize=(12, 10))
        ax.imshow(minimap_rgb)
        
        # Draw POIs
        poi_colors = {
            'waypoint': 'cyan',
            'exit': 'orange',
            'monster': 'red',
            'npc': 'green',
            'quest': 'yellow',
            'shrine': 'magenta',
        }
        
        for poi in zone_data.get("pois", []):
            poi_type = poi.get("poi_type", "unknown")
            position = poi.get("position", [0, 0])
            name = poi.get("name", poi_type)
            
            x, y = position
            color = poi_colors.get(poi_type, 'white')
            
            # Draw circle
            circle = plt.Circle((x, y), 8, color=color, fill=False, linewidth=2.5)
            ax.add_patch(circle)
            
            # Draw label
            ax.text(x + 12, y, name, color=color, fontsize=9,
                   weight='bold', va='center',
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='black', alpha=0.7))
        
        ax.set_title(f"{zone_name} - Minimap with POIs", fontsize=14, weight='bold')
        ax.axis('off')
        
        # Legend
        legend_elements = [
            mpatches.Patch(color=poi_colors['waypoint'], label='Waypoint'),
            mpatches.Patch(color=poi_colors['exit'], label='Exit'),
            mpatches.Patch(color=poi_colors['monster'], label='Monster'),
            mpatches.Patch(color=poi_colors['npc'], label='NPC'),
            mpatches.Patch(color=poi_colors['quest'], label='Quest'),
        ]
        ax.legend(handles=legend_elements, loc='upper right')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"OK Saved minimap to {save_path}")
        
        plt.show()
    
    def _visualize_poi_only(self, zone_data: dict, save_path: str = None):
        """Visualize POIs without minimap (scatter plot)."""
        pois = zone_data.get("pois", [])
        if not pois:
            print("WARNING: No POIs to visualize")
            return
        
        fig, ax = plt.subplots(figsize=(10, 8))
        
        poi_colors = {
            'waypoint': 'cyan',
            'exit': 'orange',
            'monster': 'red',
            'npc': 'green',
        }
        
        for poi_type in poi_colors.keys():
            type_pois = [p for p in pois if p.get("poi_type") == poi_type]
            if type_pois:
                xs = [p["position"][0] for p in type_pois]
                ys = [p["position"][1] for p in type_pois]
                ax.scatter(xs, ys, c=poi_colors[poi_type], label=poi_type.capitalize(),
                          s=200, alpha=0.7, edgecolors='black', linewidth=2)
        
        ax.set_title(f"{zone_data['zone_name']} - POI Locations", fontsize=14, weight='bold')
        ax.set_xlabel("X Position")
        ax.set_ylabel("Y Position")
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.invert_yaxis()  # Match image coordinates
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"OK Saved POI map to {save_path}")
        
        plt.show()
    
    def print_statistics(self):
        """Print map statistics."""
        if not self.zones:
            print("WARNING: No zones loaded")
            return
        
        print("\n" + "=" * 70)
        print("MAP STATISTICS")
        print("=" * 70)
        
        total_pois = 0
        total_waypoints = 0
        total_exits = 0
        total_connections = 0
        
        acts_count = {}
        
        for zone_name, zone_data in self.zones.items():
            act = zone_data.get("act", "unknown")
            acts_count[act] = acts_count.get(act, 0) + 1
            
            pois = zone_data.get("pois", [])
            total_pois += len(pois)
            
            for poi in pois:
                if poi.get("poi_type") == "waypoint":
                    total_waypoints += 1
                elif poi.get("poi_type") == "exit":
                    total_exits += 1
            
            total_connections += len(zone_data.get("connections", {}))
        
        print(f"\nTotal Zones: {len(self.zones)}")
        print(f"Total POIs: {total_pois}")
        print(f"  - Waypoints: {total_waypoints}")
        print(f"  - Exits: {total_exits}")
        print(f"Total Connections: {total_connections}")
        
        print("\nZones by Act:")
        for act in ['a1', 'a2', 'a3', 'a4', 'a5', 'unknown']:
            if act in acts_count:
                print(f"  {act}: {acts_count[act]} zones")
        
        print("\n" + "=" * 70 + "\n")


def main():
    """Main visualization interface."""
    print("\n" + "=" * 70)
    print("DIABLO 2 BOT - MAP VISUALIZER")
    print("=" * 70 + "\n")
    
    visualizer = MapVisualizer()
    
    if not visualizer.zones:
        print("\nNo maps found. Run the bot first to generate map data.")
        return
    
    # Print statistics
    visualizer.print_statistics()
    
    # Visualize zone graph
    print("Generating zone connection graph...")
    visualizer.visualize_zone_graph(save_path="data/maps/zone_graph.png")
    
    # Visualize first zone with POIs
    if visualizer.zones:
        first_zone = list(visualizer.zones.keys())[0]
        print(f"\nVisualizing minimap for: {first_zone}")
        visualizer.visualize_zone_minimap(first_zone, 
                                         save_path=f"data/maps/{first_zone.lower().replace(' ', '_')}_visualization.png")
    
    print("\n" + "=" * 70)
    print("Visualization complete!")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
