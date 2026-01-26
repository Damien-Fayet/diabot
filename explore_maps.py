"""
Interactive map explorer - Command-line interface for map visualization.

Usage:
    python explore_maps.py                    # Interactive menu
    python explore_maps.py --zone "ZONE_NAME" # View specific zone
    python explore_maps.py --graph            # Show zone graph only
    python explore_maps.py --stats            # Show statistics only
"""

import argparse
from pathlib import Path
from visualize_maps import MapVisualizer


def interactive_menu(visualizer: MapVisualizer):
    """Interactive menu for map exploration."""
    while True:
        print("\n" + "=" * 70)
        print("MAP EXPLORER - INTERACTIVE MENU")
        print("=" * 70)
        print("\n1. Show statistics")
        print("2. Visualize zone graph")
        print("3. Visualize specific zone minimap")
        print("4. List all zones")
        print("5. Show zone details")
        print("0. Exit")
        print("\n" + "=" * 70)
        
        choice = input("\nEnter choice: ").strip()
        
        if choice == "0":
            print("\nGoodbye!")
            break
        
        elif choice == "1":
            visualizer.print_statistics()
        
        elif choice == "2":
            print("\nGenerating zone graph...")
            visualizer.visualize_zone_graph()
        
        elif choice == "3":
            print("\nAvailable zones:")
            for i, zone_name in enumerate(visualizer.zones.keys(), 1):
                print(f"  {i}. {zone_name}")
            
            zone_input = input("\nEnter zone number or name: ").strip()
            
            try:
                zone_idx = int(zone_input) - 1
                zone_name = list(visualizer.zones.keys())[zone_idx]
            except (ValueError, IndexError):
                zone_name = zone_input
            
            if zone_name in visualizer.zones:
                visualizer.visualize_zone_minimap(zone_name)
            else:
                print(f"\nWARNING: Zone '{zone_name}' not found")
        
        elif choice == "4":
            print("\n" + "=" * 70)
            print("ALL ZONES")
            print("=" * 70)
            for zone_name, zone_data in visualizer.zones.items():
                act = zone_data.get("act", "unknown")
                poi_count = len(zone_data.get("pois", []))
                connections = len(zone_data.get("connections", {}))
                print(f"\n{zone_name}")
                print(f"  Act: {act}")
                print(f"  POIs: {poi_count}")
                print(f"  Connections: {connections}")
        
        elif choice == "5":
            print("\nAvailable zones:")
            for i, zone_name in enumerate(visualizer.zones.keys(), 1):
                print(f"  {i}. {zone_name}")
            
            zone_input = input("\nEnter zone number or name: ").strip()
            
            try:
                zone_idx = int(zone_input) - 1
                zone_name = list(visualizer.zones.keys())[zone_idx]
            except (ValueError, IndexError):
                zone_name = zone_input
            
            if zone_name in visualizer.zones:
                show_zone_details(visualizer, zone_name)
            else:
                print(f"\nWARNING: Zone '{zone_name}' not found")
        
        else:
            print("\nWARNING: Invalid choice")


def show_zone_details(visualizer: MapVisualizer, zone_name: str):
    """Show detailed information about a zone."""
    zone_data = visualizer.zones[zone_name]
    
    print("\n" + "=" * 70)
    print(f"ZONE DETAILS: {zone_name}")
    print("=" * 70)
    
    print(f"\nAct: {zone_data.get('act', 'unknown')}")
    print(f"Discovered: {zone_data.get('discovered_at', 'unknown')}")
    
    pois = zone_data.get("pois", [])
    print(f"\nPOIs: {len(pois)}")
    
    if pois:
        # Group by type
        poi_by_type = {}
        for poi in pois:
            poi_type = poi.get("poi_type", "unknown")
            if poi_type not in poi_by_type:
                poi_by_type[poi_type] = []
            poi_by_type[poi_type].append(poi)
        
        for poi_type, type_pois in poi_by_type.items():
            print(f"\n  {poi_type.upper()}: {len(type_pois)}")
            for poi in type_pois:
                name = poi.get("name", "Unknown")
                pos = poi.get("position", [0, 0])
                target = poi.get("target_zone")
                
                info = f"    - {name} @ ({pos[0]}, {pos[1]})"
                if target:
                    info += f" -> {target}"
                print(info)
    
    connections = zone_data.get("connections", {})
    print(f"\nConnections: {len(connections)}")
    for target_zone, exit_name in connections.items():
        print(f"  -> {target_zone} via '{exit_name}'")
    
    print("\n" + "=" * 70)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Explore bot-generated maps")
    parser.add_argument("--zone", type=str, help="Visualize specific zone")
    parser.add_argument("--graph", action="store_true", help="Show zone graph")
    parser.add_argument("--stats", action="store_true", help="Show statistics")
    parser.add_argument("--list", action="store_true", help="List all zones")
    
    args = parser.parse_args()
    
    print("\n" + "=" * 70)
    print("DIABLO 2 BOT - MAP EXPLORER")
    print("=" * 70 + "\n")
    
    visualizer = MapVisualizer()
    
    if not visualizer.zones:
        print("WARNING: No maps found. Run the bot first to generate map data.")
        return
    
    # Handle command-line arguments
    if args.stats:
        visualizer.print_statistics()
        return
    
    if args.graph:
        visualizer.visualize_zone_graph()
        return
    
    if args.list:
        print("ALL ZONES:")
        for zone_name, zone_data in visualizer.zones.items():
            act = zone_data.get("act", "unknown")
            poi_count = len(zone_data.get("pois", []))
            print(f"  - {zone_name} ({act}): {poi_count} POIs")
        return
    
    if args.zone:
        if args.zone in visualizer.zones:
            show_zone_details(visualizer, args.zone)
            visualizer.visualize_zone_minimap(args.zone)
        else:
            print(f"WARNING: Zone '{args.zone}' not found")
            print("\nAvailable zones:")
            for zone_name in visualizer.zones.keys():
                print(f"  - {zone_name}")
        return
    
    # No arguments: interactive mode
    interactive_menu(visualizer)


if __name__ == "__main__":
    main()
