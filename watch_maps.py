"""
Real-time map monitor - Watch for map updates while bot is running.

Usage: python watch_maps.py [--interval SECONDS]
"""

import argparse
import json
import time
from pathlib import Path
from datetime import datetime


def load_map_data(maps_file: Path) -> dict:
    """Load map data from JSON."""
    if not maps_file.exists():
        return {}
    
    try:
        with open(maps_file, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        return {}


def get_stats(data: dict) -> dict:
    """Extract statistics from map data."""
    zones = data.get("zones", [])
    
    total_pois = sum(len(z.get("pois", [])) for z in zones)
    total_waypoints = sum(
        sum(1 for poi in z.get("pois", []) if poi.get("poi_type") == "waypoint")
        for z in zones
    )
    total_connections = sum(len(z.get("connections", {})) for z in zones)
    
    return {
        "zones": len(zones),
        "pois": total_pois,
        "waypoints": total_waypoints,
        "connections": total_connections,
        "last_updated": data.get("last_updated", "unknown"),
    }


def print_status(stats: dict, changes: dict = None):
    """Print current status."""
    print("\n" + "=" * 70)
    print(f"MAP STATUS - {datetime.now().strftime('%H:%M:%S')}")
    print("=" * 70)
    
    print(f"\nZones: {stats['zones']}")
    print(f"POIs: {stats['pois']} (Waypoints: {stats['waypoints']})")
    print(f"Connections: {stats['connections']}")
    print(f"Last Updated: {stats['last_updated'][:19]}")
    
    if changes:
        print("\n" + "-" * 70)
        print("CHANGES DETECTED:")
        for key, (old, new) in changes.items():
            diff = new - old
            symbol = "+" if diff > 0 else ""
            print(f"  {key}: {old} -> {new} ({symbol}{diff})")
        print("-" * 70)
    
    print("\n(Press Ctrl+C to stop monitoring)")


def watch_maps(interval: int = 5):
    """
    Monitor map file for changes.
    
    Args:
        interval: Check interval in seconds
    """
    maps_file = Path("data/maps/zones_maps.json")
    
    print("\n" + "=" * 70)
    print("REAL-TIME MAP MONITOR")
    print("=" * 70)
    print(f"\nMonitoring: {maps_file}")
    print(f"Check interval: {interval} seconds")
    
    if not maps_file.exists():
        print("\nWARNING: Map file not found. Waiting for bot to create it...")
    
    prev_stats = None
    
    try:
        while True:
            data = load_map_data(maps_file)
            
            if data:
                stats = get_stats(data)
                
                # Detect changes
                changes = None
                if prev_stats:
                    changes = {}
                    for key in ["zones", "pois", "waypoints", "connections"]:
                        if stats[key] != prev_stats[key]:
                            changes[key] = (prev_stats[key], stats[key])
                
                # Print status
                if changes or prev_stats is None:
                    print_status(stats, changes)
                
                prev_stats = stats
            
            time.sleep(interval)
    
    except KeyboardInterrupt:
        print("\n\n" + "=" * 70)
        print("MONITORING STOPPED")
        print("=" * 70)
        
        if prev_stats:
            print("\nFinal statistics:")
            print(f"  Zones discovered: {prev_stats['zones']}")
            print(f"  POIs found: {prev_stats['pois']}")
            print(f"  Waypoints: {prev_stats['waypoints']}")
            print(f"  Connections: {prev_stats['connections']}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Monitor bot map generation in real-time")
    parser.add_argument("--interval", type=int, default=5,
                       help="Check interval in seconds (default: 5)")
    
    args = parser.parse_args()
    
    watch_maps(interval=args.interval)


if __name__ == "__main__":
    main()
