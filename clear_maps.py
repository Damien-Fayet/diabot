"""
Utility script to clear/manage saved maps.

Allows cleaning up accumulated map data to start fresh.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from pathlib import Path
import shutil


def clear_all_maps(confirm: bool = True):
    """
    Clear all saved maps from data/maps/ directory.
    
    Args:
        confirm: If True, ask for confirmation before deleting
    """
    maps_dir = Path("data/maps")
    
    if not maps_dir.exists():
        print(f"✓ No maps directory found: {maps_dir}")
        return
    
    # Find all map files
    map_files = list(maps_dir.glob("*.png")) + list(maps_dir.glob("*.json"))
    
    if not map_files:
        print(f"✓ No map files found in {maps_dir}")
        return
    
    print("="*70)
    print("CLEAR ALL SAVED MAPS")
    print("="*70)
    print(f"\nFound {len(map_files)} map files:")
    
    # Group by zone
    zones = {}
    for f in map_files:
        zone_name = f.stem.rsplit("_", 2)[0] if "_" in f.stem else "unknown"
        if zone_name not in zones:
            zones[zone_name] = []
        zones[zone_name].append(f)
    
    for zone, files in sorted(zones.items()):
        print(f"  {zone}: {len(files)} files")
    
    if confirm:
        print("\n" + "="*70)
        response = input("Delete all these files? (yes/no): ").strip().lower()
        if response not in ["yes", "y"]:
            print("✗ Cancelled")
            return
    
    # Delete files
    deleted_count = 0
    for f in map_files:
        try:
            f.unlink()
            deleted_count += 1
        except Exception as e:
            print(f"✗ Error deleting {f.name}: {e}")
    
    print(f"\n✓ Deleted {deleted_count} map files")
    print("="*70)


def clear_zone_maps(zone_name: str):
    """
    Clear saved maps for a specific zone.
    
    Args:
        zone_name: Name of zone to clear (e.g., "ROGUE_ENCAMPMENT")
    """
    maps_dir = Path("data/maps")
    
    if not maps_dir.exists():
        print(f"✓ No maps directory found")
        return
    
    # Find files matching zone
    pattern = f"{zone_name}*"
    matching_files = list(maps_dir.glob(f"{pattern}.png")) + list(maps_dir.glob(f"{pattern}.json"))
    
    if not matching_files:
        print(f"✓ No maps found for zone: {zone_name}")
        return
    
    print(f"Found {len(matching_files)} files for {zone_name}")
    
    for f in matching_files:
        try:
            f.unlink()
            print(f"  ✓ Deleted {f.name}")
        except Exception as e:
            print(f"  ✗ Error: {e}")
    
    print(f"✓ Cleared {len(matching_files)} files for {zone_name}")


def list_saved_maps():
    """List all saved maps with details."""
    maps_dir = Path("data/maps")
    
    if not maps_dir.exists():
        print("No maps directory found")
        return
    
    map_files = sorted(maps_dir.glob("*.json"))
    
    if not map_files:
        print("No saved maps found")
        return
    
    print("="*70)
    print("SAVED MAPS")
    print("="*70)
    
    import json
    for json_file in map_files:
        try:
            with open(json_file, 'r') as f:
                metadata = json.load(f)
            
            zone = metadata.get("zone", "unknown")
            timestamp = metadata.get("timestamp", "unknown")
            cells = metadata.get("cell_count", 0)
            pois = len(metadata.get("pois", []))
            
            print(f"\n{json_file.stem}")
            print(f"  Zone: {zone}")
            print(f"  Time: {timestamp}")
            print(f"  Cells: {cells}")
            print(f"  POIs: {pois}")
            
            # Show POI types
            if pois > 0:
                poi_types = {}
                for poi in metadata.get("pois", []):
                    poi_type = poi.get("type", "unknown")
                    poi_types[poi_type] = poi_types.get(poi_type, 0) + 1
                poi_summary = ", ".join([f"{count} {t}" for t, count in poi_types.items()])
                print(f"    → {poi_summary}")
        
        except Exception as e:
            print(f"\n{json_file.name}: Error reading ({e})")
    
    print("\n" + "="*70)


def main():
    """Main CLI."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Manage saved map data",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python clear_maps.py --list               # List all saved maps
  python clear_maps.py --clear-all          # Clear all maps (with confirmation)
  python clear_maps.py --clear-all --yes    # Clear all maps (no confirmation)
  python clear_maps.py --clear-zone BLOOD_MOOR  # Clear specific zone
        """
    )
    
    parser.add_argument("--list", action="store_true", help="List all saved maps")
    parser.add_argument("--clear-all", action="store_true", help="Clear all saved maps")
    parser.add_argument("--clear-zone", type=str, metavar="ZONE", help="Clear maps for specific zone")
    parser.add_argument("--yes", action="store_true", help="Skip confirmation prompt")
    
    args = parser.parse_args()
    
    if args.list:
        list_saved_maps()
    elif args.clear_all:
        clear_all_maps(confirm=not args.yes)
    elif args.clear_zone:
        clear_zone_maps(args.clear_zone)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
