"""
Navigation system using world map and POI data.

Handles:
- Pathfinding between zones
- Route planning
- Waypoint optimization
- Location caching
"""

from typing import List, Optional, Tuple, Dict
from pathlib import Path
from .world_map import WorldMapManager, ZoneMap, POI


class NavigationPath:
    """Represents a path from one location to another."""
    
    def __init__(self, destination: str, zones: List[str], waypoints: List[POI] = None):
        """
        Initialize navigation path.
        
        Args:
            destination: Final destination zone
            zones: List of zones to traverse
            waypoints: Optional waypoints to pass through
        """
        self.destination = destination
        self.zones = zones
        self.waypoints = waypoints or []
        self.current_step = 0
    
    def get_next_zone(self) -> Optional[str]:
        """Get next zone in path."""
        if self.current_step < len(self.zones):
            return self.zones[self.current_step]
        return None
    
    def advance(self):
        """Move to next zone in path."""
        self.current_step += 1
    
    def is_complete(self) -> bool:
        """Check if path is complete."""
        return self.current_step >= len(self.zones)


class Navigator:
    """Navigation system for Diablo 2."""
    
    def __init__(self, maps_dir: Path = None, debug: bool = False):
        """
        Initialize navigator.
        
        Args:
            maps_dir: Directory containing world map data
            debug: Enable debug output
        """
        self.debug = debug
        self.world_map = WorldMapManager(maps_dir=maps_dir, debug=debug)
        
        # Cache of explored zones
        self.explored_zones: set = set()
        
        if debug:
            print(f"OK Navigator initialized")
    
    def visit_zone(self, zone_name: str, act: str = "unknown"):
        """
        Mark zone as visited/explored.
        
        Args:
            zone_name: Zone name
            act: Act number
        """
        if zone_name not in self.explored_zones:
            self.world_map.register_zone(zone_name, act)
            self.explored_zones.add(zone_name)
            if self.debug:
                print(f"OK Explored: {zone_name}")
    
    def add_waypoint(self, zone_name: str, position: Tuple[int, int]):
        """
        Register waypoint location in a zone.
        
        Args:
            zone_name: Zone containing waypoint
            position: (x, y) position on minimap
        """
        self.world_map.add_poi(
            zone_name=zone_name,
            poi_name="Waypoint",
            poi_type="waypoint",
            position=position,
        )
        
        if self.debug:
            print(f"OK Registered waypoint at {position}")
    
    def add_exit(self, zone_name: str, target_zone: str, position: Tuple[int, int]):
        """
        Register exit/portal from one zone to another.
        
        Args:
            zone_name: Source zone
            target_zone: Destination zone
            position: (x, y) position on minimap
        """
        self.world_map.add_poi(
            zone_name=zone_name,
            poi_name=f"Exit to {target_zone}",
            poi_type="exit",
            position=position,
            target_zone=target_zone,
        )
        
        self.world_map.add_connection(zone_name, target_zone, f"Exit to {target_zone}")
        
        if self.debug:
            print(f"OK Registered exit: {zone_name} -> {target_zone}")
    
    def find_path(self, from_zone: str, to_zone: str) -> Optional[NavigationPath]:
        """
        Find path between two zones using BFS.
        
        Args:
            from_zone: Starting zone
            to_zone: Destination zone
            
        Returns:
            NavigationPath or None if no path found
        """
        if from_zone not in self.world_map.zones:
            if self.debug:
                print(f"WARNING Start zone {from_zone} not registered")
            return None
        
        if to_zone not in self.world_map.zones:
            if self.debug:
                print(f"WARNING Destination zone {to_zone} not registered")
            return None
        
        # BFS to find shortest path
        queue = [(from_zone, [from_zone])]
        visited = {from_zone}
        
        while queue:
            current_zone, path = queue.pop(0)
            
            if current_zone == to_zone:
                # Found path
                waypoints = []
                for zone_name in path:
                    zone_waypoints = self.world_map.get_waypoints(zone_name)
                    waypoints.extend(zone_waypoints)
                
                return NavigationPath(
                    destination=to_zone,
                    zones=path,
                    waypoints=waypoints,
                )
            
            # Explore neighbors
            zone = self.world_map.zones[current_zone]
            for neighbor_zone in zone.connections.keys():
                if neighbor_zone not in visited:
                    visited.add(neighbor_zone)
                    queue.append((neighbor_zone, path + [neighbor_zone]))
        
        # No path found
        if self.debug:
            print(f"WARNING No path found from {from_zone} to {to_zone}")
        return None
    
    def plan_route(
        self,
        current_zone: str,
        destination: str,
        via_waypoints: bool = True
    ) -> Optional[NavigationPath]:
        """
        Plan optimal route to destination.
        
        Args:
            current_zone: Current zone
            destination: Target zone
            via_waypoints: Prioritize waypoints in route
            
        Returns:
            NavigationPath or None
        """
        path = self.find_path(current_zone, destination)
        
        if path and via_waypoints:
            # Filter waypoints to only those along the route
            route_waypoints = [
                wp for wp in path.waypoints
                if wp.zone in path.zones
            ]
            path.waypoints = route_waypoints
        
        if self.debug and path:
            print(f"OK Route planned: {' -> '.join(path.zones)}")
        
        return path
    
    def get_nearest_waypoint(self, zone_name: str) -> Optional[POI]:
        """
        Get nearest waypoint in a zone.
        
        Args:
            zone_name: Zone to search
            
        Returns:
            Nearest waypoint POI or None
        """
        waypoints = self.world_map.get_waypoints(zone_name)
        if waypoints:
            return waypoints[0]  # For now, return first waypoint
        return None
    
    def save_maps(self):
        """Save all navigation data to disk."""
        self.world_map.save_all_maps()
        if self.debug:
            print(f"OK Maps saved")


# Demo code for testing
if __name__ == "__main__":
    from pathlib import Path
    
    print("\n" + "="*80)
    print("NAVIGATION SYSTEM DEMO")
    print("="*80)
    
    # Create navigator
    nav = Navigator(debug=True)
    
    # Register some test zones
    nav.visit_zone("ROGUE ENCAMPMENT", "a1")
    nav.visit_zone("BLOOD MOOR", "a1")
    nav.visit_zone("COLD PLAINS", "a1")
    
    # Add connections
    nav.add_exit("ROGUE ENCAMPMENT", "BLOOD MOOR", (300, 200))
    nav.add_exit("BLOOD MOOR", "COLD PLAINS", (400, 300))
    
    # Add waypoints
    nav.add_waypoint("ROGUE ENCAMPMENT", (250, 250))
    nav.add_waypoint("COLD PLAINS", (350, 350))
    
    # Find path
    print("\n" + "="*80)
    print("PATHFINDING TEST")
    print("="*80)
    
    path = nav.find_path("ROGUE ENCAMPMENT", "COLD PLAINS")
    
    if path:
        print(f"\nOK Path: {' -> '.join(path.zones)}")
        print(f"OK Waypoints: {len(path.waypoints)}")
        for wp in path.waypoints:
            print(f"  - {wp.name} in {wp.zone} @ {wp.position}")
    else:
        print("\nWARNING No path found")
    
    # Save
    nav.save_maps()
    print("\n" + "="*80)
