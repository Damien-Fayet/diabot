"""Game map and zone navigation system."""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import json
from pathlib import Path


@dataclass
class ZoneLocation:
    """A location/zone in the game world."""
    zone_id: str
    name: str
    act: int
    position: Tuple[float, float]  # (x, y) center of zone
    region_size: Tuple[float, float]  # (width, height) approx
    exits: List[str]  # zone_ids of adjacent zones
    npc_types: List[str]  # NPCs typically found here
    
    def contains_point(self, x: float, y: float) -> bool:
        """Check if point is in this zone."""
        cx, cy = self.position
        w, h = self.region_size
        
        return (cx - w/2 <= x <= cx + w/2 and 
                cy - h/2 <= y <= cy + h/2)


class GameMap:
    """Manages game world zones and navigation."""
    
    def __init__(self):
        """Initialize game map with Act 1 zones."""
        self.zones: Dict[str, ZoneLocation] = {}
        self.current_zone: Optional[str] = "rogue_encampment"  # Default to safe zone
        
        # Load built-in zones
        self._init_act1_zones()
    
    def _init_act1_zones(self):
        """Initialize Act 1 zones (Diablo II)."""
        # Approximate screen positions for Act 1 zones
        # Note: These are estimates - should be calibrated per game
        
        zones = [
            ZoneLocation(
                zone_id="rogue_encampment",
                name="Rogue Encampment",
                act=1,
                position=(500, 600),  # Safe hub zone
                region_size=(800, 600),
                exits=["cold_plains", "barracks"],
                npc_types=["akara", "kashya", "cain", "warriv", "charsi", "stash", "waypoint"],
            ),
            ZoneLocation(
                zone_id="cold_plains",
                name="Cold Plains",
                act=1,
                position=(400, 400),
                region_size=(600, 400),
                exits=["rogue_encampment", "stony_field", "burial_grounds"],
                npc_types=["enemy_fallen", "enemy_zombie", "enemy_quill_rat"],
            ),
            ZoneLocation(
                zone_id="burial_grounds",
                name="Burial Grounds",
                act=1,
                position=(300, 300),
                region_size=(500, 400),
                exits=["cold_plains", "crypt"],
                npc_types=["enemy_zombie", "enemy_skeleton", "boss_bloodwitch"],
            ),
            ZoneLocation(
                zone_id="stony_field",
                name="Stony Field",
                act=1,
                position=(600, 350),
                region_size=(600, 400),
                exits=["cold_plains", "dark_wood", "underground_passage"],
                npc_types=["enemy_fallen", "enemy_zombie", "treasure_chest"],
            ),
            ZoneLocation(
                zone_id="barracks",
                name="Barracks",
                act=1,
                position=(700, 650),
                region_size=(400, 400),
                exits=["rogue_encampment", "catacombs"],
                npc_types=["enemy_skeleton", "enemy_zombie", "boss_blood_raven"],
            ),
            ZoneLocation(
                zone_id="catacombs",
                name="Catacombs",
                act=1,
                position=(800, 700),
                region_size=(500, 500),
                exits=["barracks"],
                npc_types=["enemy_skeleton", "enemy_zombie", "treasure"],
            ),
        ]
        
        for zone in zones:
            self.zones[zone.zone_id] = zone
    
    def find_zone_by_position(self, x: float, y: float) -> Optional[str]:
        """Find which zone contains the given position.
        
        Args:
            x, y: Position in game world
            
        Returns:
            Zone ID or None
        """
        for zone_id, zone in self.zones.items():
            if zone.contains_point(x, y):
                return zone_id
        return None
    
    def update_current_zone(self, hero_pos: Tuple[float, float]):
        """Update current zone based on hero position.
        
        Args:
            hero_pos: (x, y) hero position in game world
        """
        zone_id = self.find_zone_by_position(hero_pos[0], hero_pos[1])
        if zone_id and zone_id != self.current_zone:
            old_zone = self.current_zone
            self.current_zone = zone_id
            print(f"Zone changed: {old_zone} → {zone_id}")
    
    def get_zone(self, zone_id: str) -> Optional[ZoneLocation]:
        """Get zone info by ID.
        
        Args:
            zone_id: Zone identifier
            
        Returns:
            ZoneLocation or None
        """
        return self.zones.get(zone_id)
    
    def get_current_zone(self) -> Optional[ZoneLocation]:
        """Get current zone.
        
        Returns:
            ZoneLocation or None
        """
        if self.current_zone:
            return self.zones.get(self.current_zone)
        return None
    
    def find_path_to_zone(self, target_zone_id: str, 
                         from_zone_id: Optional[str] = None) -> List[str]:
        """Find path to target zone using BFS.
        
        Args:
            target_zone_id: Destination zone
            from_zone_id: Starting zone (defaults to current)
            
        Returns:
            List of zone IDs forming path, or empty if unreachable
        """
        from collections import deque
        
        start = from_zone_id or self.current_zone
        if not start:
            return []
        
        if start == target_zone_id:
            return [start]
        
        # BFS
        queue = deque([(start, [start])])
        visited = {start}
        
        while queue:
            current, path = queue.popleft()
            current_zone = self.zones[current]
            
            for neighbor in current_zone.exits:
                if neighbor == target_zone_id:
                    return path + [neighbor]
                
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append((neighbor, path + [neighbor]))
        
        return []  # No path found
    
    def get_zone_entrance(self, zone_id: str, 
                         from_zone_id: Optional[str] = None) -> Optional[Tuple[float, float]]:
        """Get approximate entrance position for zone.
        
        When entering zone from another zone, use this position.
        
        Args:
            zone_id: Target zone
            from_zone_id: Zone we're coming from
            
        Returns:
            (x, y) position at zone entrance, or zone center if no specific entrance
        """
        zone = self.zones.get(zone_id)
        if not zone:
            return None
        
        # Simple heuristic: use zone center for now
        # In a real game, would use actual portal/exit positions
        return zone.position
    
    def get_all_zones(self) -> List[ZoneLocation]:
        """Get all zones.
        
        Returns:
            List of ZoneLocation objects
        """
        return list(self.zones.values())


class MapNavigator:
    """Handles navigation across zones to reach quest targets."""
    
    def __init__(self, game_map: GameMap):
        """Initialize navigator.
        
        Args:
            game_map: GameMap instance
        """
        self.map = game_map
        self.current_path: List[str] = []
        self.path_index = 0
    
    def navigate_to_quest_target(self, target_zone_id: str) -> Optional[Tuple[float, float]]:
        """Get next movement target to reach quest zone.
        
        If target zone is not visible on screen:
        1. Find path to target zone
        2. Return position of next zone entrance
        
        If target zone is visible:
        3. Return position within zone
        
        Args:
            target_zone_id: Zone containing quest target
            
        Returns:
            (x, y) position to move towards, or None if can't navigate
        """
        current = self.map.current_zone
        
        # If already in target zone, return zone center
        if current == target_zone_id:
            zone = self.map.get_zone(target_zone_id)
            return zone.position if zone else None
        
        # Need to navigate to zone
        # Find path
        path = self.map.find_path_to_zone(target_zone_id, current)
        if not path:
            print(f"Cannot path to {target_zone_id} from {current}")
            return None
        
        self.current_path = path
        self.path_index = 0
        
        # Go to next zone in path
        if len(path) > 1:
            next_zone = path[1]
            entrance = self.map.get_zone_entrance(next_zone, current)
            print(f"Path: {' → '.join(path)}, moving to {next_zone}")
            return entrance
        
        return None
    
    def update_progress(self):
        """Update position along current path."""
        # Called each frame to check if we reached zone entrance
        # and advance to next zone in path
        pass
