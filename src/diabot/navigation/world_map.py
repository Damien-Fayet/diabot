"""
World map management for offline Diablo 2 navigation.

Handles:
- Zone registration and metadata
- POI (Points of Interest) storage
- Zone connections
- Persistent JSON storage
- Minimap image caching
"""

from __future__ import annotations

import json
import hashlib
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np


@dataclass
class POI:
    """Point of Interest in a zone."""
    name: str
    poi_type: str  # 'waypoint', 'exit', 'npc', 'quest', 'monster', 'shrine'
    position: Tuple[int, int]  # (x, y) on minimap
    zone: str  # Zone name where POI is located
    target_zone: Optional[str] = None  # For exits: destination zone
    notes: str = ""


@dataclass
class ZoneMap:
    """Map data for a single zone."""
    zone_name: str
    act: str  # 'a1', 'a2', 'a3', 'a4', 'a5'
    pois: List[POI] = field(default_factory=list)
    connections: Dict[str, str] = field(default_factory=dict)  # {target_zone: exit_name}
    minimap_hash: str = ""  # Hash of minimap image for change detection
    discovered_at: str = field(default_factory=lambda: datetime.now().isoformat())

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "zone_name": self.zone_name,
            "act": self.act,
            "minimap_hash": self.minimap_hash,
            "pois": [asdict(poi) for poi in self.pois],
            "connections": self.connections,
            "discovered_at": self.discovered_at,
        }

    @staticmethod
    def from_dict(data: dict) -> ZoneMap:
        """Create ZoneMap from dictionary."""
        pois = [POI(**poi_data) for poi_data in data.get("pois", [])]
        return ZoneMap(
            zone_name=data["zone_name"],
            act=data["act"],
            pois=pois,
            connections=data.get("connections", {}),
            minimap_hash=data.get("minimap_hash", ""),
            discovered_at=data.get("discovered_at", datetime.now().isoformat()),
        )


class WorldMapManager:
    """
    Manages the complete world map for offline play.
    
    Stores zone data, POIs, and connections in JSON format.
    """

    def __init__(self, maps_dir: Path = None, debug: bool = False):
        """
        Initialize world map manager.

        Args:
            maps_dir: Directory to store map data (default: data/maps/)
            debug: Enable debug output
        """
        self.debug = debug
        
        if maps_dir is None:
            # Calculate path relative to this file
            maps_dir = Path(__file__).parent.parent.parent.parent / "data" / "maps"
        
        self.maps_dir = Path(maps_dir)
        self.maps_dir.mkdir(parents=True, exist_ok=True)
        
        self.minimap_dir = self.maps_dir / "minimap_images"
        self.minimap_dir.mkdir(parents=True, exist_ok=True)
        
        self.zones: Dict[str, ZoneMap] = {}
        
        self._load_all_maps()
        
        if self.debug:
            print(f"OK World map manager initialized with {len(self.zones)} zones")

    def _load_all_maps(self):
        """Load all zone maps from JSON."""
        maps_file = self.maps_dir / "zones_maps.json"
        
        if not maps_file.exists():
            if self.debug:
                print(f"WARNING No maps file found at {maps_file}")
            return
        
        try:
            with open(maps_file, "r", encoding="utf-8") as f:
                data = json.load(f)
            
            for zone_data in data.get("zones", []):
                zone_map = ZoneMap.from_dict(zone_data)
                self.zones[zone_map.zone_name] = zone_map
            
            if self.debug:
                print(f"OK Loaded maps for {len(self.zones)} zones")
        
        except Exception as e:
            if self.debug:
                print(f"WARNING Error loading maps: {e}")

    def save_all_maps(self):
        """Save all zone maps to JSON."""
        maps_file = self.maps_dir / "zones_maps.json"
        
        data = {
            "version": "1.0",
            "last_updated": datetime.now().isoformat(),
            "zones": [zone.to_dict() for zone in self.zones.values()],
        }
        
        try:
            with open(maps_file, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            
            if self.debug:
                print(f"OK Saved {len(self.zones)} zone maps")
        
        except Exception as e:
            if self.debug:
                print(f"WARNING Error saving maps: {e}")

    def register_zone(self, zone_name: str, act: str = "unknown"):
        """
        Register a new zone.

        Args:
            zone_name: Name of the zone
            act: Act number ('a1', 'a2', etc.)
        """
        if zone_name not in self.zones:
            self.zones[zone_name] = ZoneMap(
                zone_name=zone_name,
                act=act,
            )
            
            if self.debug:
                print(f"OK Registered zone: {zone_name} ({act})")

    def add_poi(
        self,
        zone_name: str,
        poi_name: str,
        poi_type: str,
        position: Tuple[int, int],
        target_zone: Optional[str] = None,
        notes: str = "",
    ):
        """
        Add a POI to a zone.

        Args:
            zone_name: Zone where POI is located
            poi_name: Name of the POI
            poi_type: Type ('waypoint', 'exit', 'npc', etc.)
            position: (x, y) position on minimap
            target_zone: For exits, the destination zone
            notes: Optional notes
        """
        if zone_name not in self.zones:
            if self.debug:
                print(f"WARNING Zone {zone_name} not registered")
            return
        
        zone = self.zones[zone_name]
        
        # Check for duplicate POIs nearby (within 20 pixels)
        for existing_poi in zone.pois:
            if existing_poi.poi_type == poi_type:
                dx = abs(existing_poi.position[0] - position[0])
                dy = abs(existing_poi.position[1] - position[1])
                if dx < 20 and dy < 20:
                    if self.debug:
                        print(f"OK POI '{poi_name}' already exists nearby in {zone_name}")
                    return
        
        poi = POI(
            name=poi_name,
            poi_type=poi_type,
            position=position,
            zone=zone_name,
            target_zone=target_zone,
            notes=notes,
        )
        
        zone.pois.append(poi)
        poi_id = len(zone.pois)
        
        if self.debug:
            print(f"OK Added POI #{poi_id}: {poi_name} in {zone_name} @ {position}")

    def add_connection(self, zone_name: str, target_zone: str, exit_name: str = "Exit"):
        """
        Add a connection between zones.

        Args:
            zone_name: Source zone
            target_zone: Destination zone
            exit_name: Name of the exit
        """
        if zone_name not in self.zones:
            if self.debug:
                print(f"WARNING Zone {zone_name} not registered")
            return
        
        zone = self.zones[zone_name]
        zone.connections[target_zone] = exit_name
        
        if self.debug:
            print(f"OK Connection: {zone_name} -> {target_zone} via '{exit_name}'")

    def get_zone(self, zone_name: str) -> Optional[ZoneMap]:
        """Get zone map by name."""
        return self.zones.get(zone_name)

    def get_waypoints(self, zone_name: str) -> List[POI]:
        """Get all waypoints in a zone."""
        zone = self.zones.get(zone_name)
        if zone:
            return [poi for poi in zone.pois if poi.poi_type == "waypoint"]
        return []

    def get_exits(self, zone_name: str) -> List[POI]:
        """Get all exits in a zone."""
        zone = self.zones.get(zone_name)
        if zone:
            return [poi for poi in zone.pois if poi.poi_type == "exit"]
        return []

    def save_minimap(self, zone_name: str, minimap_img: np.ndarray):
        """
        Save minimap image for a zone.

        Args:
            zone_name: Zone name
            minimap_img: Minimap image (BGR numpy array)
        """
        if minimap_img is None or minimap_img.size == 0:
            return
        
        # Generate hash for change detection
        img_hash = hashlib.md5(minimap_img.tobytes()).hexdigest()[:16]
        
        # Save image
        filename = f"{zone_name.lower().replace(' ', '_')}_{img_hash}.png"
        path = self.minimap_dir / filename
        
        try:
            cv2.imwrite(str(path), minimap_img)
            
            # Update zone hash
            if zone_name in self.zones:
                self.zones[zone_name].minimap_hash = img_hash
            
            if self.debug:
                print(f"OK Saved minimap: {path}")
        
        except Exception as e:
            if self.debug:
                print(f"WARNING Error saving minimap: {e}")
