"""
Map accumulator for building persistent world map.

Accumulates minimap observations into a global map that is stored
in memory and persisted to disk.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import json
from datetime import datetime

from .minimap_processor import MinimapGrid, CellType


@dataclass
class GlobalMapCell:
    """
    A cell in the global accumulated map.
    
    Attributes:
        cell_type: WALL, FREE, or UNKNOWN
        confidence: Number of times this cell has been observed
        last_seen: Frame number when last observed
    """
    cell_type: int  # CellType value
    confidence: int = 1
    last_seen: int = 0


@dataclass
class MapPOI:
    """
    Point of Interest detected on the map.
    
    Attributes:
        poi_type: Type of POI (npc, exit, waypoint, chest, shrine, etc.)
        position: (x, y) in global map coordinates
        label: Detected label/class name
        confidence: Detection confidence (0.0-1.0)
        frame_detected: Frame number when first detected
        last_seen: Frame number when last observed
    """
    poi_type: str
    position: Tuple[int, int]
    label: str
    confidence: float = 1.0
    frame_detected: int = 0
    last_seen: int = 0


class MapAccumulator:
    """
    Accumulates minimap observations into a persistent global map.
    
    Features:
    - Tracks visited areas in world coordinates
    - Merges new observations with existing map
    - Detects unexplored regions (frontiers)
    - Persists map to disk (JSON + PNG)
    - Handles map updates with confidence scoring
    """
    
    def __init__(
        self,
        map_size: int = 2048,
        cell_size: float = 1.0,
        save_dir: Optional[Path] = None,
        debug: bool = False
    ):
        """
        Initialize map accumulator.
        
        Args:
            map_size: Size of global map in cells (map_size x map_size)
            cell_size: Size of each cell in world units
            save_dir: Directory to save map files (default: data/maps/)
            debug: Enable debug output
        """
        self.map_size = map_size
        self.cell_size = cell_size
        self.debug = debug
        
        if save_dir is None:
            save_dir = Path("data/maps")
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        # Global map centered at (map_size//2, map_size//2)
        self.origin = (map_size // 2, map_size // 2)
        
        # Initialize empty map
        self.cells: Dict[Tuple[int, int], GlobalMapCell] = {}
        
        # POI tracking (NPCs, exits, waypoints, etc.)
        self.pois: List[MapPOI] = []
        
        # Player tracking
        self.player_world_pos = self.origin  # Current player position in global map
        self.player_path: list[Tuple[int, int]] = []  # History of positions
        
        # Frame counter for last_seen tracking
        self.frame_count = 0
        
        # Current zone name
        self.current_zone = "unknown"
        
        if self.debug:
            print(f"[MapAccumulator] Initialized {map_size}x{map_size} global map")
    
    def update(
        self,
        minimap_grid: MinimapGrid,
        player_offset: Tuple[int, int] = (0, 0)
    ):
        """
        Update global map with new minimap observation.
        
        Args:
            minimap_grid: Processed minimap grid
            player_offset: (dx, dy) movement since last update in grid cells
        """
        self.frame_count += 1
        
        # Update player position
        px, py = self.player_world_pos
        dx, dy = player_offset
        self.player_world_pos = (px + dx, py + dy)
        self.player_path.append(self.player_world_pos)
        
        # Get minimap center (player position in minimap coords)
        minimap_cx, minimap_cy = minimap_grid.center
        
        # Merge minimap into global map
        h, w = minimap_grid.shape
        for my in range(h):
            for mx in range(w):
                cell_type = minimap_grid.grid[my, mx]
                
                # Skip unknown cells
                if cell_type == CellType.UNKNOWN:
                    continue
                
                # Convert minimap coords to global coords
                # Offset from minimap center
                offset_x = mx - minimap_cx
                offset_y = my - minimap_cy
                
                # Global position
                gx = self.player_world_pos[0] + offset_x
                gy = self.player_world_pos[1] + offset_y
                
                # Bounds check
                if not (0 <= gx < self.map_size and 0 <= gy < self.map_size):
                    continue
                
                pos = (gx, gy)
                
                # Update or create cell
                if pos in self.cells:
                    cell = self.cells[pos]
                    # Increase confidence if same type observed
                    if cell.cell_type == cell_type:
                        cell.confidence = min(10, cell.confidence + 1)
                    # Override if new observation is stronger or wall
                    elif cell_type == CellType.WALL or cell.confidence < 3:
                        cell.cell_type = cell_type
                        cell.confidence = 1
                    cell.last_seen = self.frame_count
                else:
                    self.cells[pos] = GlobalMapCell(
                        cell_type=cell_type,
                        confidence=1,
                        last_seen=self.frame_count
                    )
        
        if self.debug and self.frame_count % 10 == 0:
            print(f"[MapAccumulator] Frame {self.frame_count}: {len(self.cells)} cells mapped, "
                  f"Player @ {self.player_world_pos}")
    
    def get_cell(self, x: int, y: int) -> Optional[int]:
        """
        Get cell type at global coordinates.
        
        Args:
            x, y: Global map coordinates
            
        Returns:
            CellType value or None if unknown
        """
        pos = (x, y)
        if pos in self.cells:
            return self.cells[pos].cell_type
        return None
    
    def is_explored(self, x: int, y: int) -> bool:
        """Check if a cell has been explored."""
        return (x, y) in self.cells
    
    def add_poi(
        self,
        poi_type: str,
        position: Tuple[int, int],
        label: str,
        confidence: float = 1.0
    ):
        """
        Add a Point of Interest to the map.
        
        Args:
            poi_type: Type (npc, exit, waypoint, chest, shrine, etc.)
            position: (x, y) in global map coordinates
            label: Detection label/class name
            confidence: Detection confidence (0.0-1.0)
        """
        # Validate POI is within minimap visible area (64x64 cells)
        player_x, player_y = self.player_world_pos
        poi_x, poi_y = position
        
        distance_x = abs(poi_x - player_x)
        distance_y = abs(poi_y - player_y)
        
        # Skip POIs outside 64x64 minimap grid (±32 cells from player)
        if distance_x > 32 or distance_y > 32:
            if self.debug:
                print(f"[MapAccumulator] Skipping POI outside minimap: {label} @ {position} (offset: {distance_x}, {distance_y})")
            return
        
        # Check for duplicate POIs nearby (within 5 cells)
        for existing_poi in self.pois:
            if existing_poi.poi_type == poi_type:
                dx = abs(existing_poi.position[0] - position[0])
                dy = abs(existing_poi.position[1] - position[1])
                if dx < 5 and dy < 5:
                    # Update last_seen only, keep original position
                    # NPCs move around but we keep their first detected position
                    existing_poi.last_seen = self.frame_count
                    # Update confidence if higher
                    if confidence > existing_poi.confidence:
                        existing_poi.confidence = confidence
                    if self.debug:
                        print(f"[MapAccumulator] Seen POI: {label} @ existing pos {existing_poi.position}")
                    return
        
        # Add new POI
        poi = MapPOI(
            poi_type=poi_type,
            position=position,
            label=label,
            confidence=confidence,
            frame_detected=self.frame_count,
            last_seen=self.frame_count
        )
        self.pois.append(poi)
        
        if self.debug:
            print(f"[MapAccumulator] Added POI: {label} ({poi_type}) @ {position}")
    
    def clear(self, keep_pois: bool = False):
        """
        Clear the accumulated map and reset to initial state.
        
        Args:
            keep_pois: If True, keep detected POIs; if False, clear everything
        """
        self.cells.clear()
        self.player_world_pos = self.origin
        self.player_path.clear()
        self.frame_count = 0
        
        if not keep_pois:
            self.pois.clear()
        
        if self.debug:
            poi_msg = "(kept POIs)" if keep_pois else ""
            print(f"[MapAccumulator] Map cleared {poi_msg}")
    
    def find_frontiers(self, search_radius: int = 50) -> list[Tuple[int, int]]:
        """
        Find frontier cells (boundary between known and unknown).
        
        Args:
            search_radius: Search within this radius of player
            
        Returns:
            List of (x, y) frontier positions
        """
        frontiers = []
        px, py = self.player_world_pos
        
        # Search in radius around player
        for y in range(py - search_radius, py + search_radius):
            for x in range(px - search_radius, px + search_radius):
                # Skip if out of bounds
                if not (0 <= x < self.map_size and 0 <= y < self.map_size):
                    continue
                
                # Must be free cell
                cell_type = self.get_cell(x, y)
                if cell_type != CellType.FREE:
                    continue
                
                # Check if adjacent to unknown
                is_frontier = False
                for dy in [-1, 0, 1]:
                    for dx in [-1, 0, 1]:
                        if dx == 0 and dy == 0:
                            continue
                        nx, ny = x + dx, y + dy
                        if not self.is_explored(nx, ny):
                            is_frontier = True
                            break
                    if is_frontier:
                        break
                
                if is_frontier:
                    frontiers.append((x, y))
        
        return frontiers
    
    def find_likely_exits(self, search_radius: int = 30) -> list[Tuple[int, int]]:
        """
        Find positions that look like exits (long corridors, edges).
        
        Args:
            search_radius: Search within this radius of player
            
        Returns:
            List of (x, y) positions ranked by exit likelihood
        """
        exits = []
        px, py = self.player_world_pos
        
        # Look for free cells at the edge of explored area
        for y in range(py - search_radius, py + search_radius):
            for x in range(px - search_radius, px + search_radius):
                if not (0 <= x < self.map_size and 0 <= y < self.map_size):
                    continue
                
                cell_type = self.get_cell(x, y)
                if cell_type != CellType.FREE:
                    continue
                
                # Count unknown neighbors (edge of map = likely exit)
                unknown_count = 0
                for dy in [-1, 0, 1]:
                    for dx in [-1, 0, 1]:
                        if dx == 0 and dy == 0:
                            continue
                        nx, ny = x + dx, y + dy
                        if not self.is_explored(nx, ny):
                            unknown_count += 1
                
                # Exit candidates have many unknown neighbors
                if unknown_count >= 5:
                    dist = np.sqrt((x - px)**2 + (y - py)**2)
                    exits.append((x, y, unknown_count, dist))
        
        # Sort by unknown_count (descending), then distance (ascending)
        exits.sort(key=lambda e: (-e[2], e[3]))
        
        # Return positions only
        return [(x, y) for x, y, _, _ in exits[:10]]
    
    def save_map(self, zone_name: str = ""):
        """
        Save map to disk (JSON metadata + PNG visualization).
        
        Args:
            zone_name: Name of current zone
        """
        if not zone_name:
            zone_name = self.current_zone
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base_name = f"{zone_name}_{timestamp}"
        
        # Save metadata (JSON)
        metadata = {
            "zone": zone_name,
            "timestamp": timestamp,
            "map_size": self.map_size,
            "cell_count": len(self.cells),
            "player_pos": self.player_world_pos,
            "frame_count": self.frame_count,
            "pois": [
                {
                    "type": poi.poi_type,
                    "position": poi.position,
                    "label": poi.label,
                    "confidence": poi.confidence,
                    "frame_detected": poi.frame_detected,
                }
                for poi in self.pois
            ],
        }
        
        json_path = self.save_dir / f"{base_name}_metadata.json"
        with open(json_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        # Save visualization (PNG)
        vis = self.visualize()
        png_path = self.save_dir / f"{base_name}_map.png"
        cv2.imwrite(str(png_path), vis)
        
        if self.debug:
            print(f"[MapAccumulator] Saved map to {png_path}")
    
    def visualize(self, scale: int = 4) -> np.ndarray:
        """
        Create visualization of accumulated map.
        
        Args:
            scale: Upscaling factor for visibility (default 4x)
            
        Returns:
            BGR image of the map
        """
        # Create image centered on explored area
        if not self.cells:
            return np.zeros((100, 100, 3), dtype=np.uint8)
        
        # Find bounding box of explored cells
        xs = [x for x, y in self.cells.keys()]
        ys = [y for x, y in self.cells.keys()]
        min_x, max_x = min(xs), max(xs)
        min_y, max_y = min(ys), max(ys)
        
        # Add margin
        margin = 20
        min_x -= margin
        max_x += margin
        min_y -= margin
        max_y += margin
        
        w = max_x - min_x + 1
        h = max_y - min_y + 1
        
        # Create image
        img = np.zeros((h, w, 3), dtype=np.uint8)
        img[:] = (30, 30, 30)  # Dark gray background (unknown)
        
        # Draw cells
        for (gx, gy), cell in self.cells.items():
            x = gx - min_x
            y = gy - min_y
            
            if 0 <= y < h and 0 <= x < w:
                if cell.cell_type == CellType.FREE:
                    # Free space - light gray to white (by confidence)
                    intensity = min(255, 150 + cell.confidence * 10)
                    img[y, x] = (intensity, intensity, intensity)
                elif cell.cell_type == CellType.WALL:
                    # Walls - dark red to red
                    intensity = min(255, 100 + cell.confidence * 15)
                    img[y, x] = (0, 0, intensity)
        
        # Draw player position (cross marker)
        px = self.player_world_pos[0] - min_x
        py = self.player_world_pos[1] - min_y
        if 0 <= py < h and 0 <= px < w:
            # Draw cross instead of circle for better visibility
            size = 3
            cv2.line(img, (px-size, py), (px+size, py), (0, 255, 0), 1)
            cv2.line(img, (px, py-size), (px, py+size), (0, 255, 0), 1)
            cv2.circle(img, (px, py), 1, (255, 255, 255), -1)  # Center dot
        
        # Draw player path
        for i in range(1, len(self.player_path)):
            p1 = self.player_path[i-1]
            p2 = self.player_path[i]
            pt1 = (p1[0] - min_x, p1[1] - min_y)
            pt2 = (p2[0] - min_x, p2[1] - min_y)
            cv2.line(img, pt1, pt2, (0, 200, 0), 1)
        
        # Draw POIs
        poi_colors = {
            "npc": (255, 255, 0),      # Cyan
            "exit": (0, 165, 255),     # Orange
            "waypoint": (255, 0, 255), # Magenta
            "chest": (0, 215, 255),    # Gold
            "shrine": (203, 192, 255), # Pink
            "quest": (0, 0, 255),      # Red
        }
        
        for poi in self.pois:
            px = poi.position[0] - min_x
            py = poi.position[1] - min_y
            
            if 0 <= py < h and 0 <= px < w:
                color = poi_colors.get(poi.poi_type, (255, 255, 255))
                # Draw single pixel (will be visible after scale)
                img[py, px] = color
                # Optional: small label only if scaled enough
                if scale >= 4:
                    label_text = poi.label[:2].upper()
                    cv2.putText(img, label_text, (px + 2, py + 2),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.2, color, 1)
        
        # Scale up
        img = cv2.resize(img, (w * scale, h * scale), interpolation=cv2.INTER_NEAREST)
        
        return img
