"""
Minimap SLAM (Simultaneous Localization and Mapping) for Diablo II Resurrected.

This module implements a 2D SLAM-like system using ONLY the minimap.
No in-game coordinates, memory access, or physics engine are used.

CORE CONCEPT:
- Player position is FIXED in our internal reference frame
- The WORLD moves around the player
- Movement is inferred by comparing consecutive minimap images

PIPELINE:
1. Extract and preprocess minimap (skeletonize walls)
2. Estimate relative motion between frames (cv2.phaseCorrelate)
3. Update world offset (not player position)
4. Fuse local walls into global occupancy grid
5. Track points of interest (POIs) in global coordinates
6. Detect loop closures to correct drift
7. Support multi-level maps (stairs/teleports)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntEnum
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import json
import hashlib
from datetime import datetime

import cv2
import numpy as np
from scipy.spatial.distance import cdist


class OccupancyCell(IntEnum):
    """Cell types in global occupancy grid."""
    UNKNOWN = -1
    FREE = 0
    WALL = 1


@dataclass
class POI:
    """
    Point of Interest in global coordinates.
    
    Attributes:
        poi_type: Type (npc, exit, stairs, waypoint, shrine, etc.)
        pos: (gx, gy) in global map coordinates
        confidence: Detection confidence (higher = more reliable)
        first_seen: Frame number when first detected
        last_seen: Frame number when last observed
        metadata: Additional information (e.g., NPC name, exit destination)
    """
    poi_type: str
    pos: Tuple[int, int]
    confidence: float = 1.0
    first_seen: int = 0
    last_seen: int = 0
    metadata: Dict[str, any] = field(default_factory=dict)


@dataclass
class MapSignature:
    """
    Compact signature of local minimap for loop closure detection.
    
    Used to recognize previously visited locations and correct drift.
    """
    frame_id: int
    world_offset: Tuple[int, int]
    wall_hash: str  # Hash of wall configuration
    orientation_hist: np.ndarray  # Histogram of wall orientations
    intersection_count: int  # Number of wall intersections
    
    def similarity(self, other: MapSignature) -> float:
        """
        Compute similarity score with another signature.
        
        Returns:
            Similarity score 0.0-1.0 (1.0 = identical)
        """
        # Hash match is strongest signal
        if self.wall_hash == other.wall_hash:
            return 1.0
        
        # Compare histograms
        hist_diff = np.sum(np.abs(self.orientation_hist - other.orientation_hist))
        hist_sim = 1.0 - (hist_diff / (len(self.orientation_hist) * 2))
        
        # Compare intersection count
        int_diff = abs(self.intersection_count - other.intersection_count)
        int_sim = 1.0 - min(1.0, int_diff / max(1, max(self.intersection_count, other.intersection_count)))
        
        # Weighted combination
        return 0.5 * hist_sim + 0.5 * int_sim


@dataclass
class Level:
    """
    Single level/zone in multi-level map.
    
    Each level has its own occupancy grid and POI list.
    Levels are connected via transitions (stairs, portals).
    """
    level_id: str
    occupancy_grid: np.ndarray  # [H, W] array of OccupancyCell values
    pois: List[POI] = field(default_factory=list)
    transitions: Dict[str, Tuple[int, int]] = field(default_factory=dict)  # {target_level_id: (gx, gy)}
    created_at: int = 0  # Frame number when created


class MinimapSLAM:
    """
    2D SLAM system for Diablo II using only minimap visual information.
    
    Features:
    - Visual odometry (motion estimation from minimap)
    - Global occupancy grid (walls, free space, unknown)
    - POI tracking (NPCs, exits, stairs)
    - Loop closure detection (drift correction)
    - Multi-level support (stairs/portals)
    - Persistence (save/load maps)
    
    Coordinate System:
    - Player is always at center of global map (FIXED)
    - World moves relative to player
    - world_offset tracks cumulative world displacement
    - Global coordinates: (gx, gy) = player_center + world_offset + local_offset
    """
    
    def __init__(
        self,
        map_size: int = 4096,
        player_center: Optional[Tuple[int, int]] = None,
        movement_threshold: float = 2.0,
        loop_closure_threshold: float = 0.85,
        signature_interval: int = 10,
        debug: bool = False,
        save_dir: Optional[Path] = None
    ):
        """
        Initialize SLAM system.
        
        Args:
            map_size: Size of global map grid (map_size x map_size)
            player_center: Player position in global map (default: center)
            movement_threshold: Minimum motion (pixels) to count as movement
            loop_closure_threshold: Similarity threshold for loop closure
            signature_interval: Frames between signature captures
            debug: Enable debug output
            save_dir: Directory for saving maps
        """
        self.map_size = map_size
        self.player_center = player_center or (map_size // 2, map_size // 2)
        self.movement_threshold = movement_threshold
        self.loop_closure_threshold = loop_closure_threshold
        self.signature_interval = signature_interval
        self.debug = debug
        
        # Save directory
        if save_dir is None:
            save_dir = Path("data/slam_maps")
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        # Frame counter (initialize early - needed by _init_level)
        self.frame_count = 0
        
        # World offset (cumulative displacement)
        # When player moves, world_offset changes in OPPOSITE direction
        self.world_offset_x = 0
        self.world_offset_y = 0
        
        # Previous minimap for motion estimation
        self.prev_minimap: Optional[np.ndarray] = None
        self.prev_skeleton: Optional[np.ndarray] = None
        
        # Multi-level support
        self.levels: Dict[str, Level] = {}
        self.current_level_id = "level_0"
        self._init_level(self.current_level_id)
        
        # Signature history for loop closure
        self.signatures: List[MapSignature] = []
        
        # Statistics
        self.total_movement = 0.0
        self.loop_closures = 0
        
        if self.debug:
            print(f"[MinimapSLAM] Initialized {map_size}x{map_size} map")
            print(f"[MinimapSLAM] Player fixed at {self.player_center}")
            print(f"[MinimapSLAM] World offset starts at (0, 0)")
    
    def _init_level(self, level_id: str):
        """Initialize a new level."""
        self.levels[level_id] = Level(
            level_id=level_id,
            occupancy_grid=np.full((self.map_size, self.map_size), OccupancyCell.UNKNOWN, dtype=np.int8),
            created_at=self.frame_count
        )
        if self.debug:
            print(f"[MinimapSLAM] Created level: {level_id}")
    
    @property
    def current_level(self) -> Level:
        """Get current level."""
        return self.levels[self.current_level_id]
    
    @property
    def global_map(self) -> np.ndarray:
        """Get current level's occupancy grid."""
        return self.current_level.occupancy_grid
    
    def preprocess_minimap(self, minimap: np.ndarray) -> np.ndarray:
        """
        Preprocess minimap to extract wall skeleton.
        
        Pipeline:
        1. Convert to grayscale
        2. Enhance contrast (CLAHE)
        3. Threshold to isolate walls
        4. Morphological cleanup
        5. Skeletonize walls to 1-pixel width
        
        Args:
            minimap: Raw minimap image (BGR)
            
        Returns:
            Binary skeleton image (walls = 255, free = 0)
        """
        # Convert to grayscale
        if len(minimap.shape) == 3:
            gray = cv2.cvtColor(minimap, cv2.COLOR_BGR2GRAY)
        else:
            gray = minimap.copy()
        
        # Enhance contrast
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)
        
        # Threshold: walls are typically light colored
        _, binary = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Morphological cleanup
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        morph = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
        morph = cv2.morphologyEx(morph, cv2.MORPH_CLOSE, kernel)
        
        # Skeletonize
        skeleton = cv2.ximgproc.thinning(morph)
        
        return skeleton
    
    def estimate_motion(
        self,
        current_minimap: np.ndarray,
        prev_minimap: np.ndarray
    ) -> Tuple[float, float, float]:
        """
        Estimate relative motion between two minimap frames.
        
        Uses phase correlation for sub-pixel accuracy.
        
        Args:
            current_minimap: Current minimap skeleton
            prev_minimap: Previous minimap skeleton
            
        Returns:
            (dx, dy, confidence) where:
                dx: X displacement in pixels
                dy: Y displacement in pixels
                confidence: Correlation confidence (0-1)
        """
        # Ensure same size
        if current_minimap.shape != prev_minimap.shape:
            prev_minimap = cv2.resize(prev_minimap, (current_minimap.shape[1], current_minimap.shape[0]))
        
        # Convert to float32 for phase correlation
        curr_float = np.float32(current_minimap)
        prev_float = np.float32(prev_minimap)
        
        # Phase correlation
        (dx, dy), confidence = cv2.phaseCorrelate(prev_float, curr_float)
        
        return dx, dy, confidence
    
    def update_world_offset(self, dx: float, dy: float):
        """
        Update world offset based on detected motion.
        
        IMPORTANT: World moves in OPPOSITE direction of player.
        If minimap shifted right (+dx), world moved left (-dx).
        
        Args:
            dx: X motion in pixels (positive = minimap shifted right)
            dy: Y motion in pixels (positive = minimap shifted down)
        """
        # World moves opposite to minimap shift
        self.world_offset_x -= int(dx)
        self.world_offset_y -= int(dy)
        
        # Update statistics
        motion_magnitude = np.sqrt(dx*dx + dy*dy)
        self.total_movement += motion_magnitude
        
        if self.debug and motion_magnitude > self.movement_threshold:
            print(f"[MinimapSLAM] Motion detected: dx={dx:.2f}, dy={dy:.2f}")
            print(f"[MinimapSLAM] World offset now: ({self.world_offset_x}, {self.world_offset_y})")
    
    def fuse_minimap_to_global(self, minimap_skeleton: np.ndarray):
        """
        Fuse local minimap walls into global occupancy grid.
        
        For each wall pixel in minimap:
        1. Convert to global coordinates using world_offset
        2. Update global_map cell
        
        Rules:
        - WALL is never overwritten (persistent)
        - FREE overwrites UNKNOWN
        - Multiple observations increase confidence
        
        Args:
            minimap_skeleton: Binary skeleton of walls
        """
        h, w = minimap_skeleton.shape
        cx, cy = w // 2, h // 2  # Minimap center (player position)
        
        global_map = self.global_map
        pcx, pcy = self.player_center
        
        for y in range(h):
            for x in range(w):
                # Local offset from minimap center
                local_x = x - cx
                local_y = y - cy
                
                # Global coordinates
                gx = pcx + self.world_offset_x + local_x
                gy = pcy + self.world_offset_y + local_y
                
                # Bounds check
                if not (0 <= gx < self.map_size and 0 <= gy < self.map_size):
                    continue
                
                # Determine cell type
                if minimap_skeleton[y, x] > 127:
                    # Wall detected
                    cell_type = OccupancyCell.WALL
                else:
                    # Free space
                    cell_type = OccupancyCell.FREE
                
                # Update global map
                current_type = global_map[gy, gx]
                
                # Rule: WALL is sticky (never overwritten)
                if current_type == OccupancyCell.WALL:
                    continue
                
                # Rule: FREE overwrites UNKNOWN
                if current_type == OccupancyCell.UNKNOWN and cell_type == OccupancyCell.FREE:
                    global_map[gy, gx] = OccupancyCell.FREE
                
                # Rule: WALL overwrites anything
                if cell_type == OccupancyCell.WALL:
                    global_map[gy, gx] = OccupancyCell.WALL
    
    def detect_blocked_movement(self, expected_dx: float, expected_dy: float, actual_dx: float, actual_dy: float) -> bool:
        """
        Detect if movement was blocked (collision with wall).
        
        Args:
            expected_dx, expected_dy: Expected movement from input
            actual_dx, actual_dy: Actual movement from vision
            
        Returns:
            True if movement was blocked
        """
        expected_mag = np.sqrt(expected_dx**2 + expected_dy**2)
        actual_mag = np.sqrt(actual_dx**2 + actual_dy**2)
        
        # Movement blocked if actual << expected
        if expected_mag > self.movement_threshold and actual_mag < self.movement_threshold:
            if self.debug:
                print(f"[MinimapSLAM] Movement blocked! Expected {expected_mag:.1f}, got {actual_mag:.1f}")
            return True
        
        return False
    
    def capture_signature(self, minimap_skeleton: np.ndarray) -> MapSignature:
        """
        Capture a compact signature of current minimap for loop closure.
        
        Args:
            minimap_skeleton: Binary skeleton of walls
            
        Returns:
            MapSignature object
        """
        # Hash of wall configuration
        wall_hash = hashlib.md5(minimap_skeleton.tobytes()).hexdigest()
        
        # Orientation histogram (using Sobel gradients)
        if minimap_skeleton.max() > 0:
            sobelx = cv2.Sobel(minimap_skeleton, cv2.CV_64F, 1, 0, ksize=3)
            sobely = cv2.Sobel(minimap_skeleton, cv2.CV_64F, 0, 1, ksize=3)
            angles = np.arctan2(sobely, sobelx)
            hist, _ = np.histogram(angles, bins=8, range=(-np.pi, np.pi))
            hist = hist / (hist.sum() + 1e-6)  # Normalize
        else:
            hist = np.zeros(8)
        
        # Count intersections (pixels with multiple neighbors)
        kernel = np.ones((3, 3), np.uint8)
        neighbors = cv2.filter2D(minimap_skeleton // 255, -1, kernel)
        intersections = np.sum(neighbors > 3)
        
        return MapSignature(
            frame_id=self.frame_count,
            world_offset=(self.world_offset_x, self.world_offset_y),
            wall_hash=wall_hash,
            orientation_hist=hist,
            intersection_count=intersections
        )
    
    def detect_loop_closure(self, current_sig: MapSignature) -> Optional[MapSignature]:
        """
        Detect if we've returned to a previously visited location.
        
        Args:
            current_sig: Current map signature
            
        Returns:
            Matching signature if loop closure detected, else None
        """
        if len(self.signatures) < 2:
            return None
        
        # Compare with all previous signatures (except very recent)
        for prev_sig in self.signatures[:-5]:  # Skip last 5 frames
            similarity = current_sig.similarity(prev_sig)
            
            if similarity > self.loop_closure_threshold:
                if self.debug:
                    print(f"[MinimapSLAM] Loop closure detected! Similarity={similarity:.3f}")
                    print(f"[MinimapSLAM] Matched frame {prev_sig.frame_id} (current: {current_sig.frame_id})")
                
                self.loop_closures += 1
                return prev_sig
        
        return None
    
    def correct_drift(self, current_sig: MapSignature, matched_sig: MapSignature):
        """
        Correct accumulated drift using loop closure.
        
        When we detect we've returned to a known location, adjust world_offset
        to align with the previous observation.
        
        Args:
            current_sig: Current signature
            matched_sig: Matched previous signature
        """
        # Calculate offset error
        curr_ox, curr_oy = current_sig.world_offset
        match_ox, match_oy = matched_sig.world_offset
        
        error_x = curr_ox - match_ox
        error_y = curr_oy - match_oy
        
        # Apply correction (partial to avoid jarring jumps)
        correction_factor = 0.5
        self.world_offset_x -= int(error_x * correction_factor)
        self.world_offset_y -= int(error_y * correction_factor)
        
        if self.debug:
            print(f"[MinimapSLAM] Drift corrected: ({error_x}, {error_y}) -> applied {correction_factor*100:.0f}%")
    
    def add_poi(self, poi_type: str, local_pos: Tuple[int, int], confidence: float = 1.0, metadata: Dict = None):
        """
        Add a point of interest in global coordinates.
        
        Args:
            poi_type: Type of POI (npc, exit, stairs, etc.)
            local_pos: (x, y) in local minimap coordinates
            confidence: Detection confidence
            metadata: Additional information
        """
        # Convert local position to global
        lx, ly = local_pos
        h, w = 200, 200  # Assume standard minimap size
        cx, cy = w // 2, h // 2
        
        local_offset_x = lx - cx
        local_offset_y = ly - cy
        
        gx = self.player_center[0] + self.world_offset_x + local_offset_x
        gy = self.player_center[1] + self.world_offset_y + local_offset_y
        
        # Check if POI already exists nearby
        for poi in self.current_level.pois:
            dist = np.sqrt((poi.pos[0] - gx)**2 + (poi.pos[1] - gy)**2)
            if dist < 10 and poi.poi_type == poi_type:
                # Update existing POI
                poi.confidence = min(1.0, poi.confidence + 0.1)
                poi.last_seen = self.frame_count
                if self.debug:
                    print(f"[MinimapSLAM] Updated POI: {poi_type} at ({gx}, {gy})")
                return
        
        # Add new POI
        poi = POI(
            poi_type=poi_type,
            pos=(gx, gy),
            confidence=confidence,
            first_seen=self.frame_count,
            last_seen=self.frame_count,
            metadata=metadata or {}
        )
        self.current_level.pois.append(poi)
        
        if self.debug:
            print(f"[MinimapSLAM] Added POI: {poi_type} at ({gx}, {gy})")
    
    def detect_level_change(self, current_minimap: np.ndarray, prev_minimap: np.ndarray) -> bool:
        """
        Detect if we've changed levels (stairs, portal, teleport).
        
        Indicators:
        - Very low overlap between consecutive minimaps
        - Sudden drastic change in visible area
        
        Args:
            current_minimap: Current minimap
            prev_minimap: Previous minimap
            
        Returns:
            True if level change detected
        """
        if prev_minimap is None:
            return False
        
        # Compute pixel difference
        if current_minimap.shape != prev_minimap.shape:
            prev_minimap = cv2.resize(prev_minimap, (current_minimap.shape[1], current_minimap.shape[0]))
        
        diff = cv2.absdiff(current_minimap, prev_minimap)
        change_ratio = np.sum(diff > 50) / diff.size
        
        # Threshold for level change
        if change_ratio > 0.7:
            if self.debug:
                print(f"[MinimapSLAM] Level change detected! Change ratio: {change_ratio:.2f}")
            return True
        
        return False
    
    def switch_level(self, new_level_id: str, transition_type: str = "stairs"):
        """
        Switch to a new level.
        
        Args:
            new_level_id: ID of new level
            transition_type: Type of transition (stairs, portal, waypoint)
        """
        # Store transition in current level
        current_pos = (
            self.player_center[0] + self.world_offset_x,
            self.player_center[1] + self.world_offset_y
        )
        self.current_level.transitions[new_level_id] = current_pos
        
        if self.debug:
            print(f"[MinimapSLAM] Switching from {self.current_level_id} to {new_level_id}")
        
        # Create new level if it doesn't exist
        if new_level_id not in self.levels:
            self._init_level(new_level_id)
        
        # Switch level
        old_level_id = self.current_level_id
        self.current_level_id = new_level_id
        
        # Reset world offset for new level
        self.world_offset_x = 0
        self.world_offset_y = 0
        
        # Store reverse transition
        self.current_level.transitions[old_level_id] = (
            self.player_center[0],
            self.player_center[1]
        )
        
        if self.debug:
            print(f"[MinimapSLAM] Now on level: {new_level_id}")
    
    def update(self, minimap: np.ndarray):
        """
        Main SLAM update loop.
        
        Pipeline:
        1. Preprocess minimap
        2. Estimate motion (if not first frame)
        3. Update world offset
        4. Fuse into global map
        5. Capture signature (periodically)
        6. Check loop closure
        7. Detect level changes
        
        Args:
            minimap: Raw minimap image from screen
        """
        self.frame_count += 1
        
        # Step 1: Preprocess
        skeleton = self.preprocess_minimap(minimap)
        
        # Step 2: Detect level change
        if self.prev_minimap is not None:
            if self.detect_level_change(skeleton, self.prev_skeleton):
                new_level_id = f"level_{len(self.levels)}"
                self.switch_level(new_level_id)
        
        # Step 3: Estimate motion
        if self.prev_skeleton is not None:
            dx, dy, confidence = self.estimate_motion(skeleton, self.prev_skeleton)
            
            # Only update if motion is significant
            motion_mag = np.sqrt(dx*dx + dy*dy)
            if motion_mag > self.movement_threshold:
                self.update_world_offset(dx, dy)
        
        # Step 4: Fuse into global map
        self.fuse_minimap_to_global(skeleton)
        
        # Step 5: Capture signature (periodically)
        if self.frame_count % self.signature_interval == 0:
            sig = self.capture_signature(skeleton)
            self.signatures.append(sig)
            
            # Step 6: Check loop closure
            matched = self.detect_loop_closure(sig)
            if matched is not None:
                self.correct_drift(sig, matched)
        
        # Store for next frame
        self.prev_minimap = minimap.copy()
        self.prev_skeleton = skeleton.copy()
    
    def get_visible_area(self, radius: int = 200) -> np.ndarray:
        """
        Get occupancy grid around player.
        
        Args:
            radius: Radius around player
            
        Returns:
            Cropped occupancy grid centered on player
        """
        cx, cy = self.player_center
        x1 = max(0, cx - radius)
        x2 = min(self.map_size, cx + radius)
        y1 = max(0, cy - radius)
        y2 = min(self.map_size, cy + radius)
        
        return self.global_map[y1:y2, x1:x2].copy()
    
    def render_map(self, scale: int = 1, show_pois: bool = True, show_player: bool = True) -> np.ndarray:
        """
        Render global map as image.
        
        Args:
            scale: Upscale factor for visualization
            show_pois: Draw POIs
            show_player: Draw player position
            
        Returns:
            BGR image of map
        """
        # Create visualization
        vis = np.zeros((self.map_size, self.map_size, 3), dtype=np.uint8)
        
        # Draw occupancy grid
        vis[self.global_map == OccupancyCell.WALL] = (200, 200, 200)  # White walls
        vis[self.global_map == OccupancyCell.FREE] = (50, 50, 50)     # Dark gray free
        vis[self.global_map == OccupancyCell.UNKNOWN] = (0, 0, 0)     # Black unknown
        
        # Draw POIs
        if show_pois:
            for poi in self.current_level.pois:
                gx, gy = poi.pos
                if 0 <= gx < self.map_size and 0 <= gy < self.map_size:
                    color = self._poi_color(poi.poi_type)
                    cv2.circle(vis, (gx, gy), 5, color, -1)
        
        # Draw player
        if show_player:
            px, py = self.player_center
            cv2.circle(vis, (px, py), 8, (0, 255, 255), -1)  # Yellow player
            cv2.circle(vis, (px, py), 8, (0, 0, 0), 2)
        
        # Scale up for visibility
        if scale > 1:
            vis = cv2.resize(vis, None, fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)
        
        return vis
    
    def _poi_color(self, poi_type: str) -> Tuple[int, int, int]:
        """Get color for POI type."""
        colors = {
            'npc': (255, 0, 0),        # Blue
            'exit': (0, 255, 0),       # Green
            'stairs': (0, 165, 255),   # Orange
            'waypoint': (255, 255, 0), # Cyan
            'shrine': (255, 0, 255),   # Magenta
            'chest': (0, 255, 255),    # Yellow
        }
        return colors.get(poi_type, (128, 128, 128))
    
    def save_map(self, filename: Optional[str] = None):
        """
        Save map to disk.
        
        Args:
            filename: Optional filename (default: auto-generated)
        """
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"slam_map_{timestamp}.npz"
        
        save_path = self.save_dir / filename
        
        # Save occupancy grids
        level_data = {
            f"{lid}_grid": level.occupancy_grid
            for lid, level in self.levels.items()
        }
        
        # Save metadata
        metadata = {
            'map_size': self.map_size,
            'player_center': self.player_center,
            'world_offset': (self.world_offset_x, self.world_offset_y),
            'current_level': self.current_level_id,
            'frame_count': self.frame_count,
            'total_movement': self.total_movement,
            'loop_closures': self.loop_closures,
        }
        
        np.savez_compressed(
            save_path,
            metadata=metadata,
            **level_data
        )
        
        # Save POIs as JSON
        pois_data = {
            lid: [
                {
                    'type': poi.poi_type,
                    'pos': poi.pos,
                    'confidence': poi.confidence,
                    'metadata': poi.metadata
                }
                for poi in level.pois
            ]
            for lid, level in self.levels.items()
        }
        
        pois_path = save_path.with_suffix('.json')
        with open(pois_path, 'w') as f:
            json.dump(pois_data, f, indent=2)
        
        if self.debug:
            print(f"[MinimapSLAM] Map saved to {save_path}")
    
    def load_map(self, filename: str):
        """
        Load map from disk.
        
        Args:
            filename: Path to map file
        """
        load_path = self.save_dir / filename
        
        if not load_path.exists():
            raise FileNotFoundError(f"Map file not found: {load_path}")
        
        # Load occupancy grids
        data = np.load(load_path, allow_pickle=True)
        metadata = data['metadata'].item()
        
        # Restore metadata (except current_level_id - restore that after levels)
        self.map_size = metadata['map_size']
        self.player_center = tuple(metadata['player_center'])
        self.world_offset_x, self.world_offset_y = metadata['world_offset']
        self.frame_count = metadata['frame_count']
        self.total_movement = metadata['total_movement']
        self.loop_closures = metadata['loop_closures']
        
        # Restore levels first
        self.levels = {}
        for key in data.keys():
            if key.startswith('level_') and key.endswith('_grid'):
                level_id = key.replace('_grid', '')
                self._init_level(level_id)
                self.levels[level_id].occupancy_grid = data[key]
        
        # Now set current_level_id (after levels exist)
        self.current_level_id = metadata['current_level']
        
        # Load POIs
        pois_path = load_path.with_suffix('.json')
        if pois_path.exists():
            with open(pois_path, 'r') as f:
                pois_data = json.load(f)
            
            for lid, poi_list in pois_data.items():
                if lid in self.levels:
                    self.levels[lid].pois = [
                        POI(
                            poi_type=p['type'],
                            pos=tuple(p['pos']),
                            confidence=p['confidence'],
                            metadata=p.get('metadata', {})
                        )
                        for p in poi_list
                    ]
        
        if self.debug:
            print(f"[MinimapSLAM] Map loaded from {load_path}")
            print(f"[MinimapSLAM] {len(self.levels)} levels, {self.frame_count} frames")
    
    def get_stats(self) -> Dict:
        """Get SLAM statistics."""
        known_cells = np.sum(self.global_map != OccupancyCell.UNKNOWN)
        wall_cells = np.sum(self.global_map == OccupancyCell.WALL)
        free_cells = np.sum(self.global_map == OccupancyCell.FREE)
        
        return {
            'frames': self.frame_count,
            'levels': len(self.levels),
            'current_level': self.current_level_id,
            'world_offset': (self.world_offset_x, self.world_offset_y),
            'total_movement': self.total_movement,
            'loop_closures': self.loop_closures,
            'signatures': len(self.signatures),
            'known_cells': int(known_cells),
            'wall_cells': int(wall_cells),
            'free_cells': int(free_cells),
            'pois': len(self.current_level.pois),
        }
