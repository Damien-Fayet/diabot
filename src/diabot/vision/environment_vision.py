"""
Environment Vision Module - Detects and analyzes environment elements only.

Handles:
- Enemies (monsters)
- Items (ground loot)
- Obstacles (walls, pillars)
- Doors
- Traps
- Player position

Separated from UI vision for clarity and maintainability.
"""

from dataclasses import dataclass, field
from typing import List, Tuple
import cv2
import numpy as np

from .screen_regions import ENVIRONMENT_REGIONS


@dataclass
class EnemyInfo:
    """Information about a detected enemy."""
    
    enemy_type: str         # "small", "large", "champion", etc.
    position: Tuple[int, int]  # (x, y) center
    bbox: Tuple[int, int, int, int]  # (x, y, w, h)
    confidence: float       # 0.0-1.0
    color_hsv: Tuple[int, int, int] = None  # Dominant color


@dataclass
class EnvironmentState:
    """Current state of environment elements."""
    
    enemies: List[EnemyInfo] = field(default_factory=list)
    items: List[str] = field(default_factory=list)  # ["item", "item"]
    obstacles: List[dict] = field(default_factory=list)
    doors: List[dict] = field(default_factory=list)
    player_position: Tuple[int, int] = (512, 384)  # Estimated center


class EnvironmentVisionModule:
    """
    Detects and analyzes game environment elements.
    
    Focuses on playfield only, ignoring UI.
    """
    
    def __init__(self, debug: bool = False):
        """
        Initialize environment vision module.
        
        Args:
            debug: Enable debug output
        """
        self.debug = debug
        
        # HSV ranges for environment elements (different from UI!)
        self.enemy_red_range = (
            np.array([0, 100, 100]),       # Saturated red
            np.array([10, 255, 255]),
        )
        
        self.enemy_orange_range = (
            np.array([10, 100, 100]),      # Saturated orange
            np.array([25, 255, 255]),
        )
        
        self.item_gold_range = (
            np.array([15, 150, 150]),      # Saturated gold
            np.array([35, 255, 255]),
        )
    
    def analyze(self, frame: np.ndarray) -> EnvironmentState:
        """
        Analyze environment from frame.
        
        Args:
            frame: BGR image from game
            
        Returns:
            EnvironmentState with detected environment info
        """
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # Extract playfield region only
        playfield = ENVIRONMENT_REGIONS['playfield'].extract_from_frame(hsv)
        
        if playfield.size == 0:
            return EnvironmentState()
        
        # Detect elements
        enemies = self._detect_enemies(playfield, frame)
        items = self._detect_items(playfield)
        player_pos = self._estimate_player_position(frame)
        
        return EnvironmentState(
            enemies=enemies,
            items=items,
            player_position=player_pos,
        )
    
    def _detect_enemies(self, playfield_hsv: np.ndarray, original_frame: np.ndarray) -> List[EnemyInfo]:
        """Detect enemies in playfield with a more robust pipeline.

        Combines color, shadow positioning, size/aspect filters and edge density
        to reduce false positives (UI, décor, flammes).
        """
        enemies: List[EnemyInfo] = []

        # Extract playfield from BGR frame for edges/visual cues
        playfield_bgr = ENVIRONMENT_REGIONS['playfield'].extract_from_frame(original_frame)

        play_h, play_w = playfield_hsv.shape[:2]
        if play_h == 0 or play_w == 0:
            return enemies

        playfield_area = play_h * play_w

        # 1) Couleurs possibles des monstres (élargies)
        color_ranges = {
            'red': (np.array([0, 100, 100]), np.array([10, 255, 255])),
            'orange': (np.array([10, 100, 100]), np.array([25, 255, 255])),
            'yellow': (np.array([25, 100, 100]), np.array([40, 255, 255])),
            'green': (np.array([40, 70, 80]), np.array([80, 255, 255])),
            'blue': (np.array([100, 80, 80]), np.array([130, 255, 255])),
            'purple': (np.array([130, 80, 80]), np.array([155, 255, 255])),
        }

        combined_color_mask = np.zeros(playfield_hsv.shape[:2], dtype=np.uint8)
        for lower, upper in color_ranges.values():
            mask = cv2.inRange(playfield_hsv, lower, upper)
            combined_color_mask = cv2.bitwise_or(combined_color_mask, mask)

        # 2) Ombres: V faible (zones sombres). Décor a aussi des ombres,
        # mais on exigera une couleur au-dessus et une forme cohérente.
        shadow_mask = cv2.inRange(playfield_hsv,
                                  np.array([0, 0, 0]),
                                  np.array([180, 255, 60]))

        # 3) Edges pour vérifier la structure (Canny + dilatation)
        gray = cv2.cvtColor(playfield_bgr, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 80, 180)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        edges_dilated = cv2.dilate(edges, kernel, iterations=1)

        # 4) Contours candidats issus des couleurs (moins de bruit que edges seuls)
        color_mask_clean = cv2.morphologyEx(combined_color_mask, cv2.MORPH_OPEN, kernel, iterations=1)
        contours, _ = cv2.findContours(color_mask_clean, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        min_size = 50
        max_size = playfield_area * 0.05  # 5% du playfield

        for contour in contours:
            area = cv2.contourArea(contour)
            if area < min_size or area > max_size:
                continue  # trop petit / trop grand

            x, y, w, h = cv2.boundingRect(contour)
            if h == 0 or w == 0:
                continue

            aspect_ratio = w / h
            if not 0.3 < aspect_ratio < 3.0:
                continue  # filtre décor/objets plats

            # Score couleur: proportion de pixels color_mask dans la bbox
            color_crop = combined_color_mask[y:y+h, x:x+w]
            color_ratio = cv2.countNonZero(color_crop) / (w * h)

            # Score ombre: on regarde juste sous la bbox (20% de sa hauteur)
            shadow_y1 = min(play_h, y + h)
            shadow_y2 = min(play_h, y + h + int(0.2 * h))
            shadow_crop = shadow_mask[shadow_y1:shadow_y2, x:x+w]
            shadow_area = (shadow_y2 - shadow_y1) * w if shadow_y2 > shadow_y1 else 1
            shadow_ratio = cv2.countNonZero(shadow_crop) / shadow_area

            # Score edges: densité d'edges dans la bbox
            edges_crop = edges_dilated[y:y+h, x:x+w]
            edge_ratio = cv2.countNonZero(edges_crop) / (w * h)

            # Score taille: plus la zone est proche d'une taille "typique", mieux c'est
            size_score = min(1.0, area / 800.0)

            # Score global (pondérations empiriques)
            final_score = (0.35 * color_ratio) + (0.35 * shadow_ratio) + (0.15 * edge_ratio) + (0.15 * size_score)

            if final_score < 0.08:
                # Trop faible: probablement décor / UI / flamme isolée
                continue

            enemy_type = "large_enemy" if area > 800 else "small_enemy"
            confidence = float(max(0.0, min(1.0, final_score * 2)))  # échelle simple 0-1

            enemies.append(EnemyInfo(
                enemy_type=enemy_type,
                position=(x + w // 2, y + h // 2),
                bbox=(x, y, w, h),
                confidence=confidence,
            ))

        # Cap par sécurité
        return enemies[:20]
    
    def _detect_items(self, playfield_hsv: np.ndarray) -> List[str]:
        """
        Detect items on ground.
        
        Returns:
            List of item indicators
        """
        items = []
        
        # Gold/yellow items
        gold_mask = cv2.inRange(playfield_hsv,
                               np.array([15, 150, 150]),
                               np.array([35, 255, 255]))
        
        contours, _ = cv2.findContours(gold_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        
        playfield_area = playfield_hsv.shape[0] * playfield_hsv.shape[1]
        
        for contour in contours:
            area = cv2.contourArea(contour)
            
            # Filter by size: between 20 and 2% of playfield
            if 20 < area < (playfield_area * 0.02):
                items.append("item")
        
        return items[:10]  # Cap at 10 items
    
    def _estimate_player_position(self, frame: np.ndarray) -> Tuple[int, int]:
        """
        Estimate player position.
        
        In Diablo 2 isometric view, player is typically near center-bottom
        of playfield.
        
        Returns:
            (x, y) estimated position
        """
        h, w = frame.shape[:2]
        
        # Playfield is roughly center of screen
        # Player is typically slightly above center due to isometric view
        player_x = w // 2
        player_y = int(h * 0.5)  # Slightly above center
        
        return (player_x, player_y)
