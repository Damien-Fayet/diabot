"""Advanced vision module with real game detection."""

import cv2
import numpy as np

from diabot.core.interfaces import VisionModule, Perception


class DiabloVisionModule(VisionModule):
    """
    Vision module Diablo 2 : détection d’ennemis uniquement via YOLO.
    Les anciennes méthodes (couleurs, contours) sont supprimées.
    """

    def __init__(self, debug=False, yolo_model_path=None, yolo_conf=0.35):
        """Initialise le module de vision avec YOLO uniquement pour la détection d'ennemis."""
        self.debug = debug
        self.yolo_model_path = yolo_model_path or "runs/detect/runs/train/diablo-yolo3/weights/best.pt"
        self.yolo_conf = yolo_conf
        try:
            from ultralytics import YOLO
            self.yolo = YOLO(self.yolo_model_path)
        except ImportError:
            self.yolo = None
            if self.debug:
                print("[VISION] Erreur: ultralytics non installé, YOLO indisponible.")
    
    def perceive(self, frame: np.ndarray) -> Perception:
        """Détection du nombre d'ennemis et extraction des bounding boxes YOLO pour overlay."""
        enemy_count, enemy_types, yolo_boxes = self._detect_enemies_yolo(frame)
        items = []
        player_pos = None
        raw_data = {
            "frame_shape": frame.shape,
            "detection_method": "yolo_only",
            "yolo_boxes": yolo_boxes,
        }
        return Perception(
            hp_ratio=0.0,
            mana_ratio=0.0,
            enemy_count=enemy_count,
            enemy_types=enemy_types,
            visible_items=items,
            player_position=player_pos,
            current_zone="UNKNOWN",
            raw_data=raw_data,
        )

    def _detect_enemies_yolo(self, frame: np.ndarray) -> tuple[int, list[str], list]:
        """Détecte les ennemis sur le frame via YOLO uniquement et retourne les bounding boxes pour overlay."""
        if self.yolo is None:
            if self.debug:
                print("[VISION] YOLO non initialisé, aucun ennemi détecté.")
            return 0, [], []
        results = self.yolo.predict(source=frame, conf=self.yolo_conf, verbose=False)
        enemy_count = 0
        enemy_types = []
        yolo_boxes = []
        for box in results[0].boxes:
            cls_id = int(box.cls.item())
            conf = float(box.conf.item())
            xyxy = box.xyxy.cpu().numpy().astype(int).tolist()[0]
            # Récupère le vrai nom de classe YOLO
            class_name = self.yolo.names[cls_id] if hasattr(self.yolo, 'names') and cls_id < len(self.yolo.names) else f"class_{cls_id}"
            if cls_id == 0:
                enemy_count += 1
                enemy_types.append(class_name)
            yolo_boxes.append({
                "bbox": xyxy,
                "class_id": cls_id,
                "class_name": class_name,
                "confidence": conf,
            })
        return enemy_count, enemy_types, yolo_boxes
    

    
    # Suppression de la détection ennemis par couleur. À remplacer par YOLO uniquement.
    
    def _detect_items(self, frame: np.ndarray, hsv: np.ndarray) -> list[str]:
        """Detect items (golden/yellow highlights)."""
        h, w = frame.shape[:2]
        
        # Yellow/gold range in HSV
        yellow_mask = cv2.inRange(hsv, np.array([15, 100, 100]), np.array([35, 255, 255]))
        
        # Find contours
        contours, _ = cv2.findContours(yellow_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        
        items = []
        for contour in contours:
            area = cv2.contourArea(contour)
            # Filter by size
            if 20 < area < (h * w * 0.05):
                x, y, cw, ch = cv2.boundingRect(contour)
                # Skip if in UI area
                if y > h * 0.15:
                    items.append("item")
        
        return items[:5]  # Limit to 5 items
    
    def _estimate_player_position(self, frame: np.ndarray, hsv: np.ndarray) -> tuple[int, int]:
        """Estimate player position (usually center of playfield)."""
        h, w = frame.shape[:2]
        
        # Player is typically in the center of the playable area
        # Diablo 2 has isometric view, player is roughly at center
        player_x = w // 2
        player_y = h // 2
        
        # Slight bias upward as UI takes bottom space
        player_y = int(h * 0.45)
        
        return (player_x, player_y)


