"""
Abstract base class for static map localization methods.

Provides common interface and shared functionality for different
localization approaches (ECC, RANSAC, etc).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional, Tuple
import cv2
import numpy as np

from .minimap_edge_extractor import MinimapEdgeExtractor


class StaticMapLocalizerBase(ABC):
    """
    Abstract base class for static map localization.
    
    Defines common interface and shared functionality for different
    alignment methods (ECC, RANSAC, etc).
    """
    
    def __init__(
        self,
        static_map_path: Optional[Path] = None,
        debug: bool = False,
        output_dir: Optional[Path] = None
    ):
        """
        Initialize base localizer.
        
        Args:
            static_map_path: Path to static reference map
            debug: Enable debug output and visualizations
            output_dir: Directory to save debug images
        """
        self.debug = debug
        self.output_dir = Path(output_dir) if output_dir else None
        self.static_map_path = Path(static_map_path) if static_map_path else None
        self.static_map = None
        
        # Create output dir if needed
        if self.output_dir and not self.output_dir.exists():
            self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize common components
        self.edge_extractor = MinimapEdgeExtractor(
            debug=debug,
            output_dir=output_dir
        )
        
        # Load static map if provided
        if self.static_map_path:
            self.load_static_map(self.static_map_path)
        
        if self.debug:
            print(f"[{self.__class__.__name__}] Initialized (output_dir: {output_dir})")
    
    def load_static_map(self, map_path: Path) -> bool:
        """
        Load static reference map.
        
        Args:
            map_path: Path to static map image
            
        Returns:
            True if loaded successfully
        """
        try:
            self.static_map_path = Path(map_path)
            self.static_map = cv2.imread(str(map_path))
            
            if self.static_map is None:
                if self.debug:
                    print(f"[!] Failed to load static map: {map_path}")
                return False
            
            if self.debug:
                print(f"[+] Static map loaded: {self.static_map.shape}")
            
            return True
        
        except Exception as e:
            if self.debug:
                print(f"[!] Error loading static map: {e}")
            return False
    
    @abstractmethod
    def localize(
        self,
        frame_with_minimap: np.ndarray,
        frame_without_minimap: np.ndarray,
        **kwargs
    ) -> Tuple[Optional[Tuple[float, float]], float]:
        """
        Localize player position on static map.
        
        Abstract method - must be implemented by subclasses.
        
        Args:
            frame_with_minimap: Frame with minimap visible
            frame_without_minimap: Frame with minimap hidden
            **kwargs: Method-specific parameters
            
        Returns:
            ((player_x, player_y), confidence) or (None, 0.0)
        """
        pass
    
    def extract_minimap_edges(
        self,
        frame_with_minimap: np.ndarray,
        frame_without_minimap: np.ndarray,
        use_oriented_filter: bool = True
    ) -> Optional[np.ndarray]:
        """
        Extract minimap edges from frames.
        
        Common preprocessing step for all methods.
        
        Args:
            frame_with_minimap: Frame with minimap
            frame_without_minimap: Frame without minimap
            use_oriented_filter: Use Gabor filters for isometric detection
            
        Returns:
            Edge image or None
        """
        try:
            edges = self.edge_extractor.extract_full_pipeline(
                frame_with_minimap,
                frame_without_minimap,
                use_oriented_filter=use_oriented_filter
            )
            
            if edges is None or np.count_nonzero(edges) == 0:
                if self.debug:
                    print(f"[{self.__class__.__name__}] No edges detected in minimap")
                return None
            
            if self.debug:
                print(f"[+] Edges extracted: shape={edges.shape}, non-zero={np.count_nonzero(edges)}")
            
            return edges
        
        except Exception as e:
            if self.debug:
                print(f"[!] Error extracting edges: {e}")
            return None
    
    def extract_minimap_edges_canny(
        self,
        frame_with_minimap: np.ndarray,
        frame_without_minimap: np.ndarray,
        use_oriented_filter: bool = True,
        canny_low: int = 50,
        canny_high: int = 150
    ) -> Optional[np.ndarray]:
        """
        Extract minimap edges and apply Canny edge detection.
        
        Args:
            frame_with_minimap: Frame with minimap
            frame_without_minimap: Frame without minimap
            use_oriented_filter: Use Gabor filters for isometric detection
            canny_low: Canny lower threshold
            canny_high: Canny upper threshold
            
        Returns:
            Canny edge-detected image or None
        """
        edges = self.extract_minimap_edges(
            frame_with_minimap,
            frame_without_minimap,
            use_oriented_filter=use_oriented_filter
        )
        
        if edges is None:
            return None
        
        try:
            edges_canny = cv2.Canny(edges, canny_low, canny_high)
            if self.debug:
                print(f"[+] Applied Canny edge detection")
            return edges_canny
        except Exception as e:
            if self.debug:
                print(f"[!] Error in Canny edge detection: {e}")
            return None
    
    def extract_static_map_edges_canny(
        self,
        white_threshold: int = 200,
        canny_low: int = 50,
        canny_high: int = 150
    ) -> Optional[np.ndarray]:
        """
        Extract static map edges with Canny detection.
        
        Uses Gabor filters + Canny for better edge quality, similar to minimap extraction.
        
        Args:
            white_threshold: Threshold for white pixel filtering (disabled if 0)
            canny_low: Canny lower threshold
            canny_high: Canny upper threshold
            
        Returns:
            Canny edge-detected static map or None
        """
        if self.static_map is None:
            if self.debug:
                print("[!] Static map not loaded")
            return None
        
        try:
            # Convert to grayscale if needed
            if len(self.static_map.shape) == 3:
                static_gray = cv2.cvtColor(self.static_map, cv2.COLOR_BGR2GRAY)
            else:
                static_gray = self.static_map
            
            # Option 1: Apply white pixel filtering if threshold > 0
            if white_threshold > 0:
                ret, static_map_white = cv2.threshold(static_gray, white_threshold, 255, cv2.THRESH_BINARY)
                edges_input = static_map_white
            else:
                edges_input = static_gray
            
            # Option 2: Also try Gabor-like preprocessing for better features
            # Apply Canny with multiple passes (like standard edge detection)
            edges1 = cv2.Canny(edges_input, canny_low, canny_high)
            
            # Dilate slightly to connect nearby edges
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
            edges1 = cv2.dilate(edges1, kernel, iterations=1)
            
            if self.debug:
                print(f"[+] Applied Canny + dilation edge detection")
            
            return edges1
        
        except Exception as e:
            if self.debug:
                print(f"[!] Error in static map edge detection: {e}")
            return None
    
    def visualize_player_position(
        self,
        player_pos: Tuple[float, float],
        confidence: float,
        method_name: str = "Localization",
        output_filename: str = "player_position_on_map.png"
    ) -> Optional[Path]:
        """
        Visualize player position on the static map.
        
        Draws a marker at the found position with confidence indicator.
        Position is in static map coordinates.
        
        Args:
            player_pos: (x, y) position in static map coordinates
            confidence: Confidence score (0.0-1.0)
            method_name: Name of the localization method (for label)
            output_filename: Output filename for visualization
            
        Returns:
            Path to saved visualization or None
        """
        if self.static_map is None:
            if self.debug:
                print("[!] Static map not loaded, cannot visualize")
            return None
        
        if player_pos is None:
            if self.debug:
                print("[!] Player position is None, cannot visualize")
            return None
        
        try:
            # Create a copy to draw on
            result = self.static_map.copy()
            img_h, img_w = result.shape[:2]
            
            # Convert position to integers for drawing
            px, py = int(round(player_pos[0])), int(round(player_pos[1]))
            
            # Check if position is within bounds
            if px < 0 or px >= img_w or py < 0 or py >= img_h:
                if self.debug:
                    print(f"[!] Position ({px}, {py}) out of bounds ({img_w}x{img_h})")
                return None
            
            # Calculate color based on confidence
            # Red (low confidence) to Green (high confidence)
            confidence_clamped = max(0.0, min(1.0, confidence))
            if confidence_clamped < 0.5:
                # Red to Yellow (0.0 to 0.5)
                color = (0, int(255 * (confidence_clamped * 2)), 255)  # BGR format
            else:
                # Yellow to Green (0.5 to 1.0)
                color = (0, 255, int(255 * (1 - (confidence_clamped - 0.5) * 2)))
            
            marker_size = 20
            line_thickness = 3
            
            # Draw crosshair
            cv2.line(
                result,
                (px - marker_size, py),
                (px + marker_size, py),
                color,
                line_thickness
            )
            cv2.line(
                result,
                (px, py - marker_size),
                (px, py + marker_size),
                color,
                line_thickness
            )
            
            # Draw circle around player position
            cv2.circle(result, (px, py), marker_size + 5, color, 2)
            
            # Add text info
            text_lines = [
                f"{method_name}: ({player_pos[0]:.1f}, {player_pos[1]:.1f})",
                f"Confidence: {confidence:.1%}"
            ]
            
            y_offset = 30
            for i, text in enumerate(text_lines):
                cv2.putText(
                    result,
                    text,
                    (10, y_offset + i * 25),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 0, 0),
                    1,
                    cv2.LINE_AA
                )
                cv2.putText(
                    result,
                    text,
                    (9, y_offset + i * 25 - 1),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    color,
                    2,
                    cv2.LINE_AA
                )
            
            # Save visualization
            return self._save_debug_image(result, output_filename.replace(".png", ""))
        
        except Exception as e:
            if self.debug:
                print(f"[!] Error visualizing player position: {e}")
                import traceback
                traceback.print_exc()
            return None
    
    def _save_debug_image(self, img: np.ndarray, name: str) -> Optional[Path]:
        """
        Save debug image with timestamp.
        
        Args:
            img: Image to save
            name: Base name (without extension)
            
        Returns:
            Path to saved image or None
        """
        if not self.output_dir or img is None or img.size == 0:
            return None
        
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{name}_{timestamp}.png"
        filepath = self.output_dir / filename
        
        try:
            cv2.imwrite(str(filepath), img)
            if self.debug:
                print(f"[+] Saved: {filepath.name}")
            return filepath
        except Exception as e:
            if self.debug:
                print(f"[!] Failed to save {name}: {e}")
            return None
