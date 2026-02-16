"""
Minimap edge extraction for localization.

Extracts the edge structure from minimap difference images
to use as query for static map alignment.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple
import cv2
import numpy as np

from .image_preprocessing import (
    OrientedFilterBank,
    AdaptiveThresholdProcessor,
    OrientedMorphology,
)


class MinimapEdgeExtractor:
    """Extract structural edges from minimap for localization."""
    
    def __init__(self, debug: bool = False, output_dir: Optional[Path] = None):
        """
        Initialize extractor.
        
        Args:
            debug: Enable debug output
            output_dir: Directory to save debug images
        """
        self.debug = debug
        self.output_dir = Path(output_dir) if output_dir else None
        self.filter_bank = OrientedFilterBank(angles=[60, 120])
    
    def extract_difference(
        self,
        frame_with_minimap: np.ndarray,
        frame_without_minimap: np.ndarray
    ) -> np.ndarray:
        """
        Extract minimap region as difference between two frames.
        
        Args:
            frame_with_minimap: Frame with minimap visible
            frame_without_minimap: Frame with minimap hidden (background)
            
        Returns:
            Grayscale difference image
        """
        diff = cv2.subtract(frame_with_minimap, frame_without_minimap)
        if len(diff.shape) == 3:
            diff_gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
        else:
            diff_gray = diff
        
        if self.debug and self.output_dir:
            self._save_debug("01_diff_raw", diff_gray)
        
        return diff_gray
    
    def process_difference(
        self,
        diff_image: np.ndarray,
        brightness_min: int = 20,
        npc_threshold: int = 150
    ) -> np.ndarray:
        """
        Clean up difference image (remove noise, NPCs).
        
        Args:
            diff_image: Grayscale difference image
            brightness_min: Minimum brightness threshold
            npc_threshold: NPC cross brightness threshold
            
        Returns:
            Cleaned grayscale image
        """
        processed, npc_mask = AdaptiveThresholdProcessor.process(
            diff_image,
            brightness_min=brightness_min,
            npc_threshold=npc_threshold
        )
        
        if self.debug and self.output_dir:
            self._save_debug("02_processed", processed)
            self._save_debug("02_npc_mask", npc_mask)
        
        return processed
    
    def extract_edges(
        self,
        processed_image: np.ndarray,
        use_oriented_filter: bool = True
    ) -> np.ndarray:
        """
        Extract structural edges from processed image.
        
        Optionally uses Gabor filters for isometric angle detection.
        
        Args:
            processed_image: Processed difference image
            use_oriented_filter: Use Gabor filters for isometric angles
            
        Returns:
            Edge image (uint8)
        """
        if use_oriented_filter:
            # Apply oriented Gabor filters
            oriented = self.filter_bank.apply(processed_image)
            if self.debug and self.output_dir:
                self._save_debug("03_gabor_filtered", oriented)
            
            # Threshold and thicken
            _, thresholded = cv2.threshold(oriented, 40, 255, cv2.THRESH_BINARY)
            blurred = cv2.GaussianBlur(thresholded, (5, 5), 0)
            
            # Dilate to thicken lines
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            edges = cv2.dilate(blurred, kernel, iterations=1)
            
            if self.debug and self.output_dir:
                self._save_debug("03a_thickened_edges", edges)
        else:
            # Simple Canny edge detection
            edges = cv2.Canny(processed_image, 50, 150)
            if self.debug and self.output_dir:
                self._save_debug("03_canny_edges", edges)
        
        return edges
    
    def extract_full_pipeline(
        self,
        frame_with_minimap: np.ndarray,
        frame_without_minimap: np.ndarray,
        use_oriented_filter: bool = True
    ) -> np.ndarray:
        """
        Run full extraction pipeline: difference -> clean -> edges.
        
        Args:
            frame_with_minimap: Frame with minimap
            frame_without_minimap: Frame without minimap
            use_oriented_filter: Use oriented Gabor filters
            
        Returns:
            Final edge image
        """
        diff = self.extract_difference(frame_with_minimap, frame_without_minimap)
        processed = self.process_difference(diff)
        edges = self.extract_edges(processed, use_oriented_filter)
        return edges
    
    def _save_debug(self, name: str, img: np.ndarray) -> None:
        """Save debug image."""
        if not self.output_dir:
            return
        
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{name}_{timestamp}.png"
        filepath = self.output_dir / filename
        cv2.imwrite(str(filepath), img)
        if self.debug:
            print(f"   Saved: {filename}")
