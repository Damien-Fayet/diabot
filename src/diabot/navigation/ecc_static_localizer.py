"""
ECC-based static map localization using multi-scale alignment.

Uses structural edge extraction and multi-scale ECC for robust
player localization with subpixel accuracy.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple, Literal
import cv2
import numpy as np

from .static_map_localizer_base import StaticMapLocalizerBase
from .ecc_localizer import StaticMapMatcher


class ECCStaticMapLocalizer(StaticMapLocalizerBase):
    """
    ECC-based static map localization using multi-scale alignment.
    
    Uses structural edge extraction + ECC to robustly localize player
    position on a static reference map, even with significant rotation
    or scale differences.
    """
    
    def __init__(
        self,
        static_map_path: Optional[Path] = None,
        debug: bool = False,
        output_dir: Optional[Path] = None
    ):
        """
        Initialize ECC localizer.
        
        Args:
            static_map_path: Path to static reference map
            debug: Enable debug output and visualizations
            output_dir: Directory to save debug images
        """
        super().__init__(static_map_path, debug, output_dir)
        
        # Initialize ECC-specific components
        self.matcher = StaticMapMatcher(
            debug=debug,
            output_dir=output_dir
        )
        
        # Share the static map loaded by base class
        if self.static_map is not None:
            self.matcher.static_map = self.static_map
            if self.debug:
                print("[+] Static map shared with matcher")
    
    def localize(
        self,
        frame_with_minimap: np.ndarray,
        frame_without_minimap: np.ndarray,
        motion_type: Literal['AFFINE', 'HOMOGRAPHY'] = 'HOMOGRAPHY',
        use_oriented_filter: bool = True,
        **kwargs
    ) -> Tuple[Optional[Tuple[int, int]], float]:
        """
        Localize player using ECC multi-scale alignment.
        
        Full pipeline:
        1. Extract minimap region (difference)
        2. Clean up (remove noise, NPCs)
        3. Extract structural edges (Gabor + morphology)
        4. Align with static map using ECC
        5. Project player center to static map
        
        Args:
            frame_with_minimap: Frame with minimap visible
            frame_without_minimap: Frame with minimap hidden
            motion_type: 'AFFINE' or 'HOMOGRAPHY'
            use_oriented_filter: Use Gabor filters for isometric detection
            **kwargs: Additional parameters (ignored)
            
        Returns:
            ((player_x, player_y), confidence) or (None, 0.0)
        """
        if self.static_map is None:
            if self.debug:
                print("[!] Static map not loaded")
            return None, 0.0
        
        if self.debug:
            print("\n" + "="*70)
            print("ECC STATIC MAP LOCALIZATION")
            print("="*70)
        
        try:
            # Step 1: Extract edge image from minimap
            if self.debug:
                print("\n[1/2] Extracting minimap edges...")
            
            edges = self.extract_minimap_edges(
                frame_with_minimap,
                frame_without_minimap,
                use_oriented_filter=use_oriented_filter
            )
            
            if edges is None:
                return None, 0.0
            
            # Step 2: Match with static map
            if self.debug:
                print(f"\n[2/2] Matching with static map (motion={motion_type})...")
            
            player_pos, confidence = self.matcher.match(
                edges,
                motion_type=motion_type
            )
            
            if player_pos is None:
                if self.debug:
                    print("[!] Matching failed")
                return None, 0.0
            
            if self.debug:
                print("[+] Localization successful!")
                print(f"    Position: {player_pos}")
                print(f"    Confidence: {confidence:.3f}")
            
            return player_pos, confidence
        
        except Exception as e:
            if self.debug:
                print(f"[!] Localization error: {e}")
                import traceback
                traceback.print_exc()
            return None, 0.0
    
    def load_static_map(self, map_path: Path) -> bool:
        """
        Override to sync static map with matcher.
        
        Args:
            map_path: Path to static map image
            
        Returns:
            True if loaded successfully
        """
        # Load using base class
        success = super().load_static_map(map_path)
        
        # Share with matcher
        if success and self.static_map is not None:
            self.matcher.static_map = self.static_map
            if self.debug:
                print("[+] Static map shared with matcher")
        
        return success
    
    def visualize_alignment(
        self,
        edges_img: np.ndarray,
        warp_matrix: np.ndarray,
        motion_type: Literal['AFFINE', 'HOMOGRAPHY'] = 'HOMOGRAPHY',
        output_filename: str = "alignment_result.png"
    ) -> Optional[Path]:
        """
        Visualize alignment result with player position marker.
        
        Args:
            edges_img: Edge image used for alignment
            warp_matrix: Transformation matrix from alignment
            motion_type: Type of transformation
            output_filename: Output filename
            
        Returns:
            Path to saved visualization (or None)
        """
        if self.matcher.static_map is None or not self.output_dir:
            return None
        
        # Project player position
        center = (edges_img.shape[1] / 2, edges_img.shape[0] / 2)
        player_pos = self.matcher.aligner.project_point(center, warp_matrix, motion_type)
        
        # Draw on static map
        result = self.matcher.static_map.copy()
        cv2.circle(result, player_pos, 15, (0, 0, 255), -1)
        cv2.circle(result, player_pos, 15, (255, 255, 255), 2)
        cv2.line(result, (player_pos[0]-30, player_pos[1]), (player_pos[0]+30, player_pos[1]), (0, 0, 255), 2)
        cv2.line(result, (player_pos[0], player_pos[1]-30), (player_pos[0], player_pos[1]+30), (0, 0, 255), 2)
        
        # Save
        output_path = self.output_dir / output_filename
        cv2.imwrite(str(output_path), result)
        
        if self.debug:
            print(f"[+] Visualization saved: {output_path}")
        
        return output_path
