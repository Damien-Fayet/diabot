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
from .phase_correlation_localizer import PhaseCorrelationLocalizer


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
        # Initialize phase correlation localizer (primary method)
        # MUST be done BEFORE super().__init__() because load_static_map() needs it
        self.phase_localizer = PhaseCorrelationLocalizer(
            debug=debug,
            output_dir=output_dir
        )
        
        # Initialize ECC matcher (fallback only)
        # MUST be done BEFORE super().__init__() because load_static_map() needs it
        self.matcher = StaticMapMatcher(
            debug=debug,
            output_dir=output_dir
        )
        
        # Now call parent init (which may call load_static_map)
        super().__init__(static_map_path, debug, output_dir)
        
        # Share the static map loaded by base class
        if self.static_map is not None:
            self.matcher.static_map = self.static_map
            if self.debug:
                print("[+] Static map shared with matcher")
    
    def localize(
        self,
        frame_with_minimap: np.ndarray,
        frame_without_minimap: np.ndarray,
        use_phase_correlation: bool = True,
        use_ecc_fallback: bool = False,
        motion_type: Literal['AFFINE', 'HOMOGRAPHY'] = 'HOMOGRAPHY',
        use_oriented_filter: bool = True,
        **kwargs
    ) -> Tuple[Optional[Tuple[int, int]], float]:
        """
        Localize player using phase correlation (or ECC fallback).
        
        Full pipeline:
        1. Extract minimap region (difference)
        2. Clean up (remove noise, NPCs)
        3. Extract structural edges (Gabor + morphology)
        4. Align with static map:
           - Default: Phase correlation (fast, translation only)
           - Fallback: ECC multi-scale (slower, supports rotation/scale)
        5. Return player position on static map
        
        Args:
            frame_with_minimap: Frame with minimap visible
            frame_without_minimap: Frame with minimap hidden
            use_phase_correlation: Use phase correlation (default: True)
            use_ecc_fallback: Fall back to ECC if phase correlation fails (default: False)
            motion_type: 'AFFINE' or 'HOMOGRAPHY' (for ECC only)
            use_oriented_filter: Use Gabor filters for isometric detection
            **kwargs: Additional parameters (ignored)
            
        Returns:
            ((player_x, player_y), confidence) or (None, 0.0)
        """
        if self.static_map is None:
            if self.debug:
                print("[!] Static map not loaded")
            return None, 0.0
        
        method = "PHASE CORRELATION" if use_phase_correlation else "ECC"
        
        if self.debug:
            print("\n" + "="*70)
            print(f"STATIC MAP LOCALIZATION ({method})")
            print("="*70)
        
        try:
            # Step 1: Extract edge images from minimap (both versions)
            if self.debug:
                print("\n[1/2] Extracting minimap edges...")
            
            edges_no_canny, edges_canny = self.extract_minimap_edges_canny(
                frame_with_minimap,
                frame_without_minimap,
                use_oriented_filter=use_oriented_filter,
                return_both=True
            )
            
            if edges_canny is None:
                return None, 0.0
            
            # Step 2: Remove NPC and color information
            if self.static_map is None:
                if self.debug:
                    print("[!] No static map loaded")
                return None, 0.0
            # Keep only white color from static map using threshold
            static_map_gray = cv2.cvtColor(self.static_map, cv2.COLOR_BGR2GRAY)
            _, static_map_white = cv2.threshold(static_map_gray, 200, 255, cv2.THRESH_BINARY)
       
            if self.debug and self.output_dir:
                cv2.imwrite(
                    str(self.output_dir / "static_map_white_extracted.png"),
                    static_map_white
                )
            # Step 3: Localize using chosen method
            if use_phase_correlation:
                if self.debug:
                    print("\n[2/2] Localizing with phase correlation...")
                
                # Use non-Canny edges for minimap (better for phase correlation)
                minimap_for_pc = edges_no_canny if edges_no_canny is not None else edges_canny
                
                player_pos, confidence = self.phase_localizer.localize(
                    minimap_for_pc,
                    static_map_white
                )
                
                # Fallback to ECC if requested and phase correlation failed
                if player_pos is None and use_ecc_fallback:
                    if self.debug:
                        print("\n[FALLBACK] Phase correlation failed, trying ECC...")
                    
                    player_pos, confidence = self.matcher.match(
                        edges_canny,
                        motion_type=motion_type
                    )
            else:
                # Direct ECC
                if self.debug:
                    print(f"\n[2/2] Matching with ECC (motion={motion_type})...")
                
                player_pos, confidence = self.matcher.match(
                    edges_canny,
                    motion_type=motion_type
                )
            
            if player_pos is None:
                if self.debug:
                    print("[!] Localization failed")
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
