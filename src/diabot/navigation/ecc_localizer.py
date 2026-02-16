"""
Multi-scale ECC alignment for static map localization.

Implements Enhanced Correlation Coefficient (ECC) alignment
with image pyramids for robust structural matching.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple, Literal
import cv2
import numpy as np


class ECCAligner:
    """Enhanced Correlation Coefficient aligner with multi-scale support."""
    
    def __init__(self, debug: bool = False, output_dir: Optional[Path] = None):
        """
        Initialize ECC aligner.
        
        Args:
            debug: Enable debug output
            output_dir: Directory for debug images
        """
        self.debug = debug
        self.output_dir = Path(output_dir) if output_dir else None
    
    @staticmethod
    def build_pyramid(
        img: np.ndarray,
        levels: int = 4
    ) -> list[np.ndarray]:
        """
        Build Gaussian pyramid for multi-scale alignment.
        
        Args:
            img: Input image (float32, [0,1])
            levels: Number of pyramid levels
            
        Returns:
            Pyramid levels from coarse to fine
        """
        pyramid = [img]
        current = img.copy()
        
        for i in range(levels - 1):
            current = cv2.pyrDown(current)
            pyramid.append(current)
        
        return pyramid[::-1]  # Coarse-to-fine
    
    @staticmethod
    def prepare_for_ecc(
        img: np.ndarray,
        blur_kernel: int = 5
    ) -> np.ndarray:
        """
        Prepare image for ECC (normalize, smooth).
        
        Args:
            img: Input image (binary or grayscale)
            blur_kernel: Gaussian blur kernel size
            
        Returns:
            Normalized float32 image [0, 1]
        """
        # Ensure grayscale
        if len(img.shape) == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            gray = img.copy()
        
        # Smooth edges
        smoothed = cv2.GaussianBlur(gray, (blur_kernel, blur_kernel), 1.0)
        
        # Normalize to [0, 1]
        normalized = smoothed.astype(np.float32) / 255.0
        
        return normalized
    
    def align(
        self,
        query_img: np.ndarray,
        reference_img: np.ndarray,
        motion_type: Literal['AFFINE', 'HOMOGRAPHY'] = 'HOMOGRAPHY',
        max_iterations: int = 5000,
        epsilon: float = 1e-6,
        pyramid_levels: int = 4
    ) -> Tuple[np.ndarray, float]:
        """
        Align query image to reference using multi-scale ECC.
        
        Args:
            query_img: Query image (minimap edges, float32)
            reference_img: Reference image (static map edges, float32)
            motion_type: 'AFFINE' or 'HOMOGRAPHY'
            max_iterations: Max ECC iterations per level
            epsilon: Convergence threshold
            pyramid_levels: Number of pyramid levels
            
        Returns:
            (warp_matrix, correlation_score)
        """
        if self.debug:
            print(f"\n[ECC] Starting multi-scale alignment ({motion_type})")
            print(f"      Pyramid levels: {pyramid_levels}")
            print(f"      Query shape: {query_img.shape}, Ref shape: {reference_img.shape}")
        
        # Build pyramids
        query_pyramid = self.build_pyramid(query_img, pyramid_levels)
        ref_pyramid = self.build_pyramid(reference_img, pyramid_levels)
        
        # Initialize warp matrix
        if motion_type == 'AFFINE':
            warp_mode = cv2.MOTION_AFFINE
            warp_matrix = np.eye(2, 3, dtype=np.float32)
        else:
            warp_mode = cv2.MOTION_HOMOGRAPHY
            warp_matrix = np.eye(3, 3, dtype=np.float32)
        
        # Convergence criteria
        criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, max_iterations, epsilon)
        
        cc_score = 0.0
        
        # Multi-scale alignment coarse-to-fine
        for level in range(len(query_pyramid)):
            query_level = query_pyramid[level]
            ref_level = ref_pyramid[level]
            scale = 2 ** (len(query_pyramid) - level - 1)
            
            if self.debug:
                print(f"   Level {level + 1}/{len(query_pyramid)} (scale {scale}x)")
                print(f"      Shapes - Query: {query_level.shape}, Ref: {ref_level.shape}")
                print(f"      Query range: [{query_level.min():.3f}, {query_level.max():.3f}]")
                print(f"      Ref range: [{ref_level.min():.3f}, {ref_level.max():.3f}]")
            
            try:
                cc_score, warp_matrix = cv2.findTransformECC(
                    query_level,
                    ref_level,
                    warp_matrix,
                    warp_mode,
                    criteria
                )
                if self.debug:
                    print(f"      ✓ ECC score: {cc_score:.4f}")
            except cv2.error as e:
                if self.debug:
                    print(f"      [!] ECC failed: {str(e)[:80]}")
                continue
            
            # Scale warp matrix for next level
            if level < len(query_pyramid) - 1:
                if motion_type == 'AFFINE':
                    warp_matrix[0, 2] *= 2
                    warp_matrix[1, 2] *= 2
                else:
                    warp_matrix[0, 2] *= 2
                    warp_matrix[1, 2] *= 2
                    warp_matrix[2, 0] *= 0.5
                    warp_matrix[2, 1] *= 0.5
        
        if self.debug:
            print(f"[ECC] Alignment complete (final score: {cc_score:.4f})")
        
        return warp_matrix, cc_score
    
    def project_point(
        self,
        point: Tuple[float, float],
        warp_matrix: np.ndarray,
        motion_type: Literal['AFFINE', 'HOMOGRAPHY'] = 'HOMOGRAPHY'
    ) -> Tuple[int, int]:
        """
        Project a point through the warp matrix.
        
        Args:
            point: (x, y) point to project
            warp_matrix: Warp transformation matrix
            motion_type: Type of transformation
            
        Returns:
            Projected (x, y) as integers
        """
        x, y = point
        point_3d = np.array([x, y, 1], dtype=np.float32)
        
        if motion_type == 'AFFINE':
            projected = warp_matrix @ point_3d
            proj_x, proj_y = int(projected[0]), int(projected[1])
        else:  # HOMOGRAPHY
            projected = warp_matrix @ point_3d
            proj_x = int(projected[0] / (projected[2] + 1e-6))
            proj_y = int(projected[1] / (projected[2] + 1e-6))
        
        return (proj_x, proj_y)
    
    def warp_image(
        self,
        src_img: np.ndarray,
        warp_matrix: np.ndarray,
        output_shape: Tuple[int, int],
        motion_type: Literal['AFFINE', 'HOMOGRAPHY'] = 'HOMOGRAPHY',
        border_value: int = 0
    ) -> np.ndarray:
        """
        Warp image using transformation matrix.
        
        Args:
            src_img: Source image to warp
            warp_matrix: Warp transformation
            output_shape: Output shape (H, W)
            motion_type: Type of transformation
            border_value: Value for out-of-bounds pixels
            
        Returns:
            Warped image
        """
        if motion_type == 'AFFINE':
            warped = cv2.warpAffine(
                src_img,
                warp_matrix,
                (output_shape[1], output_shape[0]),
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=border_value
            )
        else:
            warped = cv2.warpPerspective(
                src_img,
                warp_matrix,
                (output_shape[1], output_shape[0]),
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=border_value
            )
        
        return warped


class StaticMapMatcher:
    """Match minimap edges with static map using ECC."""
    
    def __init__(
        self,
        static_map_path: Optional[Path] = None,
        debug: bool = False,
        output_dir: Optional[Path] = None
    ):
        """
        Initialize matcher.
        
        Args:
            static_map_path: Path to static map image
            debug: Enable debug output
            output_dir: Directory for debug images
        """
        self.debug = debug
        self.output_dir = Path(output_dir) if output_dir else None
        self.aligner = ECCAligner(debug=debug, output_dir=output_dir)
        
        self.static_map = None
        self.static_edges = None
        
        if static_map_path:
            self.load_static_map(static_map_path)
    
    def load_static_map(self, map_path: Path) -> bool:
        """
        Load static reference map.
        
        Args:
            map_path: Path to static map image
            
        Returns:
            True if loaded successfully
        """
        map_path = Path(map_path)
        
        if not map_path.exists():
            if self.debug:
                print(f"[!] Static map not found: {map_path}")
            return False
        
        self.static_map = cv2.imread(str(map_path))
        if self.static_map is None:
            if self.debug:
                print(f"[!] Failed to load: {map_path}")
            return False
        
        if self.debug:
            h, w = self.static_map.shape[:2]
            print(f"[+] Loaded static map: {map_path.name} ({w}x{h})")
        
        return True
    
    def match(
        self,
        minimap_edges: np.ndarray,
        motion_type: Literal['AFFINE', 'HOMOGRAPHY'] = 'HOMOGRAPHY'
    ) -> Tuple[Optional[Tuple[int, int]], float]:
        """
        Match minimap edges to static map.
        
        Args:
            minimap_edges: Edge image from minimap
            motion_type: Alignment motion model
            
        Returns:
            ((player_x, player_y), confidence) or (None, 0.0)
        """
        if self.static_map is None:
            if self.debug:
                print("[!] No static map loaded")
            return None, 0.0
        
        if self.debug:
            print("\n[MATCH] Starting static map matching")
        
        # Extract edges from static map
        static_gray = cv2.cvtColor(self.static_map, cv2.COLOR_BGR2GRAY)
        static_edges = cv2.Canny(static_gray, 50, 150)
        
        # Prepare for ECC
        query_ecc = self.aligner.prepare_for_ecc(minimap_edges)
        ref_ecc = self.aligner.prepare_for_ecc(static_edges)
        
        # Resize query to match reference
        if query_ecc.shape != ref_ecc.shape:
            query_ecc = cv2.resize(query_ecc, (ref_ecc.shape[1], ref_ecc.shape[0]))
        
        # Perform alignment
        warp_matrix, cc_score = self.aligner.align(
            query_ecc,
            ref_ecc,
            motion_type=motion_type
        )
        
        # Project player position (center of minimap)
        center_orig = (minimap_edges.shape[1] / 2, minimap_edges.shape[0] / 2)
        center_resized = (
            center_orig[0] * query_ecc.shape[1] / minimap_edges.shape[1],
            center_orig[1] * query_ecc.shape[0] / minimap_edges.shape[0]
        )
        
        player_pos = self.aligner.project_point(
            center_resized,
            warp_matrix,
            motion_type
        )
        
        if self.debug:
            print(f"   Player position: {player_pos}")
            print(f"   Confidence: {cc_score:.3f}")
        
        return player_pos, cc_score
