"""
RANSAC-based static map localization.

Uses SIFT feature detection and RANSAC homography estimation
for robust player localization on static maps.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple
import cv2
import numpy as np

from .static_map_localizer_base import StaticMapLocalizerBase


class RANSACStaticMapLocalizer(StaticMapLocalizerBase):
    """
    RANSAC-based static map localization using feature matching.
    
    Uses SIFT features and RANSAC to robustly match minimap to static map,
    handling outliers and partial occlusions better than ECC in some cases.
    """
    
    def __init__(
        self,
        static_map_path: Optional[Path] = None,
        debug: bool = False,
        output_dir: Optional[Path] = None,
        sift_ratio_threshold: float = 0.65,
        ransac_threshold: float = 5.0,
        min_inliers: int = 4
    ):
        """
        Initialize RANSAC localizer.
        
        Args:
            static_map_path: Path to static reference map
            debug: Enable debug output and visualizations
            output_dir: Directory to save debug images
            sift_ratio_threshold: Lowe ratio test threshold for match quality (default 0.65 = balanced)
            ransac_threshold: RANSAC inlier distance threshold
            min_inliers: Minimum required inliers for success
        """
        super().__init__(static_map_path, debug, output_dir)
        
        self.sift_ratio_threshold = sift_ratio_threshold
        self.ransac_threshold = ransac_threshold
        self.min_inliers = min_inliers
        
        # Initialize SIFT detector
        try:
            # Try both SIFT_create and SIFT for compatibility
            if hasattr(cv2, 'SIFT_create'):
                self.sift = cv2.SIFT_create()  # type: ignore
            else:
                self.sift = cv2.SIFT()  # type: ignore
        except (AttributeError, RuntimeError) as e:
            self.sift = None  # type: ignore
            if self.debug:
                print(f"[!] SIFT not available: {e}")
    
    def localize(
        self,
        frame_with_minimap: np.ndarray,
        frame_without_minimap: np.ndarray,
        use_oriented_filter: bool = True,
        **kwargs
    ) -> Tuple[Optional[Tuple[float, float]], float]:
        """
        Localize player using RANSAC-based feature matching.
        
        Args:
            frame_with_minimap: Frame with minimap visible
            frame_without_minimap: Frame without minimap
            use_oriented_filter: Use Gabor filters for edge extraction
            **kwargs: Additional parameters (ignored)
            
        Returns:
            ((player_x, player_y), confidence) or (None, 0.0)
        """
        if self.static_map is None:
            if self.debug:
                print("[!] Static map not loaded")
            return None, 0.0
        
        if self.sift is None:
            if self.debug:
                print("[!] SIFT detector not available")
            return None, 0.0
        
        if self.debug:
            print("\n" + "="*70)
            print("RANSAC STATIC MAP LOCALIZATION")
            print("="*70)
        
        try:
            # Step 1: Extract minimap edges
            if self.debug:
                print("\n[1/3] Extracting minimap edges...")
            
            minimap_edges = self.extract_minimap_edges(
                frame_with_minimap,
                frame_without_minimap,
                use_oriented_filter=use_oriented_filter
            )
            
            if minimap_edges is None:
                return None, 0.0
            
            # Step 2: Detect features and match
            if self.debug:
                print("\n[2/3] Detecting and matching features...")
            
            good_matches, kp1, kp2 = self._match_features(minimap_edges)
            
            if len(good_matches) < self.min_inliers:
                if self.debug:
                    print(f"[!] Too few matches: {len(good_matches)} < {self.min_inliers}")
                return None, 0.0
            
            # Step 3: Find homography using RANSAC
            if self.debug:
                print("\n[3/3] Finding homography with RANSAC...")
            
            player_pos, confidence = self._ransac_alignment(
                minimap_edges,
                good_matches,
                kp1,
                kp2
            )
            
            if player_pos is None:
                if self.debug:
                    print("[!] RANSAC alignment failed")
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
    
    def _match_features(
        self,
        minimap_edges: np.ndarray
    ) -> Tuple[list, list, list]:
        """
        Match features between minimap and static map using SIFT.
        
        Args:
            minimap_edges: Edge image from minimap
            
        Returns:
            (good_matches, keypoints1, keypoints2) or empty lists
        """
        if self.sift is None:
            if self.debug:
                print("[!] SIFT detector not initialized")
            return [], [], []
        
        if self.static_map is None:
            if self.debug:
                print("[!] Static map not loaded")
            return [], [], []
        
        # Ensure images are uint8
        if minimap_edges.dtype != np.uint8:
            minimap_edges = cv2.convertScaleAbs(minimap_edges)
        
        static_gray = cv2.cvtColor(self.static_map, cv2.COLOR_BGR2GRAY)
        
        # Detect keypoints and descriptors
        kp1, des1 = self.sift.detectAndCompute(minimap_edges, None)
        kp2, des2 = self.sift.detectAndCompute(static_gray, None)
        
        if des1 is None or des2 is None:
            if self.debug:
                print("[!] Failed to compute descriptors")
            return [], [], []
        
        if len(kp1) < self.min_inliers or len(kp2) < self.min_inliers:
            if self.debug:
                print(f"[!] Insufficient keypoints: {len(kp1)}, {len(kp2)}")
            return [], [], []
        
        # Match using FLANN
        FLANN_INDEX_KDTREE = 1
        index_params = {"algorithm": FLANN_INDEX_KDTREE, "trees": 5}  # type: ignore
        search_params = {"checks": 50}  # type: ignore
        flann = cv2.FlannBasedMatcher(index_params, search_params)  # type: ignore
        
        matches = flann.knnMatch(des1, des2, k=2)
        
        # Apply Lowe's ratio test
        good_matches = []
        for match_pair in matches:
            if len(match_pair) == 2:
                m, n = match_pair
                if m.distance < self.sift_ratio_threshold * n.distance:
                    good_matches.append(m)
        
        if self.debug:
            print(f"[+] Matched features: {len(good_matches)}")
        
        return good_matches, list(kp1), list(kp2)
    
    def _ransac_alignment(
        self,
        minimap_edges: np.ndarray,
        good_matches: list,
        kp1: list,
        kp2: list
    ) -> Tuple[Optional[Tuple[float, float]], float]:
        """
        Find homography using RANSAC and estimate player position.
        
        Args:
            minimap_edges: Edge image
            good_matches: List of good feature matches
            kp1: Keypoints in minimap
            kp2: Keypoints in static map
            
        Returns:
            ((player_x, player_y), confidence) or (None, 0.0)
        """
        # Extract matching points
        src_pts_list = [kp1[m.queryIdx].pt for m in good_matches]
        dst_pts_list = [kp2[m.trainIdx].pt for m in good_matches]
        
        src_pts = np.float32(src_pts_list).reshape(-1, 1, 2)  # type: ignore
        dst_pts = np.float32(dst_pts_list).reshape(-1, 1, 2)  # type: ignore
        
        # Find homography using RANSAC
        homography, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, self.ransac_threshold)
        
        if homography is None:
            if self.debug:
                print("[!] Could not compute homography")
            return None, 0.0
        
        # Count inliers
        inliers = np.sum(mask)
        total_matches = len(good_matches)
        inlier_ratio = inliers / total_matches if total_matches > 0 else 0.0
        
        if self.debug:
            print(f"[+] Inliers: {inliers}/{total_matches} ({inlier_ratio:.2%})")
        
        # Confidence based on inlier ratio
        if inlier_ratio < 0.3:
            confidence = inlier_ratio
        else:
            confidence = min(1.0, inlier_ratio * 0.9)
        
        # Project minimap center to static map space
        minimap_center = np.array(
            [minimap_edges.shape[1] / 2, minimap_edges.shape[0] / 2],
            dtype=np.float32
        ).reshape(1, 1, 2)
        
        transformed = cv2.perspectiveTransform(minimap_center, homography)
        
        if transformed is None or transformed.size == 0:
            return None, 0.0
        
        # Convert to float tuple and ensure proper types
        transformed_point = transformed[0, 0]
        player_pos = (float(transformed_point[0]), float(transformed_point[1]))
        
        return player_pos, float(confidence)
