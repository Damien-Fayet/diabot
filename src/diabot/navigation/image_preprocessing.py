"""
Image preprocessing for localization.

Provides specialized image filtering and enhancement:
- Gabor filters for isometric geometry detection
- Oriented morphological operations
- Adaptive thresholding
- Edge extraction
"""

from __future__ import annotations

from typing import Tuple, List, Optional
import cv2
import numpy as np


class OrientedFilterBank:
    """Bank of oriented filters for isometric game structure detection."""
    
    def __init__(self, angles: Optional[List[float]] = None):
        """
        Initialize filter bank for specific angles.
        
        Args:
            angles: List of angles in degrees. Default: [60, 120] for isometric.
        """
        self.angles = angles or [60, 120]
        self.kernels = self._create_gabor_kernels()
    
    def _create_gabor_kernels(self) -> List[np.ndarray]:
        """Create Gabor filters for specified angles."""
        kernels = []
        for theta in self.angles:
            theta_rad = theta * np.pi / 180
            kernel = cv2.getGaborKernel(
                ksize=(21, 21),
                sigma=5.0,
                theta=theta_rad,
                lambd=10.0,
                gamma=0.5,
                psi=0
            )
            kernels.append(kernel)
        return kernels
    
    def apply(self, image: np.ndarray) -> np.ndarray:
        """
        Apply all filters and combine responses (max pooling).
        
        Args:
            image: Grayscale input image
            
        Returns:
            Combined response image (uint8)
        """
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        filtered_images = []
        for kernel in self.kernels:
            filtered = cv2.filter2D(gray, cv2.CV_32F, kernel)
            filtered = np.abs(filtered)
            filtered_images.append(filtered)
        
        # Combine all orientations (max response)
        combined = np.maximum.reduce(filtered_images)
        combined_vis = cv2.normalize(
            combined, None, 0, 255, cv2.NORM_MINMAX
        ).astype(np.uint8)
        
        return combined_vis


class OrientedMorphology:
    """Oriented morphological kernels for isometric wall closing."""
    
    @staticmethod
    def create_isometric_kernels(size: int = 21) -> dict:
        """
        Create morphological kernels that follow isometric wall angles.
        
        Isometric angles: 60° and 120° (true game structure angles).
        
        Args:
            size: Kernel size (should be odd)
            
        Returns:
            Dict with 'iso_60' and 'iso_120' kernels
        """
        kernels = {}
        
        # For 60°: tan(60°) ≈ sqrt(3) ≈ 3/2 ratio
        kernel_60 = np.zeros((size, size), dtype=np.uint8)
        for step in range(size):
            j = step * 4 // 3  # Horizontal
            i = size - 1 - step  # Vertical (going up)
            if 0 <= i < size and 0 <= j < size:
                kernel_60[i, j] = 1
                if i + 1 < size:
                    kernel_60[i + 1, j] = 1
        kernels['iso_60'] = kernel_60
        
        # For 120°: tan(120°) ≈ -sqrt(3) ≈ -3/2 ratio
        kernel_120 = np.zeros((size, size), dtype=np.uint8)
        for step in range(size):
            j = step * 4 // 3  # Horizontal
            i = step  # Vertical (going down)
            if 0 <= i < size and 0 <= j < size:
                kernel_120[i, j] = 1
                if i - 1 >= 0:
                    kernel_120[i - 1, j] = 1
        kernels['iso_120'] = kernel_120
        
        return kernels
    
    @staticmethod
    def apply_oriented_closing(image: np.ndarray, kernel_size: int = 21) -> np.ndarray:
        """
        Apply morphological closing with isometric kernels.
        
        Args:
            image: Binary input image
            kernel_size: Size of morphological kernels
            
        Returns:
            Image after oriented closing
        """
        kernels = OrientedMorphology.create_isometric_kernels(kernel_size)
        
        closed_images = []
        for kernel in kernels.values():
            closed = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)
            closed_images.append(closed)
        
        # Union of all closings
        combined = np.maximum.reduce(closed_images)
        return combined


class AdaptiveThresholdProcessor:
    """Adaptive thresholding with edge enhancement."""
    
    @staticmethod
    def process(
        image: np.ndarray,
        brightness_min: int = 20,
        noise_kernel: int = 1,
        npc_threshold: int = 150
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Process image with brightness filtering and NPC removal.
        
        Args:
            image: Input grayscale image
            brightness_min: Minimum brightness to keep (removes shadows)
            noise_kernel: Morphological kernel size for noise removal
            npc_threshold: Brightness threshold for NPC cross removal
            
        Returns:
            (processed_image, npc_mask) tuple
        """
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        # Step 1: Filter dark pixels
        _, dark_mask = cv2.threshold(
            gray, brightness_min, 255, cv2.THRESH_BINARY
        )
        gray_filtered = cv2.bitwise_and(gray, gray, mask=dark_mask)
        
        # Step 2: Remove small artifacts
        if noise_kernel > 0:
            kernel = np.ones((noise_kernel, noise_kernel), np.uint8)
            gray_clean = cv2.morphologyEx(
                gray_filtered, cv2.MORPH_OPEN, kernel
            )
        else:
            gray_clean = gray_filtered
        
        # Step 3: Detect and remove bright NPCs
        _, npc_mask = cv2.threshold(
            gray_clean, npc_threshold, 255, cv2.THRESH_BINARY
        )
        gray_no_npc = cv2.bitwise_and(gray_clean, cv2.bitwise_not(npc_mask))
        
        return gray_no_npc, npc_mask
