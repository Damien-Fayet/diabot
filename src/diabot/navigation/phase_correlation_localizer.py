"""
Phase correlation-based static map localization.

Fast and robust localization using FFT phase correlation
for pure translation estimation.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple
import cv2
import numpy as np


class PhaseCorrelationLocalizer:
    """
    Phase correlation localizer for fast translation estimation.
    
    Uses FFT-based phase correlation to find translation offset
    between minimap and static map. Much faster than ECC and
    sufficient when no rotation/scale differences exist.
    """
    
    def __init__(self, debug: bool = False, output_dir: Optional[Path] = None):
        """
        Initialize phase correlation localizer.
        
        Args:
            debug: Enable debug output
            output_dir: Directory for debug images
        """
        self.debug = debug
        self.output_dir = Path(output_dir) if output_dir else None
    
    @staticmethod
    def create_hann_window(shape: Tuple[int, int]) -> np.ndarray:
        """
        Create 2D Hann window for FFT preprocessing.
        
        Reduces edge artifacts in frequency domain by smoothly
        tapering image borders to zero.
        
        Args:
            shape: (height, width) of window
            
        Returns:
            2D Hann window (float32, [0, 1])
        """
        h, w = shape
        hann_h = np.hanning(h)
        hann_w = np.hanning(w)
        return np.outer(hann_h, hann_w).astype(np.float32)
    
    def localize(
        self,
        minimap_edges: np.ndarray,
        static_map_edges: np.ndarray
    ) -> Tuple[Optional[Tuple[int, int]], float]:
        """
        Localize minimap on static map using phase correlation.
        
        Pipeline:
        1. Resize minimap to match static map size
        2. Apply Hann window to both images
        3. Compute phase correlation
        4. Calculate player position from translation offset
        
        Args:
            minimap_edges: Edge image from minimap (grayscale or binary)
            static_map_edges: Edge image from static map (grayscale or binary)
            
        Returns:
            ((player_x, player_y), confidence) or (None, 0.0)
        """
        if self.debug:
            print("\n" + "="*70)
            print("PHASE CORRELATION LOCALIZATION")
            print("="*70)
            print(f"Minimap shape: {minimap_edges.shape}")
            print(f"Static map shape: {static_map_edges.shape}")
        
        try:
            # Ensure grayscale
            if len(minimap_edges.shape) == 3:
                minimap_gray = cv2.cvtColor(minimap_edges, cv2.COLOR_BGR2GRAY)
            else:
                minimap_gray = minimap_edges.copy()
            
            if len(static_map_edges.shape) == 3:
                static_gray = cv2.cvtColor(static_map_edges, cv2.COLOR_BGR2GRAY)
            else:
                static_gray = static_map_edges.copy()
            
            # Resize minimap to match static map
            h_stat, w_stat = static_gray.shape
            minimap_resized = cv2.resize(
                minimap_gray,
                (w_stat, h_stat),
                interpolation=cv2.INTER_LINEAR
            )
            
            if self.debug:
                print(f"Resized minimap to: {minimap_resized.shape}")
            
            # Convert to float32 [0, 1]
            minimap_float = minimap_resized.astype(np.float32) / 255.0
            static_float = static_gray.astype(np.float32) / 255.0
            
            # Apply Hann window to reduce FFT edge artifacts
            hann = self.create_hann_window(static_float.shape)
            minimap_windowed = minimap_float * hann
            static_windowed = static_float * hann
            
            # Phase correlation
            shift, response = cv2.phaseCorrelate(minimap_windowed, static_windowed)
            
            if self.debug:
                print(f"\nPhase correlation results:")
                print(f"  Translation: X={shift[0]:.2f}, Y={shift[1]:.2f}")
                print(f"  Response: {response:.4f}")
            
            # Calculate player position
            # Player is at center of minimap
            center_x = w_stat / 2
            center_y = h_stat / 2
            
            # Apply translation
            player_x = int(center_x - shift[0])
            player_y = int(center_y - shift[1])
            
            # Calculate confidence based on overlap
            # Warp minimap using translation
            M = np.float32([[1, 0, -shift[0]], [0, 1, -shift[1]]])
            minimap_warped = cv2.warpAffine(
                minimap_resized,
                M,
                (w_stat, h_stat),
                borderValue=0
            )
            
            # Binary threshold
            _, minimap_binary = cv2.threshold(minimap_warped, 10, 255, cv2.THRESH_BINARY)
            _, static_binary = cv2.threshold(static_gray, 10, 255, cv2.THRESH_BINARY)
            
            # Calculate overlap
            overlap_pixels = np.count_nonzero(cv2.bitwise_and(minimap_binary, static_binary))
            minimap_pixels = np.count_nonzero(minimap_binary)
            static_pixels = np.count_nonzero(static_binary)
            
            if max(minimap_pixels, static_pixels) > 0:
                overlap_ratio = overlap_pixels / max(minimap_pixels, static_pixels)
            else:
                overlap_ratio = 0.0
            
            # Hybrid confidence: 70% overlap + 30% phase correlation response
            confidence = (0.7 * overlap_ratio) + (0.3 * response)
            
            if self.debug:
                print(f"\nAlignment quality:")
                print(f"  Overlap ratio: {overlap_ratio*100:.1f}%")
                print(f"  Hybrid confidence: {confidence:.3f}")
                print(f"  Player position: ({player_x}, {player_y})")
                
                # Quality interpretation
                if overlap_ratio >= 0.70:
                    quality = "Excellent"
                elif overlap_ratio >= 0.50:
                    quality = "Good"
                elif overlap_ratio >= 0.30:
                    quality = "Moderate"
                else:
                    quality = "Low"
                print(f"  Quality: {quality}")
            
            # Save debug visualization if requested
            if self.output_dir and self.output_dir.exists():
                self._save_debug_visualization(
                    minimap_resized,
                    static_gray,
                    minimap_warped,
                    player_x,
                    player_y,
                    overlap_ratio
                )
            
            return (player_x, player_y), confidence
        
        except Exception as e:
            if self.debug:
                print(f"[!] Phase correlation failed: {e}")
                import traceback
                traceback.print_exc()
            return None, 0.0
    
    def _save_debug_visualization(
        self,
        minimap: np.ndarray,
        static_map: np.ndarray,
        minimap_warped: np.ndarray,
        player_x: int,
        player_y: int,
        overlap_ratio: float
    ):
        """Save debug visualization of alignment."""
        try:
            # Create overlay: minimap in green, static in red
            overlay = np.zeros((static_map.shape[0], static_map.shape[1], 3), dtype=np.uint8)
            overlay[:, :, 1] = minimap_warped  # Green = warped minimap
            overlay[:, :, 2] = static_map  # Red = static map
            
            # Add text
            cv2.putText(
                overlay,
                f"Overlap: {overlap_ratio*100:.1f}%",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (255, 255, 0),
                2
            )
            
            # Mark player position
            cv2.circle(overlay, (player_x, player_y), 15, (0, 0, 255), -1)
            cv2.circle(overlay, (player_x, player_y), 15, (255, 255, 255), 2)
            
            output_path = self.output_dir / "phase_correlation_result.png"
            cv2.imwrite(str(output_path), overlay)
            
            if self.debug:
                print(f"  Visualization saved: {output_path}")
        
        except Exception as e:
            if self.debug:
                print(f"  [!] Could not save visualization: {e}")
