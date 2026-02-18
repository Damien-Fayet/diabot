"""
Integration example: Using MinimapSLAM with existing bot architecture.

This script demonstrates how to integrate the SLAM system with:
- Existing minimap extraction
- Screen capture
- YOLO object detection
- Navigation systems

Can run in developer mode (static images) or runtime mode (live capture).
"""

import cv2
import numpy as np
from pathlib import Path
import time
from typing import Optional

from src.diabot.navigation import (
    MinimapSLAM,
    SLAMVisualizer,
    MinimapExtractor,
    MinimapProcessor,
)


class SLAMIntegrationExample:
    """
    Example integration of SLAM with existing bot components.
    
    Shows how to:
    - Use existing minimap extraction
    - Update SLAM in main loop
    - Track POIs from detection
    - Use SLAM for navigation decisions
    """
    
    def __init__(
        self,
        developer_mode: bool = True,
        enable_visualization: bool = True
    ):
        """
        Initialize integration.
        
        Args:
            developer_mode: Use static images instead of live capture
            enable_visualization: Show SLAM dashboard
        """
        self.developer_mode = developer_mode
        self.enable_visualization = enable_visualization
        
        # Initialize existing components
        self.minimap_extractor = MinimapExtractor(debug=False, fullscreen_mode=True)
        self.minimap_processor = MinimapProcessor(
            grid_size=64,
            wall_threshold=49,
            debug=False,
            use_background_subtraction=False
        )
        
        # Initialize SLAM
        self.slam = MinimapSLAM(
            map_size=4096,
            movement_threshold=2.0,
            loop_closure_threshold=0.80,
            signature_interval=10,
            debug=True
        )
        
        # Initialize visualizer
        if self.enable_visualization:
            self.visualizer = SLAMVisualizer(
                window_name="SLAM Integration",
                map_view_size=500
            )
        
        # Frame counter
        self.frame_count = 0
        self.fps_history = []
        
        # Current zone tracking
        self.current_zone = "unknown"
        self.zone_frame_count = 0
        
        print("[Integration] Initialized")
        print(f"  Mode: {'Developer' if developer_mode else 'Runtime'}")
        print(f"  Visualization: {'Enabled' if enable_visualization else 'Disabled'}")
    
    def process_frame(
        self,
        frame: np.ndarray,
        detected_pois: Optional[list] = None
    ) -> dict:
        """
        Process a single frame through the SLAM pipeline.
        
        Args:
            frame: Full game screen capture
            detected_pois: List of detected POIs from object detection
                          Format: [{"type": "npc", "pos": (x, y), "confidence": 0.9}, ...]
            
        Returns:
            Dictionary with SLAM state and statistics
        """
        start_time = time.time()
        self.frame_count += 1
        self.zone_frame_count += 1
        
        # Step 1: Extract minimap
        try:
            minimap = self.minimap_extractor.extract(frame)
        except Exception as e:
            print(f"[Integration] Failed to extract minimap: {e}")
            return {"error": str(e)}
        
        # Step 2: Preprocess for SLAM
        skeleton = self.slam.preprocess_minimap(minimap)
        
        # Step 3: Estimate motion (if not first frame)
        dx, dy, confidence = 0.0, 0.0, 0.0
        if self.slam.prev_skeleton is not None:
            dx, dy, confidence = self.slam.estimate_motion(skeleton, self.slam.prev_skeleton)
        
        # Step 4: Update SLAM
        self.slam.update(minimap)
        
        # Step 5: Add detected POIs
        if detected_pois:
            for poi_data in detected_pois:
                self.slam.add_poi(
                    poi_type=poi_data.get("type", "unknown"),
                    local_pos=poi_data["pos"],
                    confidence=poi_data.get("confidence", 0.5),
                    metadata=poi_data.get("metadata", {})
                )
        
        # Step 6: Check for zone change
        if self._detect_zone_change(minimap):
            self.current_zone = self._identify_zone()
            self.zone_frame_count = 0
            print(f"[Integration] Zone changed to: {self.current_zone}")
        
        # Calculate FPS
        elapsed = time.time() - start_time
        fps = 1.0 / elapsed if elapsed > 0 else 0
        self.fps_history.append(fps)
        
        # Get SLAM statistics
        stats = self.slam.get_stats()
        stats.update({
            'current_zone': self.current_zone,
            'fps': fps,
            'avg_fps': np.mean(self.fps_history[-30:]),
            'motion_dx': dx,
            'motion_dy': dy,
            'motion_confidence': confidence,
        })
        
        # Step 7: Visualize
        if self.enable_visualization:
            dashboard = self.visualizer.create_dashboard(
                slam=self.slam,
                current_minimap=minimap,
                skeleton=skeleton,
                dx=dx,
                dy=dy,
                fps=stats['avg_fps']
            )
            
            key = self.visualizer.show(dashboard, wait_key=1)
            
            if key == ord('q'):
                stats['quit_requested'] = True
            elif key == ord('s'):
                self._save_current_map()
        
        return stats
    
    def _detect_zone_change(self, current_minimap: np.ndarray) -> bool:
        """
        Detect if we've changed zones.
        
        Uses SLAM's level change detection.
        """
        if self.zone_frame_count < 5:
            return False  # Need a few frames to stabilize
        
        if self.slam.prev_minimap is None:
            return False
        
        return self.slam.detect_level_change(current_minimap, self.slam.prev_minimap)
    
    def _identify_zone(self) -> str:
        """
        Identify current zone based on SLAM data.
        
        In a real implementation, this would use:
        - OCR on zone name
        - POI signatures
        - Map template matching
        """
        # Placeholder: use frame count as proxy
        zone_id = len(self.slam.levels) - 1
        return f"zone_{zone_id}"
    
    def _save_current_map(self):
        """Save current map state."""
        filename = f"slam_map_{self.current_zone}_{int(time.time())}.npz"
        self.slam.save_map(filename)
        print(f"[Integration] Map saved: {filename}")
    
    def get_navigation_target(self, target_type: str = "npc") -> Optional[tuple]:
        """
        Get navigation target from SLAM data.
        
        Args:
            target_type: Type of POI to find
            
        Returns:
            (gx, gy) in global coordinates, or None
        """
        # Find closest POI of requested type
        pois = [poi for poi in self.slam.current_level.pois if poi.poi_type == target_type]
        
        if not pois:
            return None
        
        # Sort by confidence
        pois.sort(key=lambda p: p.confidence, reverse=True)
        
        return pois[0].pos
    
    def is_area_explored(self, radius: int = 100) -> bool:
        """
        Check if area around player is fully explored.
        
        Args:
            radius: Check radius in cells
            
        Returns:
            True if area is mostly explored
        """
        visible = self.slam.get_visible_area(radius=radius)
        
        from src.diabot.navigation.minimap_slam import OccupancyCell
        unknown_ratio = np.sum(visible == OccupancyCell.UNKNOWN) / visible.size
        
        return unknown_ratio < 0.1  # Less than 10% unknown
    
    def find_exploration_target(self) -> Optional[tuple]:
        """
        Find a good target for exploration.
        
        Returns:
            (gx, gy) in global coordinates, or None
        """
        # Get frontiers (boundaries between known and unknown)
        from src.diabot.navigation.minimap_slam import OccupancyCell
        
        global_map = self.slam.global_map
        cx, cy = self.slam.player_center
        
        # Search in a radius around player
        search_radius = 150
        frontiers = []
        
        for dy in range(-search_radius, search_radius):
            for dx in range(-search_radius, search_radius):
                gx = cx + dx
                gy = cy + dy
                
                if not (0 <= gx < self.slam.map_size and 0 <= gy < self.slam.map_size):
                    continue
                
                # Check if this is a frontier cell
                if global_map[gy, gx] == OccupancyCell.FREE:
                    # Check for unknown neighbors
                    for ndy in [-1, 0, 1]:
                        for ndx in [-1, 0, 1]:
                            ngx = gx + ndx
                            ngy = gy + ndy
                            if 0 <= ngx < self.slam.map_size and 0 <= ngy < self.slam.map_size:
                                if global_map[ngy, ngx] == OccupancyCell.UNKNOWN:
                                    frontiers.append((gx, gy))
                                    break
        
        if not frontiers:
            return None
        
        # Return closest frontier
        distances = [np.sqrt((gx - cx)**2 + (gy - cy)**2) for gx, gy in frontiers]
        min_idx = np.argmin(distances)
        
        return frontiers[min_idx]
    
    def shutdown(self):
        """Clean shutdown."""
        # Save final map
        self._save_current_map()
        
        # Close visualizer
        if self.enable_visualization:
            self.visualizer.close()
        
        # Print final stats
        stats = self.slam.get_stats()
        print("\n[Integration] Shutdown")
        print(f"  Total frames: {self.frame_count}")
        print(f"  Levels explored: {stats['levels']}")
        print(f"  POIs tracked: {stats['pois']}")
        print(f"  Loop closures: {stats['loop_closures']}")
        print(f"  Average FPS: {np.mean(self.fps_history):.1f}")


def demo_integration_developer_mode():
    """Demo using static images."""
    print("\n" + "="*70)
    print("INTEGRATION DEMO - Developer Mode")
    print("="*70)
    
    # Initialize
    integration = SLAMIntegrationExample(
        developer_mode=True,
        enable_visualization=True
    )
    
    # Load test images
    test_images_dir = Path("data/screenshots/inputs")
    
    # Fallback: create synthetic if no images available
    if not test_images_dir.exists():
        print("No test images found, using synthetic data")
        
        # Create synthetic frames
        for i in range(20):
            # Create synthetic game frame
            frame = np.random.randint(0, 50, (1080, 1920, 3), dtype=np.uint8)
            
            # Add synthetic minimap in top-right
            minimap_size = 200
            minimap = np.random.randint(100, 200, (minimap_size, minimap_size, 3), dtype=np.uint8)
            frame[-minimap_size:, -minimap_size:] = minimap
            
            # Add some synthetic POIs
            detected_pois = []
            if i % 5 == 0:
                detected_pois.append({
                    "type": "npc",
                    "pos": (100, 100),
                    "confidence": 0.9
                })
            
            if i % 7 == 0:
                detected_pois.append({
                    "type": "exit",
                    "pos": (150, 120),
                    "confidence": 0.85
                })
            
            # Process
            stats = integration.process_frame(frame, detected_pois)
            
            print(f"Frame {i}: Offset=({stats['world_offset'][0]}, {stats['world_offset'][1]}), "
                  f"Known={stats['known_cells']}, FPS={stats['fps']:.1f}")
            
            if stats.get('quit_requested'):
                break
            
            time.sleep(0.1)  # Simulate frame time
    
    # Shutdown
    integration.shutdown()


def demo_integration_navigation():
    """Demo navigation queries."""
    print("\n" + "="*70)
    print("NAVIGATION INTEGRATION DEMO")
    print("="*70)
    
    # Initialize
    integration = SLAMIntegrationExample(
        developer_mode=True,
        enable_visualization=False
    )
    
    # Simulate some frames
    for i in range(10):
        frame = np.random.randint(0, 50, (1080, 1920, 3), dtype=np.uint8)
        minimap = np.random.randint(100, 200, (200, 200, 3), dtype=np.uint8)
        frame[-200:, -200:] = minimap
        
        # Add POIs
        pois = []
        if i == 3:
            pois.append({"type": "npc", "pos": (100, 100), "confidence": 0.9})
        if i == 6:
            pois.append({"type": "waypoint", "pos": (120, 80), "confidence": 0.95})
        
        integration.process_frame(frame, pois)
    
    # Navigation queries
    print("\n--- Navigation Queries ---")
    
    # Find NPC
    npc_target = integration.get_navigation_target("npc")
    if npc_target:
        print(f"NPC location: {npc_target}")
    else:
        print("No NPC found")
    
    # Check exploration
    is_explored = integration.is_area_explored(radius=100)
    print(f"Area explored: {is_explored}")
    
    # Find exploration target
    explore_target = integration.find_exploration_target()
    if explore_target:
        print(f"Exploration target: {explore_target}")
    else:
        print("No unexplored areas nearby")
    
    integration.shutdown()


if __name__ == "__main__":
    print("SLAM Integration Examples")
    print("=========================")
    print("\n1. Developer mode demo (static images)")
    print("2. Navigation integration demo")
    
    choice = input("\nSelect demo (1-2): ").strip()
    
    if choice == "1":
        demo_integration_developer_mode()
    elif choice == "2":
        demo_integration_navigation()
    else:
        print("Invalid choice")
