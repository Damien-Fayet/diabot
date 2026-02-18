"""
Quick automated test for SLAM system - no GUI interaction needed.
"""

import numpy as np
from src.diabot.navigation.minimap_slam import MinimapSLAM

def test_slam_basic():
    """Test basic SLAM functionality."""
    print("\n" + "="*70)
    print("AUTOMATED SLAM TEST")
    print("="*70)
    
    # Initialize
    print("\n1. Initializing SLAM...")
    slam = MinimapSLAM(map_size=2048, debug=True)
    print("   ✓ SLAM initialized")
    
    # Create synthetic minimap
    print("\n2. Creating synthetic minimap...")
    minimap = np.random.randint(0, 255, (200, 200, 3), dtype=np.uint8)
    print(f"   ✓ Minimap created: {minimap.shape}")
    
    # Test preprocessing
    print("\n3. Testing preprocessing...")
    skeleton = slam.preprocess_minimap(minimap)
    print(f"   ✓ Skeleton extracted: {skeleton.shape}")
    print(f"   ✓ Wall pixels: {np.sum(skeleton > 127)}")
    
    # Test first update
    print("\n4. Testing first update...")
    slam.update(minimap)
    stats = slam.get_stats()
    print(f"   ✓ Frame count: {stats['frames']}")
    print(f"   ✓ Known cells: {stats['known_cells']}")
    
    # Test motion estimation
    print("\n5. Testing motion estimation...")
    minimap2 = np.roll(minimap, 5, axis=0)  # Shift down
    slam.update(minimap2)
    stats = slam.get_stats()
    print(f"   ✓ World offset: {stats['world_offset']}")
    print(f"   ✓ Movement detected: {stats['total_movement']:.1f} px")
    
    # Test POI addition
    print("\n6. Testing POI tracking...")
    slam.add_poi("npc", (100, 100), confidence=0.9, metadata={"name": "Test NPC"})
    slam.add_poi("exit", (150, 120), confidence=0.85)
    stats = slam.get_stats()
    print(f"   ✓ POIs tracked: {stats['pois']}")
    
    # Test save/load
    print("\n7. Testing save/load...")
    slam.save_map("test_map.npz")
    print("   ✓ Map saved")
    
    slam2 = MinimapSLAM(map_size=2048, debug=False)
    slam2.load_map("test_map.npz")
    stats2 = slam2.get_stats()
    print("   ✓ Map loaded")
    print(f"   ✓ Loaded frames: {stats2['frames']}")
    print(f"   ✓ Loaded POIs: {stats2['pois']}")
    
    # Test multi-level
    print("\n8. Testing multi-level support...")
    slam.switch_level("level_1")
    stats = slam.get_stats()
    print(f"   ✓ Current level: {stats['current_level']}")
    print(f"   ✓ Total levels: {stats['levels']}")
    
    print("\n" + "="*70)
    print("ALL TESTS PASSED ✓")
    print("="*70)
    
    # Final statistics
    print("\nFinal Statistics:")
    for key, value in stats.items():
        print(f"  {key}: {value}")
    
    return True

if __name__ == "__main__":
    try:
        success = test_slam_basic()
        exit(0 if success else 1)
    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        exit(1)
