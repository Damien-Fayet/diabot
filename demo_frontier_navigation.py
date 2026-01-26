"""
Demo script for vision-based navigation system.

Demonstrates frontier-based exploration using a static screenshot.
Works cross-platform (no Windows dependencies).

Usage:
    python demo_frontier_navigation.py --image path/to/screenshot.png
"""

import argparse
from pathlib import Path

import cv2
import numpy as np

from src.diabot.navigation import (
    FrontierNavigator,
    NavigationOverlay,
    NavigationAction
)


def main():
    parser = argparse.ArgumentParser(description="Demo vision-based navigation")
    parser.add_argument(
        "--image",
        type=str,
        default="data/screenshots/inputs/game_frame.png",
        help="Path to game screenshot"
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=10,
        help="Number of navigation update iterations"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/screenshots/outputs/nav_demo.png",
        help="Output path for visualization"
    )
    
    args = parser.parse_args()
    
    # Load screenshot
    print(f"Loading screenshot: {args.image}")
    frame = cv2.imread(args.image)
    
    if frame is None:
        print(f"ERROR: Could not load image from {args.image}")
        return
    
    print(f"Frame loaded: {frame.shape[1]}x{frame.shape[0]}")
    
    # Initialize navigator
    print("Initializing FrontierNavigator...")
    navigator = FrontierNavigator(
        minimap_grid_size=64,
        local_map_size=200,
        movement_speed=2.0,
        debug=True
    )
    
    # Initialize visualization
    overlay = NavigationOverlay(
        show_local_map=True,
        show_path=True,
        show_frontiers=True,
        show_minimap_grid=True
    )
    
    print("\n" + "=" * 60)
    print("Starting navigation demo")
    print("=" * 60)
    
    # Run several navigation updates
    for iteration in range(args.iterations):
        print(f"\n--- Iteration {iteration + 1} ---")
        
        # Update navigation
        nav_state = navigator.update(frame)
        
        # Print state
        print(f"Action: {nav_state.action}")
        print(f"Position: {nav_state.current_position}")
        print(f"Angle: {nav_state.current_angle:.1f}°")
        print(f"Frontiers: {nav_state.frontiers_available}")
        print(f"Progress: {nav_state.exploration_progress * 100:.1f}%")
        
        # Simulate movement (if action requires it)
        if nav_state.action == NavigationAction.MOVE_FORWARD:
            # Simulate 0.5 seconds of forward movement
            navigator.report_movement("forward", 0.5)
        elif nav_state.action == NavigationAction.TURN_LEFT:
            # Simulate 30 degree left turn
            navigator.report_rotation(-30)
        elif nav_state.action == NavigationAction.TURN_RIGHT:
            # Simulate 30 degree right turn
            navigator.report_rotation(30)
        
        # Create visualization for final iteration
        if iteration == args.iterations - 1:
            print("\n" + "=" * 60)
            print("Creating visualization...")
            
            vis_frame = overlay.draw(
                frame,
                nav_state,
                local_map=navigator.get_local_map(),
                minimap_grid=None  # Could pass this if we store it
            )
            
            # Save output
            output_path = Path(args.output)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(output_path), vis_frame)
            print(f"Saved visualization to: {output_path}")
            
            # Display if possible
            try:
                cv2.imshow("Navigation Demo", vis_frame)
                print("\nPress any key to close...")
                cv2.waitKey(0)
                cv2.destroyAllWindows()
            except:
                print("(Display not available in this environment)")
    
    print("\n" + "=" * 60)
    print("Demo complete!")
    print("=" * 60)
    
    # Print final statistics
    local_map = navigator.get_local_map()
    explored_cells = np.sum(local_map.visited)
    print(f"\nFinal Statistics:")
    print(f"  Explored cells: {explored_cells}")
    print(f"  Map size: {local_map.map_size}x{local_map.map_size}")
    print(f"  Exploration: {nav_state.exploration_progress * 100:.1f}%")


if __name__ == "__main__":
    main()
