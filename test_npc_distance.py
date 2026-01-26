#!/usr/bin/env python3
"""Test complete NPC under marker detection with distance tracking."""

import time
from pathlib import Path
from src.diabot.core.implementations import WindowsScreenCapture
from src.diabot.vision.yolo_detector import YOLODetector, Detection
from typing import Optional

def find_npc_under_marker(quest_marker: Detection, 
                          detections: list[Detection]) -> Optional[Detection]:
    """Find NPC directly under quest marker."""
    marker_x, marker_y = quest_marker.center
    marker_x1, marker_y1, marker_x2, marker_y2 = quest_marker.bbox
    marker_bottom = marker_y2
    
    npc_classes = ["akara", "kashya", "warriv", "cain", "charsi", 
                  "stash", "waypoint", "hero"]
    
    npcs = [d for d in detections if d.class_name in npc_classes]
    
    best_npc = None
    best_score = float('inf')
    
    for npc in npcs:
        npc_x, npc_y = npc.center
        npc_x1, npc_y1, npc_x2, npc_y2 = npc.bbox
        npc_top = npc_y1
        
        # NPC should be below marker
        vertical_gap = npc_top - marker_bottom
        if vertical_gap < -20:
            continue
        
        horizontal_overlap = abs(npc_x - marker_x)
        
        vertical_score = max(0, vertical_gap)
        horizontal_score = horizontal_overlap * 0.5
        
        score = vertical_score + horizontal_score
        
        print(f"    Candidate: {npc.class_name} at ({npc_x:.0f},{npc_y:.0f}), vgap={vertical_gap:.0f}, hscore={horizontal_score:.1f}, score={score:.1f}")
        
        if score < best_score:
            best_score = score
            best_npc = npc
    
    if best_npc:
        print(f"  ✓ Best NPC: {best_npc.class_name} at {best_npc.center} (score={best_score:.1f})")
    
    return best_npc

def calculate_distance(pos1, pos2):
    """Calculate Euclidean distance."""
    x1, y1 = pos1
    x2, y2 = pos2
    return ((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5

def main():
    print("🤖 Testing NPC detection and distance tracking...\n")
    
    screen_capture = WindowsScreenCapture(window_title="Diablo II: Resurrected")
    detector = YOLODetector(
        "runs/detect/runs/train/diablo-yolo3/weights/best.pt",
        confidence_threshold=0.35,
        debug=False
    )
    
    last_distance = 0.0
    distance_decreasing_frames = 0
    
    for i in range(5):
        print(f"[Frame {i}]")
        frame = screen_capture.get_frame()
        
        if frame is None:
            print("  ❌ Failed to capture")
            continue
        
        detections = detector.detect(frame)
        
        # Find quest marker
        quest_detections = [d for d in detections if d.class_name == "quest"]
        if not quest_detections:
            print("  ⚠️  No quest marker found")
            continue
        
        quest_marker = quest_detections[0]
        print(f"  🎯 Quest marker at {quest_marker.center} (conf={quest_marker.confidence:.2f})")
        
        # Find NPC under marker
        print(f"  Searching for NPCs...")
        npc_under_marker = find_npc_under_marker(quest_marker, detections)
        
        # Find hero
        hero_det = None
        for det in detections:
            if det.class_name == "hero":
                hero_det = det
                break
        
        # Distance tracking
        if hero_det and npc_under_marker:
            distance = calculate_distance(hero_det.center, npc_under_marker.center)
            
            if last_distance > 0 and distance < last_distance:
                distance_decreasing_frames += 1
            elif distance >= last_distance:
                distance_decreasing_frames = 0
            
            delta = distance - last_distance if last_distance > 0 else 0
            arrow = "↓" if delta < -5 else "↑" if delta > 5 else "="
            print(f"  📍 Distance: {distance:.0f}px {arrow} (Δ{delta:+.0f}px, {distance_decreasing_frames} frames closer)")
            print(f"    Hero at {hero_det.center}, NPC at {npc_under_marker.center}")
            
            last_distance = distance
        
        time.sleep(0.5)

if __name__ == "__main__":
    main()
