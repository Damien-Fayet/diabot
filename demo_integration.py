#!/usr/bin/env python3
"""
Integration demo: Item Detection + Session Logging

Shows how the item detection and logging systems work together.
"""

import numpy as np
from pathlib import Path
from diabot.items import ItemDetector, ItemClassifier
from diabot.logging import SessionLogger
from diabot.stats import SessionAnalytics


def demo_integration():
    """Demo item detection with session logging."""
    print("\n" + "="*70)
    print("INTEGRATION DEMO: Item Detection + Session Logging")
    print("="*70)
    
    # Initialize systems
    detector = ItemDetector()
    classifier = ItemClassifier()
    logger = SessionLogger()
    
    print(f"\n✅ Systems initialized")
    print(f"   ItemDetector: 5 quality types (Unique, Set, Rare, Magic, Normal)")
    print(f"   ItemClassifier: {len(classifier.database['items'])} items in database")
    print(f"   SessionLogger: Session {logger.session_id} started")
    
    # Create synthetic game scenario
    print("\n" + "-"*70)
    print("SCENARIO: Exploring Cathedral")
    print("-"*70)
    
    # Log initial decision
    logger.log_decision("start_exploration", {"location": "Cathedral"}, "success")
    print("✓ Decision logged: start_exploration")
    
    # Simulate finding items
    print("\n🏆 Items found:")
    items_found = [
        ("Harlequin Crest", "S"),
        ("Random Rare Axe", "C"),
        ("Shako", "A"),
        ("String of Ears", "A"),
        ("Magic Armor", "B"),
    ]
    
    for item_name, expected_tier in items_found:
        # Classify the item
        tier_enum = classifier.classify(item_name)
        tier = tier_enum.value  # Convert ItemTier enum to string
        color = classifier.get_tier_color(tier_enum)
        
        # Log the pickup
        logger.log_item_pickup(item_name, tier, {"location": "Cathedral"})
        
        print(f"  {color}● {item_name:25} → Tier {tier}")
    
    # Simulate combat
    print("\n⚔️  Combat encounter:")
    logger.log_decision("engage_combat", {"enemies": 3}, "success")
    print("✓ Decision logged: engage_combat")
    
    enemies = [
        ("Zombie", 50),
        ("Skeleton Archer", 75),
        ("Dark Shaman", 100),
    ]
    
    total_damage = 0
    for enemy_name, damage in enemies:
        logger.log_enemy_kill(enemy_name, damage, {"enemies_remaining": len(enemies) - 1})
        total_damage += damage
        print(f"  ⚔️  {enemy_name:20} → {damage} damage")
    
    # Use potions
    print("\n🧪 Healing:")
    logger.log_damage_taken(45)
    logger.log_potion_used("health", 0.45, 0.90, {"location": "Cathedral"})
    print(f"  Used health potion (45% → 90% HP)")
    
    logger.log_potion_used("mana", 0.20, 0.85, {"location": "Cathedral"})
    print(f"  Used mana potion (20% → 85% Mana)")
    
    # Time tracking
    logger.log_combat_time(25.0)
    logger.log_exploration_time(35.0)
    
    print("\n⏱️  Session time:")
    print(f"  Combat: 25.0s")
    print(f"  Exploration: 35.0s")
    
    # End session
    summary = logger.end_session()
    print(f"\n📊 Session saved: {Path(summary['file']).name}")
    
    # Analyze the session
    print("\n" + "="*70)
    print("SESSION ANALYSIS")
    print("="*70)
    
    analytics = SessionAnalytics(str(summary['file']))
    
    # Summary
    summary_data = analytics.get_summary()
    print(f"\n📋 Summary:")
    print(f"   Duration: {summary_data['duration_seconds']:.1f}s")
    print(f"   Total Events: {summary_data['total_events']}")
    print(f"   Survival: {summary_data['survival']}")
    
    # Items
    items_stats = analytics.get_item_statistics()
    print(f"\n🏆 Items:")
    print(f"   Total: {items_stats['total_items_picked']}")
    print(f"   By Tier:")
    for tier in ['S', 'A', 'B', 'C', 'D']:
        count = items_stats['by_tier'].get(tier, 0)
        if count > 0:
            print(f"      {tier}: {count}")
    
    # Combat
    combat = analytics.get_combat_statistics()
    print(f"\n⚔️  Combat:")
    print(f"   Enemies Killed: {combat['enemies_killed']}")
    print(f"   Damage Dealt: {combat['damage_dealt']}")
    print(f"   Damage Taken: {combat['damage_taken']}")
    print(f"   Potions Used: {combat['potions_used']}")
    print(f"   Deaths: {combat['deaths']}")
    
    # Time
    time_info = analytics.get_time_breakdown()
    print(f"\n⏱️  Time:")
    print(f"   Total: {time_info['total_seconds']:.1f}s")
    print(f"   Combat: {time_info['combat_percent']:.1f}%")
    print(f"   Exploration: {time_info['exploration_percent']:.1f}%")
    
    # Efficiency
    score = analytics.get_efficiency_score()
    print(f"\n🎯 Efficiency Score: {score:.1f}/100")
    
    # Event breakdown
    events = analytics.get_event_breakdown()
    print(f"\n📊 Events:")
    for event_type, count in events.items():
        print(f"   {event_type}: {count}")


if __name__ == "__main__":
    demo_integration()
    print("\n" + "="*70)
    print("✅ INTEGRATION DEMO COMPLETE")
    print("="*70)
