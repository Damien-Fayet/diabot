"""Demo script showing the logging and analytics system in action."""

import time
from pathlib import Path
from diabot.logging import SessionLogger
from diabot.stats import SessionAnalytics, MultiSessionAnalytics


def demo_simple_session():
    """Run a simple session simulation."""
    print("\n" + "="*70)
    print("DEMO: Simple Gaming Session with Logging")
    print("="*70)
    
    # Create a session
    logger = SessionLogger()
    print(f"\n✅ Session started: {logger.session_id}")
    
    # Simulate game events
    game_state = {
        "location": "Cathedral",
        "hp_ratio": 0.95,
        "mana_ratio": 0.90,
        "enemies": 0
    }
    
    # Log some decisions
    print("\n📍 Game Started - Cathedral")
    logger.log_decision("start_exploration", game_state, "success")
    time.sleep(0.1)
    
    # Find some items
    print("   🏆 Found: Harlequin Crest (S-tier unique)")
    logger.log_item_pickup("Harlequin Crest", "S", game_state)
    
    print("   🏆 Found: Rare Armor (C-tier)")
    logger.log_item_pickup("Random Rare Armor", "C", game_state)
    
    # Combat
    game_state["enemies"] = 2
    print("   ⚔️  Enemies spotted!")
    
    logger.log_decision("engage_combat", game_state, "success")
    logger.log_enemy_kill("zombie", 50, game_state)
    logger.log_damage_taken(15)
    
    print("   💥 Killed 1 zombie (50 dmg dealt, 15 dmg taken)")
    
    logger.log_enemy_kill("skeleton", 75, game_state)
    print("   💥 Killed 1 skeleton (75 dmg dealt)")
    
    # Healing
    game_state["hp_ratio"] = 0.70
    logger.log_potion_used("health", 0.70, 0.95, game_state)
    print("   🧪 Used health potion (70% → 95%)")
    
    # Continue exploring
    game_state["enemies"] = 0
    logger.log_exploration_time(45.0)
    logger.log_combat_time(15.0)
    
    print("   ⏱️  Session time: 45s exploration + 15s combat = 60s total")
    
    # End session and get summary
    summary = logger.end_session()
    print(f"\n📊 Session saved: {summary['file']}")
    
    return summary['file']


def demo_analytics(session_file):
    """Analyze the session."""
    print("\n" + "="*70)
    print("ANALYTICS REPORT")
    print("="*70)
    
    analytics = SessionAnalytics(str(session_file))
    
    # Overall summary
    print("\n📋 SESSION SUMMARY")
    summary = analytics.get_summary()
    print(f"   Session ID: {summary['session_id']}")
    total_time = analytics.get_time_breakdown()
    total_duration = sum(total_time.values())
    print(f"   Duration: {total_duration:.1f}s")
    print(f"   Total Events: {summary['total_events']}")
    
    # Item statistics
    print("\n🏆 ITEM STATISTICS")
    items = analytics.get_item_statistics()
    print(f"   Total Items Picked: {items['total_items_picked']}")
    print(f"   By Tier:")
    for tier, count in items['by_tier'].items():
        if count > 0:
            print(f"      Tier {tier}: {count} item(s)")
    
    # Combat statistics
    print("\n⚔️  COMBAT STATISTICS")
    combat = analytics.get_combat_statistics()
    print(f"   Enemies Killed: {combat['enemies_killed']}")
    print(f"   Total Damage Dealt: {combat['damage_dealt']}")
    print(f"   Total Damage Taken: {combat['damage_taken']}")
    print(f"   Potions Used: {combat['potions_used']}")
    print(f"   Deaths: {combat['deaths']}")
    
    # Event breakdown
    print("\n📊 EVENT BREAKDOWN")
    breakdown = analytics.get_event_breakdown()
    for event_type, count in breakdown.items():
        if count > 0:
            print(f"   {event_type}: {count}")
    
    # Time breakdown
    print("\n⏱️  TIME BREAKDOWN")
    time_breakdown = analytics.get_time_breakdown()
    for activity, time_val in time_breakdown.items():
        if time_val > 0:
            print(f"   {activity}: {time_val:.1f}s")
    
    # Efficiency score
    print("\n🎯 EFFICIENCY SCORE")
    score = analytics.get_efficiency_score()
    print(f"   Score: {score:.1f}/100")
    
    print("\n📄 Full Report:")
    print("-" * 70)
    analytics.print_report()


def demo_multiple_sessions():
    """Run multiple sessions and analyze trends."""
    print("\n" + "="*70)
    print("DEMO: Multiple Sessions with Trend Analysis")
    print("="*70)
    
    # Simulate 3 sessions with improving performance
    sessions_config = [
        {
            "name": "Session 1: Learning",
            "items": 2,
            "kills": 3,
            "deaths": 1,
            "duration": 120
        },
        {
            "name": "Session 2: Practice",
            "items": 4,
            "kills": 8,
            "deaths": 0,
            "duration": 150
        },
        {
            "name": "Session 3: Expertise",
            "items": 6,
            "kills": 12,
            "deaths": 0,
            "duration": 180
        }
    ]
    
    for config in sessions_config:
        print(f"\n🎮 {config['name']}")
        logger = SessionLogger()
        
        # Log items
        for i in range(config['items']):
            tier = 'S' if i == 0 else 'A' if i == 1 else 'B'
            logger.log_item_pickup(f"Item_{i+1}", tier, {})
        
        # Log combat
        for i in range(config['kills']):
            logger.log_enemy_kill(f"enemy_{i+1}", 50 + i*10, {})
        
        # Log deaths
        for i in range(config['deaths']):
            logger.log_death("boss", {})
        
        logger.log_combat_time(config['duration'] * 0.3)
        logger.log_exploration_time(config['duration'] * 0.7)
        
        summary = logger.end_session()
        
        print(f"   Items picked: {config['items']}")
        print(f"   Enemies killed: {config['kills']}")
        print(f"   Deaths: {config['deaths']}")
        print(f"   Session saved: {summary['file']}")
    
    # Analyze trends
    print("\n" + "="*70)
    print("TREND ANALYSIS")
    print("="*70)
    
    analytics = MultiSessionAnalytics(log_dir="logs")
    analytics.print_trend_report()


def main():
    """Run all demos."""
    print("\n" + "🎮 "*35)
    print(" " * 15 + "DIABOT LOGGING & ANALYTICS DEMO")
    print("🎮 " * 35)
    
    # Demo 1: Simple session
    session_file = demo_simple_session()
    
    # Demo 2: Analytics for single session
    demo_analytics(session_file)
    
    # Demo 3: Multiple sessions with trends
    demo_multiple_sessions()
    
    print("\n" + "="*70)
    print("✅ DEMO COMPLETE")
    print("="*70)
    print("\nKey Features Demonstrated:")
    print("  ✓ Real-time event logging during gameplay")
    print("  ✓ Item tracking by tier (S/A/B/C/D)")
    print("  ✓ Combat statistics (kills, damage, potions)")
    print("  ✓ Single-session analytics and efficiency scoring")
    print("  ✓ Multi-session trend analysis")
    print("  ✓ Persistent JSON storage for analysis")
    print("\nAll logs saved in: logs/sessions/")
    print("Stream logs saved in: logs/session.jsonl")


if __name__ == "__main__":
    main()
