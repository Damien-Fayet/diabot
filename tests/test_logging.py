"""Tests for logging and analytics systems."""

import json
import tempfile
from pathlib import Path
from diabot.logging import SessionLogger, SessionEvent
from diabot.stats import SessionAnalytics


def test_session_logger_init():
    """Test session logger initialization."""
    with tempfile.TemporaryDirectory() as tmpdir:
        logger = SessionLogger(log_dir=tmpdir)
        
        assert logger.session_id is not None
        assert logger.session_file.exists()
        assert logger.metrics.start_time > 0
        
        print("✅ Session logger initialization works")


def test_session_logger_log_decision():
    """Test logging decisions."""
    with tempfile.TemporaryDirectory() as tmpdir:
        logger = SessionLogger(log_dir=tmpdir)
        
        game_state = {
            "hp_ratio": 0.8,
            "mana_ratio": 0.7,
            "enemies": 2,
        }
        
        logger.log_decision(
            action="attack",
            game_state=game_state,
            result="hit"
        )
        
        assert logger.metrics.total_decisions == 1
        assert logger.metrics.total_events == 1
        assert len(logger.events) == 1
        
        print("✅ Decision logging works")


def test_session_logger_item_pickup():
    """Test logging item pickups."""
    with tempfile.TemporaryDirectory() as tmpdir:
        logger = SessionLogger(log_dir=tmpdir)
        
        game_state = {"hp_ratio": 0.9}
        
        logger.log_item_pickup("Harlequin Crest", "S", game_state)
        logger.log_item_pickup("Random Rare", "C", game_state)
        
        assert logger.metrics.total_items_picked == 2
        assert logger.metrics.items_by_tier["S"] == 1
        assert logger.metrics.items_by_tier["C"] == 1
        
        print("✅ Item pickup logging works")


def test_session_logger_combat():
    """Test logging combat events."""
    with tempfile.TemporaryDirectory() as tmpdir:
        logger = SessionLogger(log_dir=tmpdir)
        
        game_state = {"enemies": 1}
        
        logger.log_enemy_kill("zombie", 50, game_state)
        logger.log_damage_taken(20)
        logger.log_potion_used("health", 0.3, 0.7, game_state)
        
        assert logger.metrics.total_enemies_killed == 1
        assert logger.metrics.total_damage_dealt == 50
        assert logger.metrics.total_damage_taken == 20
        assert logger.metrics.total_potions_used == 1
        
        print("✅ Combat logging works")


def test_session_logger_death():
    """Test logging death."""
    with tempfile.TemporaryDirectory() as tmpdir:
        logger = SessionLogger(log_dir=tmpdir)
        
        logger.log_death("boss", {})
        
        assert logger.metrics.total_deaths == 1
        
        print("✅ Death logging works")


def test_session_logger_time_tracking():
    """Test time tracking."""
    with tempfile.TemporaryDirectory() as tmpdir:
        logger = SessionLogger(log_dir=tmpdir)
        
        logger.log_combat_time(10.0)
        logger.log_exploration_time(5.0)
        
        assert logger.metrics.combat_time == 10.0
        assert logger.metrics.exploration_time == 5.0
        
        print("✅ Time tracking works")


def test_session_logger_end_session():
    """Test ending a session."""
    with tempfile.TemporaryDirectory() as tmpdir:
        logger = SessionLogger(log_dir=tmpdir)
        
        logger.log_decision("test", {})
        
        summary = logger.end_session()
        
        assert summary["session_id"] is not None
        assert summary["metrics"]["total_events"] == 1
        assert "file" in summary
        
        print("✅ Session ending works")


def test_session_logger_get_recent_events():
    """Test retrieving recent events."""
    with tempfile.TemporaryDirectory() as tmpdir:
        logger = SessionLogger(log_dir=tmpdir)
        
        for i in range(5):
            logger.log_decision(f"action_{i}", {})
        
        recent = logger.get_recent_events(count=3)
        
        assert len(recent) == 3
        
        print("✅ Getting recent events works")


def test_session_analytics():
    """Test analytics on logged session."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create a session
        logger = SessionLogger(log_dir=tmpdir)
        
        logger.log_decision("attack", {"hp_ratio": 0.9})
        logger.log_item_pickup("Unique Item", "S", {})
        logger.log_enemy_kill("zombie", 50, {})
        logger.log_potion_used("health", 0.3, 0.7, {})
        
        logger.end_session()
        
        # Analyze it
        analytics = SessionAnalytics(str(logger.session_file))
        
        summary = analytics.get_summary()
        assert summary["session_id"] is not None
        
        items = analytics.get_item_statistics()
        assert items["total_items_picked"] == 1
        assert items["unique_count"] == 1
        
        combat = analytics.get_combat_statistics()
        assert combat["enemies_killed"] == 1
        assert combat["damage_dealt"] == 50
        assert combat["potions_used"] == 1
        
        print("✅ Session analytics works")


def test_session_analytics_efficiency():
    """Test efficiency score calculation."""
    with tempfile.TemporaryDirectory() as tmpdir:
        logger = SessionLogger(log_dir=tmpdir)
        
        logger.log_item_pickup("Item1", "A", {})
        logger.log_item_pickup("Item2", "B", {})
        logger.log_enemy_kill("zombie", 100, {})
        
        logger.end_session()
        
        analytics = SessionAnalytics(str(logger.session_file))
        score = analytics.get_efficiency_score()
        
        assert 0 <= score <= 100
        
        print(f"✅ Efficiency score calculation works (score: {score:.1f})")


def test_session_analytics_event_breakdown():
    """Test event type breakdown."""
    with tempfile.TemporaryDirectory() as tmpdir:
        logger = SessionLogger(log_dir=tmpdir)
        
        logger.log_decision("attack", {})
        logger.log_decision("move", {})
        logger.log_item_pickup("Item", "A", {})
        logger.log_enemy_kill("zombie", 50, {})
        
        logger.end_session()
        
        analytics = SessionAnalytics(str(logger.session_file))
        breakdown = analytics.get_event_breakdown()
        
        assert breakdown["decision"] == 2
        assert breakdown["item_pickup"] == 1
        assert breakdown["enemy_kill"] == 1
        
        print("✅ Event breakdown works")


def test_session_analytics_report():
    """Test analytics report printing."""
    with tempfile.TemporaryDirectory() as tmpdir:
        logger = SessionLogger(log_dir=tmpdir)
        
        logger.log_item_pickup("Harlequin Crest", "S", {})
        logger.log_enemy_kill("boss", 200, {})
        logger.log_potion_used("health", 0.2, 0.8, {})
        logger.log_combat_time(30.0)
        logger.log_exploration_time(20.0)
        
        logger.end_session()
        
        analytics = SessionAnalytics(str(logger.session_file))
        
        # Just test that it doesn't crash
        try:
            analytics.print_report()
            print("✅ Analytics report generation works")
        except Exception as e:
            print(f"❌ Report failed: {e}")
            raise


if __name__ == "__main__":
    test_session_logger_init()
    test_session_logger_log_decision()
    test_session_logger_item_pickup()
    test_session_logger_combat()
    test_session_logger_death()
    test_session_logger_time_tracking()
    test_session_logger_end_session()
    test_session_logger_get_recent_events()
    test_session_analytics()
    test_session_analytics_efficiency()
    test_session_analytics_event_breakdown()
    test_session_analytics_report()
    
    print("\n✅ All logging and analytics tests passed!")
