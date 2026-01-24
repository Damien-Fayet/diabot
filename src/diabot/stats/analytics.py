"""Analytics and statistics for bot performance."""

import json
from pathlib import Path
from typing import Dict, List, Any
from collections import defaultdict, Counter


class SessionAnalytics:
    """
    Analyzes session logs to provide statistics and insights.
    """
    
    def __init__(self, session_file: str):
        """
        Initialize analytics for a session.
        
        Args:
            session_file: Path to session JSON file
        """
        self.session_file = Path(session_file)
        self.data = self._load_session()
    
    def _load_session(self) -> Dict[str, Any]:
        """Load session data."""
        try:
            with open(self.session_file, 'r') as f:
                return json.load(f)
        except Exception as e:
            print(f"Error loading session: {e}")
            return {}
    
    def get_summary(self) -> Dict[str, Any]:
        """Get overall session summary."""
        metrics = self.data.get("metrics", {})
        
        # Calculate duration from time spent
        time_stats = metrics.get("time", {})
        duration = time_stats.get("combat", 0) + time_stats.get("exploration", 0)
        
        return {
            "session_id": self.data.get("session_id"),
            "created_at": self.data.get("created_at"),
            "ended_at": self.data.get("ended_at"),
            "duration_seconds": duration,
            "total_events": metrics.get("total_events", 0),
            "survival": "success" if metrics.get("deaths", 0) == 0 else "death",
        }
    
    def get_event_breakdown(self) -> Dict[str, int]:
        """Get count of each event type."""
        events = self.data.get("events", [])
        breakdown = Counter()
        
        for event in events:
            event_type = event.get("event_type", "unknown")
            breakdown[event_type] += 1
        
        return dict(breakdown)
    
    def get_item_statistics(self) -> Dict[str, Any]:
        """Get item pickup statistics."""
        metrics = self.data.get("metrics", {})
        
        return {
            "total_items_picked": metrics.get("items_picked", 0),
            "by_tier": metrics.get("items_by_tier", {}),
            "unique_count": metrics.get("items_by_tier", {}).get("S", 0),
            "set_count": metrics.get("items_by_tier", {}).get("A", 0),
        }
    
    def get_combat_statistics(self) -> Dict[str, Any]:
        """Get combat statistics."""
        metrics = self.data.get("metrics", {})
        
        return {
            "enemies_killed": metrics.get("enemies_killed", 0),
            "damage_dealt": metrics.get("damage", {}).get("dealt", 0),
            "damage_taken": metrics.get("damage", {}).get("taken", 0),
            "deaths": metrics.get("deaths", 0),
            "potions_used": metrics.get("potions_used", 0),
        }
    
    def get_time_breakdown(self) -> Dict[str, float]:
        """Get time spent in different activities."""
        metrics = self.data.get("metrics", {})
        time_stats = metrics.get("time", {})
        
        combat = time_stats.get("combat", 0)
        exploration = time_stats.get("exploration", 0)
        total = combat + exploration
        
        # Avoid division by zero
        if total == 0:
            total = 1
        
        return {
            "total_seconds": total,
            "combat_seconds": combat,
            "exploration_seconds": exploration,
            "other_seconds": max(0, total - combat - exploration),
            "combat_percent": (combat / total * 100) if total > 0 else 0,
            "exploration_percent": (exploration / total * 100) if total > 0 else 0,
        }
    
    def get_decision_log(self, limit: int = 20) -> List[Dict[str, Any]]:
        """
        Get recent decisions.
        
        Args:
            limit: Number of decisions to return
            
        Returns:
            List of decisions
        """
        events = self.data.get("events", [])
        decisions = [e for e in events if e.get("event_type") == "decision"]
        return decisions[-limit:]
    
    def get_item_log(self, limit: int = 20) -> List[Dict[str, Any]]:
        """
        Get items picked up.
        
        Args:
            limit: Number of items to return
            
        Returns:
            List of item pickup events
        """
        events = self.data.get("events", [])
        items = [e for e in events if e.get("event_type") == "item_pickup"]
        return items[-limit:]
    
    def get_best_items(self) -> List[str]:
        """Get list of best items found (S and A tier)."""
        items = self.get_item_log(limit=1000)
        
        best_items = []
        for item in items:
            action = item.get("action", "")
            if "S" in action or "A" in action:
                best_items.append(item.get("result", "Unknown"))
        
        return best_items
    
    def get_efficiency_score(self) -> float:
        """
        Calculate overall efficiency score (0-100).
        
        Factors:
        - Items per minute
        - Kills per minute
        - Deaths (negative)
        
        Returns:
            Score from 0-100
        """
        metrics = self.data.get("metrics", {})
        duration_min = metrics.get("duration_seconds", 0) / 60.0
        
        if duration_min == 0:
            return 0.0
        
        items_per_min = metrics.get("items_picked", 0) / duration_min
        kills_per_min = metrics.get("enemies_killed", 0) / duration_min
        deaths = metrics.get("deaths", 0)
        
        # Calculate score
        score = (items_per_min * 5) + (kills_per_min * 10) - (deaths * 20)
        
        # Normalize to 0-100
        return max(0, min(100, score))
    
    def print_report(self):
        """Print formatted session report."""
        summary = self.get_summary()
        print("\n" + "=" * 60)
        print("SESSION REPORT")
        print("=" * 60)
        print(f"\nSession: {summary['session_id']}")
        print(f"Duration: {summary['duration_seconds']:.1f} seconds")
        print(f"Status: {summary['survival']}")
        
        # Combat stats
        print("\n--- COMBAT ---")
        combat = self.get_combat_statistics()
        print(f"Enemies Killed: {combat['enemies_killed']}")
        print(f"Damage Dealt: {combat['damage_dealt']}")
        print(f"Damage Taken: {combat['damage_taken']}")
        print(f"Deaths: {combat['deaths']}")
        print(f"Potions Used: {combat['potions_used']}")
        
        # Items
        print("\n--- ITEMS ---")
        items = self.get_item_statistics()
        print(f"Total Items: {items['total_items_picked']}")
        print(f"S Tier: {items['unique_count']}")
        print(f"A Tier: {items['set_count']}")
        
        # Best items
        best = self.get_best_items()
        if best:
            print(f"\nBest Items Found:")
            for item in best[-5:]:
                print(f"  - {item}")
        
        # Time
        print("\n--- TIME ---")
        time_info = self.get_time_breakdown()
        print(f"Combat: {time_info['combat_percent']:.1f}%")
        print(f"Exploration: {time_info['exploration_percent']:.1f}%")
        
        # Efficiency
        print("\n--- EFFICIENCY ---")
        score = self.get_efficiency_score()
        print(f"Efficiency Score: {score:.1f}/100")
        
        print("\n" + "=" * 60)


class MultiSessionAnalytics:
    """
    Analyze multiple sessions for trends.
    """
    
    def __init__(self, log_dir: str = "logs"):
        """
        Initialize multi-session analytics.
        
        Args:
            log_dir: Logs directory
        """
        self.log_dir = Path(log_dir)
        self.sessions: List[SessionAnalytics] = []
        self._load_all_sessions()
    
    def _load_all_sessions(self):
        """Load all session files."""
        sessions_dir = self.log_dir / "sessions"
        if not sessions_dir.exists():
            return
        
        for session_file in sorted(sessions_dir.glob("session_*.json")):
            try:
                analytics = SessionAnalytics(str(session_file))
                self.sessions.append(analytics)
            except Exception as e:
                print(f"Error loading session {session_file}: {e}")
    
    def get_trend_statistics(self) -> Dict[str, Any]:
        """Get statistics across all sessions."""
        if not self.sessions:
            return {}
        
        summaries = [s.get_summary() for s in self.sessions]
        items_stats = [s.get_item_statistics() for s in self.sessions]
        combat_stats = [s.get_combat_statistics() for s in self.sessions]
        
        total_duration = sum(s.get("duration_seconds", 0) for s in summaries)
        total_items = sum(s.get("total_items_picked", 0) for s in items_stats)
        total_kills = sum(s.get("enemies_killed", 0) for s in combat_stats)
        total_deaths = sum(s.get("deaths", 0) for s in combat_stats)
        
        avg_efficiency = sum(s.get_efficiency_score() for s in self.sessions) / len(self.sessions)
        
        return {
            "session_count": len(self.sessions),
            "total_duration": total_duration,
            "total_items_found": total_items,
            "total_kills": total_kills,
            "total_deaths": total_deaths,
            "avg_efficiency": avg_efficiency,
            "survival_rate": ((len(self.sessions) - total_deaths) / len(self.sessions) * 100) if self.sessions else 0,
        }
    
    def print_trend_report(self):
        """Print multi-session trend report."""
        stats = self.get_trend_statistics()
        
        print("\n" + "=" * 60)
        print("TREND ANALYSIS")
        print("=" * 60)
        print(f"\nTotal Sessions: {stats.get('session_count', 0)}")
        print(f"Total Play Time: {stats.get('total_duration', 0):.0f} seconds")
        print(f"Items Found: {stats.get('total_items_found', 0)}")
        print(f"Enemies Killed: {stats.get('total_kills', 0)}")
        print(f"Deaths: {stats.get('total_deaths', 0)}")
        print(f"\nAverage Efficiency: {stats.get('avg_efficiency', 0):.1f}/100")
        print(f"Survival Rate: {stats.get('survival_rate', 0):.1f}%")
        print("\n" + "=" * 60)
