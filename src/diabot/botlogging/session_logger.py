


@dataclass
class SessionMetrics:
    """Metrics collected during a session."""
    
    start_time: float
    end_time: Optional[float] = None
    
    # Counters
    total_events: int = 0
    total_decisions: int = 0
    total_items_picked: int = 0
    total_enemies_killed: int = 0
    total_potions_used: int = 0
    total_deaths: int = 0
    
    # Inventory
    items_by_tier: Dict[str, int] = field(default_factory=lambda: {
        "S": 0, "A": 0, "B": 0, "C": 0, "D": 0
    })
    
    # Damage stats
    total_damage_dealt: int = 0
    total_damage_taken: int = 0
    
    # Time tracking
    combat_time: float = 0.0
    exploration_time: float = 0.0
    
    def get_duration(self) -> float:
        """Get session duration in seconds."""
        if self.end_time:
            return self.end_time - self.start_time
        return time.time() - self.start_time
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert metrics to dictionary."""
        return {
            "duration_seconds": self.get_duration(),
            "total_events": self.total_events,
            "decisions": self.total_decisions,
            "items_picked": self.total_items_picked,
            "enemies_killed": self.total_enemies_killed,
            "potions_used": self.total_potions_used,
            "deaths": self.total_deaths,
            "items_by_tier": self.items_by_tier,
            "damage": {
                "dealt": self.total_damage_dealt,
                "taken": self.total_damage_taken,
            },
            "time": {
                "combat": self.combat_time,
                "exploration": self.exploration_time,
            }
        }


class SessionLogger:
    """
    Logs bot decisions and game events for analysis.
    
    Saves logs to:
    - logs/sessions/ - Per-session JSON files
    - logs/session.jsonl - Line-delimited JSON for streaming analysis
    """
    
    def __init__(self, log_dir: str = "logs"):
        """
        Initialize session logger.
        
        Args:
            log_dir: Directory to store logs
        """
        self.log_dir = Path(log_dir)
        self.session_dir = self.log_dir / "sessions"
        self.session_dir.mkdir(parents=True, exist_ok=True)
        
        # Current session
        self.session_id = self._generate_session_id()
        self.session_file = self.session_dir / f"session_{self.session_id}.json"
        self.events: List[SessionEvent] = []
        self.metrics = SessionMetrics(start_time=time.time())
        
        # Initialize session file
        self._init_session_file()
    
    def _generate_session_id(self) -> str:
        """Generate unique session ID."""
        now = datetime.now()
        return now.strftime("%Y%m%d_%H%M%S_") + f"{now.microsecond // 1000:03d}"
    
    def _init_session_file(self):
        """Create initial session file."""
        session_data = {
            "session_id": self.session_id,
            "created_at": datetime.now().isoformat(),
            "events": [],
            "metrics": {}
        }
        
        with open(self.session_file, 'w') as f:
            json.dump(session_data, f, indent=2)
    
    def log_decision(
        self,
        action: str,
        game_state: Dict[str, Any],
        result: Optional[str] = None,
    ):
        """
        Log a bot decision.
        
        Args:
            action: Action taken (move, attack, use_skill, etc.)
            game_state: Game state at time of decision
            result: Result of the action
        """
        event = SessionEvent(
            timestamp=time.time(),
            event_type="decision",
            action=action,
            game_state=game_state,
            result=result,
        )
        
        self._log_event(event)
        self.metrics.total_decisions += 1
    
    def log_item_pickup(
        self,
        item_name: str,
        item_tier: str,
        game_state: Dict[str, Any],
    ):
        """
        Log item pickup.
        
        Args:
            item_name: Name of item picked up
            item_tier: Tier (S, A, B, C, D)
            game_state: Game state
        """
        event = SessionEvent(
            timestamp=time.time(),
            event_type="item_pickup",
            action=f"pickup_{item_tier}",
            game_state=game_state,
            result=item_name,
        )
        
        self._log_event(event)
        self.metrics.total_items_picked += 1
        self.metrics.items_by_tier[item_tier] += 1
    
    def log_enemy_kill(
        self,
        enemy_type: str,
        damage_dealt: int,
        game_state: Dict[str, Any],
    ):
        """
        Log enemy kill.
        
        Args:
            enemy_type: Type of enemy
            damage_dealt: Damage dealt
            game_state: Game state
        """
        event = SessionEvent(
            timestamp=time.time(),
            event_type="enemy_kill",
            action=f"kill_{enemy_type}",
            game_state=game_state,
            result=str(damage_dealt),
        )
        
        self._log_event(event)
        self.metrics.total_enemies_killed += 1
        self.metrics.total_damage_dealt += damage_dealt
    
    def log_potion_used(
        self,
        potion_type: str,
        hp_before: float,
        hp_after: float,
        game_state: Dict[str, Any],
    ):
        """
        Log potion usage.
        
        Args:
            potion_type: health or mana
            hp_before: HP before potion
            hp_after: HP after potion
            game_state: Game state
        """
        event = SessionEvent(
            timestamp=time.time(),
            event_type="potion_used",
            action=f"drink_{potion_type}",
            game_state=game_state,
            result=f"{hp_before:.1f}→{hp_after:.1f}",
        )
        
        self._log_event(event)
        self.metrics.total_potions_used += 1
    
    def log_death(
        self,
        killer: Optional[str],
        game_state: Dict[str, Any],
    ):
        """
        Log player death.
        
        Args:
            killer: What killed the player (if known)
            game_state: Game state
        """
        event = SessionEvent(
            timestamp=time.time(),
            event_type="death",
            action="player_death",
            game_state=game_state,
            result=killer or "unknown",
        )
        
        self._log_event(event)
        self.metrics.total_deaths += 1
    
    def log_damage_taken(self, damage: int):
        """Log damage taken."""
        self.metrics.total_damage_taken += damage
    
    def log_combat_time(self, seconds: float):
        """Add to combat time."""
        self.metrics.combat_time += seconds
    
    def log_exploration_time(self, seconds: float):
        """Add to exploration time."""
        self.metrics.exploration_time += seconds
    
    def _log_event(self, event: SessionEvent):
        """
        Add event to session.
        
        Args:
            event: Event to log
        """
        self.events.append(event)
        self.metrics.total_events += 1
        
        # Write to session file
        self._append_event_to_file(event)
        
        # Also write to jsonl stream
        self._write_to_jsonl(event)
    
    def _append_event_to_file(self, event: SessionEvent):
        """Append event to session JSON file."""
        try:
            with open(self.session_file, 'r') as f:
                data = json.load(f)
            
            data["events"].append(asdict(event))
            
            with open(self.session_file, 'w') as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            print(f"Error appending to session file: {e}")
    
    def _write_to_jsonl(self, event: SessionEvent):
        """Write event to jsonl stream."""
        try:
            jsonl_file = self.log_dir / "session.jsonl"
            with open(jsonl_file, 'a') as f:
                f.write(json.dumps({
                    "session_id": self.session_id,
                    **asdict(event)
                }) + '\n')
        except Exception as e:
            print(f"Error writing to jsonl: {e}")
    
    def end_session(self) -> Dict[str, Any]:
        """
        Finalize and save session.
        
        Returns:
            Session summary
        """
        self.metrics.end_time = time.time()
        
        # Update session file with final metrics
        try:
            with open(self.session_file, 'r') as f:
                data = json.load(f)
            
            data["metrics"] = self.metrics.to_dict()
            data["ended_at"] = datetime.now().isoformat()
            
            with open(self.session_file, 'w') as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            print(f"Error finalizing session: {e}")
        
        return {
            "session_id": self.session_id,
            "file": str(self.session_file),
            "metrics": self.metrics.to_dict()
        }
    
    def get_recent_events(self, count: int = 10) -> List[Dict[str, Any]]:
        """
        Get recent events.
        
        Args:
            count: Number of events to return
            
        Returns:
            List of recent events
        """
        recent = self.events[-count:]
        return [asdict(e) for e in recent]
    
    def get_session_summary(self) -> Dict[str, Any]:
        """
        Get current session summary.
        
        Returns:
            Session summary
        """
        return {
            "session_id": self.session_id,
            "duration": self.metrics.get_duration(),
            "events_logged": len(self.events),
            "metrics": self.metrics.to_dict(),
        }
    
    @staticmethod
    def load_session(session_file: str) -> Dict[str, Any]:
        """
        Load a previous session from file.
        
        Args:
            session_file: Path to session JSON file
            
        Returns:
            Session data
        """
        try:
            with open(session_file, 'r') as f:
                return json.load(f)
        except Exception as e:
            print(f"Error loading session: {e}")
            return {}
    
    @staticmethod
    def list_sessions(log_dir: str = "logs") -> List[str]:
        """
        List all saved sessions.
        
        Args:
            log_dir: Logs directory
            
        Returns:
            List of session file paths
        """
        sessions_dir = Path(log_dir) / "sessions"
        if not sessions_dir.exists():
            return []
        
        return sorted(sessions_dir.glob("session_*.json"))
