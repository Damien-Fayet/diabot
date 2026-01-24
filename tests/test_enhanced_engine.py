"""Integration tests for enhanced decision engine."""

from diabot.models.state import GameState, EnemyInfo
from diabot.models.inventory import Inventory, Item, ItemType, PotionSize
from diabot.models.skills import SkillBar, Skill, SkillType, CharacterClass
from diabot.skills import SkillManager
from diabot.decision.diablo_fsm import DiabloFSM
from diabot.decision.enhanced_engine import EnhancedDecisionContext, EnhancedDecisionEngine


def test_enhanced_engine_basic():
    """Test basic enhanced engine operation."""
    engine = EnhancedDecisionEngine()
    
    # Safe state
    state = GameState(
        hp_ratio=0.9,
        mana_ratio=0.8,
        enemies=[],
        threat_level="none",
    )
    
    action = engine.decide(state)
    assert action is not None
    print(f"✅ Safe state action: {action.action_type}")


def test_enhanced_engine_with_potion():
    """Test automatic potion usage."""
    # Setup inventory with health potion
    inventory = Inventory()
    health_pot = Item(
        item_type=ItemType.HEALTH_POTION,
        name="Greater Health",
        potion_size=PotionSize.GREATER,
    )
    inventory.add_to_belt(health_pot, slot=0)
    
    # Create engine with inventory
    context = EnhancedDecisionContext(inventory=inventory)
    engine = EnhancedDecisionEngine(context)
    
    # Critical HP state
    critical_state = GameState(
        hp_ratio=0.2,  # 20% HP
        mana_ratio=0.8,
        enemies=[EnemyInfo(type="zombie", position=(100, 100))],
        threat_level="critical",
    )
    
    # Should drink potion
    action = engine.decide(critical_state)
    assert action.action_type == "drink_potion"
    assert "health" in action.params.get("potion_type", "")
    
    # Potion should be consumed
    assert not inventory.has_health_potion()
    
    print("✅ Potion usage works correctly")


def test_enhanced_engine_skill_selection():
    """Test skill selection based on FSM state."""
    # Create custom skills
    attack_skill = Skill(
        name="Attack",
        skill_type=SkillType.ATTACK,
        char_class=CharacterClass.SORCERESS,
        mana_cost=10,
        damage=50,
    )
    
    aoe_skill = Skill(
        name="AOE",
        skill_type=SkillType.AOE,
        char_class=CharacterClass.SORCERESS,
        mana_cost=20,
        damage=80,
        is_aoe=True,
    )
    
    skill_bar = SkillBar(
        left_click=attack_skill,
        right_click=aoe_skill,
    )
    
    skill_manager = SkillManager(skill_bar)
    context = EnhancedDecisionContext(skill_manager=skill_manager)
    engine = EnhancedDecisionEngine(context)
    
    # Combat state with many enemies
    combat_state = GameState(
        hp_ratio=0.8,
        mana_ratio=0.9,
        enemies=[
            EnemyInfo(type="zombie", position=(i * 20, i * 20))
            for i in range(5)
        ],
        threat_level="high",
    )
    
    # Should engage and use AOE
    action = engine.decide(combat_state)
    
    # Should be using skill or attacking
    assert action.action_type in ["use_skill", "attack", "engage"]
    
    print(f"✅ Skill selection works: {action.action_type}")


def test_enhanced_engine_panic_behavior():
    """Test panic state behavior."""
    # Add teleport for escape
    teleport = Skill(
        name="Teleport",
        skill_type=SkillType.UTILITY,
        char_class=CharacterClass.SORCERESS,
        mana_cost=30,
    )
    
    attack = Skill(
        name="Attack",
        skill_type=SkillType.ATTACK,
        char_class=CharacterClass.SORCERESS,
        mana_cost=10,
        damage=50,
    )
    
    skill_bar = SkillBar(
        left_click=attack,
        hotkeys=[teleport, None, None, None, None, None, None, None],
    )
    
    # Add health potion
    inventory = Inventory()
    inventory.add_to_belt(
        Item(
            item_type=ItemType.HEALTH_POTION,
            name="HP",
            potion_size=PotionSize.REGULAR,
        )
    )
    
    context = EnhancedDecisionContext(
        skill_manager=SkillManager(skill_bar),
        inventory=inventory,
    )
    engine = EnhancedDecisionEngine(context)
    
    # Panic state
    panic_state = GameState(
        hp_ratio=0.15,  # Critical HP
        mana_ratio=0.9,
        enemies=[
            EnemyInfo(type="boss", position=(50, 50))
            for _ in range(3)
        ],
        threat_level="critical",
    )
    
    # Should drink potion (emergency)
    action = engine.decide(panic_state)
    
    # First action should be potion
    assert action.action_type == "drink_potion"
    print(f"First action (with HP={panic_state.hp_ratio}): {action.action_type}")
    
    # After drinking potion, HP is still low but no more potions
    # Next decision should show PANIC state
    panic_state2 = GameState(
        hp_ratio=0.20,  # Still low
        mana_ratio=0.9,
        enemies=[
            EnemyInfo(type="boss", position=(50, 50))
            for _ in range(3)
        ],
        threat_level="critical",
    )
    
    action2 = engine.decide(panic_state2)
    
    # FSM should be in panic
    print(f"FSM state: {context.fsm.current_state.name}")
    assert context.fsm.current_state.name == "PANIC"
    
    print("✅ Panic behavior works correctly")


def test_enhanced_engine_status_summary():
    """Test status summary."""
    engine = EnhancedDecisionEngine()
    
    # Get status
    status = engine.get_status_summary()
    
    assert "fsm_state" in status
    assert "skills" in status
    assert "inventory" in status
    assert "recent_transitions" in status
    
    print(f"✅ Status summary: FSM={status['fsm_state']}, Skills={status['skills']['total_skills']}")


def test_enhanced_engine_mana_management():
    """Test mana potion usage."""
    inventory = Inventory()
    
    # Add many mana potions
    for i in range(5):
        inventory.add_to_belt(
            Item(
                item_type=ItemType.MANA_POTION,
                name=f"Mana {i}",
                potion_size=PotionSize.GREATER,
            ),
            slot=i % 4,
        )
    
    context = EnhancedDecisionContext(inventory=inventory)
    engine = EnhancedDecisionEngine(context)
    
    # Combat with low mana
    low_mana_combat = GameState(
        hp_ratio=0.9,
        mana_ratio=0.15,  # Low mana
        enemies=[EnemyInfo(type="zombie", position=(100, 100))],
        threat_level="medium",
    )
    
    # Should drink mana potion
    action = engine.decide(low_mana_combat)
    assert action.action_type == "drink_potion"
    assert "mana" in action.params.get("potion_type", "")
    
    print("✅ Mana management works")


if __name__ == "__main__":
    test_enhanced_engine_basic()
    test_enhanced_engine_with_potion()
    test_enhanced_engine_skill_selection()
    test_enhanced_engine_panic_behavior()
    test_enhanced_engine_status_summary()
    test_enhanced_engine_mana_management()
    
    print("\n✅ All enhanced engine integration tests passed!")
