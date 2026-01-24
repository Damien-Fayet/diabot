"""
Enhanced Decision Engine Demo.

Demonstrates the full decision system integrating FSM, Skills, and Inventory.
"""

from diabot.models.state import GameState, EnemyInfo, ItemInfo
from diabot.models.skills import SkillBar, Skill, SkillType, CharacterClass
from diabot.models.inventory import Inventory, Item, ItemType, PotionSize
from diabot.skills import SkillManager
from diabot.decision.enhanced_engine import EnhancedDecisionEngine, EnhancedDecisionContext


def create_sorceress_setup():
    """Create a typical Sorceress skill and inventory setup."""
    # Skills
    frozen_orb = Skill(
        name="Frozen Orb",
        skill_type=SkillType.AOE,
        char_class=CharacterClass.SORCERESS,
        mana_cost=25,
        cooldown=1.0,
        damage=100,
        is_aoe=True,
    )
    
    glacial_spike = Skill(
        name="Glacial Spike",
        skill_type=SkillType.ATTACK,
        char_class=CharacterClass.SORCERESS,
        mana_cost=15,
        damage=80,
    )
    
    teleport = Skill(
        name="Teleport",
        skill_type=SkillType.UTILITY,
        char_class=CharacterClass.SORCERESS,
        mana_cost=35,
        range_=30,
    )
    
    skill_bar = SkillBar(
        left_click=glacial_spike,
        right_click=frozen_orb,
        hotkeys=[teleport, None, None, None, None, None, None, None],
    )
    
    # Inventory
    inventory = Inventory()
    
    # Add potions to belt
    inventory.add_to_belt(
        Item(
            item_type=ItemType.HEALTH_POTION,
            name="Greater Health",
            potion_size=PotionSize.GREATER,
        ),
        slot=0,
    )
    
    inventory.add_to_belt(
        Item(
            item_type=ItemType.HEALTH_POTION,
            name="Super Health",
            potion_size=PotionSize.SUPER,
        ),
        slot=1,
    )
    
    inventory.add_to_belt(
        Item(
            item_type=ItemType.MANA_POTION,
            name="Greater Mana",
            potion_size=PotionSize.GREATER,
        ),
        slot=2,
    )
    
    inventory.gold = 5000
    
    return SkillManager(skill_bar), inventory


def simulate_scenario(name: str, game_state: GameState, engine: EnhancedDecisionEngine):
    """Simulate a game scenario and show decision process."""
    print(f"\n{'=' * 60}")
    print(f"SCENARIO: {name}")
    print('=' * 60)
    
    # Show game state
    print(f"\n📊 Game State:")
    print(f"  HP: {game_state.hp_ratio * 100:.0f}%")
    print(f"  Mana: {game_state.mana_ratio * 100:.0f}%")
    print(f"  Enemies: {len(game_state.enemies)}")
    print(f"  Threat: {game_state.threat_level}")
    
    # Get decision
    action = engine.decide(game_state)
    
    # Show decision
    print(f"\n🎯 Decision:")
    print(f"  Action: {action.action_type}")
    if action.skill_name:
        print(f"  Skill: {action.skill_name}")
    if action.item_name:
        print(f"  Item: {action.item_name}")
    if action.target:
        print(f"  Target: {action.target}")
    
    # Show engine status
    status = engine.get_status_summary()
    print(f"\n🤖 Bot Status:")
    print(f"  FSM State: {status['fsm_state']}")
    print(f"  Skills Ready: {status['skills']['ready_skills']}/{status['skills']['total_skills']}")
    print(f"  HP Potions: {status['inventory']['health_potions']}")
    print(f"  MP Potions: {status['inventory']['mana_potions']}")
    
    if status['recent_transitions']:
        print(f"\n📝 Recent Transitions:")
        for t in status['recent_transitions'][-2:]:
            print(f"  {t['from']} → {t['to']}: {t['reason']}")


def main():
    """Run demo scenarios."""
    print("=" * 60)
    print("ENHANCED DECISION ENGINE DEMO")
    print("=" * 60)
    
    # Setup
    skill_manager, inventory = create_sorceress_setup()
    context = EnhancedDecisionContext(
        skill_manager=skill_manager,
        inventory=inventory,
    )
    engine = EnhancedDecisionEngine(context)
    
    # Scenario 1: Safe exploration
    simulate_scenario(
        "Safe Exploration",
        GameState(
            hp_ratio=0.95,
            mana_ratio=0.90,
            enemies=[],
            threat_level="none",
        ),
        engine,
    )
    
    # Scenario 2: Few enemies
    simulate_scenario(
        "Small Combat",
        GameState(
            hp_ratio=0.85,
            mana_ratio=0.80,
            enemies=[
                EnemyInfo(type="zombie", position=(100, 100)),
                EnemyInfo(type="zombie", position=(120, 105)),
            ],
            threat_level="low",
        ),
        engine,
    )
    
    # Scenario 3: Many enemies (AOE situation)
    simulate_scenario(
        "Large Pack",
        GameState(
            hp_ratio=0.75,
            mana_ratio=0.70,
            enemies=[
                EnemyInfo(type="zombie", position=(i * 20, i * 20))
                for i in range(8)
            ],
            threat_level="high",
        ),
        engine,
    )
    
    # Scenario 4: Critical HP
    simulate_scenario(
        "Emergency (Low HP)",
        GameState(
            hp_ratio=0.25,  # Critical!
            mana_ratio=0.60,
            enemies=[
                EnemyInfo(type="boss", position=(80, 80)),
                EnemyInfo(type="minion", position=(100, 90)),
            ],
            threat_level="critical",
        ),
        engine,
    )
    
    # Scenario 5: After drinking potion (PANIC state)
    simulate_scenario(
        "Escape After Potion",
        GameState(
            hp_ratio=0.35,  # Still low
            mana_ratio=0.85,
            enemies=[
                EnemyInfo(type="boss", position=(80, 80)),
            ],
            threat_level="critical",
        ),
        engine,
    )
    
    # Final status
    print("\n" + "=" * 60)
    print("DEMO COMPLETE")
    print("=" * 60)
    
    final_status = engine.get_status_summary()
    print(f"\nFinal FSM State: {final_status['fsm_state']}")
    print(f"Remaining HP Potions: {final_status['inventory']['health_potions']}")
    print(f"Remaining MP Potions: {final_status['inventory']['mana_potions']}")
    print(f"\nTotal Transitions: {len(engine.context.fsm.transition_history)}")


if __name__ == "__main__":
    main()
