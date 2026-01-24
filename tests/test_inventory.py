"""Tests for inventory system."""

from diabot.models.inventory import (
    Item,
    ItemType,
    PotionSize,
    Inventory,
)


def test_item_creation():
    """Test item creation and properties."""
    potion = Item(
        item_type=ItemType.HEALTH_POTION,
        name="Greater Health Potion",
        potion_size=PotionSize.GREATER,
        value=50,
    )
    
    assert potion.is_usable()
    assert potion.item_type == ItemType.HEALTH_POTION
    assert potion.potion_size == PotionSize.GREATER
    
    weapon = Item(
        item_type=ItemType.WEAPON,
        name="Sword",
        value=100,
    )
    
    assert not weapon.is_usable()
    
    print("✅ Item creation works")


def test_inventory_potions():
    """Test potion management in inventory."""
    inv = Inventory()
    
    # Add health potions
    minor_hp = Item(
        item_type=ItemType.HEALTH_POTION,
        name="Minor Health",
        potion_size=PotionSize.MINOR,
    )
    
    greater_hp = Item(
        item_type=ItemType.HEALTH_POTION,
        name="Greater Health",
        potion_size=PotionSize.GREATER,
    )
    
    # Add to belt
    inv.add_to_belt(minor_hp, slot=0)
    inv.add_to_belt(greater_hp, slot=1)
    
    # Check counts
    assert inv.has_health_potion()
    assert len(inv.get_health_potions()) == 2
    
    # Get best should return greater
    best = inv.get_best_health_potion()
    assert best == greater_hp
    assert best.potion_size == PotionSize.GREATER
    
    print("✅ Inventory potion management works")


def test_inventory_belt():
    """Test belt management."""
    inv = Inventory()
    
    # Belt starts empty
    assert not inv.is_belt_full()
    
    # Add items to belt
    for i in range(4):
        item = Item(
            item_type=ItemType.HEALTH_POTION,
            name=f"Potion {i}",
            potion_size=PotionSize.REGULAR,
        )
        added = inv.add_to_belt(item)
        assert added
        assert item.slot == i
    
    # Belt should be full now
    assert inv.is_belt_full()
    
    # Try to add one more - should fail
    extra = Item(
        item_type=ItemType.HEALTH_POTION,
        name="Extra",
        potion_size=PotionSize.REGULAR,
    )
    added = inv.add_to_belt(extra)
    assert not added
    
    print("✅ Belt management works")


def test_inventory_use_item():
    """Test using items from inventory."""
    inv = Inventory()
    
    potion = Item(
        item_type=ItemType.HEALTH_POTION,
        name="Test Potion",
        potion_size=PotionSize.REGULAR,
    )
    
    # Add to belt
    inv.add_to_belt(potion, slot=0)
    assert inv.belt_slots[0] == potion
    
    # Use it
    used = inv.use_item(potion)
    assert used
    assert inv.belt_slots[0] is None
    
    # Can't use non-usable items
    weapon = Item(
        item_type=ItemType.WEAPON,
        name="Sword",
    )
    inv.backpack.append(weapon)
    
    used = inv.use_item(weapon)
    assert not used
    assert weapon in inv.backpack
    
    print("✅ Item usage works")


def test_inventory_mana_potions():
    """Test mana potion management."""
    inv = Inventory()
    
    # Add different mana potions
    potions = [
        Item(
            item_type=ItemType.MANA_POTION,
            name="Minor Mana",
            potion_size=PotionSize.MINOR,
        ),
        Item(
            item_type=ItemType.MANA_POTION,
            name="Super Mana",
            potion_size=PotionSize.SUPER,
        ),
        Item(
            item_type=ItemType.MANA_POTION,
            name="Regular Mana",
            potion_size=PotionSize.REGULAR,
        ),
    ]
    
    # Add to inventory
    for i, pot in enumerate(potions):
        inv.add_to_belt(pot, slot=i)
    
    # Check counts
    assert inv.has_mana_potion()
    assert len(inv.get_mana_potions()) == 3
    
    # Best should be super
    best = inv.get_best_mana_potion()
    assert best.potion_size == PotionSize.SUPER
    
    print("✅ Mana potion management works")


def test_inventory_summary():
    """Test inventory summary reporting."""
    inv = Inventory()
    
    # Add various items
    inv.add_to_belt(
        Item(
            item_type=ItemType.HEALTH_POTION,
            name="HP",
            potion_size=PotionSize.REGULAR,
        ),
        slot=0,
    )
    
    inv.add_to_belt(
        Item(
            item_type=ItemType.MANA_POTION,
            name="MP",
            potion_size=PotionSize.REGULAR,
        ),
        slot=1,
    )
    
    inv.backpack.append(
        Item(
            item_type=ItemType.WEAPON,
            name="Sword",
        )
    )
    
    inv.gold = 500
    
    # Get summary
    summary = inv.get_inventory_summary()
    
    assert summary["health_potions"] == 1
    assert summary["mana_potions"] == 1
    assert summary["belt_slots_used"] == 2
    assert summary["backpack_items"] == 1
    assert summary["gold"] == 500
    
    print("✅ Inventory summary works")


def test_inventory_mixed_storage():
    """Test items in both belt and backpack."""
    inv = Inventory()
    
    # Add potion to belt
    belt_hp = Item(
        item_type=ItemType.HEALTH_POTION,
        name="Belt HP",
        potion_size=PotionSize.REGULAR,
    )
    inv.add_to_belt(belt_hp, slot=0)
    
    # Add potion to backpack
    backpack_hp = Item(
        item_type=ItemType.HEALTH_POTION,
        name="Backpack HP",
        potion_size=PotionSize.GREATER,
    )
    inv.backpack.append(backpack_hp)
    
    # Should find both
    all_hp = inv.get_health_potions()
    assert len(all_hp) == 2
    
    # Best should be the greater one from backpack
    best = inv.get_best_health_potion()
    assert best == backpack_hp
    
    print("✅ Mixed storage works")


if __name__ == "__main__":
    test_item_creation()
    test_inventory_potions()
    test_inventory_belt()
    test_inventory_use_item()
    test_inventory_mana_potions()
    test_inventory_summary()
    test_inventory_mixed_storage()
    
    print("\n✅ All inventory tests passed!")
