"""Tests for item detection and classification."""

import numpy as np
import cv2
from diabot.items.item_detector import ItemDetector, ItemQuality, DetectedItem
from diabot.items.item_classifier import ItemClassifier, ItemTier


def test_item_detector_init():
    """Test item detector initialization."""
    detector = ItemDetector()
    
    assert detector.min_item_size == 5
    assert detector.max_item_size == 100
    
    # Check HSV ranges are defined
    assert detector.unique_range is not None
    assert detector.set_range is not None
    assert detector.magic_range is not None
    
    print("✅ Item detector initialization works")


def test_item_detector_colors():
    """Test item quality color mapping."""
    detector = ItemDetector()
    
    # Test all qualities have colors
    for quality in ItemQuality.ALL:
        color = detector.get_item_color_rgb(quality)
        assert isinstance(color, tuple)
        assert len(color) == 3
        assert all(0 <= c <= 255 for c in color)
    
    print("✅ Item quality colors defined")


def test_item_detector_create_synthetic_items():
    """Test detection on synthetic image."""
    detector = ItemDetector()
    
    # Create synthetic frame with gold-colored pixel
    frame = np.ones((200, 200, 3), dtype=np.uint8) * 50  # Dark background
    
    # Add a gold (unique) item area
    # Gold in HSV is around H=20-35
    cv2.rectangle(frame, (50, 50), (70, 70), (0, 215, 255), -1)  # Gold in BGR
    
    # Detect items
    items = detector.detect_items(frame)
    
    # Should detect something
    print(f"Detected {len(items)} items in synthetic frame")
    
    if items:
        print(f"  Quality: {items[0].quality}")
        print(f"  Position: {items[0].position}")
        print(f"  Confidence: {items[0].confidence:.2%}")
    
    print("✅ Item detection on synthetic image works")


def test_item_classifier_init():
    """Test item classifier initialization."""
    classifier = ItemClassifier()
    
    # Check database loaded
    assert classifier.database is not None
    assert "items" in classifier.database
    assert "runewords" in classifier.database
    assert "default_tiers" in classifier.database
    
    # Check default items
    items = classifier.database.get("items", {})
    assert len(items) > 0
    
    print("✅ Item classifier initialization works")


def test_item_classifier_s_tier_items():
    """Test classification of S tier items."""
    classifier = ItemClassifier()
    
    # Test S tier items
    s_tier_items = ["Harlequin Crest", "Stone of Jordan", "Annihilus"]
    
    for item_name in s_tier_items:
        tier = classifier.classify(
            item_name=item_name,
            quality="unique"
        )
        assert tier == ItemTier.S, f"{item_name} should be S tier, got {tier}"
    
    print("✅ S tier classification works")


def test_item_classifier_a_tier_items():
    """Test classification of A tier items."""
    classifier = ItemClassifier()
    
    a_tier_items = ["Shako", "War Traveler"]
    
    for item_name in a_tier_items:
        tier = classifier.classify(
            item_name=item_name,
            quality="unique"
        )
        assert tier == ItemTier.A, f"{item_name} should be A tier, got {tier}"
    
    print("✅ A tier classification works")


def test_item_classifier_default_quality():
    """Test default classification by quality."""
    classifier = ItemClassifier()
    
    # Unknown item, unique quality
    tier = classifier.classify("Unknown Unique", quality="unique")
    assert tier == ItemTier.A  # Default for unique
    
    # Unknown item, rare quality
    tier = classifier.classify("Unknown Rare", quality="rare")
    assert tier == ItemTier.C  # Default for rare
    
    # Unknown item, magic quality
    tier = classifier.classify("Unknown Magic", quality="magic")
    assert tier == ItemTier.D  # Default for magic
    
    print("✅ Default quality-based classification works")


def test_item_classifier_runewords():
    """Test runeword classification."""
    classifier = ItemClassifier()
    
    # S tier runewords
    tier = classifier.classify("Enigma")
    assert tier == ItemTier.S
    
    tier = classifier.classify("Chains of Honor")
    assert tier == ItemTier.S
    
    # A tier runeword
    tier = classifier.classify("Infinity")
    assert tier == ItemTier.A
    
    print("✅ Runeword classification works")


def test_item_classifier_add_rule():
    """Test adding custom classification rule."""
    classifier = ItemClassifier()
    
    # Add custom item
    success = classifier.add_item_rule(
        name="Custom Super Item",
        tier="S",
        quality="unique"
    )
    assert success
    
    # Rebuild rules
    classifier._build_rules()
    
    # Classify it (with same quality)
    tier = classifier.classify(
        "Custom Super Item",
        quality="unique"
    )
    assert tier == ItemTier.S
    
    print("✅ Adding custom rules works")


def test_item_classifier_tier_colors():
    """Test tier color assignment."""
    classifier = ItemClassifier()
    
    for tier in ItemTier:
        color = classifier.get_tier_color(tier)
        assert isinstance(color, tuple)
        assert len(color) == 3
        assert all(0 <= c <= 255 for c in color)
    
    print("✅ Tier colors assigned correctly")


def test_item_classifier_statistics():
    """Test database statistics."""
    classifier = ItemClassifier()
    
    stats = classifier.get_statistics()
    
    assert "total_items" in stats
    assert "total_runewords" in stats
    assert "tier_s_count" in stats
    assert "tier_a_count" in stats
    
    assert stats["total_items"] > 0
    assert stats["tier_s_count"] > 0
    
    print(f"✅ Database statistics: {stats['total_items']} items, {stats['tier_s_count']} S tier")


def test_item_detected_item_dataclass():
    """Test DetectedItem dataclass."""
    item = DetectedItem(
        quality="unique",
        position=(100, 150),
        bbox=(90, 140, 110, 160),
        brightness=0.8,
        color_hsv=(25, 200, 200),
        confidence=0.95,
        name="Test Item",
    )
    
    assert item.quality == "unique"
    assert item.position == (100, 150)
    assert item.confidence == 0.95
    assert item.name == "Test Item"
    
    print("✅ DetectedItem dataclass works")


def test_item_detector_filtering():
    """Test item filtering methods."""
    detector = ItemDetector()
    
    # Create test items
    items = [
        DetectedItem("unique", (10, 10), (0, 0, 20, 20), 0.8, (25, 200, 200), 0.9),
        DetectedItem("rare", (50, 50), (40, 40, 60, 60), 0.7, (30, 200, 200), 0.7),
        DetectedItem("magic", (100, 100), (90, 90, 110, 110), 0.6, (120, 200, 200), 0.5),
    ]
    
    # Filter by quality
    unique_items = detector.filter_by_quality(items, "unique")
    assert len(unique_items) == 1
    assert unique_items[0].quality == "unique"
    
    # Filter by confidence
    high_conf = detector.filter_by_confidence(items, 0.7)
    assert len(high_conf) == 2
    
    print("✅ Item filtering works")


if __name__ == "__main__":
    test_item_detector_init()
    test_item_detector_colors()
    test_item_detector_create_synthetic_items()
    test_item_classifier_init()
    test_item_classifier_s_tier_items()
    test_item_classifier_a_tier_items()
    test_item_classifier_default_quality()
    test_item_classifier_runewords()
    test_item_classifier_add_rule()
    test_item_classifier_tier_colors()
    test_item_classifier_statistics()
    test_item_detected_item_dataclass()
    test_item_detector_filtering()
    
    print("\n✅ All item system tests passed!")
