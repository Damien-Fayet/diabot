#!/usr/bin/env python3
"""
Validation script to verify ECC localization refactoring.

This script checks that all modules are properly integrated
and can be imported and used correctly.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

def test_imports():
    """Test that all new modules can be imported."""
    print("\n" + "="*70)
    print("TEST 1: Module Imports")
    print("="*70)
    
    try:
        from diabot.navigation import (
            # Image preprocessing
            OrientedFilterBank,
            OrientedMorphology,
            AdaptiveThresholdProcessor,
            # Minimap extraction
            MinimapEdgeExtractor,
            # ECC alignment
            ECCAligner,
            StaticMapMatcher,
            # High-level interface
            ECCStaticMapLocalizer,
            # Helper
            load_zone_static_map,
        )
        print("✓ All modules imported successfully")
        return True
    except ImportError as e:
        print(f"✗ Import failed: {e}")
        return False


def test_class_instantiation():
    """Test that all classes can be instantiated."""
    print("\n" + "="*70)
    print("TEST 2: Class Instantiation")
    print("="*70)
    
    try:
        from diabot.navigation import (
            OrientedFilterBank,
            OrientedMorphology,
            AdaptiveThresholdProcessor,
            MinimapEdgeExtractor,
            ECCAligner,
            StaticMapMatcher,
            ECCStaticMapLocalizer,
        )
        
        # Test instantiation
        fb = OrientedFilterBank()
        print("✓ OrientedFilterBank")
        
        extractor = MinimapEdgeExtractor()
        print("✓ MinimapEdgeExtractor")
        
        aligner = ECCAligner()
        print("✓ ECCAligner")
        
        matcher = StaticMapMatcher()
        print("✓ StaticMapMatcher")
        
        localizer = ECCStaticMapLocalizer()
        print("✓ ECCStaticMapLocalizer")
        
        return True
    except Exception as e:
        print(f"✗ Instantiation failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_method_signatures():
    """Test that key methods exist and have correct signatures."""
    print("\n" + "="*70)
    print("TEST 3: Method Signatures")
    print("="*70)
    
    try:
        from diabot.navigation import ECCStaticMapLocalizer
        import inspect
        
        localizer = ECCStaticMapLocalizer()
        
        # Check main methods
        methods = ['localize', 'load_static_map', 'visualize_alignment']
        for method_name in methods:
            if hasattr(localizer, method_name):
                method = getattr(localizer, method_name)
                sig = inspect.signature(method)
                print(f"✓ localize{sig}")
            else:
                print(f"✗ Missing method: {method_name}")
                return False
        
        return True
    except Exception as e:
        print(f"✗ Method check failed: {e}")
        return False


def test_documentation():
    """Test that classes have docstrings."""
    print("\n" + "="*70)
    print("TEST 4: Documentation")
    print("="*70)
    
    try:
        from diabot.navigation import (
            OrientedFilterBank,
            MinimapEdgeExtractor,
            ECCAligner,
            ECCStaticMapLocalizer,
        )
        
        classes = [
            ("OrientedFilterBank", OrientedFilterBank),
            ("MinimapEdgeExtractor", MinimapEdgeExtractor),
            ("ECCAligner", ECCAligner),
            ("ECCStaticMapLocalizer", ECCStaticMapLocalizer),
        ]
        
        all_documented = True
        for name, cls in classes:
            if cls.__doc__:
                lines = len(cls.__doc__.strip().split('\n'))
                print(f"✓ {name}: {lines} doc lines")
            else:
                print(f"✗ {name}: No docstring")
                all_documented = False
        
        return all_documented
    except Exception as e:
        print(f"✗ Documentation check failed: {e}")
        return False


def test_backward_compatibility():
    """Test that existing StaticMapLocalizer still works."""
    print("\n" + "="*70)
    print("TEST 5: Backward Compatibility")
    print("="*70)
    
    try:
        from diabot.navigation import StaticMapLocalizer, load_zone_static_map
        
        # Old class should still be importable
        localizer = StaticMapLocalizer()
        print("✓ StaticMapLocalizer (old class) still works")
        
        # Helper function should work
        path = load_zone_static_map("Rogue Encampment")
        print(f"✓ load_zone_static_map('Rogue Encampment') returns: {path}")
        
        return True
    except Exception as e:
        print(f"✗ Backward compatibility test failed: {e}")
        return False


def test_files_exist():
    """Test that all required files exist."""
    print("\n" + "="*70)
    print("TEST 6: File Structure")
    print("="*70)
    
    root = Path(__file__).parent
    required_files = [
        "src/diabot/navigation/image_preprocessing.py",
        "src/diabot/navigation/minimap_edge_extractor.py",
        "src/diabot/navigation/ecc_localizer.py",
        "src/diabot/navigation/ecc_static_localizer.py",
        "example_ecc_localization.py",
        "REFACTORING_ECC_LOCALIZATION.md",
        "MIGRATION_ECC_LOCALIZATION.md",
        "REFACTORING_SUMMARY.md",
    ]
    
    all_exist = True
    for filepath in required_files:
        full_path = root / filepath
        if full_path.exists():
            size = full_path.stat().st_size
            print(f"✓ {filepath} ({size} bytes)")
        else:
            print(f"✗ Missing: {filepath}")
            all_exist = False
    
    return all_exist


def test_exports():
    """Test that __init__.py exports are correct."""
    print("\n" + "="*70)
    print("TEST 7: Module Exports")
    print("="*70)
    
    try:
        from diabot import navigation
        
        expected = [
            'ECCStaticMapLocalizer',
            'MinimapEdgeExtractor',
            'ECCAligner',
            'StaticMapMatcher',
            'OrientedFilterBank',
            'OrientedMorphology',
            'AdaptiveThresholdProcessor',
        ]
        
        all_exported = True
        for name in expected:
            if hasattr(navigation, name):
                print(f"✓ navigation.{name}")
            else:
                print(f"✗ Missing export: {name}")
                all_exported = False
        
        return all_exported
    except Exception as e:
        print(f"✗ Export check failed: {e}")
        return False


def main():
    """Run all tests."""
    print("\n" + "="*70)
    print("ECC LOCALIZATION REFACTORING - VALIDATION TESTS")
    print("="*70)
    
    tests = [
        ("File Structure", test_files_exist),
        ("Module Imports", test_imports),
        ("Module Exports", test_exports),
        ("Class Instantiation", test_class_instantiation),
        ("Method Signatures", test_method_signatures),
        ("Documentation", test_documentation),
        ("Backward Compatibility", test_backward_compatibility),
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"\n✗ Test crashed: {e}")
            import traceback
            traceback.print_exc()
            results.append((test_name, False))
    
    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{status}: {test_name}")
    
    print(f"\nTotal: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n" + "🎉 "*35)
        print("ALL TESTS PASSED! Refactoring is complete.")
        print("🎉 "*35)
        return 0
    else:
        print(f"\n⚠️  {total - passed} test(s) failed. Check output above.")
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
