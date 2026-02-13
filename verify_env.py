#!/usr/bin/env python3
"""
Script de vérification de l'environnement Diabot.
Vérifie que toutes les dépendances sont correctement installées.
"""
import sys
from pathlib import Path

def check_import(module_name, display_name=None, optional=False):
    """Tente d'importer un module et affiche le résultat."""
    display = display_name or module_name
    symbol = "ℹ" if optional else "✗"
    try:
        __import__(module_name)
        print(f"✓ {display}")
        return True
    except (ImportError, AssertionError) as e:
        error_msg = str(e).split('\n')[0][:80]
        print(f"{symbol} {display}: {error_msg}")
        return optional

def main():
    """Vérifie toutes les dépendances principales."""
    print("Vérification de l'environnement Diabot...")
    print(f"Python: {sys.version}")
    print(f"Executable: {sys.executable}")
    print()
    
    checks = [
        ("cv2", "OpenCV (cv2)", False),
        ("numpy", "NumPy", False),
        ("PIL", "Pillow (PIL)", False),
        ("matplotlib", "Matplotlib", False),
        ("pytesseract", "PyTesseract", False),
        ("ultralytics", "Ultralytics YOLO", False),
        ("pyautogui", "PyAutoGUI (Windows/macOS)", True),  # Optional on macOS
        ("pytest", "Pytest", False),
        ("dataclasses_json", "Dataclasses JSON", False),
    ]
    
    results = []
    print("Modules requis:")
    for module, name, optional in checks:
        results.append(check_import(module, name, optional))
    
    print()
    success = all(results)
    if success:
        print("✓ Toutes les dépendances sont installées correctement!")
        
        # Test rapide d'import du package diabot
        src_path = Path(__file__).parent / "src"
        if src_path.exists():
            sys.path.insert(0, str(src_path))
            try:
                import diabot
                print("✓ Package diabot importable")
            except ImportError as e:
                print(f"ℹ Package diabot: {e}")
    else:
        print("✗ Certaines dépendances sont manquantes!")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
