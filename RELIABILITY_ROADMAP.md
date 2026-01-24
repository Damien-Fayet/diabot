# 🎯 Guide Étape-par-Étape: Fiabiliser la Reconnaissance

## Vue d'ensemble

Vous avez maintenant une architecture **propre** (UI vs Environment séparé).
Maintenant, faire en sorte qu'elle soit **fiable**.

Fiabilité = **Confiance** que la détection fonctionne correctement.

---

## Étape 1: Configuration Externalisée

### Problème actuel
```python
# Hardcodé partout:
cv2.inRange(hsv, np.array([0, 100, 100]), np.array([10, 255, 255]))
```

### Solution: vision_config.yaml

Créer fichier:
```yaml
# data/vision_config.yaml

screen:
  # Résolution référence (pour bien adapter les ratios)
  reference_width: 1024
  reference_height: 768
  
regions:
  ui_top_left:
    description: "Health bar, mana bar, effects"
    x_ratio: 0.0
    y_ratio: 0.0
    w_ratio: 0.3
    h_ratio: 0.2
  
  playfield:
    description: "Main game area where player and enemies are"
    x_ratio: 0.0
    y_ratio: 0.15
    w_ratio: 1.0
    h_ratio: 0.7

detection:
  ui:
    health_bar:
      color: "red"
      hsv_lower: [0, 80, 100]
      hsv_upper: [10, 255, 255]
      min_pixels: 50
      description: "Red health bar in UI"
    
    mana_bar:
      color: "blue"
      hsv_lower: [100, 80, 100]
      hsv_upper: [130, 255, 255]
      min_pixels: 50
      description: "Blue mana bar in UI"
  
  environment:
    enemies:
      red:
        color: "red"
        hsv_lower: [0, 100, 100]
        hsv_upper: [10, 255, 255]
        min_area: 50
        max_area: 5000
        description: "Saturated red enemies"
      
      orange:
        color: "orange"
        hsv_lower: [10, 100, 100]
        hsv_upper: [25, 255, 255]
        min_area: 50
        max_area: 5000
        description: "Orange enemies"
    
    items:
      gold:
        color: "gold"
        hsv_lower: [15, 150, 150]
        hsv_upper: [35, 255, 255]
        min_area: 20
        max_area: 2000
        description: "Unique/rare items on ground"
```

### Code pour charger

```python
import yaml
from pathlib import Path

class VisionConfig:
    def __init__(self, config_file: str = "data/vision_config.yaml"):
        with open(config_file, 'r') as f:
            self.data = yaml.safe_load(f)
    
    def get_region(self, name: str):
        """Get region definition."""
        return self.data['regions'][name]
    
    def get_ui_config(self, element: str):
        """Get UI element config."""
        return self.data['detection']['ui'][element]
    
    def get_env_config(self, element: str, subtype: str = None):
        """Get environment element config."""
        config = self.data['detection']['environment'][element]
        if subtype:
            return config[subtype]
        return config

# Utilisation:
config = VisionConfig()
health_config = config.get_ui_config('health_bar')
enemy_red_config = config.get_env_config('enemies', 'red')
```

### Bénéfices

✓ Pas de hardcodage dans le code
✓ Facile à ajuster (edit YAML)
✓ Documenté (descriptions)
✓ Portable (changeable par utilisateur)
✓ Testable (différentes configs)

---

## Étape 2: Confidence Scores

### Problème actuel
```python
# On retourne juste un nombre, pas de confiance
hp_ratio = 0.85  # C'est 85% confident? 50%? On ne sait pas!
```

### Solution: Ajouter confidence

```python
@dataclass
class DetectionResult:
    """Detection result with confidence."""
    value: float           # 0.0-1.0 ou count
    confidence: float      # 0.0-1.0, how sure are we?
    method: str           # "hsv_threshold", "contour", etc.
    timestamp: float      # When detected
    debug_info: dict = field(default_factory=dict)

# Utilisation:
health_result = detect_health_bar(...)
print(f"Health: {health_result.value:.1%}")
print(f"Confidence: {health_result.confidence:.1%}")
print(f"Method: {health_result.method}")

if health_result.confidence < 0.5:
    print("⚠️ Low confidence - might be wrong!")
```

### Calculer confidence

```python
def _detect_health_bar(self, ui_area, config) -> DetectionResult:
    """Detect health bar with confidence."""
    
    # Get config
    hsv_lower = config['hsv_lower']
    hsv_upper = config['hsv_upper']
    min_pixels = config['min_pixels']
    
    # Detect
    mask = cv2.inRange(ui_area, np.array(hsv_lower), np.array(hsv_upper))
    red_pixels = cv2.countNonZero(mask)
    total_pixels = ui_area.shape[0] * ui_area.shape[1]
    
    # Calculate metrics
    pixel_ratio = red_pixels / total_pixels if total_pixels > 0 else 0
    meets_min = red_pixels >= min_pixels
    
    # Health ratio
    hp_ratio = min(1.0, pixel_ratio * 10)  # Scale to 0-1
    
    # Confidence: higher if more pixels, lower if too few
    if red_pixels == 0:
        confidence = 0.0
    elif red_pixels < min_pixels:
        confidence = 0.3  # Low confidence
    elif red_pixels > min_pixels * 10:
        confidence = 0.9  # High confidence
    else:
        confidence = 0.7  # Medium confidence
    
    return DetectionResult(
        value=hp_ratio,
        confidence=confidence,
        method="hsv_threshold",
        timestamp=time.time(),
        debug_info={
            "red_pixels": red_pixels,
            "total_pixels": total_pixels,
            "pixel_ratio": pixel_ratio,
        }
    )
```

### Utiliser confidence pour décisions

```python
ui_state = ui_module.analyze(frame)

# Exemple: Si confiance basse, ignorer la détection
if health_result.confidence < 0.5:
    # Ne pas faire confiance à cette détection
    # Utiliser la dernière valeur connue au lieu
    hp = last_known_hp  
else:
    hp = health_result.value
```

---

## Étape 3: Debug Visualization

### Problème actuel
```
On sait pas ce qui s'est passé.
La détection a raté? Pourquoi?
```

### Solution: Vision Debugger

```python
class VisionDebugger:
    """Visualize vision detection for debugging."""
    
    def __init__(self, output_dir: str = "debug_vision"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
    
    def debug_ui_detection(self, frame, ui_region, detection_result):
        """Save debug visualization of UI detection."""
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        ui_area = ui_region.extract_from_frame(hsv)
        
        # Get the mask
        config = VisionConfig().get_ui_config('health_bar')
        mask = cv2.inRange(ui_area, 
                          np.array(config['hsv_lower']),
                          np.array(config['hsv_upper']))
        
        # Create composite image
        combined = np.hstack([
            cv2.cvtColor(ui_area, cv2.COLOR_HSV2BGR),  # Original
            cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR),    # Mask
        ])
        
        # Save
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = self.output_dir / f"ui_detection_{timestamp}.png"
        cv2.imwrite(str(output_file), combined)
        
        # Log
        print(f"Debug image saved: {output_file}")
        print(f"  Health: {detection_result.value:.1%}")
        print(f"  Confidence: {detection_result.confidence:.1%}")
    
    def debug_environment_detection(self, frame, playfield_region, detection_result):
        """Save debug visualization of environment detection."""
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        play_area = playfield_region.extract_from_frame(hsv)
        
        # Draw detections on frame
        output_frame = frame.copy()
        
        for enemy in detection_result.enemies:
            x, y, w, h = enemy.bbox
            # Convert from region-local to frame coords
            play_x, play_y, _, _ = playfield_region.get_bounds(frame.shape[0], frame.shape[1])
            x += play_x
            y += play_y
            
            # Draw bbox
            color = (0, 255, 0) if enemy.confidence > 0.7 else (0, 165, 255)
            cv2.rectangle(output_frame, (x, y), (x+w, y+h), color, 2)
            cv2.putText(output_frame, f"{enemy.enemy_type} {enemy.confidence:.0%}",
                       (x, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        
        # Save
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = self.output_dir / f"env_detection_{timestamp}.png"
        cv2.imwrite(str(output_file), output_frame)
        
        print(f"Debug image saved: {output_file}")
        for enemy in detection_result.enemies:
            print(f"  - {enemy.enemy_type}: confidence {enemy.confidence:.0%}")

# Utilisation:
debugger = VisionDebugger()
if ui_result.confidence < 0.5:
    debugger.debug_ui_detection(frame, ui_region, ui_result)
```

---

## Étape 4: Test Suite

### Créer tests

```python
# tests/test_ui_vision_reliable.py

import pytest
import cv2
from pathlib import Path
from diabot.vision import UIVisionModule, UIState
from diabot.vision.config import VisionConfig

class TestUIVisionReliability:
    """Test that UI vision is reliable."""
    
    @pytest.fixture
    def ui_module(self):
        return UIVisionModule()
    
    @pytest.fixture
    def test_images_dir(self):
        return Path("test_data/ui_vision")
    
    def test_health_bar_detection_high_health(self, ui_module):
        """Test detecting high health (full bar)."""
        frame = cv2.imread("test_data/ui_vision/full_health.png")
        ui_state = ui_module.analyze(frame)
        
        assert ui_state.hp_ratio > 0.8
        print(f"✓ Full health detected: {ui_state.hp_ratio:.1%}")
    
    def test_health_bar_detection_low_health(self, ui_module):
        """Test detecting low health."""
        frame = cv2.imread("test_data/ui_vision/low_health.png")
        ui_state = ui_module.analyze(frame)
        
        assert ui_state.hp_ratio < 0.3
        print(f"✓ Low health detected: {ui_state.hp_ratio:.1%}")
    
    def test_no_false_positives_dark_screen(self, ui_module):
        """Test that dark screen doesn't trigger false detection."""
        frame = np.zeros((768, 1024, 3), dtype=np.uint8)
        ui_state = ui_module.analyze(frame)
        
        # No UI elements on black screen
        assert ui_state.hp_ratio < 0.3
        assert ui_state.mana_ratio < 0.3
        print("✓ No false positives on dark screen")
    
    def test_mana_bar_detection(self, ui_module):
        """Test mana bar detection."""
        frame = cv2.imread("test_data/ui_vision/half_mana.png")
        ui_state = ui_module.analyze(frame)
        
        assert 0.4 < ui_state.mana_ratio < 0.6
        print(f"✓ Mana detected: {ui_state.mana_ratio:.1%}")

# tests/test_environment_vision_reliable.py

class TestEnvironmentVisionReliability:
    """Test that environment vision is reliable."""
    
    @pytest.fixture
    def env_module(self):
        return EnvironmentVisionModule()
    
    def test_single_enemy_detection(self, env_module):
        """Test detecting a single enemy."""
        frame = cv2.imread("test_data/env_vision/one_enemy.png")
        env_state = env_module.analyze(frame)
        
        assert len(env_state.enemies) == 1
        assert env_state.enemies[0].confidence > 0.7
        print(f"✓ Single enemy detected")
    
    def test_multiple_enemies_detection(self, env_module):
        """Test detecting multiple enemies."""
        frame = cv2.imread("test_data/env_vision/three_enemies.png")
        env_state = env_module.analyze(frame)
        
        assert len(env_state.enemies) >= 3
        print(f"✓ Multiple enemies detected: {len(env_state.enemies)}")
    
    def test_item_detection(self, env_module):
        """Test detecting items on ground."""
        frame = cv2.imread("test_data/env_vision/item_on_ground.png")
        env_state = env_module.analyze(frame)
        
        assert len(env_state.items) > 0
        print(f"✓ Item detected")
    
    def test_no_false_positives_empty_room(self, env_module):
        """Test empty room has no detections."""
        frame = cv2.imread("test_data/env_vision/empty_room.png")
        env_state = env_module.analyze(frame)
        
        assert len(env_state.enemies) == 0
        assert len(env_state.items) == 0
        print("✓ No false positives in empty room")
```

### Courir les tests

```bash
pytest tests/test_ui_vision_reliable.py -v
pytest tests/test_environment_vision_reliable.py -v
```

---

## Étape 5: Calibration Tool (Interactif)

### Script pour calibrer

```python
# tools/calibration_tool.py

import cv2
import numpy as np
import argparse

class CalibrationTool:
    """Interactive tool to calibrate HSV ranges."""
    
    def __init__(self, image_path: str):
        self.image = cv2.imread(image_path)
        self.hsv = cv2.cvtColor(self.image, cv2.COLOR_BGR2HSV)
        
        # Default ranges
        self.h_min = 0
        self.s_min = 80
        self.v_min = 80
        self.h_max = 10
        self.s_max = 255
        self.v_max = 255
        
        self.window_name = "Calibration Tool"
    
    def run(self):
        """Run interactive calibration."""
        cv2.namedWindow(self.window_name)
        
        # Create trackbars
        cv2.createTrackbar("H Min", self.window_name, self.h_min, 180, self.on_change)
        cv2.createTrackbar("H Max", self.window_name, self.h_max, 180, self.on_change)
        cv2.createTrackbar("S Min", self.window_name, self.s_min, 255, self.on_change)
        cv2.createTrackbar("S Max", self.window_name, self.s_max, 255, self.on_change)
        cv2.createTrackbar("V Min", self.window_name, self.v_min, 255, self.on_change)
        cv2.createTrackbar("V Max", self.window_name, self.v_max, 255, self.on_change)
        
        while True:
            # Get trackbar values
            self.h_min = cv2.getTrackbarPos("H Min", self.window_name)
            self.h_max = cv2.getTrackbarPos("H Max", self.window_name)
            self.s_min = cv2.getTrackbarPos("S Min", self.window_name)
            self.s_max = cv2.getTrackbarPos("S Max", self.window_name)
            self.v_min = cv2.getTrackbarPos("V Min", self.window_name)
            self.v_max = cv2.getTrackbarPos("V Max", self.window_name)
            
            # Create mask
            lower = np.array([self.h_min, self.s_min, self.v_min])
            upper = np.array([self.h_max, self.s_max, self.v_max])
            mask = cv2.inRange(self.hsv, lower, upper)
            
            # Display
            result = cv2.bitwise_and(self.image, self.image, mask=mask)
            cv2.imshow(self.window_name, result)
            
            # Info
            pixels = cv2.countNonZero(mask)
            print(f"Pixels detected: {pixels} | Range: H[{self.h_min},{self.h_max}] S[{self.s_min},{self.s_max}] V[{self.v_min},{self.v_max}]", end="\r")
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                # Save config
                self.save_config()
        
        cv2.destroyAllWindows()
    
    def save_config(self):
        """Save calibrated values to YAML."""
        config = {
            'hsv_lower': [self.h_min, self.s_min, self.v_min],
            'hsv_upper': [self.h_max, self.s_max, self.v_max],
        }
        print(f"\n✓ Saved config: {config}")

# Utilisation:
# python tools/calibration_tool.py --image test_screenshot.png
# → Adjust sliders
# → Press 's' to save
```

---

## 🎯 Résumé Étapes

| Étape | Action | Bénéfice |
|-------|--------|---------|
| 1 | Config YAML | Pas de hardcode |
| 2 | Confidence scores | Savoir si c'est fiable |
| 3 | Debug vizualization | Comprendre les problèmes |
| 4 | Test suite | Vérifier que ça marche |
| 5 | Calibration tool | Tuner facilement |

## ⏱️ Temps estimé

- Étape 1: 1-2 heures (créer YAML, loader)
- Étape 2: 2-3 heures (ajouter confidence)
- Étape 3: 1-2 heures (debugger)
- Étape 4: 2-3 heures (tests + test data)
- Étape 5: 1-2 heures (calibration UI)

**Total: 1-1.5 semaines** pour fiabilisation complète

---

## ✅ Checklist

- [ ] Créer `data/vision_config.yaml`
- [ ] Loader config dans UIVisionModule
- [ ] Loader config dans EnvironmentVisionModule
- [ ] Ajouter `DetectionResult` dataclass
- [ ] Retourner `DetectionResult` au lieu de float
- [ ] Créer `VisionDebugger` class
- [ ] Test sur 5 screenshots réelles
- [ ] Créer `test_ui_vision_reliable.py`
- [ ] Créer `test_environment_vision_reliable.py`
- [ ] Créer `calibration_tool.py`
- [ ] Tuner les ranges HSV
- [ ] Documenter les ranges finaux

**État**: Architecture prête, fiabilisation à implémenter
