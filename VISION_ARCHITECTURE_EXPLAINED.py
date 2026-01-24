"""
ARCHITECTURE VISION ACTUELLE - Explication & Fiabilisation

Ce document explique le système de reconnaissance en place et comment le fiabiliser.
"""

# ============================================================================
# 1. ARCHITECTURE ACTUELLE DE RECONNAISSANCE
# ============================================================================

"""
Le système de vision actuel fonctionne en 2 couches:

┌─────────────────────────────────────────────────────────────────┐
│                  FRAME SCREENSHOT (pixel array)                │
└────────────────────┬────────────────────────────────────────────┘
                     │
                     ▼
        ┌────────────────────────────┐
        │   Conversion HSV           │
        │  (meilleure que RGB pour   │
        │   détecter les couleurs)   │
        └────────────────────────────┘
                     │
        ┌────────────┼────────────┐
        │            │            │
        ▼            ▼            ▼
    [UI BARS]   [PLAYFIELD]  [ITEMS]
    Santé/Mana  Ennemis      Trésors
    
   DiabloVisionModule.perceive():
   - _detect_health_bar()     → hp_ratio (0.0-1.0)
   - _detect_mana_bar()       → mana_ratio (0.0-1.0) 
   - _detect_enemies()        → enemy_count + types
   - _detect_items()          → liste d'items
   - _estimate_player_position() → (x, y)
   
   ↓↓↓ Retourne
   
   Perception dataclass:
   {
     hp_ratio: 0.8,
     mana_ratio: 0.6,
     enemy_count: 2,
     enemy_types: ["large_enemy", "small_enemy"],
     visible_items: ["item", "item"],
     player_position: (640, 360),
   }

┌─────────────────────────────────────────────────────────────────┐
│            COUCHE 2: ItemDetector (plus spécialisée)           │
│                                                                 │
│  Détection fine des items par couleur HSV:                     │
│  - Unique (Gold)   - H: 15-35                                 │
│  - Set (Green)     - H: 60-90                                 │
│  - Rare (Yellow)   - H: 20-40                                 │
│  - Magic (Blue)    - H: 100-130                               │
│  - Normal (White)  - Low saturation                           │
│                                                                 │
│  → Returns: list[DetectedItem] avec:                           │
│    - quality, position, bbox, confidence, name                │
└─────────────────────────────────────────────────────────────────┘
"""

# ============================================================================
# 2. PROBLÈMES ACTUELS (Défis de fiabilité)
# ============================================================================

CURRENT_ISSUES = """

A) CONFUSION UI vs PLAYFIELD
   - Les barres UI (santé, mana) sont rouges/bleues
   - Les ennemis sont aussi rouges/oranges
   - Les items sont jaunes/or
   
   Problème: Les régions UI et playfield ne sont pas bien séparées
   - Code actuel utilise: "skip if y > h*0.2" (heuristique basique)
   - Fragile si la résolution change
   - Confond parfois UI avec éléments du jeu

B) DÉTECTION BASÉE UNIQUEMENT SUR LA COULEUR
   - HSV thresholding = fragile aux variations de luminosité
   - Si l'écran est plus/moins lumineux, ça rate
   - Les ennemis peuvent avoir plusieurs couleurs
   - Les items au sol à côté d'autres objets = ambigüité
   
C) ABSENCE DE CONTEXTE SPATIAL
   - On compte juste les pixels rouges
   - Pas de vraie détection d'objets
   - Pas de distinction: "1 grand ennemi" vs "10 petits pixels rouges"
   
D) PARAMÈTRES HARDCODÉS
   - Les seuils HSV sont constants
   - Les aires min/max des contours sont fixes
   - Les régions UI sont fixes en pourcentages
   
E) PAS DE SÉPARATION UI vs ENVIRONNEMENT
   - Tout dans VisionModule
   - Difficile à tester isolément
   - Difficile à fiabiliser sans refactorer
"""

# ============================================================================
# 3. VOTRE IDÉE: SÉPARER UI vs ENVIRONNEMENT ✨
# ============================================================================

"""
EXCELLENTE IDÉE! Voici comment:

AVANT (mélange):
┌──────────────────────────────┐
│    DiabloVisionModule        │
│  - detect_health_bar()       │  ← UI
│  - detect_mana_bar()         │  ← UI
│  - detect_enemies()          │  ← ENVIRONNEMENT
│  - detect_items()            │  ← ENVIRONNEMENT
│  - estimate_player_position()│  ← UI/Position
└──────────────────────────────┘

APRÈS (bien séparé):
┌──────────────────────────┐
│   UIVisionModule         │  ← Gère UNIQUEMENT UI
│ - health_bar()           │
│ - mana_bar()             │
│ - experience_bar()       │
│ - potions_ready()        │
│ - spell_cooldowns()      │
└──────────────────────────┘
           ↑
           │
      [Frame HSV]
           │
           ↓
┌──────────────────────────┐
│  EnvironmentVisionModule │  ← Gère ENVIRONNEMENT
│ - enemies()              │
│ - items()                │
│ - obstacles()            │
│ - doors()                │
│ - traps()                │
│ - player_position()      │
└──────────────────────────┘

Avantages:
✓ Chaque module indépendant
✓ Easier to debug (fixer UI n'impacte pas Env)
✓ Easier to test isolément
✓ Paramètres différents par module
✓ Peut améliorer l'un sans toucher l'autre
"""

# ============================================================================
# 4. PLAN DE FIABILISATION (Par couche)
# ============================================================================

RELIABILITY_PLAN = """

ÉTAPE 1: Refactoriser en UIVisionModule + EnvironmentVisionModule
─────────────────────────────────────────────────────────────────────

src/diabot/vision/
├── ui_vision.py              [NEW]
│   ├── UIVisionModule
│   │   ├── detect_health_bar()
│   │   ├── detect_mana_bar()
│   │   ├── detect_potions_bar()
│   │   └── detect_effects()  # buff/debuff icons
│   │
│   └── UIState dataclass
│       ├── hp: (current, max)
│       ├── mana: (current, max)
│       ├── potions: dict
│       ├── effects: list
│
├── environment_vision.py     [NEW]
│   ├── EnvironmentVisionModule
│   │   ├── detect_enemies()
│   │   ├── detect_items()
│   │   ├── detect_obstacles()
│   │   ├── detect_doors()
│   │   ├── detect_traps()
│   │   └── estimate_player_position()
│   │
│   └── EnvironmentState dataclass
│       ├── enemies: list[EnemyInfo]
│       ├── items: list[ItemInfo]
│       ├── obstacles: list[Obstacle]
│       ├── doors: list[Door]
│       ├── player_pos: (x, y)
│
└── vision_config.py          [NEW]
    ├── HSV_RANGES
    ├── SIZE_THRESHOLDS
    ├── REGION_DEFINITIONS (UI bounds, playfield bounds)


ÉTAPE 2: Créer des régions précises
─────────────────────────────────────────────────────────────────────

Au lieu de hardcoder "y > h*0.2", créer une classe Region:

class ScreenRegion:
    \"\"\"Définit une région de l'écran.\"\"\"
    def __init__(self, name, x_ratio, y_ratio, w_ratio, h_ratio):
        self.name = name
        self.x_ratio = x_ratio  # % du frame
        self.y_ratio = y_ratio
        self.w_ratio = w_ratio
        self.h_ratio = h_ratio
    
    def get_bounds(self, frame_h, frame_w):
        return (
            int(self.x_ratio * frame_w),
            int(self.y_ratio * frame_h),
            int(self.w_ratio * frame_w),
            int(self.h_ratio * frame_h),
        )

# Utilisation:
REGIONS = {
    'ui_top_left': ScreenRegion('ui_top_left', 0.0, 0.0, 0.25, 0.15),
    'ui_bottom': ScreenRegion('ui_bottom', 0.0, 0.85, 1.0, 0.15),
    'playfield': ScreenRegion('playfield', 0.0, 0.15, 1.0, 0.85),
    'minimap': ScreenRegion('minimap', 0.85, 0.0, 0.15, 0.15),
}


ÉTAPE 3: Fiabiliser HSV ranges avec une config externalisée
─────────────────────────────────────────────────────────────────────

vision_config.yaml:
---
ui:
  health_bar:
    color: red
    hsv_range: [[0, 100, 100], [10, 255, 255]]
    min_pixels: 50
  mana_bar:
    color: blue
    hsv_range: [[100, 100, 100], [140, 255, 255]]
    min_pixels: 50

environment:
  enemies:
    - name: "small_enemy"
      colors: [red, orange]
      hsv_ranges:
        - [[0, 80, 80], [10, 255, 255]]     # red
        - [[10, 80, 80], [25, 255, 255]]    # orange
      min_area: 50
      max_area: 10000
  
  items:
    - name: "unique"
      color: gold
      hsv_range: [[15, 100, 100], [35, 255, 255]]
    - name: "set"
      color: green
      hsv_range: [[60, 100, 100], [90, 255, 255]]


ÉTAPE 4: Calibration & Tuning
─────────────────────────────────────────────────────────────────────

Créer un outil de calibration interactif:

calibration_tool.py:
  - Charge une screenshot
  - Permet d'ajuster HSV ranges en temps réel (sliders)
  - Montre les masques détectés
  - Exporte les paramètres qui marchent
  - Test sur plusieurs screenshots

➜ Permet de tuner les ranges une fois, réutilisables


ÉTAPE 5: Validation & Tests
─────────────────────────────────────────────────────────────────────

Créer une test suite:

test_ui_vision.py:
  - Charge screenshots variées
  - Vérifie: health bar détecté correctement
  - Vérifie: mana bar détecté correctement
  - Vérifie: pas de faux positifs (confusions UI/playfield)

test_environment_vision.py:
  - Charge screenshots avec ennemis
  - Vérifie: ennemi détecté
  - Vérifie: comptage correct
  - Vérifie: pas de confusions avec UI


ÉTAPE 6: Logging détaillé pour debug
─────────────────────────────────────────────────────────────────────

class DetectionDebugger:
    def log_detection(self, region_name, mask, contours, results):
        # Sauvegarde une image debug avec:
        # - Région originale
        # - Mask détecté
        # - Contours trouvés
        # - Résultats finaux
        
➜ Permet de voir EXACTEMENT ce qui est détecté
"""

# ============================================================================
# 5. EXEMPLE CODE: Séparation UI vs Environment
# ============================================================================

REFACTOR_EXAMPLE = """

AVANT (mélange):
────────────────

class DiabloVisionModule(VisionModule):
    def perceive(self, frame):
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # UI
        hp = self._detect_health_bar(frame, hsv)
        mana = self._detect_mana_bar(frame, hsv)
        
        # Environment
        enemies = self._detect_enemies(frame, hsv)
        items = self._detect_items(frame, hsv)
        
        return Perception(...)


APRÈS (bien séparé):
────────────────────

class UIVisionModule:
    def __init__(self, config):
        self.config = config
        self.ui_region = REGIONS['ui_top_left']
    
    def analyze(self, frame):
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # Extraire SEULEMENT la région UI
        x, y, w, h = self.ui_region.get_bounds(frame.shape)
        ui_area = hsv[y:y+h, x:x+w]
        
        hp = self._detect_health_bar(ui_area)
        mana = self._detect_mana_bar(ui_area)
        potions = self._detect_potions(ui_area)
        
        return UIState(hp=hp, mana=mana, potions=potions)
    
    def _detect_health_bar(self, ui_area):
        # Configuration spécifique à la barre de santé
        config = self.config['ui']['health_bar']
        hsv_range = config['hsv_range']
        
        mask = cv2.inRange(ui_area, np.array(hsv_range[0]), np.array(hsv_range[1]))
        pixels = cv2.countNonZero(mask)
        
        # Normalize
        ratio = pixels / (ui_area.shape[0] * ui_area.shape[1])
        return min(1.0, ratio * 10)  # Scale factor


class EnvironmentVisionModule:
    def __init__(self, config):
        self.config = config
        self.playfield_region = REGIONS['playfield']
    
    def analyze(self, frame):
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # Extraire SEULEMENT la région playfield
        x, y, w, h = self.playfield_region.get_bounds(frame.shape)
        play_area = hsv[y:y+h, x:x+w]
        
        enemies = self._detect_enemies(play_area)
        items = self._detect_items(play_area)
        obstacles = self._detect_obstacles(play_area)
        player_pos = self._estimate_player_position(frame)
        
        return EnvironmentState(
            enemies=enemies,
            items=items,
            obstacles=obstacles,
            player_pos=player_pos
        )
    
    def _detect_enemies(self, play_area):
        # Configuration spécifique aux ennemis
        results = []
        
        for enemy_config in self.config['environment']['enemies']:
            for hsv_range in enemy_config['hsv_ranges']:
                mask = cv2.inRange(play_area, np.array(hsv_range[0]), np.array(hsv_range[1]))
                contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
                
                for contour in contours:
                    area = cv2.contourArea(contour)
                    if enemy_config['min_area'] < area < enemy_config['max_area']:
                        x, y, cw, ch = cv2.boundingRect(contour)
                        results.append(EnemyInfo(
                            name=enemy_config['name'],
                            pos=(x, y),
                            size=(cw, ch),
                            confidence=area / enemy_config['max_area']
                        ))
        
        return results


# Utilisation combinée:
ui_module = UIVisionModule(config)
env_module = EnvironmentVisionModule(config)

ui_state = ui_module.analyze(frame)
env_state = env_module.analyze(frame)

full_perception = Perception(
    ui=ui_state,
    environment=env_state
)
"""

# ============================================================================
# 6. ROADMAP DE FIABILISATION
# ============================================================================

ROADMAP = """

Phase 1: ARCHITECTURE (1-2 jours)
──────────────────────────────────
[x] Comprendre l'actuel (this document!)
[ ] Créer UIVisionModule
[ ] Créer EnvironmentVisionModule
[ ] Créer ScreenRegion class
[ ] Refactorer pour utiliser les régions

Phase 2: CONFIGURATION (1 jour)
───────────────────────────────
[ ] vision_config.yaml avec tous les ranges
[ ] Charger config depuis fichier
[ ] Paramètres pour chaque élément

Phase 3: CALIBRATION (2-3 jours)
────────────────────────────────
[ ] Créer outil interactif de calibration
[ ] Tester sur 5-10 screenshots variées
[ ] Ajuster les ranges HSV
[ ] Documenter les ranges finaux

Phase 4: FIABILISATION (2-3 jours)
──────────────────────────────────
[ ] Créer test suite (test_ui_vision.py)
[ ] Créer test suite (test_environment_vision.py)
[ ] Ajouter logging détaillé
[ ] Créer debug visualizer

Phase 5: EXTENSIONS (future)
────────────────────────────
[ ] Détecter obstacles (murs, pièges)
[ ] Détecter portes
[ ] Détecter effets spéciaux (feu, froid)
[ ] Détecter ressources (potions, scrolls)

Effort total: ~1 semaine pour une vraie fiabilisation
"""

if __name__ == "__main__":
    print(CURRENT_ISSUES)
    print()
    print(RELIABILITY_PLAN)
    print()
    print(ROADMAP)
