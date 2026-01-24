# Plan de développement - Diablo 2 Bot

## 🎯 Vision
Un agent IA basé sur la vision pour jouer à Diablo 2 en mode developer (screenshots statiques sur macOS).

## 📋 Étapes de démarrage (Phase 1 - Foundation)

### Étape 1: ✅ Infrastructure de base
- [x] Structure Python modulaire (src/, tests/, data/)
- [x] Environnement virtuel et dépendances (OpenCV, NumPy, Pytest)
- [x] Configuration pyproject.toml
- [ ] **PROCHAINE**: Créer interfaces principales

### Étape 2: ✅ Interfaces abstraites (Architecture)
Créer les abstractions pour l'inversion de dépendances:
- `ImageSource` (interface pour capturer images)
- `VisionModule` (interface pour perception)
- `ActionExecutor` (interface pour actions)
- Implémentations concrètes:
  - `ScreenshotFileSource` → charger images du disque
  - `WindowsScreenCapture` → placeholder (non-fonctionnel sur macOS)
  - `RuleBasedVisionModule` → perception simple
  - `DummyActionExecutor` → placeholder

**Fichiers**: `src/diabot/core/interfaces.py`, `src/diabot/core/implementations.py`

### Étape 3: ✅ État et Perception (COMPLETED)
- [x] GameState dataclass → représentation abstraite
- [x] Perception dataclass → résultats de vision
- [x] RuleBasedVisionModule → placeholder
- [x] **DiabloVisionModule** → détection réelle (color thresholding)
- [x] **FastVisionModule** → version optimisée
- [x] **EnhancedStateBuilder** → conversion perception→state
- [x] Tests unitaires de vision
- [x] **Tests d'intégration** → pipeline complet
- [x] DebugOverlay avec visualisation d'état

**Files**: `src/diabot/core/vision_advanced.py`, `src/diabot/builders/state_builder.py`
- Créer `GameState` dataclass (représentation abstraite du jeu)
- Créer `Perception` dataclass (résultats de vision: hp_ratio, enemy_count, etc.)
- Implémenter `StateBuilder` → convertir perception en GameState
- Tests unitaires pour vérifier les conversions

**Fichiers**: `src/diabot/models/state.py`, `src/diabot/builders/state_builder.py`

### Étape 4: ✅ Décision et Action (DONE)
- [x] RuleBasedDecisionEngine basic
- [x] **AdvancedDecisionEngine** → threat-aware decisions
- [x] **Skill system** → spell selection, cooldowns, mana management
- [x] **Inventory system** → items, potions, belt management
- [x] **Enhanced Decision Engine** → FSM + Skills + Inventory integration
- [x] Tests de décision (19 tests passent)

**Fichiers créés**:
- `src/diabot/models/skills.py` - Système de skills complet
- `src/diabot/models/inventory.py` - Gestion d'inventory
- `src/diabot/skills/skill_manager.py` - Sélection intelligente de skills
- `src/diabot/decision/enhanced_engine.py` - Moteur de décision avancé
- `tests/test_skills.py` - Tests du système de skills (6 tests)
- `tests/test_inventory.py` - Tests de l'inventory (7 tests)
- `tests/test_enhanced_engine.py` - Tests d'intégration (6 tests)

### Étape 5: ✅ Item Detection & Classification (COMPLETED)
**Option 1 du plan d'amélioration - Détection d'items par couleur**
- [x] Item detection par HSV (Unique/Set/Rare/Magic/Normal)
- [x] Item classification en tiers S/A/B/C/D
- [x] Base de données JSON configurable (12 items + 6 runewords)
- [x] Tests de détection (13 tests passent)

**Fichiers créés**:
- `src/diabot/items/item_detector.py` - ItemDetector avec HSV ranges
- `src/diabot/items/item_classifier.py` - Classification par tiers
- `data/items_database.json` - Règles configurables
- `tests/test_items.py` - 13 tests (tous passent ✅)
- `src/diabot/items/__init__.py`

### Étape 6: ✅ Session Logging & Analytics (COMPLETED)
**Option 5 du plan d'amélioration - Logging persistant + Analytics**
- [x] SessionLogger avec 5 types d'événements
- [x] SessionMetrics avec tracking détaillé
- [x] SessionAnalytics pour rapports single-session
- [x] MultiSessionAnalytics pour tendances cross-session
- [x] Tests de logging (12 tests passent)
- [x] Demo script complet

**Fichiers créés**:
- `src/diabot/logging/session_logger.py` - SessionLogger avec événements
- `src/diabot/stats/analytics.py` - Analytics simple et multi-session
- `tests/test_logging.py` - 12 tests de logging/analytics (tous passent ✅)
- `demo_logging_system.py` - Demo avec 3 sessions + trends
- `src/diabot/logging/__init__.py`
- `src/diabot/stats/__init__.py`

**Résultats de la démo**:
- Session simple: 2 items S/C tier, 2 kills, 125 dmg dealt, efficiency 100/100
- 3 sessions de trend: 14 items total, 25 kills, survival rate 75%
- Rapports détaillés avec breakdown par type d'événement

- Créer `Action` dataclass (what to do)
- Connecter: GameState → Decision → Action

**Fichiers**: `src/diabot/engines/decision_engine.py`

### Étape 5 (ancienne): ✅ Debugging & Visualisation (DONE)
- Créer `DebugOverlay` utility pour afficher l'état sur l'image
- Tester avec une screenshot en mode développeur

**Fichiers**: `src/diabot/debug/overlay.py`

### Étape 6 (ancienne): ✅ Main runnable (DONE)
- Créer `main.py` en mode developer qui:
  1. Charge une screenshot
  2. Lance perception
  3. Construit état
  4. Prend décision
  5. Affiche overlay de debug
  
**Fichiers**: `src/diabot/main.py`, `scripts/run_dev.py`

---

## 🔧 Dépendances clés
- **OpenCV** (4.13+): manipulation images
- **NumPy** (2.4+): calculs matriciels
- **Pytest** (9.0+): tests
- **Dataclasses**: typing et structure données

## 🎮 Mode operandi (pour maintenant)
- **macOS only** pour Phase 1
- **Pas de ML/RL** encore
- **Pas de logique Diablo-spécifique** réelle
- Focus: scaffolding, architecture, extensibilité

## 📁 Structure finale Phase 1
```
diabot/
├── src/diabot/
│   ├── __init__.py
│   ├── core/
│   │   ├── interfaces.py      (abstractions)
│   │   └── implementations.py (concrètes)
│   ├── models/
│   │   └── state.py           (dataclasses)
│   ├── builders/
│   │   └── state_builder.py   (Perception → State)
│   ├── engines/
│   │   └── decision_engine.py (State → Action)
│   ├── debug/
│   │   └── overlay.py         (visualisation)
│   └── main.py               (orchestration)
├── scripts/
│   └── run_dev.py            (entrypoint)
├── tests/
│   ├── test_state_builder.py
│   └── test_decision_engine.py
├── data/
│   └── screenshots/          (test images)
├── pyproject.toml
└── requirements.txt
```

---

**Prochaine action**: Commencer Étape 4 (Décision Avancée) ✨
