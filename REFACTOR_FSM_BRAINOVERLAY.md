# Améliorations - FSM et BrainOverlay ✅

## 📝 Résumé des Modifications

Suite aux instructions ajoutées dans `project.instructions.md`, le code a été adapté pour implémenter:
1. **Finite State Machine (FSM)** - Diablo-inspired decision engine
2. **BrainOverlay** - Advanced visual debug system
3. **Réorganisation des fichiers** - Séparation inputs/outputs

---

## 🗂️ Structure des Screenshots

### Avant
```
data/screenshots/
├── char_menu.jpg
├── game_screen_2.jpg
├── output_debug.png
└── integration_*.png
```

### Après ✅
```
data/screenshots/
├── inputs/              ← Images sources (originales)
│   ├── char_menu.jpg
│   └── game.jpg
└── outputs/             ← Images générées par tests
    ├── brain_overlay.png
    ├── integration_char_menu.png
    └── integration_game.png
```

**Bénéfices**:
- Séparation claire sources/résultats
- Pas de mélange entre inputs et outputs
- Facile à nettoyer (supprimer outputs sans toucher sources)

---

## 🤖 Finite State Machine (FSM)

### Fichier: `src/diabot/decision/diablo_fsm.py`

**États implémentés**:
```python
FSMState.IDLE      # No enemies, waiting/scanning
FSMState.EXPLORE   # Moving to unexplored area
FSMState.ENGAGE    # Enemies detected, attacking
FSMState.KITE      # Too close, repositioning
FSMState.PANIC     # Low HP or surrounded
FSMState.RECOVER   # Regaining resources
```

**Priorités de transition** (plus haute à plus basse):
1. **PANIC** - Survie critique (HP<30% + menacé)
2. **RECOVER** - Récupération post-danger
3. **KITE** - Trop d'ennemis (≥5) ou menace critique
4. **ENGAGE** - Ennemis présents mais gérables
5. **EXPLORE** - Zone sûre, exploration
6. **IDLE** - État par défaut

**Features**:
- ✅ Transitions déterministes
- ✅ Historique des transitions avec raisons
- ✅ Durée dans chaque état
- ✅ Actions mappées par état
- ✅ Facilement extensible

**Exemple de transition**:
```
IDLE → PANIC: Critical: HP=0%, Enemies=10
```

### Tests FSM (`tests/test_fsm.py`)

6 tests unitaires créés:
- ✅ `test_fsm_panic_transition` - Transition vers PANIC
- ✅ `test_fsm_engage_transition` - Transition vers ENGAGE
- ✅ `test_fsm_kite_transition` - Transition vers KITE
- ✅ `test_fsm_explore_transition` - Transition vers EXPLORE
- ✅ `test_fsm_transition_history` - Historique des transitions
- ✅ `test_fsm_state_duration` - Tracking durée

**Résultat**: 6/6 tests PASSING ✓

---

## 🧠 BrainOverlay (Visual Debug System)

### Fichier: `src/diabot/debug/overlay.py`

**Nouvelle classe**: `BrainOverlay`

**Ce qu'elle affiche**:

1. **Top-Left Panel**:
   - État FSM (avec couleur selon état)
   - Action décidée + cible
   - HP ratio (perception)
   - Mana ratio (perception)
   - Nombre d'ennemis
   - Threat level (avec couleur)
   - Location

2. **Bottom-Left**: 
   - Barre de santé dynamique (vert→rouge)
   - Texte HP %

3. **Top-Right**:
   - Indicateur circulaire de menace
   - Couleur: vert (safe) → orange (warning) → rouge (danger)
   - Nombre d'ennemis dans le cercle

**Couleurs utilisées**:
- 🟢 Green = Safe/OK
- 🔴 Red = Danger/Critical
- 🟠 Orange = Warning
- 🔵 Blue = Target
- ⚪ White = Info

**Configuration**:
```python
brain_overlay = BrainOverlay(enabled=True)
output = brain_overlay.draw(
    frame=frame,
    perception=perception,
    state=state,
    action=action,
    fsm_state=fsm_state.name,
)
```

**Avantages**:
- ✅ Comprendre ce que l'agent perçoit
- ✅ Voir pourquoi il prend une décision
- ✅ Debugger les transitions FSM
- ✅ Valider les détections visuelles
- ✅ Aucun couplage avec vision logic
- ✅ Toggleable (enabled=True/False)

---

## 🔄 Modifications des Scripts

### `scripts/run_dev_advanced.py`

**Changements**:
- ✅ Import `DiabloFSM` et `BrainOverlay`
- ✅ Utilise `EnhancedStateBuilder` (au lieu de SimpleStateBuilder)
- ✅ FSM pour décision (au lieu de RuleBasedDecisionEngine)
- ✅ BrainOverlay (au lieu de DebugOverlay)
- ✅ Chemin par défaut: `inputs/game.jpg`
- ✅ Sortie: `outputs/brain_overlay.png`
- ✅ Affiche état FSM et transitions

### `tests/test_integration.py`

**Changements**:
- ✅ Import `DiabloFSM` et `BrainOverlay`
- ✅ Utilise FSM pour décision
- ✅ BrainOverlay pour visualisation
- ✅ Chemins: `inputs/*.jpg` → `outputs/*.png`
- ✅ Affiche transitions FSM dans output

### `tests/test_vision.py`

**Changements**:
- ✅ Chemins: `inputs/*.jpg`

### `scripts/analyze_screenshots.py`

**Changements**:
- ✅ Analyse seulement `inputs/`

---

## 📊 Résultats des Tests

### Tests Unitaires FSM
```bash
$ python tests/test_fsm.py
✓ test_fsm_panic_transition
✓ test_fsm_engage_transition
✓ test_fsm_kite_transition
✓ test_fsm_explore_transition
✓ test_fsm_transition_history
✓ test_fsm_state_duration
✅ All FSM tests passed!
```

### Tests d'Intégration
```bash
$ python tests/test_integration.py
📸 TEST: Character Menu
  FSM State: PANIC
  Action: drink_potion
  Transition: IDLE → PANIC: Critical: HP=8%, Enemies=1
  ✅ Pipeline test passed!

📸 TEST: Game Screen
  FSM State: PANIC
  Action: drink_potion
  Transition: IDLE → PANIC: Critical: HP=0%, Enemies=10
  ✅ Pipeline test passed!

📊 RESULTS: 2 passed, 0 failed
```

### Script Dev Mode
```bash
$ python scripts/run_dev_advanced.py
✓ Loaded frame: (720, 1280, 3)
✓ Perception: HP=0.0%, Enemies=10, Items=5
✓ State built: Threat Level=critical
✓ FSM State: PANIC
✓ Decision: drink_potion
    Transition: IDLE → PANIC: Critical: HP=0%, Enemies=10
✓ Debug overlay saved: brain_overlay.png
✅ Bot cycle complete!
```

---

## 🎯 Conformité aux Instructions

### Instruction: BrainOverlay

**Requirements** ✅:
- [x] Implement BrainOverlay class using OpenCV
- [x] Receives: frame, perception, state, action
- [x] Draws: text overlay (FSM state, action, hp/mana, enemy count)
- [x] Optional bounding boxes (préparé pour futures détections)
- [x] Colored indicators (red=danger, green=safe, blue=target)
- [x] Purely visual (no game interaction)
- [x] Toggleable via configuration
- [x] Isolated in debug/overlay.py
- [x] NOT coupled to vision logic
- [x] Clear docstrings

### Instruction: Diablo FSM

**Requirements** ✅:
- [x] States reflect human gameplay intuition
- [x] Prioritize survival over optimization
- [x] Easy to extend
- [x] Required states: IDLE, EXPLORE, ENGAGE, KITE, PANIC, RECOVER
- [x] State transitions driven by abstract state (not raw pixels)
- [x] Clear, readable conditions (hp_ratio thresholds, enemy_count)
- [x] Transitions logged for debugging
- [x] State enum created
- [x] DiabloFSM class created
- [x] Exposes: update(state_data), decide_action(state_data)
- [x] Deterministic and testable
- [x] No RL at this stage
- [x] Implementation in decision/diablo_fsm.py
- [x] Transition table with readable logic
- [x] Example integration in scripts
- [x] Inline comments explaining gameplay reasoning
- [x] BrainOverlay displays FSM state and decision driver

---

## 📁 Nouveaux Fichiers Créés

1. **`src/diabot/decision/diablo_fsm.py`** (230 lignes)
   - FSM implementation complète
   - 6 états + transitions
   - Historique et durée tracking

2. **`src/diabot/decision/__init__.py`**
   - Module init

3. **`tests/test_fsm.py`** (195 lignes)
   - 6 tests unitaires FSM
   - Couvre toutes les transitions importantes

4. **`REFACTOR_FSM_BRAINOVERLAY.md`** (ce fichier)
   - Documentation complète des changements

---

## 📁 Fichiers Modifiés

1. **`src/diabot/debug/overlay.py`**
   - Ajout classe `BrainOverlay` (150+ lignes)
   - Conserve `DebugOverlay` pour compatibilité

2. **`scripts/run_dev_advanced.py`**
   - Intégration FSM + BrainOverlay
   - Chemins inputs/outputs

3. **`tests/test_integration.py`**
   - Utilise FSM + BrainOverlay
   - Chemins inputs/outputs

4. **`tests/test_vision.py`**
   - Chemins inputs

5. **`scripts/analyze_screenshots.py`**
   - Chemins inputs

---

## 🎨 Outputs Visuels

**Fichiers générés dans `outputs/`**:

1. **`brain_overlay.png`** (run_dev_advanced.py)
   - Affiche FSM state en haut
   - Action décidée
   - Perception data (HP, Mana, Enemies)
   - Threat level avec couleur
   - Barre de santé en bas
   - Indicateur circulaire de menace en haut-droite

2. **`integration_char_menu.png`** (test character menu)
   - FSM: PANIC
   - Action: drink_potion
   - HP: 8% (rouge)

3. **`integration_game.png`** (test game screen)
   - FSM: PANIC
   - Action: drink_potion
   - HP: 0% (rouge critique)
   - Enemies: 10 (dans cercle rouge)

---

## 💡 Bénéfices de ces Changements

### Architecture
- ✅ FSM rend les décisions explicites et traçables
- ✅ BrainOverlay sépare debugging de logique métier
- ✅ Code découplé et testable
- ✅ Facile à étendre (nouveaux états FSM, nouvelles visualisations)

### Developer Experience
- ✅ Comprendre pourquoi l'agent prend une décision
- ✅ Visualiser transitions FSM
- ✅ Debugger perception vs décision
- ✅ Screenshots organisés (sources vs outputs)

### Qualité Code
- ✅ 100% tests passing (FSM + intégration)
- ✅ Docstrings complètes
- ✅ Type hints partout
- ✅ Clean architecture respectée

---

## 🚀 Prochaines Étapes

Maintenant que FSM + BrainOverlay sont implémentés, on peut:

1. **Étape 4**: Décision & Action avancée
   - Skill selection logic
   - Pathfinding/movement
   - Inventory management
   - Advanced threat assessment

2. **Améliorer FSM**:
   - Ajouter sous-états (PANIC_FLEE, PANIC_DRINK, etc.)
   - Cooldowns sur certaines transitions
   - Historique d'état pour patterns

3. **Améliorer BrainOverlay**:
   - Bounding boxes pour ennemis détectés
   - Trajectoires de mouvement prévues
   - Skill cooldowns display
   - Minimap overlay

---

## ✅ Checklist Finale

- [x] FSM implémentée avec 6 états
- [x] BrainOverlay créé avec visualisation complète
- [x] Tests FSM (6 tests) ✓
- [x] Tests d'intégration mis à jour ✓
- [x] Scripts mis à jour (chemins + FSM + BrainOverlay)
- [x] Screenshots réorganisés (inputs/outputs)
- [x] Documentation complète
- [x] Tous tests passing (100%)
- [x] Conformité aux instructions ✓

**Statut**: ✅ READY FOR STEP 4
