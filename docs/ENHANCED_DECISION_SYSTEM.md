# Enhanced Decision System

## Vue d'ensemble

L'**Enhanced Decision Engine** est le cerveau du bot Diablo 2. Il intègre trois systèmes majeurs :

1. **FSM (Finite State Machine)** - États de jeu de haut niveau
2. **Skill System** - Gestion des compétences et sorts
3. **Inventory System** - Gestion des items et potions

## Architecture

```
EnhancedDecisionEngine
├── FSM (DiabloFSM)
│   ├── IDLE - Aucun danger
│   ├── EXPLORE - Exploration sécurisée
│   ├── ENGAGE - Combat actif
│   ├── KITE - Repositionnement tactique
│   ├── PANIC - Urgence (HP critique)
│   └── RECOVER - Récupération post-combat
│
├── Skills (SkillManager)
│   ├── Combat skill selection
│   ├── Escape skill selection
│   ├── Defensive skill selection
│   └── Mana management
│
└── Inventory
    ├── Belt (4 slots quick access)
    ├── Backpack
    └── Potion management
```

## Système de Skills

### Types de Skills

```python
class SkillType(Enum):
    ATTACK = auto()      # Dégâts directs
    DEFENSIVE = auto()   # Buffs, boucliers
    UTILITY = auto()     # Téléportation, mouvement
    SUMMON = auto()      # Minions
    AOE = auto()         # Dégâts de zone
    DOT = auto()         # Dégâts sur la durée
```

### Exemple d'utilisation

```python
from diabot.models.skills import Skill, SkillBar, SkillType, CharacterClass

# Créer un skill
frozen_orb = Skill(
    name="Frozen Orb",
    skill_type=SkillType.AOE,
    char_class=CharacterClass.SORCERESS,
    mana_cost=25,
    cooldown=1.0,
    damage=100,
    is_aoe=True,
)

# Créer une skill bar
skill_bar = SkillBar(
    left_click=basic_attack,
    right_click=frozen_orb,
    hotkeys=[teleport, None, None, None, None, None, None, None],
)

# Utiliser le skill manager
from diabot.skills import SkillManager

manager = SkillManager(skill_bar)
best_skill = manager.select_combat_skill(game_state)
```

### Sélection intelligente

Le `SkillManager` choisit automatiquement le meilleur skill selon :
- **Nombre d'ennemis** : AOE vs single-target
- **HP actuel** : Défensif si HP bas
- **Mana disponible** : Conservation intelligente
- **État FSM** : Combat vs escape vs défense

## Système d'Inventory

### Structure

```python
from diabot.models.inventory import Inventory, Item, ItemType, PotionSize

inventory = Inventory()

# Ajouter potion à la ceinture
health_pot = Item(
    item_type=ItemType.HEALTH_POTION,
    name="Greater Health Potion",
    potion_size=PotionSize.GREATER,
)
inventory.add_to_belt(health_pot, slot=0)

# Récupérer la meilleure potion
best_hp = inventory.get_best_health_potion()
inventory.use_item(best_hp)
```

### Types d'items

- **Potions** : Health, Mana, Rejuvenation
- **Equipment** : Weapon, Armor
- **Consumables** : Scrolls
- **Resources** : Gold, Gems, Runes

### Priorisation automatique

L'inventory sélectionne automatiquement :
- Plus grande potion disponible en priorité
- Belt avant backpack pour accès rapide
- Gestion intelligente de l'espace

## Enhanced Decision Engine

### Flux de décision

```
GameState
    ↓
1. Urgence potions ? → Action (drink_potion)
    ↓ non
2. Mise à jour FSM
    ↓
3. Action FSM de base
    ↓
4. Enhancement avec skills
    ↓
Action finale
```

### Exemple complet

```python
from diabot.decision.enhanced_engine import (
    EnhancedDecisionEngine,
    EnhancedDecisionContext,
)
from diabot.models.state import GameState, EnemyInfo

# Créer le moteur
engine = EnhancedDecisionEngine()

# État de jeu
state = GameState(
    hp_ratio=0.25,  # HP critique !
    mana_ratio=0.8,
    enemies=[
        EnemyInfo(type="boss", position=(100, 100)),
        EnemyInfo(type="zombie", position=(150, 120)),
    ],
    threat_level="critical",
)

# Décision
action = engine.decide(state)
# → drink_potion (urgence HP)

# Prochain tour (après potion)
state2 = GameState(
    hp_ratio=0.35,  # Toujours bas
    mana_ratio=0.8,
    enemies=[...],
)

action2 = engine.decide(state2)
# → use_skill (Teleport pour s'échapper, FSM=PANIC)
```

### État du système

```python
status = engine.get_status_summary()

# {
#   "fsm_state": "PANIC",
#   "state_duration": 5,
#   "skills": {
#     "total_skills": 3,
#     "ready_skills": 2,
#     "on_cooldown": 1,
#   },
#   "inventory": {
#     "health_potions": 2,
#     "mana_potions": 1,
#     "belt_slots_used": 3,
#   },
#   "recent_transitions": [
#     {"from": "IDLE", "to": "ENGAGE", "reason": "Enemies detected"},
#     {"from": "ENGAGE", "to": "PANIC", "reason": "Critical HP"},
#   ]
# }
```

## Tests

### Lancer les tests

```bash
# Tests skills
python tests/test_skills.py

# Tests inventory
python tests/test_inventory.py

# Tests intégration
python tests/test_enhanced_engine.py

# Tous les tests
pytest tests/
```

### Couverture

- ✅ 6 tests skills (cooldowns, sélection, mana)
- ✅ 7 tests inventory (potions, belt, usage)
- ✅ 6 tests integration (FSM+Skills+Inventory)
- ✅ **19 tests au total**

## Extensibilité

### Ajouter un nouveau skill

```python
from diabot.models.skills import Skill, SkillType, CharacterClass

new_skill = Skill(
    name="Chain Lightning",
    skill_type=SkillType.AOE,
    char_class=CharacterClass.SORCERESS,
    mana_cost=30,
    cooldown=1.5,
    damage=120,
    is_aoe=True,
    range_=25,
)
```

### Ajouter un nouvel état FSM

```python
from diabot.decision.diablo_fsm import FSMState

class CustomFSM(DiabloFSM):
    def update(self, game_state: GameState):
        # Logique custom
        if custom_condition:
            self._transition_to(FSMState.CUSTOM_STATE, "Custom reason")
```

### Créer un SkillManager custom

```python
from diabot.skills import SkillManager

class AdvancedSkillManager(SkillManager):
    def select_combat_skill(self, game_state):
        # Logique améliorée
        if game_state.is_boss_fight:
            return self.boss_specific_skill
        return super().select_combat_skill(game_state)
```

## Prochaines étapes

- [ ] **Pathfinding** - Navigation intelligente
- [ ] **Item pickup logic** - Prioriser les items à ramasser
- [ ] **Boss-specific strategies** - Tactiques par boss
- [ ] **Multi-target selection** - Ciblage intelligent
- [ ] **Resource prediction** - Anticiper les besoins en mana/HP

## Références

- [FSM Documentation](../decision/diablo_fsm.py)
- [Skills Model](../models/skills.py)
- [Inventory Model](../models/inventory.py)
- [Enhanced Engine](../decision/enhanced_engine.py)
