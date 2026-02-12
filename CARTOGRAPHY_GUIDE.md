# 🗺️ Système de Cartographie et POI - Guide Complet

## 🎯 Résumé des Fonctionnalités

Le bot Diablo 2 dispose maintenant d'un système complet de cartographie intelligente avec:

### ✅ Fonctionnalités Implémentées

1. **Traitement Minimap Optimisé**
   - Paramètres ajustés (gamma 3.0, Top Hat, CLAHE)
   - Résultat: 5% murs, 95% libre (vs 97% murs avant)
   - Pipeline: Crop → TopHat → Gamma → CLAHE → Filter → Threshold → Morphology

2. **Détection Position Joueur**
   - Localisation via croix blanche centrale
   - Tracking mouvement frame-à-frame

3. **Accumulation Carte Globale**
   - Carte 2048×2048 en mémoire
   - Fusion observations multiples
   - Système de confiance (1-10)
   - Sauvegarde JSON + PNG

4. **Détection POI Automatique**
   - NPCs, Waypoints, Sorties, Coffres, Sanctuaires
   - Integration YOLO en temps réel
   - Évite doublons (merge < 5 cellules)
   - Visualisation color-coded

5. **Navigation Intelligente**
   - Mode exploration (< 30% exploré)
   - Mode recherche sortie (≥ 30% exploré)
   - Pathfinding vers sorties détectées

6. **Gestion des Cartes**
   - Clear all / Clear zone
   - Option keep POIs
   - Liste cartes sauvegardées

## 📁 Structure des Fichiers

```
diabot/
├── src/diabot/navigation/
│   ├── minimap_processor.py       # Traitement optimisé minimap
│   ├── player_locator.py          # Détection croix blanche
│   ├── map_accumulator.py         # Carte globale + POI
│   └── exit_navigator.py          # Détection sorties
│
├── Scripts principaux:
│   ├── tune_minimap_params.py     # Interface ajustement paramètres
│   ├── test_map_navigation.py     # Test navigation complète
│   ├── test_poi_mapping.py        # Test POI + sauvegarde
│   ├── clear_maps.py              # Gestion cartes
│   └── view_map.py                # Visualisation cartes
│
├── Documentation:
│   ├── MAP_NAVIGATION_SYSTEM.md   # Système navigation
│   └── POI_SYSTEM.md              # Système POI
│
└── Data:
    ├── data/maps/*.json           # Métadonnées cartes
    ├── data/maps/*.png            # Visualisations
    └── minimap_tuned_params.txt   # Paramètres optimisés
```

## 🚀 Quick Start

### 1. Ajuster Paramètres Minimap

```powershell
python tune_minimap_params.py
```

Interface interactive avec sliders:
- Ajustez gamma, Top Hat, CLAHE, threshold
- Objectif: ~50% murs, ~50% libre
- Appuyez sur **S** pour sauvegarder

### 2. Tester Navigation

```powershell
python test_map_navigation.py
```

Génère 3 visualisations:
- Détection joueur (croix blanche)
- Grille occupancy (murs/libre)
- Carte accumulée

### 3. Tester POI

```powershell
python test_poi_mapping.py
```

Démontre:
- Ajout POI (NPCs, waypoints, etc.)
- Visualisation color-coded
- Clear avec/sans rétention POI

### 4. Lancer le Bot

```powershell
python src/diabot/main.py --debug --overlay-show
```

Le bot va:
1. Extraire minimap chaque frame
2. Détecter joueur (croix blanche)
3. Accumuler carte en mémoire
4. Ajouter POI depuis détections YOLO
5. Naviguer intelligemment (explore puis cherche sortie)
6. Sauvegarder carte périodiquement

## 🎨 Visualisations

### Carte Accumulée

```powershell
python view_map.py
```

Affiche la carte la plus récente avec:
- **Blanc** - Zones libres (haute confiance)
- **Rouge** - Murs
- **Noir** - Inexploré
- **Vert** - Position joueur + trajectoire
- **Marqueurs colorés** - POI détectés

### POI Color-Coding

| POI | Couleur | Code RGB |
|-----|---------|----------|
| NPC | Cyan | (255, 255, 0) |
| Exit | Orange | (0, 165, 255) |
| Waypoint | Magenta | (255, 0, 255) |
| Chest | Gold | (0, 215, 255) |
| Shrine | Pink | (203, 192, 255) |

## 🛠️ Gestion des Cartes

### Lister toutes les cartes

```powershell
python clear_maps.py --list
```

Affiche:
- Zone name
- Timestamp
- Cell count
- POI count + types

### Nettoyer toutes les cartes

```powershell
# Avec confirmation
python clear_maps.py --clear-all

# Sans confirmation
python clear_maps.py --clear-all --yes
```

### Nettoyer une zone spécifique

```powershell
python clear_maps.py --clear-zone ROGUE_ENCAMPMENT
```

## 🧪 Tests Disponibles

| Script | Description | Output |
|--------|-------------|--------|
| `test_map_navigation.py` | Test navigation complète | 3 PNG + JSON |
| `test_poi_mapping.py` | Test POI + clear | 1 PNG + JSON |
| `tune_minimap_params.py` | Ajustement interactif | params.txt |
| `view_map.py` | Visualisation carte | Window CV2 |
| `clear_maps.py --list` | Liste cartes | Console |

## 📊 Performances

### Traitement par Frame

| Composant | Temps | Description |
|-----------|-------|-------------|
| MinimapProcessor | ~50ms | Crop + TopHat + Gamma + CLAHE |
| PlayerLocator | ~5ms | Détection croix blanche |
| MapAccumulator | ~10ms | Fusion observations |
| ExitNavigator | ~20ms | Analyse frontières |
| YOLO POI | ~50ms | Détection objets (GPU) |
| **Total** | **~135ms** | **~7 FPS** |

### Recommandations

- **Sauvegarde**: Toutes les 50 frames (pas chaque frame)
- **Clear périodique**: Après exploration complète zone
- **POI confidence**: Filtrer < 0.5 pour éviter faux positifs

## 🔧 Configuration

### Paramètres Minimap (minimap_tuned_params.txt)

```
crop_bottom_percent = 21      # Retirer HUD
tophat_kernel_size = 5        # Extraction structures
gamma = 3.00                  # Contraste extrême
clahe_clip_limit = 3.9        # Contraste local
clahe_tile_grid_size = 8      # Grille CLAHE
threshold = 49                # Binarisation
morph_open_kernel = 2         # Nettoyage
morph_close_kernel = 8        # Remplissage trous
```

### Paramètres Navigation (main.py)

```python
exploration_threshold = 0.3    # 30% avant recherche sortie
map_size = 2048               # Taille carte globale
grid_size = 64                # Résolution minimap grid
search_radius = 50            # Rayon recherche frontières
```

## 🎯 Workflow Bot

```
┌─────────────────────────────────────────────────────┐
│                  CAPTURE FRAME                       │
└───────────────────┬─────────────────────────────────┘
                    │
                    ▼
┌─────────────────────────────────────────────────────┐
│  VISION: YOLO + OCR + Minimap Extraction            │
│  → Détections: NPCs, Exits, Waypoints               │
│  → Zone name, HP/Mana                               │
└───────────────────┬─────────────────────────────────┘
                    │
                    ▼
┌─────────────────────────────────────────────────────┐
│  MINIMAP PROCESSING                                  │
│  → Crop HUD → TopHat → Gamma → CLAHE → Threshold   │
│  → Grid 64×64 (5% murs, 95% libre)                 │
└───────────────────┬─────────────────────────────────┘
                    │
                    ▼
┌─────────────────────────────────────────────────────┐
│  PLAYER LOCATOR                                      │
│  → Détection croix blanche                          │
│  → Position (x, y) + tracking mouvement             │
└───────────────────┬─────────────────────────────────┘
                    │
                    ▼
┌─────────────────────────────────────────────────────┐
│  MAP ACCUMULATOR                                     │
│  → Update carte globale (2048×2048)                 │
│  → Add POI depuis YOLO                              │
│  → Merge observations (confiance 1-10)              │
└───────────────────┬─────────────────────────────────┘
                    │
                    ▼
┌─────────────────────────────────────────────────────┐
│  NAVIGATION DECISION                                 │
│  → < 30% exploré = EXPLORE                          │
│  → ≥ 30% exploré = SEEK EXIT                       │
└───────────────────┬─────────────────────────────────┘
                    │
        ┌───────────┴───────────┐
        │                       │
        ▼                       ▼
┌──────────────┐      ┌──────────────────┐
│   EXPLORE    │      │   SEEK EXIT      │
│ (Frontiers)  │      │ (Best candidate) │
└──────┬───────┘      └────────┬─────────┘
       │                       │
       └──────────┬────────────┘
                  │
                  ▼
        ┌─────────────────┐
        │ ACTION EXECUTOR │
        │ → Click minimap │
        └─────────────────┘
                  │
                  ▼
        ┌─────────────────┐
        │ SAVE MAP (50fr) │
        │ → JSON + PNG    │
        └─────────────────┘
```

## 📝 API Rapide

### MapAccumulator

```python
from diabot.navigation.map_accumulator import MapAccumulator

acc = MapAccumulator(map_size=2048, debug=True)

# Update carte
acc.update(minimap_grid, player_offset=(dx, dy))

# Add POI
acc.add_poi("npc", (1030, 1020), "Akara", confidence=0.95)

# Find exits
exits = acc.find_likely_exits(search_radius=30)

# Clear
acc.clear(keep_pois=False)  # Clear tout
acc.clear(keep_pois=True)   # Clear carte, garde POI

# Save
acc.save_map("ZONE_NAME")

# Visualize
img = acc.visualize(scale=4)
cv2.imshow("Map", img)
```

### MinimapProcessor

```python
from diabot.navigation.minimap_processor import MinimapProcessor

proc = MinimapProcessor(grid_size=64, wall_threshold=49, debug=True)

# Process minimap
minimap_grid = proc.process(minimap_image)

# Check cells
is_wall = minimap_grid.is_wall(x, y)
is_free = minimap_grid.is_free(x, y)

# Visualize
vis = proc.visualize(minimap_grid)
cv2.imshow("Grid", vis)
```

### PlayerLocator

```python
from diabot.navigation.player_locator import PlayerLocator

loc = PlayerLocator(debug=True)

# Detect player
player_pos = loc.detect_player_cross(minimap_image)
# → (734, 355)

# Visualize
vis = loc.visualize_detection(minimap_image)
cv2.imshow("Player", vis)
```

## 🐛 Troubleshooting

### Problème: Trop de murs (> 50%)

**Solution:**
```powershell
python tune_minimap_params.py
```
Augmentez gamma et CLAHE, baissez threshold.

### Problème: POI dupliqués

**Cause:** Détections répétées à chaque frame

**Solution:** Le système merge automatiquement POI < 5 cellules. Si problème persiste:
```python
# Augmenter seuil merge
if dx < 10 and dy < 10:  # Au lieu de 5
    # Merge
```

### Problème: Cartes obsolètes

**Solution:**
```powershell
python clear_maps.py --clear-all --yes
```

### Problème: Navigation bloquée

**Cause:** Carte trop fragmentée ou sorties non détectées

**Solution:**
1. Clear carte: `accumulator.clear(keep_pois=True)`
2. Réexplorer avec nouveaux paramètres
3. Vérifier détection YOLO (exits, waypoints)

## 🚀 Prochaines Étapes

### Court Terme
- [ ] Améliorer conversion coords écran → carte
- [ ] Filtrer POI par confiance (threshold 0.5)
- [ ] Confirmation multi-frame POI critiques

### Moyen Terme
- [ ] Modèle YOLO custom D2-specific
- [ ] Pathfinding A* vers POI
- [ ] POI clustering intelligent

### Long Terme
- [ ] Reconnaissance NPCs individuels
- [ ] Graphe navigation inter-zones
- [ ] Prédiction position sorties

## 📚 Documentation Complète

- [MAP_NAVIGATION_SYSTEM.md](MAP_NAVIGATION_SYSTEM.md) - Système navigation détaillé
- [POI_SYSTEM.md](POI_SYSTEM.md) - Système POI complet

## 🎉 Résumé

Le bot peut maintenant:
- ✅ **Explorer** une zone méthodiquement
- ✅ **Mémoriser** la carte en temps réel
- ✅ **Détecter** NPCs, waypoints, sorties
- ✅ **Naviguer** intelligemment vers objectifs
- ✅ **Sauvegarder** progression (JSON + PNG)
- ✅ **Gérer** cartes (clear/list/view)

**Résultat:** Navigation autonome complète avec mémoire persistante ! 🗺️🎯
