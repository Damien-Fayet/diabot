# Système de Navigation Intelligent avec Cartographie

## Vue d'ensemble

Le bot Diablo 2 dispose maintenant d'un système de navigation avancé qui :
- **Traite la minimap** avec des paramètres optimisés (gamma 3.0, Top Hat, CLAHE)
- **Détecte la position du joueur** via la croix blanche centrale
- **Accumule une carte globale** en mémoire et sur disque
- **Identifie les sorties** en analysant les zones inexplorées
- **Navigue intelligemment** entre exploration et recherche de sortie

## Architecture

```
Frame → Minimap Extraction → MinimapProcessor (optimisé)
                                     ↓
                          MinimapGrid (64×64)
                                     ↓
                          PlayerLocator (croix blanche)
                                     ↓
                          MapAccumulator (carte 2048×2048)
                                     ↓
                          ExitNavigator (détection sorties)
                                     ↓
                          ActionExecutor (clics)
```

## Composants

### 1. MinimapProcessor (Optimisé)
**Fichier:** `src/diabot/navigation/minimap_processor.py`

**Pipeline de traitement:**
1. **Crop HUD** (21% bas) - Retire l'interface
2. **Top Hat** (kernel 5) - Extrait structures brillantes (murs)
3. **Gamma** (3.0) - Contraste extrême
4. **CLAHE** (clip 3.9, grid 8) - Contraste local
5. **Bilateral Filter** - Réduction bruit
6. **Threshold** (49) - Binarisation
7. **Morphology Open** (2) - Supprime petits points
8. **Morphology Close** (8) - Remplit trous

**Résultats:**
- ✓ **5% murs, 95% libre** (avant: 97% murs)
- ✓ Détection précise des chemins praticables
- ✓ Paramètres issus de `tune_minimap_params.py`

### 2. PlayerLocator
**Fichier:** `src/diabot/navigation/player_locator.py`

**Fonctionnalité:**
- Détecte la croix blanche (+) au centre de la minimap
- Utilise seuillage (threshold 220) pour isoler pixels blancs
- Trouve le contour le plus proche du centre
- Retourne position (x, y) exacte du joueur

**Exemple:**
```python
locator = PlayerLocator(debug=True)
player_pos = locator.detect_player_cross(minimap)
# → (734, 355)
```

### 3. MapAccumulator
**Fichier:** `src/diabot/navigation/map_accumulator.py`

**Fonctionnalité:**
- Carte globale 2048×2048 cellules
- Accumule observations minimap frame par frame
- Suit position joueur en coordonnées monde
- Détecte frontières (explorées vs inexplorées)
- Sauvegarde carte en mémoire + disque (JSON + PNG)

**Système de confiance:**
- Chaque cellule a un score de confiance (1-10)
- Observations répétées → confiance accrue
- Murs prioritaires (toujours conservés)

**Méthodes clés:**
```python
accumulator.update(minimap_grid, player_offset)
frontiers = accumulator.find_frontiers(search_radius=50)
exits = accumulator.find_likely_exits(search_radius=30)
accumulator.save_map(zone_name)
```

### 4. ExitNavigator
**Fichier:** `src/diabot/navigation/exit_navigator.py`

**Stratégie de détection des sorties:**
1. Cherche cellules libres à la bordure explorée
2. Compte voisins inconnus (edge_score)
3. Favorise distance du spawn (distance_score)
4. Score combiné: `0.7×edge + 0.3×distance`

**Décision exploration vs sortie:**
- Calcule ratio zone explorée (rayon 50)
- Si < 30% exploré → **EXPLORE**
- Si ≥ 30% exploré → **SEEK EXIT**

**Navigation:**
- Convertit position sortie en clic minimap
- Gère sorties hors minimap (projection directionnelle)
- Évite obstacles (cherche cellule libre proche)

### 5. Intégration Bot Principal
**Fichier:** `src/diabot/main.py`

**Flux d'exécution:**
```python
# Chaque frame:
minimap_grid = minimap_processor.process(minimap_image)
player_pos = player_locator.detect_player_cross(minimap_image)
map_accumulator.update(minimap_grid, player_offset)

# Décision navigation:
if exit_navigator.should_explore_instead(map_accumulator):
    mode = "explore"  # Frontier-based
else:
    mode = "exit_seek"  # Navigate to best exit
    
# Exécution:
if mode == "exit_seek":
    exits = exit_navigator.find_exit_candidates(map_accumulator)
    target = exit_navigator.get_navigation_target(best_exit, ...)
    executor.execute(click_action, target)
else:
    # Fallback to frontier exploration
    frontier_navigator.update(frame)
```

## Paramètres Optimisés

**Fichier source:** `minimap_tuned_params.txt`

```
crop_bottom_percent = 21
tophat_kernel_size = 5
gamma = 3.00
clahe_clip_limit = 3.9
clahe_tile_grid_size = 8
threshold = 49
morph_open_kernel = 2
morph_close_kernel = 8
```

Ces paramètres ont été obtenus via l'outil interactif `tune_minimap_params.py`.

## Utilisation

### Test du système
```powershell
python test_map_navigation.py
```

Génère 4 visualisations:
1. **test_player_detection.png** - Croix blanche détectée
2. **test_occupancy_grid.png** - Grille 64×64 (rouge=murs, gris=libre)
3. **test_accumulated_map.png** - Carte globale accumulée
4. **test_navigation_pipeline.png** - Vue combinée

### Bot en direct
```powershell
python src/diabot/main.py --debug --overlay-show
```

**Logs attendus:**
```
[MinimapProcessor] Crop: 21%, Gamma: 3.0, TopHat: 5
[MinimapProcessor] Grid: 3892 free, 204 walls
[PlayerLocator] Cross detected at (734, 355)
[MapAccumulator] Frame 10: 4096 cells mapped, Player @ (1024, 1024)
[ExitNavigator] Explored 41.0% of nearby area
[ExitNavigator] Decision: SEEK EXIT
[EXIT_NAV] Moving to exit @ (0.68, 0.32) score=0.71
```

### Tuning des paramètres
```powershell
python tune_minimap_params.py
```

Interface interactive avec sliders:
- **Crop Bottom %** (0-30) - Retirer HUD
- **Gamma** (0.1-3.0) - Contraste
- **Top Hat** (1-15) - Extraction structures
- **CLAHE** - Contraste local
- **Threshold** (0-255) - Binarisation
- **Morphology** - Nettoyage

Touches:
- **S** - Sauvegarder paramètres
- **R** - Reset défauts
- **Q** - Quitter

## Visualisations

### Carte Accumulée
- **Blanc brillant** - Zones libres (haute confiance)
- **Gris clair** - Zones libres (faible confiance)
- **Rouge foncé** - Murs (faible confiance)
- **Rouge vif** - Murs (haute confiance)
- **Noir** - Inexploré
- **Vert** - Position joueur
- **Ligne verte** - Trajectoire joueur

Sauvegardée dans: `data/maps/{zone}_{timestamp}_map.png`

### Grille Occupancy
- **Rouge** - Murs (5%)
- **Gris clair** - Libre (95%)
- **Cercle vert** - Centre (joueur)

Résolution: 64×64 cellules, redimensionné 256×256 pour affichage.

## Métriques de Performance

**Avant optimisation:**
- 97% murs, 3% libre (minimap inutilisable)
- Gamma 0.5, CLAHE 3.0, threshold 55

**Après optimisation:**
- 5% murs, 95% libre (excellente détection)
- Gamma 3.0, Top Hat 5, threshold 49

**Temps de traitement:**
- MinimapProcessor: ~50ms/frame
- PlayerLocator: ~5ms/frame
- MapAccumulator: ~10ms/frame
- ExitNavigator: ~20ms/frame
- **Total: ~85ms/frame (12 FPS)**

## Améliorations Futures

1. **Détection direction joueur**
   - Analyser bras de la croix (+)
   - Orientation 0-360°

2. **Template matching sorties**
   - Détecter icônes portes/waypoints
   - Reconnaissance pattern "sortie"

3. **Pathfinding A***
   - Navigation optimale vers sortie
   - Évitement obstacles dynamiques

4. **Fusion carte multizone**
   - Connexions entre zones
   - Graphe de navigation global

5. **Deep Learning**
   - CNN pour classification sortie
   - Segmentation sémantique minimap

## Fichiers Clés

```
src/diabot/navigation/
├── minimap_processor.py      # Traitement minimap optimisé
├── player_locator.py          # Détection croix blanche
├── map_accumulator.py         # Carte globale persistante
└── exit_navigator.py          # Détection + navigation sorties

Outils:
├── tune_minimap_params.py     # Interface tuning interactif
├── test_map_navigation.py     # Test système complet
└── minimap_tuned_params.txt   # Paramètres optimaux

Données:
├── data/maps/                 # Cartes sauvegardées (JSON+PNG)
└── data/screenshots/outputs/  # Visualisations debug
```

## Résumé Technique

Le système de navigation repose sur **vision pure** (pas d'accès mémoire):

1. **Perception** - Traitement minimap optimisé → grille occupancy
2. **Localisation** - Détection croix blanche → position joueur
3. **Cartographie** - Accumulation observations → carte monde
4. **Planification** - Analyse frontières → identification sorties
5. **Exécution** - Conversion coordonnées → clics souris

**Avantages:**
- ✓ Fonctionne avec Diablo 2 Resurrected (offline/online)
- ✓ Pas de détection anti-cheat (vision seulement)
- ✓ Carte persistante (apprentissage zones)
- ✓ Navigation intelligente (explore puis cherche sortie)
- ✓ Paramètres ajustables (tune_minimap_params.py)

**Résultat:** Le bot peut maintenant **explorer méthodiquement une zone, mémoriser la carte, et naviguer vers la sortie de manière autonome**.
