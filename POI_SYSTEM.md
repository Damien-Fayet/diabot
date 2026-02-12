# POI Detection & Map Management

## Vue d'ensemble

Le système de cartographie intègre maintenant la détection de Points d'Intérêt (POI) via YOLO et autres détecteurs, avec gestion complète du cycle de vie des cartes.

## Nouveautés

### 1. Détection POI Automatique

Les POI sont automatiquement détectés et ajoutés à la carte accumulée :
- **NPCs** - Personnages non-joueurs (Akara, Kashya, etc.)
- **Sorties** - Portails et passages vers autres zones
- **Waypoints** - Points de téléportation
- **Coffres** - Chests et conteneurs
- **Sanctuaires** - Shrines avec bonus temporaires
- **Quêtes** - Objectifs de quête

### 2. Intégration YOLO

Le bot analyse chaque frame avec YOLO et :
1. Détecte les objets (classe + confiance + bbox)
2. Convertit les coordonnées écran → carte globale
3. Ajoute automatiquement les POI détectés
4. Évite les duplicatas (merge si < 5 cellules de distance)

### 3. Gestion des Cartes

Nouveau système de nettoyage et maintenance :
- **Clear all** - Supprime toutes les cartes
- **Clear zone** - Supprime une zone spécifique
- **Keep POIs** - Option pour garder les POI lors du clear
- **List maps** - Liste toutes les cartes sauvegardées

## Utilisation

### Ajouter des POI Manuellement

```python
from diabot.navigation.map_accumulator import MapAccumulator

accumulator = MapAccumulator(debug=True)

# Ajouter un NPC
accumulator.add_poi(
    poi_type="npc",
    position=(1030, 1020),  # Coords globales
    label="Akara",
    confidence=0.95
)

# Ajouter une sortie
accumulator.add_poi(
    poi_type="exit",
    position=(990, 1055),
    label="Blood Moor Exit",
    confidence=0.88
)
```

### Nettoyer les Cartes

**Lister toutes les cartes :**
```powershell
python clear_maps.py --list
```

**Supprimer toutes les cartes (avec confirmation) :**
```powershell
python clear_maps.py --clear-all
```

**Supprimer sans confirmation :**
```powershell
python clear_maps.py --clear-all --yes
```

**Supprimer une zone spécifique :**
```powershell
python clear_maps.py --clear-zone ROGUE_ENCAMPMENT
```

### Clear Programmatique

```python
# Clear complet
accumulator.clear(keep_pois=False)

# Clear mais garder les POIs
accumulator.clear(keep_pois=True)
```

## Structure des POI

### Classe MapPOI

```python
@dataclass
class MapPOI:
    poi_type: str              # npc, exit, waypoint, chest, shrine, quest
    position: Tuple[int, int]  # (x, y) en coordonnées globales
    label: str                 # Nom détecté ("Akara", "Waypoint", etc.)
    confidence: float          # Confiance détection (0.0-1.0)
    frame_detected: int        # Frame de première détection
    last_seen: int            # Frame de dernière observation
```

### Types de POI

| Type | Couleur | Description |
|------|---------|-------------|
| `npc` | Cyan (255, 255, 0) | Personnages non-joueurs |
| `exit` | Orange (0, 165, 255) | Sorties et portails |
| `waypoint` | Magenta (255, 0, 255) | Points de téléportation |
| `chest` | Gold (0, 215, 255) | Coffres et conteneurs |
| `shrine` | Pink (203, 192, 255) | Sanctuaires |
| `quest` | Red (0, 0, 255) | Objectifs de quête |

## Mapping YOLO → POI

Le bot convertit automatiquement les classes YOLO en types POI :

```python
poi_type_map = {
    "npc": "npc",
    "waypoint": "waypoint",
    "exit": "exit",
    "portal": "exit",
    "chest": "chest",
    "shrine": "shrine",
    "quest": "quest",
}
```

Classes YOLO génériques (ex: "person") sont mappées à "npc".

## Conversion Coordonnées

### Écran → Carte Globale

Les détections YOLO sont en coordonnées écran :
```python
bbox = [x1, y1, x2, y2]  # Pixels absolus
center = ((x1+x2)/2, (y1+y2)/2)
```

Conversion en coordonnées carte :
```python
# Offset depuis centre écran
offset_x = (cx - frame_width/2) / scale_factor
offset_y = (cy - frame_height/2) / scale_factor

# Position globale
global_x = player_world_pos[0] + offset_x
global_y = player_world_pos[1] + offset_y
```

Le `scale_factor` (défaut: 20) définit combien de pixels écran = 1 cellule carte.

## Visualisation

### Carte avec POI

Les POI apparaissent sur la carte accumulée :
- **Marqueur circulaire** coloré selon le type
- **Label texte** (3 premiers caractères)
- **Bordure noire** pour contraste

Génération :
```python
map_vis = accumulator.visualize(scale=4)
cv2.imshow("Map with POIs", map_vis)
```

### Légende

```
● NPC       (Cyan)
● Exit      (Orange)
● Waypoint  (Magenta)
● Chest     (Gold)
● Shrine    (Pink)
```

## Sauvegarde JSON

Les POI sont inclus dans les métadonnées JSON :

```json
{
  "zone": "ROGUE_ENCAMPMENT",
  "timestamp": "20260127_100945",
  "map_size": 2048,
  "cell_count": 4096,
  "player_pos": [1024, 1024],
  "frame_count": 150,
  "pois": [
    {
      "type": "npc",
      "position": [1030, 1020],
      "label": "Akara",
      "confidence": 0.95,
      "frame_detected": 12
    },
    {
      "type": "waypoint",
      "position": [1010, 1030],
      "label": "Waypoint",
      "confidence": 0.98,
      "frame_detected": 45
    }
  ]
}
```

## Tests

### Test Complet POI

```powershell
python test_poi_mapping.py
```

Démontre :
- ✓ Détection POI avec YOLO
- ✓ Ajout automatique à la carte
- ✓ Visualisation color-coded
- ✓ Sauvegarde JSON avec métadonnées
- ✓ Clear avec/sans rétention POI

### Test Navigation avec POI

```powershell
python test_map_navigation.py
```

Montre la carte accumulée avec tous les POI détectés pendant l'exploration.

## Workflow Bot

Le bot principal intègre maintenant la détection POI :

```
1. Frame capturé
   ↓
2. Vision (YOLO + OCR)
   ↓
3. Détections extraites (yolo_boxes dans raw_data)
   ↓
4. Carte mise à jour (minimap → grid → accumulator)
   ↓
5. POI ajoutés depuis détections YOLO
   ↓
6. Navigation utilise carte + POI
   ↓
7. Sauvegarde périodique (JSON + PNG)
```

### Code Bot

Dans `src/diabot/main.py` :

```python
# STEP 4b: Add detected POIs to map
if perception.raw_data and "yolo_boxes" in perception.raw_data:
    for detection in perception.raw_data["yolo_boxes"]:
        class_name = detection["class_name"]
        confidence = detection["confidence"]
        bbox = detection["bbox"]
        
        # Map to POI type
        poi_type = map_class_to_poi(class_name)
        
        if poi_type:
            # Convert screen → global coords
            global_pos = screen_to_global(bbox, player_pos)
            
            accumulator.add_poi(
                poi_type=poi_type,
                position=global_pos,
                label=class_name,
                confidence=confidence
            )
```

## Maintenance Cartes

### Problèmes Courants

**Cartes obsolètes après changement de paramètres :**
```powershell
python clear_maps.py --clear-all --yes
```

**Réinitialiser une zone spécifique :**
```powershell
python clear_maps.py --clear-zone BLOOD_MOOR
```

**Garder les POI, recalculer la carte :**
```python
accumulator.clear(keep_pois=True)
# Re-run exploration
```

### Best Practices

1. **Clear périodique** - Nettoyer cartes de test
2. **Backup important** - Sauvegarder zones complétées
3. **POI unique** - Éviter doublons (merge auto < 5 cells)
4. **Confidence threshold** - Filtrer détections faibles (< 0.5)

## Performance

### Détection POI

- YOLO inference : ~50ms/frame (GPU)
- POI ajout : < 1ms
- Check duplicata : < 5ms (scan radius 5)

**Total overhead : ~55ms/frame**

### Sauvegarde

- JSON write : ~10ms
- PNG write : ~50ms (dépend de la taille)

**Recommandation :** Sauvegarder toutes les 50 frames (pas chaque frame).

## Limitations Actuelles

1. **Conversion coords approximative** - Scale factor fixe (20)
   - *Solution future :* Projection minimap précise

2. **Détection limitée par YOLO** - Classes génériques (person, door)
   - *Solution future :* Modèle custom D2-specific

3. **Pas de filtrage temporel** - POI ajoutés immédiatement
   - *Solution future :* Confirmation multi-frame

4. **Pas de POI suppression** - POI persistent indéfiniment
   - *Solution future :* Expiration automatique (last_seen > threshold)

## Évolutions Futures

### Court terme
- [ ] Améliorer conversion coords (projection minimap)
- [ ] Filtrer POI par confiance minimale
- [ ] Confirmation multi-frame pour POI critiques

### Moyen terme
- [ ] Modèle YOLO custom pour D2 (waypoints, shrines, etc.)
- [ ] POI clustering intelligent (merge nearby similar)
- [ ] Pathfinding vers POI détectés

### Long terme
- [ ] Reconnaissance NPCs spécifiques (Akara vs Kashya)
- [ ] POI temporels (shrine buff expiration)
- [ ] Graphe de navigation inter-POI

## Fichiers Clés

```
src/diabot/navigation/
├── map_accumulator.py          # +POI tracking +clear()
│   ├── MapPOI dataclass
│   ├── add_poi()
│   ├── clear(keep_pois)
│   └── visualize() avec POI

src/diabot/main.py              # +YOLO→POI integration
└── STEP 4b: POI detection loop

Scripts:
├── clear_maps.py               # Gestion cartes (list/clear)
├── test_poi_mapping.py         # Test POI complet
└── test_map_navigation.py      # Test navigation+POI

Data:
├── data/maps/*.json            # +pois[] array
└── data/maps/*.png             # +POI markers
```

## Exemples

### Clear Toutes les Cartes

```powershell
> python clear_maps.py --clear-all

======================================================================
CLEAR ALL SAVED MAPS
======================================================================

Found 6 map files:
  TEST_POI_ZONE: 2 files
  TEST_ZONE: 2 files
  zones_maps: 1 files

======================================================================
Delete all these files? (yes/no): yes

✓ Deleted 6 map files
======================================================================
```

### Lister les Cartes

```powershell
> python clear_maps.py --list

======================================================================
SAVED MAPS
======================================================================

TEST_POI_ZONE_20260127_100945_metadata
  Zone: TEST_POI_ZONE
  Time: 20260127_100945
  Cells: 4096
  POIs: 5
    → 2 npc, 1 waypoint, 1 exit, 1 chest
======================================================================
```

### Bot avec POI Tracking

```powershell
> python src/diabot/main.py --debug --overlay-show

[VISION] YOLO detected 3 objects
[MapAccumulator] Added POI: npc (npc) @ (1035, 1022)
[MapAccumulator] Updated POI: Waypoint @ (1012, 1028)
[EXIT_NAV] Found exit POI @ (992, 1055)
[NAV_ACTION] Moving to exit
```

## Résumé

Le système de cartographie offre maintenant :

✓ **Détection automatique POI** via YOLO
✓ **Mapping intelligent** (évite doublons)
✓ **Visualisation color-coded** 
✓ **Persistence JSON** (métadonnées complètes)
✓ **Gestion cartes** (clear/list/backup)
✓ **Integration bot** (navigation vers POI)

Le bot peut désormais **mémoriser les positions des NPCs, waypoints, et sorties** et les utiliser pour optimiser sa navigation ! 🎯
