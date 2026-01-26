# 🗺️ Visualisation des Cartes du Bot

Ce système permet de visualiser les cartes générées automatiquement par le bot pendant qu'il explore Diablo 2.

## 📊 Outils de Visualisation

### 1. **visualize_maps.py** - Visualisation automatique

Génère automatiquement toutes les visualisations disponibles.

```bash
python visualize_maps.py
```

**Sortie :**
- `data/maps/zone_graph.png` - Graphe des zones et leurs connexions
- `data/maps/{zone}_visualization.png` - Minimap de chaque zone avec POIs

**Ce qui est affiché :**
- 📈 Graphe des zones colorées par acte
- 🎯 POIs (waypoints, exits, monsters, NPCs)
- 🔗 Connexions entre les zones
- 📊 Statistiques complètes

---

### 2. **explore_maps.py** - Explorateur interactif

Interface en ligne de commande pour explorer les cartes.

#### Mode interactif (menu)
```bash
python explore_maps.py
```

**Menu disponible :**
1. Afficher les statistiques
2. Visualiser le graphe des zones
3. Visualiser une zone spécifique
4. Lister toutes les zones
5. Afficher les détails d'une zone

#### Commandes directes
```bash
# Lister toutes les zones
python explore_maps.py --list

# Afficher les statistiques
python explore_maps.py --stats

# Visualiser le graphe
python explore_maps.py --graph

# Détails d'une zone spécifique
python explore_maps.py --zone "ROGUE ENCAMPMENT"
```

---

## 🎨 Types de Visualisations

### Graphe des Zones
![Zone Graph](zone_graph_example.png)

**Légende :**
- 🔴 **Act 1** - Rouge
- 🔵 **Act 2** - Cyan
- 🟢 **Act 3** - Bleu clair
- 🟠 **Act 4** - Orange
- 🟣 **Act 5** - Vert menthe
- ⭐ **Or** - Zone avec waypoint
- Lignes noires - Connexions entre zones

### Minimap avec POIs
![Minimap](minimap_example.png)

**POI Colors:**
- 🔵 **Cyan** - Waypoint
- 🟠 **Orange** - Exit/Portal
- 🔴 **Rouge** - Monster
- 🟢 **Vert** - NPC
- 🟡 **Jaune** - Quest
- 🟣 **Magenta** - Shrine

---

## 📁 Structure des Données

### Fichiers générés par le bot

```
data/maps/
├── zones_maps.json          # Données complètes des zones
├── zone_graph.png           # Graphe de navigation
├── rogue_encampment_visualization.png
└── minimap_images/          # Images des minimaps capturées
    ├── rogue_encampment_abc123.png
    └── blood_moor_def456.png
```

### Format JSON (zones_maps.json)

```json
{
  "version": "1.0",
  "last_updated": "2026-01-24T20:24:52.542244",
  "zones": [
    {
      "zone_name": "ROGUE ENCAMPMENT",
      "act": "a1",
      "pois": [
        {
          "name": "Waypoint",
          "poi_type": "waypoint",
          "position": [395, 385],
          "zone": "ROGUE ENCAMPMENT",
          "target_zone": null
        }
      ],
      "connections": {
        "BLOOD MOOR": "Exit to Blood Moor"
      },
      "discovered_at": "2026-01-24T19:24:47.855165"
    }
  ]
}
```

---

## 🚀 Utilisation Avancée

### Intégration dans le workflow

1. **Lancer le bot** pour générer les données
   ```bash
   python src/diabot/main.py
   ```

2. **Visualiser en temps réel** (pendant que le bot tourne)
   ```bash
   python explore_maps.py --stats
   ```

3. **Analyser après exploration**
   ```bash
   python visualize_maps.py
   ```

### Automatisation

Créer un script batch pour visualisation automatique :

```batch
@echo off
echo Generating map visualizations...
python visualize_maps.py
echo.
echo Opening explorer...
python explore_maps.py --graph
pause
```

---

## 📊 Statistiques Affichées

**Exemple de sortie :**

```
======================================================================
MAP STATISTICS
======================================================================

Total Zones: 15
Total POIs: 47
  - Waypoints: 12
  - Exits: 18
  - Monsters: 8
  - NPCs: 5
  - Quests: 4
Total Connections: 23

Zones by Act:
  a1: 8 zones
  a2: 5 zones
  a3: 2 zones

======================================================================
```

---

## 🔧 Dépendances

```bash
pip install matplotlib opencv-python numpy
```

Déjà inclus dans `requirements.txt` du bot.

---

## 💡 Conseils

### Pour de meilleures visualisations :

1. **Laisser le bot explorer plusieurs zones** avant de visualiser
2. **Les minimaps sont générées automatiquement** pendant le jeu
3. **Utiliser `--zone` pour voir les détails** d'une zone spécifique
4. **Le graphe se met à jour automatiquement** à chaque visualisation

### Debugging

Si aucune carte n'apparaît :
```bash
# Vérifier que le fichier existe
dir data\maps\zones_maps.json

# Vérifier le contenu
python -c "import json; print(json.load(open('data/maps/zones_maps.json')))"
```

---

## 🎯 Fonctionnalités Futures

- [ ] Visualisation 3D des zones
- [ ] Animation du parcours du bot
- [ ] Export en HTML interactif
- [ ] Heatmap des zones visitées
- [ ] Comparaison de sessions différentes
- [ ] Export pour outils externes (Graphviz, D3.js)

---

## 📝 Notes

- Les positions des POIs sont en coordonnées de minimap (pixels)
- Les couleurs des actes sont configurables dans `visualize_maps.py`
- Les graphes utilisent un layout horizontal par acte
- La taille des nœuds reflète le nombre de POIs

Enjoy exploring! 🎮
