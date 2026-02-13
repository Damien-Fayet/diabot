# Configuration de l'environnement Diabot

## Environnement virtuel Python

Un environnement virtuel Python a été créé avec toutes les dépendances nécessaires.

### Activation de l'environnement

**macOS/Linux:**
```bash
source venv/bin/activate
# ou
./activate.sh
```

**Windows:**
```cmd
venv\Scripts\activate
```

### Dépendances installées

- **opencv-python** (4.8.0+): Traitement d'image et vision par ordinateur
- **numpy** (1.24.0+): Calcul numérique
- **Pillow** (10.0.0+): Manipulation d'images
- **matplotlib** (3.7.0+): Visualisation de données
- **ultralytics** (8.2.0+): YOLO pour la détection d'objets
- **pytesseract** (0.3.10+): OCR (reconnaissance de texte)
- **pyautogui** (0.9.54+): Automatisation de la souris/clavier
- **dataclasses-json** (0.6.1+): Sérialisation de dataclasses
- **pytest** (7.4.0+): Framework de tests
- **pytest-cov** (4.1.0+): Couverture de code

### Vérification de l'environnement

Pour vérifier que toutes les dépendances sont installées :
```bash
./venv/bin/python verify_env.py
```

### Mise à jour des dépendances

Pour mettre à jour les packages :
```bash
source venv/bin/activate
pip install --upgrade -r requirements.txt
```

### Ajout de nouvelles dépendances

1. Ajouter le package dans `requirements.txt`
2. Installer :
   ```bash
   source venv/bin/activate
   pip install -r requirements.txt
   ```

### Configuration VS Code

Le fichier `.vscode/settings.json` est configuré pour utiliser automatiquement le venv.
VS Code devrait détecter et utiliser l'interpréteur Python du venv.

### Désactivation

Pour quitter l'environnement virtuel :
```bash
deactivate
```

## Note sur pywin32 (Windows uniquement)

Le package `pywin32` est commenté dans `requirements.txt` car il n'est nécessaire que sur Windows.
Pour l'installer sur Windows :
```bash
pip install pywin32>=306
```
