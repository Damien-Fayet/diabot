#!/bin/bash
# Script d'activation rapide du venv Diabot

# Activer le venv
source venv/bin/activate

# Afficher la version Python
echo "Python activé: $(python --version)"
echo "Environnement: $(which python)"
echo ""
echo "Utilisez 'deactivate' pour quitter le venv"
echo "Pour exécuter le bot: python main.py"
