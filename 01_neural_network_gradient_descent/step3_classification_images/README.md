# Step 03 — Classification d’images avec un réseau de neurones

L’objectif de cette troisième étape est d’entraîner un réseau de neurones à trois couches
pour distinguer automatiquement une image de chat d’une image de chien.
Un dataset d’images est importé, prétraité puis utilisé pour l’apprentissage.

## NOUVEAUTÉS ET AMÉLIORATIONS

1. Nouvelle organisation du projet
----------------------------------
Le code est désormais structuré en trois fichiers :

- NeuralNetwork.py
    Contient l’implémentation complète du réseau de neurones
    (architecture, propagation avant, rétropropagation).

- DatasetLoader.py
    Gère le chargement, la mise à l’échelle et la normalisation
    des images du dataset (chats / chiens).

- TrainOrPredict.py
    Permet d’entraîner le modèle ou d’effectuer des prédictions
    à partir de paramètres sauvegardés.

Cette séparation rend le projet plus lisible, modulaire et maintenable.


--------------------------------
2. Sauvegarde des poids et biais
--------------------------------
Après chaque entraînement, les poids (W) et biais (b) sont sauvegardés.
Lors d’une prédiction, il suffit de charger ces paramètres.

Avantages :
- plus besoin de réentraîner le réseau à chaque exécution
- prédictions instantanées
- reproductibilité améliorée
