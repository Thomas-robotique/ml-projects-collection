# Step 03 — Classification d’images avec un réseau de neurones

L’objectif de cette troisième étape est d’entraîner un réseau de neurones à trois couches
pour distinguer automatiquement une image de chat d’une image de chien.
Le dataset d'images permettant l'entraînement a été importé avec TensorFlow.

## NOUVEAUTÉS ET AMÉLIORATIONS

### 1. Nouvelle organisation du projet
----------------------------------
Le code est désormais structuré en trois fichiers :

- `NeuralNetwork.py`
    Contient l’implémentation complète du réseau de neurones
    (architecture, propagation avant, rétropropagation).

- `load_cat_and_dog.py`
    Gère le chargement, la mise à l’échelle et la normalisation
    des images du dataset (chats / chiens) et des images à prédire.

- `cat_and_dog.py`
    Permet d’entraîner le modèle ou d’effectuer des prédictions
    à partir de paramètres sauvegardés.
  
- `dataset2`
  Contient toutes les photos pour l'entraînement du réseau de neurones

- `dataset_tensorflow.py `
  Contient les fonctions permettant de trier les photos, de supprimer celles qui sont inutiles, puis de récupérer les images restantes pour l’entraînement.

Cette séparation rend le projet plus lisible et modulaire 


### 2. Sauvegarde des poids et biais
--------------------------------
Après chaque entraînement, les poids (W) et biais (b) sont sauvegardés.
Lors d’une prédiction, il suffit de charger ces paramètres.

Avantages :
- plus besoin de réentraîner le réseau à chaque exécution
- prédictions instantanées
- reproductibilité améliorée
