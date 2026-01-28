# Step 02 — Réseau de neurones

Dans cette deuxième étape, je passe d’un neurone unique à un réseau de neurones, construit entièrement à la main en Python.  
L’objectif est de comprendre comment fonctionne un réseau avec une couche cachée, comment les informations circulent à travers lui, et comment les poids s’ajustent pendant l’apprentissage.


## Ce que j’ai implémenté
Dans cette étape, j’ai ajouté plusieurs éléments importants :

- 2 couches de neurones  
- des fonctions d’activation  
- la propagation avant sur tout le réseau  
- la rétropropagation pour mettre à jour les poids  
- l’entraînement complet du modèle sur un dataset simple  

L’idée est de comprendre comment un réseau apprend, comment l’erreur circule en arrière, et comment les paramètres s’ajustent petit à petit.

##  Organisation des fichiers
Comme pour l'étape 1, l'organisation des fichiers est assez similaire.

####  `Neuron_Network2.py`
Contient :
- le code complet du réseau de neurone
- l’implémentation de la descente de gradient
- les fonctions d’entraînement et de prédiction

####  `Exécution2.md`
Contient :
- les images de la base de données générée
- les graphes de l’apprentissage (log loss, accuracy)
- les visualisations permettant de suivre l’évolution du modèle
#### `Vidéo.md`
 Pour cette deuxième étape, j’ai trouvé pertinent de visualiser l’apprentissage du réseau de neurones sur la base de données.  
J’ai donc enregistré quelques exemples, disponibles dans ce dossier.

---

## Suite du projet
La prochaine étape sera d’aller plus loin :  
tester d’autres architectures, utiliser des datasets plus complexes, et comparer les résultats avec un framework comme PyTorch ou TensorFlow.

Cette étape 2 pose les bases d’un vrai réseau de neurones, et prépare la suite du projet.

## Structure du projet


