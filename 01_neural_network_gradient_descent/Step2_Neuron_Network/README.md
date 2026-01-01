# Step 02 — Réseau de neurones

Dans cette deuxième étape, je passe d’un neurone unique à un réseau de neurones, construit entièrement à la main en Python.  
L’objectif est de comprendre comment fonctionne un réseau avec plusieurs couches, comment les informations circulent à travers lui, et comment les poids s’ajustent pendant l’apprentissage.


## Ce que j’ai implémenté
Dans cette étape, j’ai ajouté plusieurs éléments importants :

- 2 couches de neurones  
- des fonctions d’activation  
- la propagation avant sur tout le réseau  
- la rétropropagation pour mettre à jour les poids  
- l’entraînement complet du modèle sur un dataset simple  

L’idée est de comprendre comment un réseau apprend, comment l’erreur circule en arrière, et comment les paramètres s’ajustent petit à petit.

## Visualisations
Comme dans l’étape 1, j’utilise un dataset artificiel généré avec scikit‑learn.  
Il ne représente rien de physique : il sert uniquement à tester et visualiser l’apprentissage.

J’affiche ensuite :
- la fonction coût au fil des itérations  
- l’accuracy du modèle  
- la prédiction sur un point jamais vu auparavant  

Ces graphiques permettent de voir si le réseau apprend correctement et s’il généralise bien.

## Suite du projet
La prochaine étape sera d’aller plus loin :  
tester d’autres architectures, utiliser des datasets plus complexes, et comparer les résultats avec un framework comme PyTorch ou TensorFlow.

Cette étape 2 pose les bases d’un vrai réseau de neurones, et prépare la suite du projet.

