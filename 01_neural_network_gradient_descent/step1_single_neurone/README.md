#  Step 01 — Single Neuron

##  Objectif du projet  
Dans cette première étape, j’implémente un neurone unique en Python, en utilisant uniquement l’algorithme de descente de gradient.  
L’objectif est de comprendre en profondeur les mécanismes fondamentaux du Deep Learning, sans aucun framework.



---

##  Concepts abordés

Ce projet permet d’explorer les bases essentielles du fonctionnement d’un neurone artificiel :

- Propagation avant (forward propagation)
  Calcul de la sortie du neurone à partir des entrées et des poids.

- Calcul du gradient 
  Détermination de la direction dans laquelle ajuster les paramètres.

- Rétropropagation (backpropagation)
  Mise à jour des poids en fonction de l’erreur.

- Descente de gradient
  Optimisation itérative pour réduire la fonction coût.

- Comportement d’apprentissage d’un neurone 
  Observation de la convergence et de la capacité de généralisation.


##  Organisation des fichiers

####  `Neuron_Network.md`
Contient :
- le code complet du neurone
- l’implémentation de la descente de gradient
- les fonctions d’entraînement et de prédiction

####  `Exécution.md`
Contient :
- les images de la base de données générée
- les graphes de l’apprentissage (log loss, accuracy)
- les visualisations permettant de suivre l’évolution du modèle

---

##  Étape suivante  
Dans l’étape 2, je passerai à la construction d’un réseau de neurones complet, en réutilisant les fondations posées ici.
