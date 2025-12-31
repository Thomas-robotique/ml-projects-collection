```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.metrics import accuracy_score

# Création de la base de données d'entraînement fictive
X, y = make_blobs(n_samples=200, n_features=2, centers=2, random_state=0)
y = y.reshape((y.shape[0], 1))

```

<img src="https://github.com/user-attachments/assets/6f226d4a-a99b-41a5-89d8-6853e8283eea" width="500" alt="screen_de_la_database">

##  Visualisation du dataset

La base de données ci‑dessous a été générée avec la bibliothèque scikit‑learn.  
Elle est entièrement artificielle et ne représente aucun phénomène physique réel ; elle sert uniquement de support pour l’entraînement du neurone.

Un point rouge a été ajouté manuellement :  
il ne fait pas partie de la base d’entraînement et sert d’exemple pour évaluer la capacité du modèle à prédire correctement un échantillon jamais vu auparavant.






```python
def Neuron_network(X, y, a=0.1, n_inter=1000):
    w, b = Initialisation(X)
    loss = []
    acc =[]
    for i in range(n_inter):
        A = model(X, w, b)
        loss.append(Logg_loss(A, y))
        dw, db = gradiant(y, X, A)
        w, b = uptade(w, b, dw, db, a)
        y_pred= predict(X,w,b)
        acc.append(accuracy_score(y, y_pred))  # Calcule le taux de précision entre les vraies étiquettes (y) et les prédictions (y_pred)
        
 
    
    # Affichage de la fonction log_loss
    plt.figure(figsize=(8, 5))
    plt.plot(loss, color='blue', linewidth=2)
    plt.title("Évolution du Log Loss pendant l'apprentissage", fontsize=14)
    plt.xlabel("Itérations", fontsize=12)
    plt.ylabel("Log Loss", fontsize=12)
    plt.grid(True)
    plt.plot(acc)  # Affiche la courbe d'accuracy sur le graphe, utile pour repérer un éventuel surapprentissage (overfitting)

    plt.show()
    return w, b
    




Y = np.array([3, 5])   # point à prédire
plt.scatter(Y[0], Y[1], color='r')


w, b = Neuron_network(X, y)
y_pred = model(Y, w, b)

print("La plante est toxique à " + str(y_pred * 100) + "%")

# Exécution avec ce point : La plante est toxique à [3.36220185]%

```

<img width="700" height="698" alt="screen_log_loss" src="https://github.com/user-attachments/assets/455e98db-dd68-456e-aaa7-49aa2fde0b6f" />

##  Visualisation de l’apprentissage

L’exécution de la fonction `Neuron_network` permet d’observer l’évolution de l’apprentissage du modèle.

- La courbe de la fonction coût (log loss) diminue progressivement jusqu’à tendre vers 0, ce qui indique que le neurone apprend correctement au fil des itérations.  
- La courbe d’accuracy augmente quant à elle jusqu’à atteindre 1, montrant que le modèle généralise bien et ne présente pas de signe d’overfitting.

Ces deux courbes offrent une vision claire et intuitive de la convergence du modèle.




