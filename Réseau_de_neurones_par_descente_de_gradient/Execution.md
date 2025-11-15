```cpp

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.metrics import accuracy_score

X, y = make_blobs(n_samples=200, n_features=2, centers=2, random_state=0)
y = y.reshape((y.shape[0], 1))

plt.scatter(X[:,0], X[:, 1], c=y, cmap='summer')


```

<img width="803" height="687" alt="screen_de_la_database" src="https://github.com/user-attachments/assets/22093e28-c995-4153-bd3a-d309177ccb75" />



Voici la base de données fictive que j'ai générée grâce à la bibliothèque scikit-learn.



```cpp




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
    acc.append(accuracy_score(y, y_pred))  # Calcule le taux de précision entre les vraies étiquettes(y)                                               et les prédictions (y_pred)

# Affichage de la fonction log_loss
    plt.figure(figsize=(8, 5))
    plt.plot(loss, color='blue', linewidth=2)
    plt.title("Évolution du Log Loss pendant l'apprentissage", fontsize=14)
    plt.xlabel("Itérations", fontsize=12)
    plt.ylabel("Log Loss", fontsize=12)
    plt.grid(True)
    plt.plot(acc)  # Affiche la courbe d'accuracy sur le graphe, utile pour repérer un éventuel                         surapprentissage (overfitting)

    plt.show()
    return w, b
    
```



<img width="981" height="698" alt="screen_log_loss" src="https://github.com/user-attachments/assets/097a0329-72fb-463e-96a3-49c47763f208" />





Voici le code de l’entraînement du neurone avec l’exécution associée.  
On constate que l’erreur entre les données d’entraînement et de test diminue fortement avec le temps, en se rapprochant de plus en plus de 0, ce qui montre que le neurone apprend correctement.

```cpp
Y = np.array([3, 5])   # point à prédirek
plt.scatter(Y[0], Y[1], color='r')


w, b = Neuron_network(X, y)
y_pred = predict(Y, w, b)


    print("La plante est toxique à " + str(y_pred * 100) + "%")

```
Ce code permet de créer un nouveau point (le point rouge sur la 1ère image), puis le neurone prédit la probabilité que ce point appartienne à l’une des catégories.

Dans cet exemple, le contexte choisi est une classification de plantes en fonction de leur toxicité.  
La toxicité repose sur les dimensions de leurs feuilles, représentées respectivement par l’abscisse et l’ordonnée du graphique.

