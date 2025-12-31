```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.metrics import accuracy_score

X, y = make_blobs(n_samples=200, n_features=2, centers=2, random_state=0)
y = y.reshape((y.shape[0], 1))
```

<img src="https://github.com/user-attachments/assets/6f226d4a-a99b-41a5-89d8-6853e8283eea" width="500" alt="screen_de_la_database">

Voici la base de données générée avec la bibliothèque sklearn.  
Elle est entièrement artificielle et ne représente aucun phénomène physique réel ; elle sert uniquement de support pour l’entraînement du neurone.





```python
def Neuron_network(X, y, a=0.1, n_inter=200):
    w, b = Initialisation(X)
    loss = []
    for i in range(n_inter):
        A = model(X, w, b)
        loss.append(Logg_loss(A, y))
        dw, db = gradiant(y, X, A)
        w, b = uptade(w, b, dw, db, a)

    # Affichage de la fonction log loss
    plt.figure(figsize=(8, 5))
    plt.plot(loss, color='blue', linewidth=2)
    plt.title("Évolution du Log Loss pendant l'apprentissage", fontsize=14)
    plt.xlabel("Itérations", fontsize=12)
    plt.ylabel("Log Loss", fontsize=12)
    plt.grid(True)
    plt.show()
    return w, b

w, b = Neuron_network(X, y)
y_pred = model(X, w, b)
print(accuracy_score(y, y_pred))

```
