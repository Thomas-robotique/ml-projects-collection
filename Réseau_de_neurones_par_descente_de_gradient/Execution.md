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
def Neuron_network(X, y, a=0.01, n_inter=3000):
    acc= []
    w, b = Initialisation(X)
    loss = []
    for i in range(n_inter):
        A = model(X, w, b)
        loss.append(Logg_loss(A, y))
        dw, db = gradiant(y, X, A)
        w, b = uptade(w, b, dw, db, a)
        y_pred= predict(X,w,b)
        #acc.append(accuracy_score(y,y_pred))
    
 
    
    # Affichage de la fonction log_loss
    plt.figure(figsize=(8, 5))
    plt.plot(loss, color='blue', linewidth=2)
    plt.title("Évolution du Log Loss pendant l'apprentissage", fontsize=14)
    plt.xlabel("Itérations", fontsize=12)
    plt.ylabel("Log Loss", fontsize=12)
    plt.grid(True)
    # plt.plot(acc)
    plt.show()
    return w, b
```


<img width="990" height="700" alt="screen_log_loss" src="https://github.com/user-attachments/assets/b027aa0e-963b-476d-b574-8789270b587d" />
