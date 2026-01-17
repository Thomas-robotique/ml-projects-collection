```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.metrics import accuracy_score
from tqdm import tqdm
from matplotlib.animation import FuncAnimation



def make_two_concentric_circles(n_points=600, noise=0.05, r1=2.0, r2=4.0):


    

   n = n_points // 2
   theta = np.sqrt(np.random.rand(n)) * 2 * np.pi

   r_a = 2 * theta + np.pi
   data_a = np.array([np.cos(theta) * r_a, np.sin(theta) * r_a]).T
   data_a += np.random.randn(n, 2) * noise

   r_b = -2 * theta - np.pi
   data_b = np.array([np.cos(theta) * r_b, np.sin(theta) * r_b]).T
   data_b += np.random.randn(n, 2) * noise

   X = np.vstack([data_a, data_b])
   y = np.hstack([np.zeros(n), np.ones(n)])

   return X.T, y.reshape(1, -1)



X, y = make_two_concentric_circles()



plt.scatter(X[0,:], X[1,:], c=y.flatten(), cmap='summer')
plt.xlabel("X1")
plt.ylabel("X2")
plt.title("Cercle concentriques")
plt.show()
```


Cette partie du code permet de générer deux classe de points en forme de spirale.
<img width="796" height="677" alt="Screen_dataset_spires" src="https://github.com/user-attachments/assets/6014cc6b-7a76-4422-a265-084650d45478" />


```python

# Fonction finale qui combine toutes les fonctions précédentes dans une boucle avec un nombre d’itérations fixé. Plus ce nombre est grand, plus le modèle s’entraîne longtemps.
def Neuron_network(X, y, a=0.1, n_inter=3000):
    NC0= X.shape[0]
    NC1=300
    NC2=1
    parametre = Initialisation(X,NC0, NC1, NC2)
    w1= parametre["w1"]
    w2=parametre["w2"]
    b1=parametre["b1"]
    b2=parametre["b2"]
    loss = []
    acc =[]
    for i in tqdm(range(n_inter)):
        activation = model(X, parametre)
        loss.append(Logg_loss(activation, y))
        grad = gradiant(y, X, parametre, activation)
        parametre = uptade(a,grad, parametre)
        y_pred= predict(X,parametre)
        acc.append(accuracy_score(y.flatten(), y_pred.flatten()))  # Calcule le taux de précision entre les vraies étiquettes (y) et les prédictions (y_pred)
        if i % 10 == 0:   # Sauvegarde toutes les 10 itérations
         
         history.append({
         "w1": parametre["w1"].copy(),
         "b1": parametre["b1"].copy(),
         "w2": parametre["w2"].copy(),
         "b2": parametre["b2"].copy()
    })
         
         


 
    
        
    # Affichage de la fonction log_loss
    plt.figure(figsize=(8, 5))
    plt.plot(loss, color='blue', linewidth=2)
    plt.title("Évolution du Log Loss pendant l'apprentissage", fontsize=14)
    plt.xlabel("Itérations", fontsize=12)
    plt.ylabel("Log Loss", fontsize=12)
    plt.grid(True)
    plt.plot(acc)  # Affiche la courbe d'accuracy sur le graphe, utile pour repérer un éventuel surapprentissage (overfitting)

    plt.show()
    return parametre, history

```

## Analyse de l’apprentissage du réseau de neurones
Cette section présente l’entraînement d’un réseau de neurones à trois couches :
- une couche d’entrée,
 - une couche cachée (300 neurones),
- une couche de sortie (1 neurone).


Le suivi de l’apprentissage est assuré via la LogLoss et l’accuracy.
La courbe ci‑dessous montre que la LogLoss diminue progressivement pour tendre vers 0, indiquant que le modèle apprend efficacement la fonction cible. En parallèle, l’accuracy converge vers 1, confirmant la bonne qualité de l’apprentissage.
<img width="998" height="707" alt="screen_LogLoss_300_neurones" src="https://github.com/user-attachments/assets/afb7962e-7d82-45db-bcf0-e071f31f0884" />

## Comparaison avec une architecture réduite (10 neurones)

Pour évaluer l’impact de la capacité du modèle, le même réseau a été entraîné avec seulement **10 neurones** dans la couche cachée. La convergence est plus lente et l’erreur finale plus élevée, ce qui illustre clairement les limites d’un modèle moins expressif.

<img width="993" height="713" alt="screen_LogLoss_10_neurones" src="https://github.com/user-attachments/assets/5af6647a-975b-4b2a-9a5c-2511155192db" />








