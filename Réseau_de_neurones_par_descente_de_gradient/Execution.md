'''cpp

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.metrics import accuracy_score

X, y = make_blobs(n_samples=200, n_features=2, centers=2, random_state=0)
y = y.reshape((y.shape[0], 1))

plt.scatter(X[:,0], X[:, 1], c=y, cmap='summer')


'''

<img width="803" height="687" alt="screen_de_la_database" src="https://github.com/user-attachments/assets/22093e28-c995-4153-bd3a-d309177ccb75" />



Voici la base de données fictive que j'ai générée grâce à la bibliothèque scikit-learn.
