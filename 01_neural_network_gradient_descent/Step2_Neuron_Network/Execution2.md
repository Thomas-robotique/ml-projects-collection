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

