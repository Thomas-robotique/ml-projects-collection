import numpy as np
import matplotlib.pyplot as plt


def make_two_spirales(n_points=600, noise=0.05, r1=2.0, r2=4.0):


    

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

X_circle, y_circle = make_two_spirales()

plt.scatter(X_circle[0,:], X_circle[1,:], c=y_circle.flatten(), cmap='summer')
plt.xlabel("X1")
plt.ylabel("X2")
plt.title("2 spirales")
plt.show()