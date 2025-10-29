import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.metrics import accuracy_score

X, y = make_blobs(n_samples=200, n_features=2, centers=2, random_state=0)
y = y.reshape((y.shape[0], 1))

plt.scatter(X[:,0], X[:, 1], c=y, cmap='summer')

# Fonction d’initialisation qui permet d’initialiser les paramètres w et b
def Initialisation(X):                 
    w = np.random.rand(X.shape[1], 1)
    b = np.random.rand(1)
    return (w, b)

# Fonction modèle : calcule la prédiction grâce à la fonction sigmoïde
# et prédit la position du point sur le graphe
def model(X, w, b):
    z = X.dot(w) + b
    A = 1 / (1 + np.exp(-z))  

    return A

# Fonction log_loss qui calcule à quel point la prédiction est proche de la réalité
def Logg_loss(A, y):
    espilon= pow(10,-15)
    return 1 / len(y) * np.sum(-y * np.log(A+espilon) - (1 - y) * np.log(1 - A+espilon))

# Fonction qui calcule les dérivées partielles de la fonction log_loss
# par rapport à w et b
def gradiant(y, X, A):
    

     dw = 1 / len(y) * np.dot(X.T, A - y)
     db = 1 / len(y) * np.sum(A - y)
     return (dw, db)
    
        

# Fonction qui ajuste les paramètres w et b en fonction de la fonction log_loss
def uptade(w, b, dw, db, a):
    w = w - a * dw
    b = b - a * db
    return (w, b)


# Fonction de prédiction : renvoie True (1) si la probabilité >= 0.5, sinon False (0)
def predict(Y, w, b):
    A = model(Y, w, b)
    return A >= 0.5


def Neuron_network(X, y, a=0.1, n_inter=1000):
    w, b = Initialisation(X)
    loss = []
    for i in range(n_inter):
        A = model(X, w, b)
        loss.append(Logg_loss(A, y))
        dw, db = gradiant(y, X, A)
        w, b = uptade(w, b, dw, db, a)
 
    
    # Affichage de la fonction log_loss
    plt.figure(figsize=(8, 5))
    plt.plot(loss, color='blue', linewidth=2)
    plt.title("Évolution du Log Loss pendant l'apprentissage", fontsize=14)
    plt.xlabel("Itérations", fontsize=12)
    plt.ylabel("Log Loss", fontsize=12)
    plt.grid(True)
    plt.show()
    return w, b
    




Y = np.array([2, 1])   # point à prédire
plt.scatter(Y[0], Y[1], color='r')

'''
w, b = Neuron_network(X, y)
y_pred = model(Y, w, b)

print("La plante est toxique à " + str(y_pred * 100) + "%")
'''

   


    
