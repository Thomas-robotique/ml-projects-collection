import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.metrics import accuracy_score

# Création de la base de données d'entraînement fictive

# Génère un jeu de données artificiel composé de 200 points en 2D, 
# # répartis en 2 groupes, avec une graine aléatoire fixée pour 
# obtenir toujours les mêmes données.
X, y = make_blobs(n_samples=200, n_features=2, centers=2, random_state=0) 
y = y.reshape((y.shape[0], 1))



plt.scatter(X[:,0], X[:, 1], c=y, cmap='summer')
X = X.T
y = y.reshape(1, -1)


# Fonction d’initialisation qui permet d’initialiser les paramètres w et b
def Initialisation(X,NC0, NC1, NC2):                 #NC0 = Nombre d'entrées               
    w1 = np.random.rand(NC1, X.shape[0])             #NC1= Nombre de neurones couche1
    b1 = np.random.rand(NC1,1)                       #NC2= Nombre de neurones couche2 qui doit être fixé à 1 pour avoir une sortie binaire


    w2=np.random.rand(NC2,NC1)
    b2=np.random.rand(NC2,1)

    parametre = {
            "w1": w1,
            "b1": b1,
            "w2":w2,
            "b2":b2

    }

    return parametre

# Fonction modèle : calcule la prédiction grâce à la fonction sigmoïde
# et prédit la position du point sur le graphe
def model(X, parametre):
    w1=parametre["w1"]
    b1=parametre["b1"]
    w2=parametre["w2"]
    b2=parametre["b2"]

     

    z1 = w1.dot(X) + b1

    A1 = 1 / (1 + np.exp(-z1))  

   

  

    z2= w2.dot(A1)+b2
    A2= 1/(1+np.exp(-z2))

    activation ={          #Tableau qui contient les activations des neurones

        "z1": z1,
      
        "A1": A1,
        "A2":A2,
        "z2":z2
       
    }
   
    return activation

# Fonction log_loss qui calcule à quel point la prédiction est proche de la réalité
def Logg_loss(activation, y):
    epsilon= pow(10,-15)
    A2= activation["A2"]

    return 1 / len(y) * np.sum(-y * np.log(A2+epsilon) - (1 - y) * np.log(1 - A2+epsilon))

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

# Fonction finale qui combine toutes les fonctions précédentes dans une boucle avec un nombre d’itérations fixé. Plus ce nombre est grand, plus le modèle s’entraîne longtemps.
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


   


    