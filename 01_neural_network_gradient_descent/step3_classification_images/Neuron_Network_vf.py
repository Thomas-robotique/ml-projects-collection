import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from tqdm import tqdm
from PIL import Image
import pickle






# Fonction d’initialisation qui permet d’initialiser les paramètres w et b
def initialize_parameters(X,NC0, NC1, NC2):                 #NC0 = Nombre d'entrées               
    b1 = np.random.randn(NC1,1)                       #NC2= Nombre de neurones couche2 qui doit être fixé à 1 pour avoir une sortie binaire


    w1 = np.random.randn(NC1, X.shape[0]) * 0.1         # np.random.randn et pas rand pour éviter d'avoir des poids trop petits
    w2 = np.random.randn(NC2, NC1) * 0.1

    b2=np.random.randn(NC2,1)

    parametre = {
            "w1": w1,
            "b1": b1,
            "w2":w2,
            "b2":b2

    }

    return parametre

# Fonction modèle : calcule la prédiction grâce à la fonction sigmoïde
def model(X, parametre):
    w1=parametre["w1"]
    b1=parametre["b1"]
    w2=parametre["w2"]
    b2=parametre["b2"]

     

    z1 = w1.dot(X) + b1 # Broadcasting : NumPy étire b1 pour qu'il corresponde aux dimensions de w1


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
    m=y.shape[1]         # m correspond au nombre de données

    A2= activation["A2"]

    return 1 / m * np.sum(-y * np.log(A2+epsilon) - (1 - y) * np.log(1 - A2+epsilon))

# Fonction qui calcule les dérivées partielles de la fonction log_loss
# par rapport à w et b
def compute_gradients(y, X, parametre, activation):
    
     w1=parametre["w1"]    
     w2= parametre["w2"]
     b1=parametre["b1"]
     b2=parametre["b2"]


     A1= activation["A1"]
     A2=activation["A2"]
     z1=activation["z1"]
     z2=activation["z2"]


     dz2=A2-y                                           # Formules mathématiques des dérivées partielles

     m=y.shape[1]
  
     dw2 = 1 / m * dz2.dot(A1.T)
     db2 = 1 / m * np.sum(A2 - y, axis=1, keepdims=True)
  
     dz1=np.dot(w2.T, dz2)*A1*(1-A1)
     db1 = 1 / m * np.sum(dz1, axis=1, keepdims=True)
     dw1=1 / m * dz1.dot(X.T)

     gradiant = {
          "dw1":dw1,
          "dw2":dw2,
          "db1":db1,
          "db2": db2
        }

        
     

     return gradiant
    
        

# Fonction qui ajuste les paramètres w et b en fonction de la fonction log_loss
def update_parameters(a,gradiant,parametre ):

 dw1=gradiant["dw1"]
 dw2=gradiant["dw2"]
 db1=gradiant["db1"]
 db2=gradiant["db2"]

 w1=parametre["w1"]
 w2=parametre["w2"] 
 b1=parametre["b1"]
 b2=parametre["b2"]

 w1 = w1 - a * dw1
 b1 = b1 - a * db1

 w2 = w2 - a * dw2
 b2 = b2 - a * db2

 parametre ={
     "w1":w1,
     "w2":w2,
     "b1":b1,
     "b2":b2
   
 }
 return  parametre


# Fonction de prédiction : renvoie True (1) si la probabilité >= 0.5, sinon False (0)
def predict(Y, parametre, predict): # la fonction prend un bool en paramètre pour définir le mode ( prédiction ou entrainement) 
    if(predict):
           
       activation = model(Y, parametre)
       A2= activation["A2"]
       print("c'est un chien à",A2*100, "%")
       return 
    
    else:
      activation = model(Y, parametre)
      A2= activation["A2"]
      print("prediction",A2)
      return A2>=0.5 
        

# Fonction finale qui combine toutes les fonctions précédentes dans une boucle avec un nombre d’itérations fixé. Plus ce nombre est grand, plus le modèle s’entraîne longtemps.
def Neuron_network(X, y, a=0.01, n_inter=6000):
    NC0= X.shape[0]
    NC1=300
    NC2=1
    parametre = initialize_parameters(X,NC0, NC1, NC2)
    w1= parametre["w1"]
    w2=parametre["w2"]
    b1=parametre["b1"]
    b2=parametre["b2"]
    loss = []
    acc =[]
    for i in tqdm(range(n_inter)):
        activation = model(X, parametre)
        loss.append(Logg_loss(activation, y))
        grad = compute_gradients(y, X, parametre, activation)
        parametre = update_parameters(a,grad, parametre)
        if i % 100 == 0:
          y_pred = predict(X, parametre)
          acc.append(accuracy_score(y.flatten(), y_pred.flatten()))

         # Calcule le taux de précision entre les vraies étiquettes (y) et les prédictions (y_pred)
    
         


 
    
        
    # Affichage de la fonction log_loss
    plt.figure(figsize=(8, 5))
    plt.plot(loss, color='blue', linewidth=2)
    plt.title("Évolution du Log Loss pendant l'apprentissage", fontsize=14)
    plt.xlabel("Itérations", fontsize=12)
    plt.ylabel("Log Loss", fontsize=12)
    plt.grid(True)
    plt.plot(acc)  # Affiche la courbe d'accuracy sur le graphe, utile pour repérer un éventuel surapprentissage (overfitting)

    plt.show()
    # On ouvre un fichier en mode écriture binaire ("wb")
    with open("parametres_nn_300_a_0.01.pkl", "wb") as f:
     pickle.dump(parametre, f)

     print("Paramètres sauvegardés avec succès !")
    return parametre
    




