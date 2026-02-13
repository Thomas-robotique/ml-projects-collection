import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from tqdm import tqdm

from utilities import *

import pickle





# Fonction d’initialisation qui permet d’initialiser les paramètres w et b
def initialize_parameters(X, dimensions):
    
       parametre={}
    
       lenght= len(dimensions)
       for i in range (1,lenght) :
        
            parametre["w"+str(i)]=np.random.randn(  dimensions[i], dimensions[i-1])
            
            parametre["b"+str(i)] = np.random.randn(dimensions[i],1)
           # print(parametre)

    return parametre

# Fonction modèle : calcule la prédiction grâce à la fonction sigmoïde
# et prédit la position du point sur le graphe
def model( X, parametre):
  
  activation={}
  activation["A0"]=X
  A=X

  lenght=len(parametre)//2   
  for i in range (1,lenght+1) :  
       if(i==lenght):
         w = parametre["w"+str(i)]
         b=parametre["b"+str(i)]
    
         z = w.dot(A) + b # Broadcasting : NumPy étire b1 pour qu'il corresponde aux dimensions de w1
         A = 1 / (1 + np.exp(-z))  #On calcule l'activation de chaque couche puis on l'ajoute dans le dictionnaire
         #print("shape du Afinal", A.shape)
         activation["Afinal"]=A         
       else:
         w = parametre["w"+str(i)]
         b=parametre["b"+str(i)]
        # print("b dans la fonction model",b)
         '''
         print("iteration",i)
         print(b.shape)
         print(w.shape)
         print(A.shape)
         '''
         z = w.dot(A) + b # Broadcasting : NumPy étire b1 pour qu'il corresponde aux dimensions de w1
         A = 1 / (1 + np.exp(-z))  #On calcule l'activation de chaque couche puis on l'ajoute dans le dictionnaire
         
         activation["A"+str(i)]=A

  return activation




'''
  for key in activation:
    print("dimension de l'activation")
    print(np.shape(activation[key]))
         
'''
   
   


         

     
    

# Fonction log_loss qui calcule à quel point la prédiction est proche de la réalité
def Logg_loss(activation, y):
    epsilon= pow(10,-15)
    m=y.shape[1]         # m correspond au nombre de données
    Afinal= activation["Afinal"]
   
    #print(Afinal.shape)
    #print ("logg loss"+str(1 / m * np.sum(-y * np.log(Afinal+epsilon) - (1 - y) * np.log(1 - Afinal+epsilon))))
    return 1 / m * np.sum(-y * np.log(Afinal+epsilon) - (1 - y) * np.log(1 - Afinal+epsilon))


# Fonction qui calcule les dérivées partielles de la fonction log_loss
# par rapport à w et b
def compute_gradients(X,y, parametre, activation):
    gradiant={}
    m=y.shape[1]         # m correspond au nombre de données
    lenght=len(parametre)//2 
    '''
    print("la lenght",lenght) 
    print("taille de l'activation finale", activation["Afinal"].shape )
    print("y",y)
    print("taille du y final", y.shape)
   '''
    dz= activation["Afinal"]-y
  #  print("taille dz final", dz.shape)
    gradiant["dz"+str(lenght)]=activation["Afinal"]-y 
     
                # On divise par 2 pour avoir le nombre de couches, car 1 couche contient 2 paramètres
    for i in reversed(range(1 ,lenght+1)) : 

         gradiant["dw"+str(i)]= 1/m*np.dot(dz,activation["A"+str(i-1)].T)
         gradiant["db"+str(i)] = np.sum(dz, axis=1, keepdims=True) #keepdim evite les erreurs de broadcasting
        # print("taille de dz :", dz.shape,i)
         #print("taille de A avant dernier :", activation["A"+str(i-1)].shape)
         dz= np.dot(parametre["w"+str(i)].T,dz)*activation["A"+str(i-1)]*(1-activation["A"+str(i-1)])
         
  
       
    return gradiant, lenght
    
        

# Fonction qui ajuste les paramètres w et b en fonction de la fonction log_loss
def update_parameters(a,gradiant,parametre,lenght):
 
 for i in range(1,lenght+1):
     parametre["w"+str(i)]= parametre["w"+str(i)]-a*gradiant["dw"+str(i)]
     parametre["b"+str(i)]=parametre["b"+str(i)]-a*gradiant["db"+str(i)]

 return  parametre


def predict(Y, parametre, predict, dataset):
    
 

  if((predict)&(dataset=="cat_dog")):
           
       activation = model(Y, parametre)
       A2= activation["Afinal"]
       print("c'est un chat à",A2*100, "%")
       return 
  
  if ((predict)&(dataset=="points")):
      activation=model(Y,parametre)
      A2=activation["Afinal"]
      print("le point est jaunne à", A2*100,"%")
      return      
  
  else:
      activation = model(Y, parametre)
      A2= activation["Afinal"]
      return A2>=0.5  

# Fonction finale qui combine toutes les fonctions précédentes dans une boucle avec un nombre d’itérations fixé. Plus ce nombre est grand, plus le modèle s’entraîne longtemps.
def Neuron_network(X, y,dimensions,dataset,a=0.01, n_inter=3500):
  
    parametre = initialize_parameters(X,dimensions)
    print("en entraînement")
    loss = []
    acc =[]
    for i in tqdm(range(n_inter)):
        
        activation = model(X, parametre)
        loss.append(Logg_loss(activation, y))
        grad,lenght = compute_gradients(X,y, parametre, activation)
        parametre = update_parameters(a,grad, parametre,lenght)
        if i % 100 == 0:
          y_pred = predict(X, parametre,False, dataset)
          acc.append(accuracy_score(y.flatten(), y_pred.flatten()))

         # Calcule le taux de précision entre les vraies étiquettes (y) et les prédictions (y_pred)
        if(i==n_inter-1):
           print("L'erreur finale est de",loss[n_inter-1]*100,"%")
         


 
    
        
    # Affichage de la fonction log_loss
    plt.figure(figsize=(8, 5))
    plt.plot(loss, color='blue', linewidth=2)
    plt.title("Évolution du Log Loss pendant l'apprentissage, dataset : 2 spirales, architecture : 2,120,120,120,120,1  nombre itérations=3500 a=0.01", fontsize=14)
    plt.xlabel("Itérations", fontsize=12)
    plt.ylabel("Log Loss", fontsize=12)
    plt.grid(True)
    plt.plot(acc)  # Affiche la courbe d'accuracy sur le graphe, utile pour repérer un éventuel surapprentissage (overfitting)

    plt.show()
    # On ouvre un fichier en mode écriture binaire ("wb")
    
    with open("parametres_nn_multicouche_2_spirales_2_120_120_120_120_1_a_0.01_ité_3500.pkl", "wb") as f:
     pickle.dump(parametre, f)

     print("Paramètres sauvegardés avec succès !")
    return parametre
    


'''
print("la taille du réseau est" ,len(dimensions))
parametre=initialize_parameters(X,dimensions)
activation=model(X,parametre)
Logg_loss(activation,y)
print("taille de y ligne 39",y.shape)

print(y)
compute_gradients(X,y,parametre,activation)
'''



