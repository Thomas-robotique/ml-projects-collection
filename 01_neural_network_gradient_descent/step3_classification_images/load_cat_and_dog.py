from PIL import Image
from Neurone_Network_vf import predict
import pickle
import numpy as np



 



def load_parametres():
 # Charger les paramètres depuis le fichier
 with open("parametres_nn_128_a_0.01_ite(50000).pkl", "rb") as f:
    parametre = pickle.load(f)      # Charge l'objet Python sauvegardé dans le fichier 'f'

 print("Paramètres chargés !")
 return parametre


def load_image(path):
 img = Image.open(path) 
 img = img.convert("L")             # Convertit l'image en niveaux de gris (1 canal)    
 img = img.resize((64,64))

 img= np.array(img)
 img_flat=img.reshape(-1,1)
 img_flat = img_flat/255 # normalisation de l'image

 print("Image convertie avec succès")
 return img_flat







