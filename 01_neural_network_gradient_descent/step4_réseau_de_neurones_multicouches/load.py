from PIL import Image
import pickle
import numpy as np




def load_parametres():
 # Charger les paramètres depuis le fichier
 with open("parametres_nn_multicouche_cat_dog_4096_500_500_500_1_a_0.001_ité_2500.pkl", "rb") as f:
    parametre = pickle.load(f)      # Charge l'objet Python sauvegardé dans le fichier 'f'

 print("Paramètres chargés !")
 #for key in parametre:
  #  print(parametre[key].shape)

 #print(parametre)
 
 #print("fin")

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







