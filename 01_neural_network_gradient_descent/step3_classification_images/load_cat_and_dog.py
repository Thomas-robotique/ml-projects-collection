from utilities import *
from PIL import Image
from Neurone_Network_vf import predict
import pickle




def load():
 # Chargement des données
 X_train, y_train = load_data()

 print(X_train.shape)   # (1000, 64, 64)
 print(y_train.shape)   # (1000, 1)

 # 1️ Flatten des images
 X_train_flat = X_train.reshape(X_train.shape[0], -1)  # (1000, 4096)
 print(X_train_flat.shape)

 # 2️ Normalisation
 X_train_flat = (X_train_flat - X_train_flat.min()) / (X_train_flat.max() - X_train_flat.min())

 # 3️ transposition 
 X_train_nn = X_train_flat.T    # (4096, 1000)
 y_train_nn = y_train.T  
 return X_train_nn,X_train_flat, y_train_nn   



def load_parametres():
 # Charger les paramètres depuis le fichier
 with open("parametres_nn.pkl", "rb") as f:
    parametre = pickle.load(f)      # Charge l'objet Python sauvegardé dans le fichier 'f'

 print("Paramètres chargés !")
 return parametre


def load_image(path,X_train_flat):
 img = Image.open(path) 
 img = img.convert("L")             # Convertit l'image en niveaux de gris (1 canal)    
 img = img.resize((64,64))

 img= np.array(img)
 img_flat=img.reshape(-1,1)
 img_flat = (img_flat - X_train_flat.min()) / (X_train_flat.max() - X_train_flat.min()) # normalisation de l'image

 print("Image convertie avec succès")
 return img_flat







