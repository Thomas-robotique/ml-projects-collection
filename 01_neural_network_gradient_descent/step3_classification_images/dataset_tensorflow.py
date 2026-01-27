import tensorflow as tf
import numpy as np
import os
from PIL import Image

import os
import tensorflow as tf


# Fonction permettant de trier les photos pour ne garder que celles facilement utilisables
def clean_dataset():
 root = "datasets2"

 deleted = 0
 checked = 0

 for subdir, dirs, files in os.walk(root):
    for file in files:
        path = os.path.join(subdir, file)

        # Si ce n'est pas un jpg/jpeg/png, la fonction supprime la photo
        if not file.lower().endswith((".jpg", ".jpeg", ".png")):
            print(" Fichier non supporté → suppression :", path)
            os.remove(path)
            deleted += 1
            continue

        # Sinon, on vérifie que TensorFlow peut le lire
        checked += 1
        try:
            raw = tf.io.read_file(path)
            img = tf.io.decode_image(raw, channels=3)
        except:
            print(" Image illisible → suppression :", path)
            os.remove(path)
            deleted += 1

 print(f"\n Terminé. {checked} images vérifiées, {deleted} fichiers supprimés.")



def load_tensorflow_data():
 train_ds = tf.keras.preprocessing.image_dataset_from_directory(
        "datasets2",
        image_size=(64, 64),
        batch_size=32,    #batch de 32, on découpe les 25000 images en 32 paquets
    )
 label=[]
 X_train_list=[]
 for images_batch, labels_batch in train_ds: 
      
        images= tf.image.rgb_to_grayscale(images_batch)

        # images_batch : (32, 128, 128, 3)
        images_flat = tf.reshape(images, (images_batch.shape[0], -1))
        X_train_list.append(images_flat.numpy())
        label.append(labels_batch.numpy())


 X_train=np.concatenate(X_train_list)
 y_train= np.concatenate(label) # y_train.shape = (25000,)

 y_train=y_train.reshape(-1,1)  
 print(X_train.shape)
 print(y_train.shape)

 X_train= (X_train/255).T #normalisation plus robuste que le min-max scaling

 X_train = X_train[:, :1000] # On garde que 1000 images, 25000 c'est beaucoup trop
 y_train = y_train[:1000].T

 print(X_train.shape)
 print(y_train.shape)     # y_train.shape = (25000,1)

 return X_train, y_train



    
