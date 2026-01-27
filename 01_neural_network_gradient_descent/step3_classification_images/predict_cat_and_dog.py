
from load_cat_and_dog import*

# Chargement paramètres obtenus lors l'entraînement
parametre= load_parametres()

# Chargement des les images à prédire
img_flat= load_image("photo_chat.jpg")
img_flat1=load_image("photo_chat_2.jpg")
img_flat2= load_image("photo_chien_2.jpg")
img_flat3= load_image("photo_chien_1.jpg")
img_flat4=load_image("photo_chien_3.jpg")



# Appelle de la fonction predict pour effectuer une prédiction sur les images
predict(img_flat,parametre,True)
predict(img_flat1,parametre,True)
predict(img_flat2,parametre,True)
predict(img_flat3,parametre,True)
predict(img_flat4,parametre,True)






