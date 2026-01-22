
from load_cat_and_dog import*
from Neurone_Network_vf import Neuron_network


X_train_nn, X_train_flat, y_train = load()

parametre= load_parametres()
img_flat= load_image("photo_chat.jpg",X_train_flat)
img_flat1= load_image("photo_chien_1.jpg", X_train_flat)





print(np.shape(X_train_flat))
print(np.shape(y_train))



predict(img_flat,parametre)


##Neuron_network(X_train_nn,y_train)

