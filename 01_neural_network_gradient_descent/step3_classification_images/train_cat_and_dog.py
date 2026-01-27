
from load_cat_and_dog import*
from Neurone_Network_vf import Neuron_network
from dataset_tensorflow import load_tensorflow_data

X_train, y_train=load_tensorflow_data()


Neuron_network(X_train,y_train)

