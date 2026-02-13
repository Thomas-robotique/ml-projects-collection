
from load import*
import numpy as np
from Neuron_Network_multi_couche_v1 import Neuron_network
from dataset_tensorflow import load_tensorflow_data

from dataset_two_circles import make_two_concentric_circles


dataset="cat_dog"


if( dataset=="cat_dog"):
 X_train, y_train=load_tensorflow_data()
 dimensions=np.array([X_train.shape[0],500,500,500,500,1])
 Neuron_network(X_train,y_train, dimensions,dataset)



if(dataset=="points"):

 X_spirale, y_spirale = make_two_concentric_circles()
 dimensions=np.array([X_spirale.shape[0],120,120,120,120,1])
 Neuron_network(X_spirale,y_spirale, dimensions, dataset)
