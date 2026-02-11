
from load import*
import numpy as np
from Neuron_Network_multi_couche_v1 import Neuron_network
from dataset_tensorflow import load_tensorflow_data


X_train, y_train=load_tensorflow_data()
dimensions=np.array([X_train.shape[0],500,500,500,500,1])
Neuron_network(X_train,y_train, dimensions)

