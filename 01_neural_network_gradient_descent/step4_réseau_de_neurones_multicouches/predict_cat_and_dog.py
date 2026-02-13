
from load import*
#from Neurone_Network_vf import predict

from Neuron_Network_multi_couche_v1 import predict

dataset= "cat_dog"
new_point= np.array([9.42,9.23])
new_point= new_point.reshape(-1,1)
parametre= load_parametres()






if(dataset=="cat_dog"):
 img_flat= load_image("photo_test/photo_chat_3.jpg")
 img_flat1=load_image("photo_test/photo_chat_2.jpg")
 img_flat2= load_image("photo_test/photo_tigre_1.jpg")
 img_flat3= load_image("photo_test/photo_noir_1.jpg")
 img_flat4=load_image("photo_test/photo_chien_3.jpg")

 predict(img_flat,parametre,True,dataset)
 predict(img_flat1,parametre,True,dataset)
 predict(img_flat2,parametre,True,dataset)
 predict(img_flat3,parametre,True,dataset)
 predict(img_flat4,parametre,True,dataset)

if (dataset=="points"):
    predict(new_point,parametre,True,dataset)





