import numpy as np
from frankefunction import Frankefunction
from neural_network import NeuralNetwork



x= np.linspace(0,1,1000)
y= np.linspace(0,1,1000)

my_dataset = Frankefunction(x,y)

my_dataset.dataset()
my_dataset.designMatrix()

print(my_dataset.X)


print(my_dataset.data_output)

my_dataset.split_data()#this ravels the output
my_dataset.scale()

print(my_dataset.data_output_train)

#Assuming the beta values(NN_beta) are the output of the NN
NN = NeuralNetwork(my_dataset.X_train, my_dataset.data_output_train)
"""
NeuralNetwork:
    def __init__(
            self,
            X_data,
            Y_data,
            n_hidden_neurons=50,
            n_categories=10,
            epochs=10,
            batch_size=100,
            eta=0.1,
            lmbd=0.0)
"""
NN.train()


NN_beta = NN.output #The prediction of the network



z_tilde = my_dataset.X_train @ NN_beta # z_prediction of the train data
z_predict = my_dataset.X_test @ NN_beta # z_prediction of the test data
predict_error = (z_tilde- z_predict)**2
