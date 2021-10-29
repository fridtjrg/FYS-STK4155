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