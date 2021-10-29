import numpy as np
from frankefunction import Frankefunction
from neural_network import NeuralNetwork



x= np.linspace(0,1,1000)
y= np.linspace(0,1,1000)

my_dataset = Frankefunction(x,y)

outputs = my_dataset.dataset()
X_d = my_dataset.designMatrix()

print(X_d)

print(my_dataset.data_output)