
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
from NeuralNetworkRegression import DenseLayer, NeuralNetwork

#====================== DATA
import sys
sys.path.append("../Data")
from DataRegression import X, X_test, X_train, x, x_mesh, y_mesh, z_test, z_train, plotSave, plotFunction, x_y_test, x_y_train, x_y, z, MSE

n_hidden_neurons = 50
batch_size = 5

Epochs = [50, 100, 150, 200, 250,  300, 350, 400]
mse_epochs = []

_lambda = 1e-5
eta = 1e-5
for i, epoch in enumerate(Epochs):

    #===============================#
    #           Training            #
    #===============================#

    # ======= OWN NN
    print("step: ", i+1,"/", len(Epochs))

    nn = NeuralNetwork(x_y_train, z_train, n_epochs=epoch, batch_size=batch_size)
    nn.add_layer(DenseLayer(x_y_train.shape[1], n_hidden_neurons, 'sigmoid', lmbda=_lambda, eta=eta))
    nn.add_layer(DenseLayer(n_hidden_neurons, n_hidden_neurons, 'relu', lmbda=_lambda, eta=eta))
    nn.add_layer(DenseLayer(n_hidden_neurons, 1, None, lmbda=_lambda, eta=eta))
    nn.train()

    #===============================#
    #          Testing              #
    #===============================#

    # ======= OWN NN

    ytilde_test = nn.predict(x_y_test)
    mse_epochs.append(MSE(ytilde_test, z_test))

plot, ax = plt.subplots()
plt.plot(Epochs, mse_epochs, 'k-o', label='MSE')
plt.xlabel('nb_epochs')
plt.ylabel('MSE')
plt.legend()
plt.subplots_adjust(left=0.2, bottom=0.2, right=0.9)
plt.savefig('../Figures/NN/MSE_for_NN_for_nb_epochs.pdf')

plt.show()