
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
epoch = 350

batchs = [5, 10, 20, 30, 40, 50, 100, 150, 200]
mse_batch = []

_lambda = 1e-5
eta = 1e-5
for i, batch in enumerate(batchs):

    #===============================#
    #           Training            #
    #===============================#

    # ======= OWN NN
    print("step: ", i+1,"/", len(batchs))

    nn = NeuralNetwork(x_y_train, z_train, n_epochs=epoch, batch_size=batch)
    nn.add_layer(DenseLayer(x_y_train.shape[1], n_hidden_neurons, 'sigmoid', lmbda=_lambda, eta=eta))
    nn.add_layer(DenseLayer(n_hidden_neurons, n_hidden_neurons, 'relu', lmbda=_lambda, eta=eta))
    nn.add_layer(DenseLayer(n_hidden_neurons, 1, None, lmbda=_lambda, eta=eta))
    nn.train()

    #===============================#
    #          Testing              #
    #===============================#

    # ======= OWN NN

    ytilde_test = nn.predict(x_y_test)
    mse_batch.append(MSE(ytilde_test, z_test))

plot, ax = plt.subplots()
plt.plot(batchs, mse_batch, 'k-o', label='MSE')
plt.xlabel('batch_size')
plt.ylabel('MSE')
plt.legend()
plt.subplots_adjust(left=0.2, bottom=0.2, right=0.9)
plt.savefig('../Figures/NN/MSE_for_NN_for_batch_size.pdf')

plt.show()