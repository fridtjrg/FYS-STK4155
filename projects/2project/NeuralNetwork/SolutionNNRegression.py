import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
from NeuralNetworkRegression import DenseLayer, NeuralNetwork

#====================== DATA
import sys
sys.path.append("../Data")
from DataRegression import X, X_test, X_train, x, x_mesh, y_mesh, z_test, z_train, plotFunction, x_y_test, x_y_train, x_y, z, MSE




n_hidden_neurons = 50
batch_size = 5
epochs = 100

Eta = np.logspace(-5, -3, 5)
Lambda = np.logspace(-5, -3, 5)

train_mse_keras = np.zeros((len(Eta), len(Lambda)))
test_mse_keras = np.zeros((len(Eta), len(Lambda)))
train_mse = np.zeros((len(Eta), len(Lambda)))
test_mse = np.zeros((len(Eta), len(Lambda)))
compt = 0

best_learning_rate_NN = Eta[0]
best_lambda_rate_NN = Lambda[0]
best_mse_NN = 1e10

for i, eta in enumerate(Eta):
    for j, _lambda in enumerate(Lambda):

        #===============================#
        #           Training            #
        #===============================#

        #======= OWN NN

        nn = NeuralNetwork(x_y_train, z_train, n_epochs = epochs, batch_size = batch_size)
        nn.add_layer(DenseLayer(x_y_train.shape[1], n_hidden_neurons, 'relu', lmbda= _lambda, eta=eta))
        nn.add_layer(DenseLayer(n_hidden_neurons, n_hidden_neurons, 'relu', lmbda= _lambda, eta=eta))
        nn.add_layer(DenseLayer(n_hidden_neurons, 1, None, lmbda= _lambda, eta=eta))
        nn.train()


        #===============================#
        #           Testing             #
        #===============================#

        #======= OWN NN

        ytilde_test = nn.predict(x_y_test)
        ytilde_train = nn.predict(x_y_train)

        train_mse[i][j] = MSE(z_train, ytilde_train)
        test_mse[i][j] = MSE(z_test, ytilde_test)

        if MSE(z_train, ytilde_train) <= best_mse_NN:
            best_lambda_rate_NN = _lambda
            best_learning_rate_NN = eta

        compt += 1
        print("step : " + str(compt) + "/" + str(len(Eta) * len(Lambda)))


#===============================#
#      Training  for the best   #
#     learning rate and lambda  #
#===============================#


#======= OWN NN

nn = NeuralNetwork(x_y_train, z_train, n_epochs = epochs, batch_size = batch_size)
nn.add_layer(DenseLayer(x_y_train.shape[1], n_hidden_neurons, 'sigmoid', lmbda= best_lambda_rate_NN, eta=best_lambda_rate_NN))
nn.add_layer(DenseLayer(n_hidden_neurons, n_hidden_neurons, 'sigmoid', lmbda= best_lambda_rate_NN, eta=best_lambda_rate_NN))
nn.add_layer(DenseLayer(n_hidden_neurons, 1, None, lmbda= best_lambda_rate_NN, eta=best_lambda_rate_NN))
nn.train()


z_pred_NN = nn.predict(x_y)

plotFunction(x_mesh, y_mesh, z, 'FranckFunction')
plotFunction(x_mesh, y_mesh, z_pred_NN.reshape(len(x), len(x)), 'Regression with NN')


fig, ax = plt.subplots(figsize = (5, 5))
sns.heatmap(train_mse, annot=True, ax=ax, cmap="viridis")
ax.set_title("Training mse")
ax.set_ylabel("$\eta$")
ax.set_xlabel("$\lambda$")


fig, ax = plt.subplots(figsize = (5, 5))
sns.heatmap(test_mse, annot=True, ax=ax, cmap="viridis")
ax.set_title("Test mse")
ax.set_ylabel("$\eta$")
ax.set_xlabel("$\lambda$")

plt.show()


