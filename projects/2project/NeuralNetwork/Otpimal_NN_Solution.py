import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
from NeuralNetworkRegression import DenseLayer, NeuralNetwork

#====================== DATA
import sys
sys.path.append("../Data")
from DataRegression import X, X_test, X_train, x, x_mesh, y_mesh, z_test, z_train, plotSave, plotFunction, x_y_test, x_y_train, x_y, z, MSE


#===============================#
#           Parameters          #
#===============================#

optimal_epochs = 350
optimal_batch_size = 5
optimal_learning_rate = 0.0001778
optimal_lambda = 0.0001778

n_hidden_neurons = 50

#===============================#
#      Training  for the best   #
#     learning rate and lambda  #
#===============================#


#======= OWN NN (With optimal paramaters)

nn = NeuralNetwork(x_y_train, z_train, n_epochs = optimal_epochs, batch_size = optimal_batch_size)
nn.add_layer(DenseLayer(x_y_train.shape[1], n_hidden_neurons, 'sigmoid', lmbda= optimal_lambda, eta=optimal_learning_rate))
nn.add_layer(DenseLayer(n_hidden_neurons, n_hidden_neurons, 'tanh', lmbda= optimal_lambda, eta=optimal_learning_rate))
nn.add_layer(DenseLayer(n_hidden_neurons, 1, None, lmbda= optimal_lambda, eta=optimal_learning_rate))
nn.train()


z_pred_NN = nn.predict(x_y)
ytilde_test = nn.predict(x_y_test)
ytilde_train = nn.predict(x_y_train)

print(MSE(ytilde_test, z_test))

#title_NN = 'prediction_NN' + '_lamda_' + str(best_lambda_rate_NN) + '_eta_' + str(best_learning_rate_NN)
title_NN = 'NN_prediction_optimal'
plotSave(x_mesh, y_mesh, z_pred_NN.reshape(len(x), len(x)),'../Figures/NN/',title_NN)

plt.show()
