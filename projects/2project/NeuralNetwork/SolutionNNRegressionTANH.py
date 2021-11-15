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




n_hidden_neurons = 50
batch_size = 5
epochs = 100

Eta = np.logspace(-5, -2, 5)
Lambda = np.logspace(-5, -2, 5)

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
        nn.add_layer(DenseLayer(x_y_train.shape[1], n_hidden_neurons, 'sigmoid', lmbda= _lambda, eta=eta))
        nn.add_layer(DenseLayer(n_hidden_neurons, n_hidden_neurons, 'relu', lmbda= _lambda, eta=eta))
        nn.add_layer(DenseLayer(n_hidden_neurons, 1, None, lmbda= _lambda, eta=eta))
        nn.train()


        #===============================#
        #           Testing             #
        #===============================#

        #======= OWN NN

        ytilde_test = nn.predict(x_y_test)
        ytilde_train = nn.predict(x_y_train)

        mse_test = MSE(z_test, ytilde_test)

        if MSE(z_train, ytilde_train) < 1e10:
            train_mse[i][j] = MSE(z_train, ytilde_train)
            test_mse[i][j] = MSE(z_test, ytilde_test)
        else:
            train_mse[i][j] = np.inf
            test_mse[i][j] = np.inf

        if  mse_test < best_mse_NN:
            best_lambda_rate_NN = j
            best_learning_rate_NN = i
            best_mse_NN = mse_test

        compt += 1
        print("step : " + str(compt) + "/" + str(len(Eta) * len(Lambda)))


#===============================#
#      Training  for the best   #
#     learning rate and lambda  #
#===============================#


#======= OWN NN (With optimal paramaters)

nn = NeuralNetwork(x_y_train, z_train, n_epochs = epochs, batch_size = batch_size)
nn.add_layer(DenseLayer(x_y_train.shape[1], n_hidden_neurons, 'tanh', lmbda= Lambda[best_lambda_rate_NN], eta=Eta[best_learning_rate_NN]))
nn.add_layer(DenseLayer(n_hidden_neurons, n_hidden_neurons, 'tanh', lmbda= Lambda[best_lambda_rate_NN], eta=Eta[best_learning_rate_NN]))
nn.add_layer(DenseLayer(n_hidden_neurons, 1, None, lmbda= Lambda[best_lambda_rate_NN], eta=Eta[best_learning_rate_NN]))
nn.train()


z_pred_NN = nn.predict(x_y)

ytilde_test = nn.predict(x_y_test)
ytilde_train = nn.predict(x_y_train)


#title_NN = 'prediction_NN' + '_lamda_' + str(best_lambda_rate_NN) + '_eta_' + str(best_learning_rate_NN)
title_NN = 'NN_prediction_tanh'
plotSave(x_mesh, y_mesh, z,'../Figures/NN/', 'Noisy_dataset' )
plotSave(x_mesh, y_mesh, z_pred_NN.reshape(len(x), len(x)),'../Figures/NN/',title_NN)

fig, ax = plt.subplots(figsize = (6, 5))
sns.heatmap(train_mse,vmin=0,vmax=0.3, annot=True, ax=ax)
#ax.set_title("Training mse")
ax.set_ylabel("$\eta$")
ax.set_xlabel("$\lambda$")
plt.savefig('../Figures/NN/train_heatmap_using_tanh.pdf')


fig, ax = plt.subplots(figsize = (6, 5))
sns.heatmap(test_mse,vmin=0,vmax=0.3, annot=True, ax=ax)
#ax.set_title("Test mse")
ax.set_ylabel("$\eta$")
ax.set_xlabel("$\lambda$")
plt.savefig('../Figures/NN/test_heatmap_using_tanh.pdf')

print('==========================================================')
print('Our final model is built with the following hyperparmaters:')
print('Lambda = ', best_lambda_rate_NN)
print('Eta = ', best_lambda_rate_NN)
print('Epochs = ', epochs)
print('Batch size = ', batch_size)
print('----------------------------------------------------------')
print('The Eta and Lambda values we tested for are as follows:')
print('Lambda = ',Lambda)
print('Eta = ', Eta)
print('----------------------------------------------------------')
print('Mean square error of prediction:')
print('Train MSE = ', MSE(z_train, ytilde_train))
print('Test MSE = ', MSE(z_test, ytilde_test))
print('==========================================================')

plt.show()


