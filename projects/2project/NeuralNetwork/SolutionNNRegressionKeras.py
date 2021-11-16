import tensorflow as tf
import numpy as np
from tensorflow.keras.layers import Dense, Activation
from tensorflow.keras.models import Sequential
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

#====================== DATA
import sys
sys.path.append("../Data")
from DataRegression import X, X_test, X_train, x, x_mesh, y_mesh, z_test, z_train, plotSave, plotFunction, x_y_test, x_y_train, x_y, z, MSE




n_hidden_neurons = 50
batch_size = 5
epochs = 100

Eta = np.logspace(-5, -3, 10)



train_mse = [0]*len(Eta)
test_mse = [0]*len(Eta)
compt = 0

best_learning_rate_NN = Eta[0]
best_mse_NN = 1e10

for i, eta in enumerate(Eta):
    #===============================#
    #           Training            #
    #===============================#

    #======= Keras NN

    optimizer = tf.keras.optimizers.Adam(learning_rate=eta)
    model = Sequential()
    model.add(Dense(n_hidden_neurons, activation='sigmoid', input_dim=X_train.shape[1]))
    model.add(Dense(units=n_hidden_neurons, activation='sigmoid'))
    model.add(Dense(units=1))
    model.compile(optimizer=optimizer, loss='mean_squared_error')
    model.fit(X_train, z_train, batch_size=batch_size, epochs=epochs)

    #===============================#
    #           Testing             #
    #===============================#

    #======= OWN NN

    ytilde_test = model.predict(X_test)
    ytilde_train = model.predict(X_train)

    mse_test = MSE(z_test, ytilde_test)

    if MSE(z_train, ytilde_train) < 1e10:
        train_mse[i] = MSE(z_train, ytilde_train)
        test_mse[i] = MSE(z_test, ytilde_test)
    else:
        train_mse[i] = np.inf
        test_mse[i]= np.inf

    if  mse_test < best_mse_NN:
        best_learning_rate_NN = i
        best_mse_NN = mse_test

    compt += 1
    print("step : " + str(compt) + "/" + str(len(Eta)))


#===============================#
#      Training  for the best   #
#     learning rate and lambda  #
#===============================#



#======= OWN NN (With optimal paramaters)
optimizer = tf.keras.optimizers.Adam(learning_rate=Eta[best_learning_rate_NN])
#sgd = tf.keras.optimizers.SGD(lr=Eta[best_learning_rate_NN], momentum=Lambda[best_lambda_rate_NN], nesterov=True)

#======= Keras NN (With optimal paramaters)


model = Sequential()
model.add(Dense(n_hidden_neurons, activation='sigmoid', input_dim=X_train.shape[1]))
model.add(Dense(units=n_hidden_neurons, activation='sigmoid'))
model.add(Dense(units=1))
model.compile(optimizer=optimizer, loss='mean_squared_error')
model.fit(X_train, z_train, batch_size=batch_size, epochs=epochs)

z_pred_NN = model.predict(X)
ytilde_test = model.predict(X_test)
ytilde_train = model.predict(X_train)


#title_NN = 'prediction_NN' + '_lamda_' + str(best_lambda_rate_NN) + '_eta_' + str(best_learning_rate_NN)
title_NN = 'NN_prediction_keras'
plotSave(x_mesh, y_mesh, z,'../Figures/NN/', 'Noisy_dataset' )
plotSave(x_mesh, y_mesh, z_pred_NN.reshape(len(x), len(x)),'../Figures/NN/',title_NN)

fig, ax = plt.subplots(figsize = (6, 5))
plt.semilogx(Eta, train_mse, 'k-o', label = 'train MSE')
plt.semilogx(Eta, test_mse, 'r-o', label = 'test MSE')
plt.xlabel("$\eta$")
plt.ylabel('MSE')
# ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
plt.legend()
plt.subplots_adjust(
    top=0.933,
    bottom=0.129,
    left=0.121,
    right=0.95,
    hspace=0.2,
    wspace=0.2
)
plt.savefig('../Figures/NN/MSE_keras.pdf')

print('==========================================================')
print('Our final model is built with the following hyperparmaters:')
print('Lambda = ', best_learning_rate_NN)
print('Epochs = ', epochs)
print('Batch size = ', batch_size)
print('----------------------------------------------------------')
print('The Eta and Lambda values we tested for are as follows:')
print('Eta = ', Eta)
print('----------------------------------------------------------')
print('Mean square error of prediction:')
print('Train MSE = ', MSE(z_train, ytilde_train))
print('Test MSE = ', MSE(z_test, ytilde_test))
print('==========================================================')

plt.show()


