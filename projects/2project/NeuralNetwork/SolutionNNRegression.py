import tensorflow as tf
import numpy as np
from keras.layers import Dense, Activation
from keras.models import Sequential
import matplotlib.pyplot as plt
from NeuralNetworkRegression import DenseLayer, NeuralNetwork

#====================== DATA
import sys
sys.path.append("../Data")
from DataRegression import X, X_test, X_train, x, x_mesh, y_mesh, z_test, z_train, plotFunction, x_y_test, x_y_train, x_y, z

N = len(x)

# Hessian Matrix
H = (2.0/N)* X_train.T @ X_train
# Get the eigenvalues
EigValues, EigVectors = np.linalg.eig(H)
eta_max = 1.0/np.max(EigValues)

n_hidden_neurons = 50
batch_size = 5
epochs = 100

activation_function = 'sigmoid'
optimizer = tf.keras.optimizers.Adam(learning_rate= eta_max)


model = Sequential()
model.add(Dense(n_hidden_neurons, activation = activation_function, input_dim = X_train.shape[1]))
model.add(Dense(units = n_hidden_neurons, activation = activation_function))
model.add(Dense(units = 1))
model.compile(optimizer = optimizer, loss = 'mean_squared_error')
model.fit(X_train, z_train, batch_size = batch_size, epochs = epochs)


nn = NeuralNetwork(x_y_train, z_train, n_epochs = epochs, batch_size = batch_size)
nn.add_layer(DenseLayer(x_y_train.shape[1], n_hidden_neurons, 'sigmoid', lmbda=0.001, eta=eta_max))
nn.add_layer(DenseLayer(n_hidden_neurons, n_hidden_neurons, 'sigmoid', lmbda=0.001, eta=eta_max))
nn.add_layer(DenseLayer(n_hidden_neurons, 1, None, lmbda= 0.001, eta=eta_max))
nn.train()



loss_values = model.history.history['loss']

z_plot = model.predict(X)
ytilde = nn.predict(x_y)


plotFunction(x_mesh, y_mesh, z.reshape(len(x), len(x)), "data")
plotFunction(x_mesh, y_mesh, ytilde.reshape(len(x), len(x)), "prediction")
plotFunction(x_mesh, y_mesh, z_plot.reshape(len(x), len(x)), "Prediction")


MSE_nn_val = nn.getLoss()
plot, ax = plt.subplots()
plt.title('Training MSE ')
plt.plot(range(1, epochs+1), MSE_nn_val, 'b-o', label='Training MSE own NN')
plt.plot(range(1, epochs+1), loss_values, 'r-o', label='Training MSE keras')
plt.title('Training loss keras')
plt.xlabel('Epochs')
plt.ylabel('MSE')
plt.legend()



plt.show()
