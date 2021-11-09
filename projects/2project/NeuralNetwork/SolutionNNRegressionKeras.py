import tensorflow as tf
import numpy as np
from keras.layers import Dense, Activation
from keras.models import Sequential
import matplotlib.pyplot as plt

#====================== DATA
import sys
sys.path.append("../Data")
from DataRegression import X, X_test, X_train, x, x_mesh, y_mesh, z_test, z_train, plotFunction, z

N = len(x)

# Hessian Matrix
H = (2.0/N)* X_train.T @ X_train
# Get the eigenvalues
EigValues, EigVectors = np.linalg.eig(H)
eta_max = 1.0/np.max(EigValues)

n_hidden_neurons = 16
batch_size = 10
epochs = 100
activation_function = 'sigmoid'
optimizer = tf.keras.optimizers.Adam(learning_rate= eta_max)


model = Sequential()
model.add(Dense(n_hidden_neurons, activation = activation_function, input_dim = X_train.shape[1]))
model.add(Dense(units = n_hidden_neurons, activation = activation_function))
model.add(Dense(units = 1))
model.compile(optimizer = optimizer, loss = 'mean_squared_error')
model.fit(X_train, z_train, batch_size = batch_size, epochs = epochs)
print(model.history.history.keys())
loss_values = model.history.history['loss']
epochs = range(1, epochs + 1)

z_plot = model.predict(X)


plt.plot(epochs, loss_values, 'r-o', label='Training loss')
plt.title('Training loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plotFunction(x_mesh, y_mesh, z.reshape(len(x), len(x)), "Data")
plotFunction(x_mesh, y_mesh, z_plot.reshape(len(x), len(x)), "Prediction")


plt.show()
