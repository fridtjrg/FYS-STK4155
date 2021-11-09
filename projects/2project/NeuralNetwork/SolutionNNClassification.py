import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import tensorflow as tf
from keras.layers import Dense
from keras.models import Sequential
from NeuralNetworkClassification import DenseLayer, NeuralNetwork

sns.set()


#====================== DATA
import sys
sys.path.append("../Data")

from DataClassification import X_test, X_train, Y_train_onehot, Y_test_onehot, accuracy_score_numpy, X, y_test, y_train


n_hidden_neurons = 16
batch_size = 10
epochs = 100

Eta = np.logspace(-5, -3, 7)
Lambda = np.logspace(-5, -3, 7)

train_accuracy_keras = np.zeros((len(Eta), len(Lambda)))
test_accuracy_keras = np.zeros((len(Eta), len(Lambda)))
train_accuracy = np.zeros((len(Eta), len(Lambda)))
test_accuracy = np.zeros((len(Eta), len(Lambda)))
compt = 0


for i, eta in enumerate(Eta):
    for j, _lambda in enumerate(Lambda):

        #===============================#
        #           Training            #
        #===============================#

        #======== KERAS

        sgd = tf.keras.optimizers.SGD(lr=eta, momentum=_lambda, nesterov=True)
        model = Sequential()
        model.add(Dense(n_hidden_neurons, activation = 'relu', input_dim = X_train.shape[1]))
        model.add(Dense(units = n_hidden_neurons, activation = 'relu'))
        model.add(Dense(units = 1, activation = None))
        model.compile(optimizer = sgd, loss = 'binary_crossentropy', metrics=['BinaryAccuracy'])
        model.fit(X_train, y_train, batch_size = batch_size, epochs = epochs)

        #======= OWN NN

        nn = NeuralNetwork(X_train, Y_train_onehot, n_epochs = epochs, batch_size = batch_size)
        nn.add_layer(DenseLayer(X_train.shape[1], n_hidden_neurons, 'relu', lmbda=_lambda, eta=eta))
        nn.add_layer(DenseLayer(n_hidden_neurons, n_hidden_neurons, 'relu', lmbda=_lambda, eta=eta))
        nn.add_layer(DenseLayer(n_hidden_neurons, 2, None, lmbda=_lambda, eta=eta))
        nn.train()

        #===============================#
        #           Testing             #
        #===============================#

        #======== KERAS

        score_test = model.evaluate(X_test, y_test, verbose=1)
        score_train = model.evaluate(X_train, y_train, verbose=1)

        #======= OWN NN

        ytilde_test = nn.predict(X_test)
        ytilde_train = nn.predict(X_train)

        train_accuracy[i][j] = accuracy_score_numpy(y_train, ytilde_train)
        test_accuracy[i][j] = accuracy_score_numpy(y_test, ytilde_test)
        train_accuracy_keras[i][j] = score_train[1]
        test_accuracy_keras[i][j] = score_test[1]

        compt += 1
        print("step : " + str(compt) + "/" + str(len(Eta) * len(Lambda)))


fig, ax = plt.subplots(figsize=(5, 5))
sns.heatmap(train_accuracy_keras, annot=True, ax=ax, cmap="viridis")
ax.set_title("Training accuracy keras")
ax.set_ylabel("$\eta$")
ax.set_xlabel("$\lambda$")

fig, ax = plt.subplots(figsize=(5, 5))
sns.heatmap(test_accuracy_keras, annot=True, ax=ax, cmap="viridis")
ax.set_title("Test accuracy keras")
ax.set_ylabel("$\eta$")
ax.set_xlabel("$\lambda$")

fig, ax = plt.subplots(figsize = (5, 5))
sns.heatmap(train_accuracy, annot=True, ax=ax, cmap="viridis")
ax.set_title("Training Accuracy")
ax.set_ylabel("$\eta$")
ax.set_xlabel("$\lambda$")


fig, ax = plt.subplots(figsize = (5, 5))
sns.heatmap(test_accuracy, annot=True, ax=ax, cmap="viridis")
ax.set_title("Test Accuracy")
ax.set_ylabel("$\eta$")
ax.set_xlabel("$\lambda$")

plt.show()





