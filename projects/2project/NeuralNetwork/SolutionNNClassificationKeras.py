import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import tensorflow as tf
from keras.layers import Dense
from keras.models import Sequential

sns.set()


#====================== DATA
import sys
sys.path.append("../Data")

from DataClassification import X_test, X_train, y_test, y_train, X



N = len(X)

# Hessian Matrix
H = (2.0/N)* X_train.T @ X_train
# Get the eigenvalues
EigValues, EigVectors = np.linalg.eig(H)
eta_max = 1.0/np.max(EigValues)


n_hidden_neurons = 16
batch_size = 10
epochs = 100

Eta = np.logspace(-5, -3, 10)
Lambda = np.logspace(-5, -3, 10)

train_accuracy = np.zeros((len(Eta), len(Lambda)))
test_accuracy = np.zeros((len(Eta), len(Lambda)))

for i, eta in enumerate(Eta):
    for j, _lambda in enumerate(Lambda):
        sgd = tf.keras.optimizers.SGD(lr=eta, momentum=_lambda, nesterov=True)
        model = Sequential()
        model.add(Dense(n_hidden_neurons, activation = 'relu', input_dim = X_train.shape[1]))
        model.add(Dense(units = n_hidden_neurons, activation = 'relu'))
        model.add(Dense(units = 1, activation = 'sigmoid'))
        model.compile(optimizer = sgd, loss = 'binary_crossentropy', metrics=['BinaryAccuracy'])
        model.fit(X_train, y_train, batch_size = batch_size, epochs = epochs)
        score_test = model.evaluate(X_test, y_test, verbose=1)
        score_train = model.evaluate(X_train, y_train, verbose=1)
        train_accuracy[i][j] = score_train[1]
        test_accuracy[i][j] = score_test[1]


fig, ax = plt.subplots(figsize=(5, 5))
sns.heatmap(train_accuracy, annot=True, ax=ax, cmap="viridis")
ax.set_title("Training accuracy keras")
ax.set_ylabel("$\eta$")
ax.set_xlabel("$\lambda$")

fig, ax = plt.subplots(figsize=(5, 5))
sns.heatmap(test_accuracy, annot=True, ax=ax, cmap="viridis")
ax.set_title("Test accuracy keras")
ax.set_ylabel("$\eta$")
ax.set_xlabel("$\lambda$")

plt.show()





