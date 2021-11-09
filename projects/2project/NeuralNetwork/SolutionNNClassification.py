from NeuralNetworkClassification import DenseLayer, NeuralNetwork
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()


#====================== DATA
import sys
sys.path.append("../Data")
from DataClassification import X_test, X_train, Y_train_onehot, Y_test_onehot, accuracy_score_numpy, X, y_test, y_train


#====================== NN

n_epochs = 100
batch_size = 10
nb_hidden_neurons = 16
eta_vals = np.logspace(-5, -3, 10)
lmbd_vals = np.logspace(-5, -3, 10)



train_accuracy = np.zeros((len(eta_vals), len(lmbd_vals)))
test_accuracy = np.zeros((len(eta_vals), len(lmbd_vals)))
compt = 0


for i, eta in enumerate(eta_vals):
    for j, lmbd in enumerate(lmbd_vals):

        nn = NeuralNetwork(X_train, Y_train_onehot, n_epochs = n_epochs, batch_size = batch_size)
        nn.add_layer(DenseLayer(X_train.shape[1], nb_hidden_neurons, 'sigmoid', lmbda=lmbd, eta=eta))
        nn.add_layer(DenseLayer(nb_hidden_neurons, nb_hidden_neurons, 'sigmoid', lmbda=lmbd, eta=eta))
        nn.add_layer(DenseLayer(nb_hidden_neurons, 2, None, lmbda=lmbd, eta=eta))
        nn.train()

        ytilde_test = nn.predict(X_test)
        ytilde_train = nn.predict(X_train)

        train_accuracy[i][j] = accuracy_score_numpy(y_train, ytilde_train)
        test_accuracy[i][j] = accuracy_score_numpy(y_test, ytilde_test)
        compt += 1
        print("step : " + str(compt) + "/" + str(len(eta_vals) * len(lmbd_vals)))

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