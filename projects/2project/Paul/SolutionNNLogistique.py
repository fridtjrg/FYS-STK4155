from DataLogistique import X_test, X_train, y_test, y_train, accuracy_score_numpy
from NeuralNetwork import NeuralNetwork
import matplotlib.pyplot as plt
from Layer import Layer
import numpy as np
import seaborn as sns
sns.set()

#-------------------  Parameters

eta_vals = np.logspace(-7, 1, 10)
lmbd_vals = np.logspace(-7, 1, 10)

#-------------------

train_accuracy = np.zeros((len(eta_vals), len(lmbd_vals)))
test_accuracy = np.zeros((len(eta_vals), len(lmbd_vals)))
compt = 0

for i, eta in enumerate(eta_vals):
    for j, lmbd in enumerate(lmbd_vals):
        nn = NeuralNetwork()
        nn.add_layer(Layer(X_train.shape[1], 32, 'sigmoid'))
        nn.add_layer(Layer(32, 32, 'sigmoid'))
        nn.add_layer(Layer(32, 2, None))
        train = nn.train(X_train, y_train, learning_rate= eta, nb_epochs = 500, batch_size = 5,  lmbd= lmbd, _type= 'classification')
        ytilde_test = nn.predict(X_test, 'classification')
        ytilde_train = nn.predict(X_train, 'classification')
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