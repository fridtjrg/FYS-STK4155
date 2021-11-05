import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
import numpy as np

#======================= NeuralNetwork

from NeuralNetwork import NeuralNetwork
from Layer import Layer

#====================== Data

from DataRegression import X, X_test, X_train, x, x_mesh, y_mesh, z_test, z_train, plotFunction, x_y_test, x_y_train, x_y



# Hessian Matrix
H = (2.0/len(x))* X_train.T @ X_train
# Get the eigenvalues
EigValues, EigVectors = np.linalg.eig(H)
eta_max = 1.0/np.max(EigValues)


def solution_nn():

    MSE_nn_val = []

    #eta = np.logspace(-5, -3, 1)
    eta = [eta_max]
    for i in eta:
        nn = NeuralNetwork()
        nn.add_layer(Layer(x_y_train.shape[1], 32, 'sigmoid'))
        nn.add_layer(Layer(32, 32, 'sigmoid'))
        nn.add_layer(Layer(32, 32, 'sigmoid'))
        nn.add_layer(Layer(32, 1, None))
        train = nn.train(x_y_train, z_train, i, nb_epochs = 100, batch_size = 5,  lmbd=0.01, _type='regression')
        y_pred = nn.predict(x_y_test, 'regression')
        z_pred = nn.predict(x_y, 'regression')
        MSE_nn_val.append(nn.MSE((y_pred.flatten()), z_test))
        plotFunction(x_mesh, y_mesh, z_pred.reshape(len(x), len(x)), "prediction")

    print(MSE_nn_val)
    plot, ax = plt.subplots()
    plt.title('MSE for the OLS, Ridge and Neural Networks (validation data)')
    plt.semilogx(eta, MSE_nn_val, 'b-o', label='Neural Networks')
    plt.xlabel(r'Learning rate $\eta$')
    plt.ylabel('MSE')
    #ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    plt.legend()
    plt.subplots_adjust(left=0.2, bottom=0.2, right=0.9)

    return plt.show()



solution_nn()

