import numpy as np
import matplotlib.pyplot as plt

#====================== Data

from DataRegression import X, X_test, X_train, x, x_mesh, y_mesh, z_test, z_train, plotFunction, z


def predict(beta, X):
    return X @ beta


def gradientsRidge(N, lmbda, X, y, theta):
    return 2.0/N*X.T @ (X @ (theta)-y)+2*lmbda*theta

################### DG Ridge ##################

Niterations = 10000
Lambdas = np.logspace(-3, 1, 10)
N = len(x)

#############################################

for lmbda in Lambdas:
    # Hessian matrix
    H = (2.0/N)* X_train.T @ X_train
    # Get the eigenvalues
    EigValues, EigVectors = np.linalg.eig(H)
    beta = np.random.randn(X_train.shape[1])
    eta = 1.0/np.max(EigValues)

    for iter in range(Niterations):
        beta = beta - eta*gradientsRidge(N, lmbda, X_train, z_train, beta)


    print("beta from own dg")
    print(beta)
    ztildeDG = predict(beta, X_train)
    ztestDG = predict(beta, X_test)
    z_pred = predict(beta, X)
    MSE_train_dg = np.mean((z_train - ztildeDG)**2, keepdims=True )
    MSE_test_dg = np.mean((z_test - ztestDG)**2, keepdims=True )
    print("MSE_train")
    print(MSE_train_dg)
    print("MSE_test")
    print(MSE_test_dg)
    print("\n")
    print("-----------------------------")
    print("\n")
    title = "plot of regression with DG with lambda = " + str(lmbda)
    plotFunction(x_mesh, y_mesh, z_pred.reshape(len(x), len(x)), title)
plt.show()