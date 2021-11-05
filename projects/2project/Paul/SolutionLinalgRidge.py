import numpy as np
import matplotlib.pyplot as plt

#====================== Data

from DataRegression import X, X_test, X_train, x, x_mesh, y_mesh, z_test, z_train, plotFunction, z


################### linalg Ridge #################

Lambdas = np.logspace(-3, 1, 10)

#############################################

for lmbda in Lambdas:
    beta_linreg = np.linalg.pinv(X_train.T.dot(X_train) + lmbda * np.eye(X_train.T.dot(X_train).shape[0])).dot(X_train.T).dot(z_train)
    print("beta from linalg")
    print(beta_linreg)
    ztildeLinreg = X_train @ beta_linreg
    ztestLinreg = X_test @ beta_linreg
    MSE_train_linreg = np.mean((z_train - ztildeLinreg)**2, keepdims=True )
    MSE_test_linreg = np.mean((z_test - ztestLinreg)**2, keepdims=True )
    print("MSE_train")
    print(MSE_train_linreg)
    print("MSE_test")
    print(MSE_test_linreg)
    print("\n")
    print("-----------------------------")
    print("\n")
    title = "plot of regression with linalg with lambda = " + str(lmbda)
    plotFunction(x_mesh, y_mesh, (X @ beta_linreg).reshape(len(x), len(x)), title)
plt.show()



