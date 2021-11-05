
import numpy as np
import matplotlib.pyplot as plt

#====================== Data

from DataRegression import X, X_test, X_train, x, x_mesh, y_mesh, z_test, z_train, plotFunction, z


def gradientOLS(N, x, y, theta):
    return (2.0/N)*x.T @ (x @ theta-y)


################### DG OLS ##################

Niterations = 100000
N = len(x)
############################################
# Hessian matrix
H = (2.0/N)* X_train.T @ X_train
# Get the eigenvalues
EigValues, EigVectors = np.linalg.eig(H)
beta = np.random.randn(X_train.shape[1])
eta = 1.0/np.max(EigValues)
for iter in range(Niterations):
    gradient = gradientOLS(N, X_train, z_train, beta)
    beta -= eta*gradient
print("beta from own dg")
print(beta)
ztildeDG = X_train @ beta
ztestDG = X_test @ beta
MSE_train_dg = np.mean((z_train - ztildeDG)**2, keepdims=True )
MSE_test_dg = np.mean((z_test - ztestDG)**2, keepdims=True )
print("MSE_train")
print(MSE_train_dg)
print("MSE_test")
print(MSE_test_dg)
print("\n")
print("-----------------------------")
print("\n")
plotFunction(x_mesh, y_mesh, z, "Plot of our data")
plotFunction(x_mesh, y_mesh, (X @ beta).reshape(len(x), len(x)), "Plot of regression with DG ols")
plt.show()