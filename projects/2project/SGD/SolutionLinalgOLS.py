import numpy as np
import matplotlib.pyplot as plt

#====================== DATA
import sys
sys.path.append("../Data")
from DataRegression import X, X_test, X_train, x, x_mesh, y_mesh, z_test, z_train, plotSave, plotFunction, z



################### linalg  OLS #################
beta_linreg = np.linalg.pinv(X_train.T.dot(X_train)).dot(X_train.T).dot(z_train)

print("beta from linalg")
print(beta_linreg)

#Prediction
ztildeLinreg = X_train @ beta_linreg
ztestLinreg = X_test @ beta_linreg

#MSE
MSE_train_linreg = np.mean((z_train - ztildeLinreg)**2, keepdims=True )
MSE_test_linreg = np.mean((z_test - ztestLinreg)**2, keepdims=True )
print("MSE_train")
print(MSE_train_linreg)
print("MSE_test")
print(MSE_test_linreg)
print("\n")
print("-----------------------------")
print("\n")

#Plot function
plotFunction(x_mesh, y_mesh, z, "Plot of our data")
plotFunction(x_mesh, y_mesh, (X @ beta_linreg).reshape(len(x), len(x)), "Plot of regression with linreg")
plt.show()
plotSave(x_mesh, y_mesh, (X @ beta_linreg).reshape(len(x), len(x)), '../Figures/GD/LinalgOLS.pdf')
