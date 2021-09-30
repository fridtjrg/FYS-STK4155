import numpy as np
from random import random, seed
from linear_regression import FrankeFunction, create_X, Split_and_Scale, OLS_solver, MSE, R2


"""
Task 1 comments:
We still need to find the variance of beta.



What to plot? (use mesh, x,y, z and z_tilda?)
How to find confidence? y-y_tilda = sigma
Sima is the stardard deviation of the error?

print("Beta(ols) variance:") //variance of beta? or = np.mean( np.var(y_pred, axis=1, keepdims=True) )
print(statistics.variance(ols_beta))


plt.plot(X_train,ztilde, label ="u values")   


"""

# Create vanilla dataset:
np.random.seed(3155)

n = 1000

x = np.linspace(0,1,n)
y = np.linspace(0,1,n) 

sigma_N = 0.1; mu_N = 0 #change for value of sigma_N to appropriate values
z = FrankeFunction(x,y) + np.random.normal(mu_N,sigma_N,n)	#adding noise to the dataset

degree=5

# OLS
X = create_X(x, y, degree)
X_train, X_test, z_train, z_test = Split_and_Scale(X,z) #StardardScaler, test_size=0.2, scale=true
ols_beta, z_tilde,z_predict = OLS_solver(X_train, X_test, z_train, z_test)

print("Training MSE", MSE(z_train,z_tilde))
print("Test MSE", MSE(z_test,z_predict))
print("-------------------------------------")
print("Training R2", R2(z_train,z_tilde))
print("Test R2", R2(z_test,z_predict))

# Missing confidence interval
# I would plot the data anyway
