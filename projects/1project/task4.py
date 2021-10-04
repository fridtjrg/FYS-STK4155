import numpy as np
from random import random, seed
from linear_regression import FrankeFunction, create_X, Split_and_Scale, OLS_solver, MSE, R2, ridge_reg



degree=5

# Create vanilla dataset:
np.random.seed(1234)

n = 25

x = np.linspace(0,1,n)
y = np.linspace(0,1,n) 
x, y = np.meshgrid(x,y)

sigma_N = 0.1; mu_N = 0 #change for value of sigma_N to appropriate values
z = FrankeFunction(x,y) +mu_N+sigma_N*np.random.randn(n,n)#+ np.random.normal(mu_N,sigma_N,n**2)  #adding noise to the dataset

# Ridge
X = create_X(x, y, degree)
X_train, X_test, z_train, z_test = Split_and_Scale(X,np.ravel(z)) #StardardScaler, test_size=0.2, scale=true

ridge_beta, opt_lambda, z_tilde,z_predict = ridge_reg(X_train, X_test, z_train, z_test)

print("-------------Ridge-------------")
print("Lambda: ", opt_lambda)
print("Training MSE", MSE(z_train,z_tilde))
print("Test MSE", MSE(z_test,z_predict))
print("-------------------------------------")
print("Training R2", R2(z_train,z_tilde))
print("Test R2", R2(z_test,z_predict))