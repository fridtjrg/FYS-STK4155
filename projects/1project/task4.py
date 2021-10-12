import numpy as np
from random import random, seed
from regan import FrankeFunction, create_X, Split_and_Scale, OLS_solver, MSE, R2, ridge_reg, lasso_reg
import matplotlib.pyplot as plt



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


n_lambdas = 30
lmd_start = -10
lmd_end = 10

MSE_lmd_ridge_train, MSE_lmd_ridge_test, lmd = ridge_reg(X_train, X_test, z_train, z_test, nlambdas=n_lambdas, lmbd_start = lmd_start, lmbd_end =lmd_end, return_MSE_lmb = True)
MSE_lmd_lasso_train, MSE_lmd_lasso_test, lmd = lasso_reg(X_train, X_test, z_train, z_test, nlambdas=n_lambdas, lmbd_start = lmd_start, lmbd_end =lmd_end, return_MSE_lmb = True)


fig = plt.figure()
ax = fig.add_subplot(1,1,1)
ax.plot(lmd,MSE_lmd_ridge_train, label ="Ridge train", color = 'red', linestyle = 'dashed')  
ax.plot(lmd,MSE_lmd_ridge_test, label ="Ridge test", color = 'red')
ax.plot(lmd,MSE_lmd_lasso_train, label ="Lasso train", color = 'green', linestyle = 'dashed')  
ax.plot(lmd,MSE_lmd_lasso_test, label ="Lasso test", color = 'green')  
ax.set_xscale('log')

plt.xlabel("$\lambda$")
plt.ylabel("MSE")
plt.title("Plot of the MSE for different $\lambda$")
plt.legend()
plt.grid()
plt.savefig("./1project/Figures/Task4/MSE.png")
plt.show() 