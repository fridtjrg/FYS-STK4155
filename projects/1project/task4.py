# The MIT License (MIT)
#
# Copyright © 2021 Fridtjof Gjengset, Adele Zaini, Gaute Holen
#
# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
# documentation files (the “Software”), to deal in the Software without restriction, including without limitation the
# rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software,
# and to permit persons to whom the Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all copies or substantial portions of
# the Software. THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT
# LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT
# SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF
# CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
# IN THE SOFTWARE.

import numpy as np
from random import random, seed
from regan import create_xyz_dataset, create_X, Split_and_Scale, OLS_solver, MSE, R2, ridge_reg, lasso_reg
import matplotlib.pyplot as plt

savefigure = False

degree=5
np.random.seed(1234)

# Datapoints (squared root of datapoints -> meshgrid)
n = 25
# Paramaters of noise distribution
mu_N = 0; sigma_N = 0.2

# Create vanilla dataset:
x,y,z = create_xyz_dataset(n,mu_N, sigma_N)

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
if savefigure:
    plt.savefig("./1project/Figures/Task4/MSE.png")
plt.show() 
