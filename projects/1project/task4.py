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
from regan import FrankeFunction, create_X, Split_and_Scale, OLS_solver, MSE, R2, ridge_reg



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

ridge_beta, z_tilde,z_predict, opt_lambda, = ridge_reg(X_train, X_test, z_train, z_test)

print("-------------Ridge-------------------")
print("Lambda: ", opt_lambda)
print("Training MSE", MSE(z_train,z_tilde))
print("Test MSE", MSE(z_test,z_predict))
print("-------------------------------------")
print("Training R2", R2(z_train,z_tilde))
print("Test R2", R2(z_test,z_predict))
