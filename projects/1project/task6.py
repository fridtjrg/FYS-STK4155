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

from math import trunc
import numpy as np

from numpy.core.numeric import True_
from regan import *
from imageio import imread
import matplotlib.pyplot as plt




# Load the terrain
terrain = imread('./DataFiles/SRTM_data_Norway_2.tif')
N = 25

#---------------------------------------------------
#RUN cv and bootstrap

run_plot_compare(terrain,'terrain in Norway', 100, N=N, n_lambdas=30, k=5,poly_degree = 10,plot=True,saveplots=True_)


#----------------------------------------------------
#Get lambda vs MSE
z = terrain[:N,:N]
# Creates mesh of image pixels
x = np.linspace(0,1, np.shape(z)[0])
y = np.linspace(0,1, np.shape(z)[1])
x,y = np.meshgrid(x,y)
z = z.ravel()

degree = 5

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
plt.title(f"Plot of the MSE for different $\lambda$ for {degree} degree polynomial")
plt.legend()
plt.grid()
plt.savefig("./Figures/Task6/MSE.png")
plt.show() 
