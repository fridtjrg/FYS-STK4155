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

savefigures = True


# Load the terrain
terrain = imread('../reports/Proj1_DataFiles/SRTM_data_Norway_2.tif')
N = 30
n = N

z = terrain[:N,:N]
# Creates mesh of image pixels
x = np.linspace(0,1, np.shape(z)[0])
y = np.linspace(0,1, np.shape(z)[1])
x,y = np.meshgrid(x,y)

degree = 20

#---------------------------------------------------
#RUN cv and bootstrap

lambdas = [10**x for x in [-12, -6, -3, 0, 3]]

foldername = 'Task6'

run_plot_compare(z, 100, N=n, k=5,poly_degree = 18,plot=True,saveplots=savefigures, foldername = foldername)

compare_lmd_BS(z, n, lambdas, degree, solver = 'RIDGE', n_resampling = 100, saveplots = savefigures, foldername = foldername)
compare_lmd_CV(z, n, 5, lambdas, degree, solver = 'RIDGE', saveplots = savefigures, foldername = foldername)
compare_lmd_CV(z, n, 10, lambdas, degree, solver = 'RIDGE', saveplots = savefigures, foldername = foldername)

compare_lmd_BS(z, n, lambdas, degree, solver = 'LASSO', n_resampling = 100, saveplots = savefigures, foldername = foldername)
compare_lmd_CV(z, n, 5, lambdas, degree, solver = 'LASSO', saveplots = savefigures, foldername = foldername)
compare_lmd_CV(z, n, 5, lambdas, degree, solver = 'LASSO', saveplots = savefigures, foldername = foldername)


#----------------------------------------------------
#Get MSE vs lambdas

degree = 5
lambdas = np.logspace(-30,20,num=50)

z = z.ravel()

X = create_X(x, y, degree)
X_train, X_test, z_train, z_test = Split_and_Scale(X,np.ravel(z)) #StardardScaler, test_size=0.2, scale=true

MSE_lmd_ridge_train = []
MSE_lmd_ridge_test = []
MSE_lmd_lasso_train = []
MSE_lmd_lasso_test = []

for lmd in lambdas:
    _,ridge_model ,ridge_predict  = ridge_reg(X_train, X_test, z_train, z_test, lmd=lmd)
    lasso_model,lasso_predict  = lasso_reg(X_train, X_test, z_train, z_test, lmd=lmd)
    MSE_lmd_ridge_train.append(MSE(ridge_model,z_train))
    MSE_lmd_ridge_test.append(MSE(ridge_predict,z_test))
    MSE_lmd_lasso_train.append(MSE(lasso_model,z_train))
    MSE_lmd_lasso_test.append(MSE(lasso_predict,z_test))


fig = plt.figure()
ax = fig.add_subplot(1,1,1)
ax.plot(lambdas,MSE_lmd_ridge_train, label ="Ridge train", color = 'red', linestyle = 'dashed')  
ax.plot(lambdas,MSE_lmd_ridge_test, label ="Ridge test", color = 'red')
ax.plot(lambdas,MSE_lmd_lasso_train, label ="Lasso train", color = 'green', linestyle = 'dashed')  
ax.plot(lambdas,MSE_lmd_lasso_test, label ="Lasso test", color = 'green')  
ax.set_xscale('log')

plt.xlabel("$\lambda$")
plt.ylabel("MSE")
plt.title(f"Plot of the MSE for different $\lambda$ with complexity {degree}")
plt.legend()
plt.grid()
if savefigures:
    plt.savefig("../reports/Proj1_Plots/Task6/MSE_lambdas.png")
plt.show() 
