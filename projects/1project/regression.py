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
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
from sklearn.model_selection import train_test_split
from sklearn import linear_model
from sklearn.preprocessing import StandardScaler
from sklearn.utils import resample


# FrankeFunction: a two-variables function to create the dataset of our vanilla problem
def FrankeFunction(x,y):
	term1 = 0.75*np.exp(-(0.25*(9*x-2)**2) - 0.25*((9*y-2)**2))
	term2 = 0.75*np.exp(-((9*x+1)**2)/49.0 - 0.1*(9*y+1))
	term3 = 0.5*np.exp(-(9*x-7)**2/4.0 - 0.25*((9*y-3)**2))
	term4 = -0.2*np.exp(-(9*x-4)**2 - (9*y-7)**2)
	return term1 + term2 + term3 + term4
 
# 3D plot of FrankeFunction
def Plot_FrankeFunction(x,y,z, title="Dataset"):
    fig = plt.figure()
    ax = fig.gca(projection="3d")

    # Plot the surface.
    surf = ax.plot_surface(x, y, z, cmap=cm.coolwarm, linewidth=0, antialiased=False)

    # Customize the z axis.
    ax.set_zlim(-0.10, 1.40)
    ax.zaxis.set_major_locator(LinearLocator(10))
    ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))

    # Add a color bar which maps values to colors.
    fig.colorbar(surf, shrink=0.5, aspect=5)
    plt.title(title)
    plt.show()
    
# Create xyz dataset from the FrankeFunction with a added normal distributed noise
def create_xyz_dataset(n,mu_N, sigma_N):
    x = np.linspace(0,1,n)
    y = np.linspace(0,1,n)

    x,y = np.meshgrid(x,y)
    z = FrankeFunction(x,y) +mu_N +sigma_N*np.random.randn(n,n)
    
    return x,y,z

# Error analysis: MSE and R2 score
def R2(y_data, y_model):
    return 1 - np.sum((y_data - y_model) ** 2) / np.sum((y_data - np.mean(y_data)) ** 2)
def MSE(y_data,y_model):
    n = np.size(y_model)
    return np.sum((y_data-y_model)**2)/n

# SVD theorem
def SVD(A):
    U, S, VT = np.linalg.svd(A,full_matrices=True)
    D = np.zeros((len(U),len(VT)))
    print("shape D= ", np.shape(D))
    print("Shape S= ",np.shape(S))
    print("lenVT =",len(VT))
    print("lenU =",len(U))
    D = np.eye(len(U),len(VT))*S
    """
    for i in range(0,VT.shape[0]): #was len(VT)
        D[i,i]=S[i]
        print("i=",i)"""
    return U @ D @ VT
    
# SVD inversion
def SVDinv(A):
    U, s, VT = np.linalg.svd(A)
    # reciprocals of singular values of s
    d = 1.0 / s
    # create m x n D matrix
    D = np.zeros(A.shape)
    # populate D with n x n diagonal matrix
    D[:A.shape[1], :A.shape[1]] = np.diag(d)
    UT = np.transpose(U)
    V = np.transpose(VT)
    return np.matmul(V,np.matmul(D.T,UT))


# Design matrix for two indipendent variables x,y
def create_X(x, y, n):
	if len(x.shape) > 1:
		x = np.ravel(x)
		y = np.ravel(y)

	N = len(x)
	l = int((n+1)*(n+2)/2)		# Number of elements in beta, number of feutures (degree of polynomial)
	X = np.ones((N,l))

	for i in range(1,n+1):
		q = int((i)*(i+1)/2)
		for k in range(i+1):
			X[:,q+k] = (x**(i-k))*(y**k)

	return X

def scale_Xz(X_train, X_test, z_train, z_test):
    scaler_X = StandardScaler(with_std=False)
    scaler_X.fit(X_train)
    X_train = scaler_X.transform(X_train)
    X_test = scaler_X.transform(X_test)

    scaler_z = StandardScaler(with_std=False)
    z_train = np.squeeze(scaler_z.fit_transform(z_train.reshape(-1, 1))) #scaler_z.fit_transform(z_train) #
    z_test = np.squeeze(scaler_z.transform(z_test.reshape(-1, 1))) #scaler_z.transform(z_test) #  
    return X_train, X_test, z_train, z_test

# Splitting and rescaling data (rescaling is optional)
# Default values: 20% of test data and the scaler is StandardScaler without std.dev.
def Split_and_Scale(X,z,test_size=0.2, scale=True):

    #Splitting training and test data
    X_train, X_test, z_train, z_test = train_test_split(X, z, test_size=test_size)

    # Rescaling X and z (optional)
    if scale:
        X_train, X_test, z_train, z_test = scale_Xz(X_train, X_test, z_train, z_test)
      
    return X_train, X_test, z_train, z_test

# OLS equation
def OLS_solver(X_train, X_test, z_train, z_test):

	# Calculating Beta Ordinary Least Square Equation with matrix pseudoinverse
    # Altervatively to Numpy pseudoinverse it is possible to use the SVD theorem to evalute the inverse of a matrix (even in case it is singular). Just replace 'np.linalg.pinv' with 'SVDinv'.
	ols_beta = np.linalg.pinv(X_train.T @ X_train) @ X_train.T @ z_train

	z_tilde = X_train @ ols_beta # z_prediction of the train data
	z_predict = X_test @ ols_beta # z_prediction of the test data
  
	return ols_beta, z_tilde, z_predict
 
def Confidence_Interval(beta, X, sigma=1):
    #Calculates variance of beta, extracting just the diagonal elements of the matrix
    #var(B_j)=sigma^2*(X^T*X)^{-1}_{jj}
    beta_variance = np.diag(sigma**2 * np.linalg.pinv(X.T @ X))
    ci1 = beta - 1.96 * np.sqrt(beta_variance)/(X.shape[0])
    ci2 = beta + 1.96 * np.sqrt(beta_variance)/(X.shape[0])
    print('Confidence interval of β-estimator at 95 %:')
    ci_df = {r'$β_{-}$': ci1,
             r'$β_{ols}$': beta,
             r'$β_{+}$': ci2}
    ci_df = pd.DataFrame(ci_df)
    display(np.round(ci_df,3))
    return ci1, ci2

# Plot MSE in function of complexity of the model
def plot_ols_complexity(x, y, z, complexity = np.arange(2,21), title="MSE as a function of model complexity"):

    MSE_train_set = []
    MSE_test_set = []

    for degree in complexity:

        X = create_X(x, y, degree)
        X_train, X_test, z_train, z_test = Split_and_Scale(X,np.ravel(z)) #StardardScaler, test_size=0.2, scale=true
        ols_beta, z_tilde,z_predict = OLS_solver(X_train, X_test, z_train, z_test)

        MSE_train_set.append(MSE(z_train,z_tilde))
        MSE_test_set.append(MSE(z_test,z_predict))

    plt.plot(complexity,MSE_train_set, label ="train")  
    plt.plot(complexity,MSE_test_set, label ="test")  
     
    plt.xlabel("complexity")
    plt.ylabel("MSE")
    plt.title("Plot of the MSE as a function of complexity of the model")
    plt.legend()
    plt.grid()     
    #plt.savefig('Task2plot(n='+str(n)+').pdf')
    plt.show() 

def lasso_reg(X_train, X_test, z_train, z_test, nlambdas=20, lmbd_start = -20, lmbd_end = 20):
    """Lasso regression using sklearn

    Args:
        X_train ([type]): [description]
        X_test ([type]): [description]
        z_train ([type]): [description]
        z_test ([type]): [description]
        nlambdas (int, optional): [description]. Defaults to 1000.
        lmbd_start (int, optional): [description]. Defaults to -20.
        lmbd_end (int, optional): [description]. Defaults to 20.

    Returns:
        [type]: [description]
    """

    lambdas = np.logspace(lmbd_start, lmbd_end, nlambdas)
    MSE_values = np.zeros(nlambdas)

    for i in range(nlambdas):
        lmb = lambdas[i]
        RegLasso = linear_model.Lasso(lmb)
        RegLasso.fit(X_train,z_train)
        z_model = RegLasso.predict(X_train)
        z_predict = RegLasso.predict(X_test)

        MSE_values[i] = MSE(z_train,z_model)    #calculates MSE
    
    #Find best lamda
    best_lamda = lambdas[np.argmin(MSE_values)]
    if best_lamda.__class__ == np.ndarray and len(best_lamda) > 1:
        print("NB: No unique value for lamda gets best MSE, multiple lamda gives smallest MSE")
        best_lamda = best_lamda[0]

    if best_lamda == lambdas[0]:
        print("NB, the best lambda was the was the first lambda value")

    if best_lamda == lambdas[-1]:
        print("NB, the best lambda was the was the last lambda value")
    
    RegLasso = linear_model.Lasso(best_lamda)
    RegLasso.fit(X_train,z_train)
    z_model = RegLasso.predict(X_train)
    beta = RegLasso.alpha
    z_predict = RegLasso.predict(X_test)

    
    return z_model, z_predict, best_lamda

def ridge_reg(X_train, X_test, z_train, z_test, nlambdas=20, lmbd_start = -4, lmbd_end = 4):


    MSEPredict = np.zeros(nlambdas)
    lambdas = np.logspace(lmbd_start, lmbd_end, nlambdas)

    MSE_values = np.zeros(nlambdas)

    for i in range(nlambdas):
        # Optimal paramaters for Ridge
        #Not sure about np.eye(len(X_train.T)), just to get right size
        ridge_beta = np.linalg.pinv(X_train.T @ X_train + lambdas[i]*np.eye(len(X_train.T))) @ X_train.T @ z_train #psudoinverse
        z_model = X_train @ ridge_beta #calculates model
        MSE_values[i] = MSE(z_train,z_model)    #calculates MSE


    #finds the lambda that gave the best MSE
    #best_lamda = lambdas[np.where(MSE_values == np.min(MSE_values))[0]]
    best_lamda = lambdas[np.argmin(MSE_values)]
    if best_lamda.__class__ == np.ndarray and len(best_lamda) > 1:
        print("NB: No unique value for lamda gets best MSE, multiple lamda gives smallest MSE")
        best_lamda = best_lamda[0]

    if best_lamda == lambdas[0]:
        print("NB, the best lambda was the was the first lambda value")

    if best_lamda == lambdas[-1]:
        print("NB, the best lambda was the was the last lambda value")

    #Calculates this Ridge_beta again
    ridge_beta_opt = np.linalg.pinv(X_train.T @ X_train + best_lamda*np.eye(len(X_train.T))) @ X_train.T @ z_train #psudoinverse

    """
    print(np.min(MSE_values))
    print(MSE_values)
    print(lambdas)
    print(best_lamda)
    """
    z_model = X_train @ ridge_beta_opt
    z_predict = X_test @ ridge_beta_opt


    return ridge_beta_opt, z_model, z_predict, best_lamda
 

# Bootstrap resampling
# Return a (m x n_bootstraps) matrix with the column vectors z_pred for each bootstrap iteration.
def bootstrap(X_train, X_test, z_train, z_test, n_boostraps=100):
    z_pred_boot = np.empty((z_test.shape[0], n_boostraps))
    for i in range(n_boostraps):
        # Draw a sample of our dataset
        X_sample, z_sample = resample(X_train, z_train)
        # Perform OLS equation
        beta, z_tilde, z_pred = OLS_solver(X_sample, X_test, z_sample, z_test)
        # Evaluate the new model on the same test data each time.
        z_pred_boot[:, i] = z_pred.ravel()
    return z_pred_boot
    
# Bias-variance tradeoff

# Note: Expectations and variances taken w.r.t. different training
# data sets, hence the axis=1. Subsequent means are taken across the test data
# set in order to obtain a total value, but before this we have error/bias/variance
# calculated per data point in the test set.
# Note 2: The use of keepdims=True is important in the calculation of bias as this
# maintains the column vector form. Dropping this yields very unexpected results.

# conclude with cross validation

def bias_variance_analysis(X_train, X_test, z_train, z_test, resampling="bootstrap", n_resampling = 100):
    if(resampling=="bootstrap"):
        z_pred = bootstrap(X_train, X_test, z_train, z_test, n_boostraps = n_resampling)
    """ else:
        z_pred = crossvalidation(X_train, X_test, z_train, z_test, n_resampling)
    """

    error = np.mean( np.mean((z_test.reshape(-1,1) - z_pred)**2, axis=1, keepdims=True) ) # MSE
    bias2 = np.mean( (z_test.reshape(-1,1) - np.mean(z_pred, axis=1, keepdims=True))**2 ) # bias^2
    variance = np.mean( np.var(z_pred, axis=1, keepdims=True) )

    return error, bias2, variance
    
# Plot bias-variance tradeoff in function of complexity of the model
def bias_variance_complexity(x, y, z, complexity = np.arange(1,15), n_resampling = 100, test_size = 0.2, plot=True, title="Bias-variance analysis: MSE as a function of model complexity"):

    error = np.zeros(complexity.size)
    bias = np.zeros(complexity.size)
    variance = np.zeros(complexity.size)

    for degree in complexity:

        X = create_X(x, y, degree)
        X_train, X_test, z_train, z_test = Split_and_Scale(X,z,test_size=test_size) #StardardScaler, test_size=0.2, scale=true
        error[degree], bias[degree], variance[degree] = bias_variance_analysis(X_train, X_test, z_train, z_test, n_resampling = n_resampling)
    
        # For debugging
        print('Error:', error[degree])
        print('Bias^2:', bias[degree])
        print('Var:', variance[degree])
        print('{} >= {} + {} = {}'.format(error[degree], bias[degree], variance[degree], bias[degree]+variance[degree]))

        # test: close to the precision...
        
    if (plot==True):
        plt.plot(complexity, error, label='Error')
        plt.plot(complexity, bias, label=r'$Bias^2$')
        plt.plot(complexity, variance, label='Variance')
        plt.xlabel("complexity")
        plt.ylabel("MSE")
        plt.title(title)
        plt.legend()
        plt.show()
    
    return error, bias, variance
