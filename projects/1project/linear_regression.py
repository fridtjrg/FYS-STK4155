import numpy as np
from random import random, seed
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter


# FrankeFunction: a two-variables function to create the dataset of our vanilla problem
def FrankeFunction(x,y): #code from task
	term1 = 0.75*np.exp(-(0.25*(9*x-2)**2) - 0.25*((9*y-2)**2))
	term2 = 0.75*np.exp(-((9*x+1)**2)/49.0 - 0.1*(9*y+1))
	term3 = 0.5*np.exp(-(9*x-7)**2/4.0 - 0.25*((9*y-3)**2))
	term4 = -0.2*np.exp(-(9*x-4)**2 - (9*y-7)**2)
	return term1 + term2 + term3 + term4
 
# 3D plot of FrankeFunction
def Plot_FrankeFunction(): #code from task
  fig = plt.figure()
  ax = fig.gca(projection="3d")

  # Make data.
  x = np.arange(0, 1, 0.05)
  y = np.arange(0, 1, 0.05)
  x, y = np.meshgrid(x,y)
  z = FrankeFunction(x, y)

  # Plot the surface.
  surf = ax.plot_surface(x, y, z, cmap=cm.coolwarm,
  linewidth=0, antialiased=False)

  # Customize the z axis.
  ax.set_zlim(-0.10, 1.40)
  ax.zaxis.set_major_locator(LinearLocator(10))
  ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))

  # Add a color bar which maps values to colors.
  fig.colorbar(surf, shrink=0.5, aspect=5)
  plt.show()

# Error analysis: MSE and R2 score
def R2(y_data, y_model): #week 35 exercise
    return 1 - np.sum((y_data - y_model) ** 2) / np.sum((y_data - np.mean(y_data)) ** 2)
def MSE(y_data,y_model):
    n = np.size(y_model)
    return np.sum((y_data-y_model)**2)/n

# SVD theorem
def SVD(A): #week35 SVD change to week 36
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

# Design matrix
def create_X(x, y, n): # week 35-36 lecture slides
	if len(x.shape) > 1:
		x = np.ravel(x)
		y = np.ravel(y)

	N = len(x)
	l = int((n+1)*(n+2)/2)		# Number of elements in beta, number of feutures (order-degree of polynomial)
	X = np.ones((N,l))

	for i in range(1,n+1):
		q = int((i)*(i+1)/2)
		for k in range(i+1):
			X[:,q+k] = (x**(i-k))*(y**k)

	return X
  
# Splitting and rescaling data (rescaling is optional)
# Default values: 20% of test data and the scaler is StandardScaler without std.dev.
def Split_and_Scale(X,z,test_size=0.2, scale=True):

    #Splitting training and test data
    X_train, X_test, z_train, z_test = train_test_split(X, z, test_size=test_size)

    # Rescaling X and z (optional)
    if scale==True:
        scaler_X = StandardScaler(with_std=False)
        scaler_X.fit(X_train)
        X_train = scaler_X.transform(X_train)
        X_test = scaler_X.transform(X_test)

        scaler_z = StandardScaler(with_std=False)
        z_train = np.squeeze(scaler_z.fit_transform(z_train.reshape(-1, 1)))
        z_test = np.squeeze(scaler_z.transform(z_test.reshape(-1, 1)))
      
    return X_train, X_test, z_train, z_test

# OLS equation
def OLS_solver(X_train, X_test, z_train, z_test):

	# Calculating Beta Ordinary Least Square with matrix inversion
	ols_beta = np.linalg.pinv(X_train.T @ X_train) @ X_train.T @ z_train #psudoinverse
  
	z_tilde = X_train @ ols_beta
	z_predict = X_test @ ols_beta

	#beta_ols_variance = z_sigma**2 @ np.linalg.pinv(X_train.T @ X_train) #Agree correct?
	return ols_beta, z_tilde, z_predict

def plot_ols_compelxity():

    complexity = []
    MSE_train_set = []
    MSE_test_set = []
    #not working as intended
    for degree in range(2,30):

        X = create_X(x, y, degree)
        print(np.shape(X))
        X_train, X_test, z_train, z_test = Split_and_Scale(X,z) #StardardScaler, test_size=0.2, scale=true
        ols_beta, z_tilde,z_predict = OLS_solver(X_train, X_test, z_train, z_test)

        complexity.append(degree)
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