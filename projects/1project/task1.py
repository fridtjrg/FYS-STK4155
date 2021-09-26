from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import numpy as np
from random import random, seed
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler



def FrankeFunction(x,y): #code from task
	term1 = 0.75*np.exp(-(0.25*(9*x-2)**2) - 0.25*((9*y-2)**2))
	term2 = 0.75*np.exp(-((9*x+1)**2)/49.0 - 0.1*(9*y+1))
	term3 = 0.5*np.exp(-(9*x-7)**2/4.0 - 0.25*((9*y-3)**2))
	term4 = -0.2*np.exp(-(9*x-4)**2 - (9*y-7)**2)
	return term1 + term2 + term3 + term4

#calculates R2 score and MSE
def R2(y_data, y_model): #week 35 exercise
    return 1 - np.sum((y_data - y_model) ** 2) / np.sum((y_data - np.mean(y_data)) ** 2)
def MSE(y_data,y_model):
    n = np.size(y_model)
    return np.sum((y_data-y_model)**2)/n


def SVD(A): #week35 SVD
    U, S, VT = np.linalg.svd(A,full_matrices=True)
    D = np.zeros((len(U),len(VT)))
    for i in range(0,len(VT)):
        D[i,i]=S[i]
    return U @ D @ VT

#Makes a 3d plot of the franke function
def Plot_franke_function(): #code from task
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


#setting up data
n = 100

x = np.linspace(0,1,n)
y = np.linspace(0,1,n) 

sigma_N = 0.1; mu_N = 0 #change for value of sigma_N to appropriate values
z = FrankeFunction(x,y) + sigma_N*np.random.randn(100)	#adding noise to the dataset


#Setting up design matrix from week 35-36 lecture slides
def create_X(x, y, n):
	if len(x.shape) > 1:
		x = np.ravel(x)
		y = np.ravel(y)

	N = len(x)
	l = int((n+1)*(n+2)/2)		# Number of elements in beta
	X = np.ones((N,l))

	for i in range(1,n+1):
		q = int((i)*(i+1)/2)
		for k in range(i+1):
			X[:,q+k] = (x**(i-k))*(y**k)

	return X

highest_order = 5 #5th ordere polynomial
X = create_X(x, y, highest_order)






#Splitting training and test data
X_train, X_test, z_train, z_test = train_test_split(X, z, test_size=0.2)

#scaling the the input with standardscalar (week35)
scaler = StandardScaler(with_std=False)
scaler.fit(X_train)

X_scaled = scaler.transform(X_train)

#used to scale train and test
z_mean = np.mean(z_train)
z_sigma = np.std(z_train)

z_train = (z_train- z_mean)/z_sigma




#Singular value decomposition
X_train = SVD(X_train) 


# Calculating Beta Ordinary Least Square with matrix inversion
beta = np.linalg.pinv(X_train.T @ X_train) @ X_train.T @ z_train #psudoinverse

z_test = (z_test- z_mean)/z_sigma

X_test = scaler.transform(X_test)

ztilde = X_train @ beta
print("Training R2")
print(R2(z_train,ztilde))
print("Training MSE")
print(MSE(z_train,ztilde))
zpredict = X_test @ beta
print("Test R2")
print(R2(z_test,zpredict))
print("Test MSE")
print(MSE(z_test,zpredict))

plt.plot()


