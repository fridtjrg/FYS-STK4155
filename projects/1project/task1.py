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
import statistics
from time import time
from scipy.stats import norm
import matplotlib.pyplot as plt



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



def OLS_solver(designmatrix, datapoints):
	X = designmatrix
	z = datapoints


	#Splitting training and test data (20%test)
	X_train, X_test, z_train, z_test = train_test_split(X, z, test_size=0.2)

	#scaling the the input with standardscalar (week35)
	scaler = StandardScaler()
	scaler.fit(X_train)

	X_scaled = scaler.transform(X_train)

	#used to scale train and test
	z_mean = np.mean(z_train)
	z_sigma = np.std(z_train)

	z_train = (z_train- z_mean)/z_sigma




	#Singular value decomposition (removed as it doesn't work ref group teacher)
	#X_train = SVD(X_train) 


	# Calculating Beta Ordinary Least Square with matrix inversion
	ols_beta = np.linalg.pinv(X_train.T @ X_train) @ X_train.T @ z_train #psudoinverse

	#Scaling test data
	z_test = (z_test- z_mean)/z_sigma

	X_test = scaler.transform(X_test)

	ztilde = X_train @ ols_beta
	print("Training R2")
	print(R2(z_train,ztilde))
	print("Training MSE")
	print(MSE(z_train,ztilde))


	zpredict = X_test @ ols_beta
	print("Test R2")
	print(R2(z_test,zpredict))
	print("Test MSE")
	print(MSE(z_test,zpredict))

	#beta_ols_variance = z_sigma**2 @ np.linalg.pinv(X_train.T @ X_train) #Agree correct?
	return ols_beta, MSE(z_train,ztilde), MSE(z_test,zpredict)


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


#------Task 2------

#setting up data
n = 500 #does it matter?

x = np.linspace(0,1,n)
y = np.linspace(0,1,n) 

sigma_N = 0.1; mu_N = 0 #change for value of sigma_N to appropriate values
z = FrankeFunction(x,y) + sigma_N*np.random.randn(n)	#adding noise to the dataset

#gives a weird graph which does not bahve as expected
#Because bootsatrap is not implemented?
complexity = []
MSE_train_set = []
MSE_test_set = []


X = create_X(x, y, 40)
ols_beta, MSE_train, MSE_test = OLS_solver(X,z)




#not working as intended
for i in range(2,30): #goes out of range for high i?
	
	X = create_X(x, y, i)
	ols_beta, MSE_train, MSE_test = OLS_solver(X,z)
	complexity.append(i)
	MSE_train_set.append(MSE_train)
	MSE_test_set.append(MSE_test)




plt.plot(complexity,MSE_train_set, label ="train")  
plt.plot(complexity,MSE_test_set, label ="test")  
 

plt.xlabel("complexity")
plt.ylabel("MSE")
plt.title("Plot of the MSE as a function of complexity of the model")
plt.legend()
plt.grid()     
#plt.savefig('Task2plot(n='+str(n)+').pdf')
plt.show() 




#How to use combine bootstrap with OLS?


# Returns mean of bootstrap samples 
# Bootstrap algorithm, returns estimated mean values for each bootstrap operation
def bootstrap(designmatrix, data, bootstrap_operations): #from week 37 lecture notes

    new_dataset_mean = np.zeros(bootstrap_operations) 
    n = len(data)			 #Data should z from my understanding
    # non-parametric bootstrap         
    for i in range(bootstrap_operations):
        new_dataset_mean[i] = np.mean(data[np.random.randint(0,n,n)]) #is this beta*(is of size n)
   		#Do we have ro relate back to fetch the OLS beta values here?
   		#In that case, do we remove the mean of the values or take mean of the OLS values?

    # analysis    
    print("Bootstrap Statistics :")
    print("original           bias      std. error")
    print("%8g %8g %14g %15g" % (np.mean(data), np.std(data),np.mean(new_dataset_mean),np.std(new_dataset_mean)))
    return new_dataset_mean


n_bootstrap = len(z) #number of bootstrap operations

bootstrap_means = bootstrap(X,z,n_bootstrap)


#from week 37 notes
n, binsboot, patches = plt.hist(bootstrap_means, 50, density=True, facecolor='red', alpha=0.75)
# add a 'best fit' line  
y = norm.pdf(binsboot, np.mean(bootstrap_means), np.std(bootstrap_means))
lt = plt.plot(binsboot, y, 'b', linewidth=1)
plt.xlabel('x')
plt.ylabel('Probability')
plt.grid(True)
plt.show()



"""




plt.xlabel("x-values")
plt.ylabel("u and v values")
plt.title("Comparison between u and v values (n="+str(n)+")")
plt.legend()
plt.grid()     
#plt.savefig('u_and_v_plotls(n='+str(n)+').pdf')
plt.show()     

"""