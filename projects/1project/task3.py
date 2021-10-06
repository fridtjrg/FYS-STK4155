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

from ols_solver import FrankeFunction, create_X
from crossvalidation import cross_validation


n = 30 #does it matter?

x = np.linspace(0,1,n)
y = np.linspace(0,1,n) 

sigma_N = 0.1; mu_N = 0 #change for value of sigma_N to appropriate values
z = FrankeFunction(x,y) + sigma_N*np.random.randn(n)	#adding noise to the dataset
print(z.shape)
#gives a weird graph which does not bahve as expected
#Because bootsatrap is not implemented?
complexity = []
MSE_train_set = []
MSE_test_set = []


for i in range(2,50): #goes out of range for high i?
	
	X = create_X(x, y, i)
	ols_beta, MSE_train, MSE_test = cross_validation(5,X,z)
	complexity.append(i)
	MSE_train_set.append(MSE_train)
	MSE_test_set.append(MSE_test)


fig = plt.figure()
ax = fig.add_subplot(1,1,1)
ax.plot(complexity,MSE_train_set, label ="train")  
ax.plot(complexity,MSE_test_set, label ="test")  
ax.set_yscale('log')

plt.xlabel("complexity")
plt.ylabel("MSE")
plt.title("Plot of the MSE as a function of complexity of the model")
plt.legend()
plt.grid()     
#plt.savefig('Task2plot(n='+str(n)+').pdf')
plt.show() 