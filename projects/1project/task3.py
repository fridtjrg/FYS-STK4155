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

from random import random, seed
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from linear_regression import FrankeFunction, create_X
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
plt.show() 
