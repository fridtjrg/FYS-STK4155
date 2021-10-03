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
from linear_regression import FrankeFunction, create_X, Split_and_Scale, OLS_solver, MSE, R2, plot_ols_complexity


# Create vanilla dataset:
np.random.seed(1234)

n = 25

x = np.linspace(0,1,n)
y = np.linspace(0,1,n) 

x,y = np.meshgrid(x,y)
sigma_N = 0.2; mu_N = 0 #change for value of sigma_N to appropriate values
z = FrankeFunction(x,y) +mu_N +sigma_N*np.random.randn(n,n)	#adding noise to the dataset

complexity = range(2,20)
plot_ols_complexity(x,y,z)


"""



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





plt.xlabel("x-values")
plt.ylabel("u and v values")
plt.title("Comparison between u and v values (n="+str(n)+")")
plt.legend()
plt.grid()     
#plt.savefig('u_and_v_plotls(n='+str(n)+').pdf')
plt.show()     

"""
