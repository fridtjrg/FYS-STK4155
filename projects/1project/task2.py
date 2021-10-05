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
from linear_regression import plot_ols_complexity, create_xyz_dataset, plot_bias_variance_complexity

def train_n(n,test_size):
    return n*n*(1-test_size)
    
# Create vanilla dataset:
np.random.seed(1234)

# Datapoints (squared root of datapoints -> meshgrid)
n = 25
# Paramaters of noise distribution
mu_N = 0; sigma_N = 0.2
# Parameter of splitting data
test_size=0.2

x,y,z = create_xyz_dataset(n,mu_N, sigma_N)
z = z.reshape(n*n,1)

print(r"Part 1: $MSE_{train}$ and $MSE_{test}$ in function of the complexity of the model (degree-order of polynomial)")
complexity = np.arange(2,21)
plot_ols_complexity(x,y,z, complexity)

print("Part 2: perform a bias-variance tradeoff analysis")
complexity = np.arange(0,15)
print("Datapoints:", train_n(n,test_size), "– Complexity:", complexity)
plot_bias_variance_complexity(x, y, z, complexity)

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
