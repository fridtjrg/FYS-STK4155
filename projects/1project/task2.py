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
from regan import plot_ols_complexity, create_xyz_dataset, bias_variance_complexity, Plot_FrankeFunction
import matplotlib.pyplot as plt

def train_n(n,test_size):
    return int(n*n*(1-test_size))
    
def test_n(n,test_size):
    return int(n*n*test_size)
    
np.random.seed(1234)

# Datapoints (squared root of datapoints -> meshgrid)
n = 25
# Paramaters of noise distribution
mu_N = 0; sigma_N = 0.2
# Parameter of splitting data
test_size = 0.2

# Create vanilla dataset:
x,y,z = create_xyz_dataset(n,mu_N, sigma_N); z = z.ravel()

print("Part 1: $MSE_{train}$ and $MSE_{test}$ in function of the complexity of the model (degree-order of polynomial) \n")
complexity = np.arange(0,21)
plot_ols_complexity(x,y,z, complexity)

print("Part 2: perform a bias-variance tradeoff analysis \n")
complexity = np.arange(0,13)
print("Train datapoints:", train_n(n,test_size))
print("Test datapoints:", test_n(n,test_size))
bias_variance_complexity(x, y, z, complexity, test_size=test_size)

print("Bias-variance tradeoff analysis with variation in training and testing datapoints")
n_ = [25,40]
test_size_ = [0.2, 0.33]
complexity = np.arange(0,18)

fig=plt.figure(figsize=(15, 10))

for i in n_:

    x,y,z = create_xyz_dataset(i,mu_N, sigma_N); z = z.ravel()
    
    for ts in test_size_:
        print("Datapoints:", i*i, "– Test size:", round(ts,3))
        
        error, bias, variance = bias_variance_complexity(x, y, z, complexity, test_size=ts, plot=False)
        plt.plot(complexity, error, label='Error-n_train'+str(train_n(i,ts))+'-n_test'+str(test_n(i,ts)))
        plt.plot(complexity, bias, label=r'$Bias^2$-n_train'+str(train_n(i,ts))+'-n_test'+str(test_n(i,ts)))
        plt.plot(complexity, variance, label='Variance-n_train'+str(train_n(i,ts))+'-n_test'+str(test_n(i,ts)))
        
plt.xlabel("complexity")
plt.ylabel("MSE")
plt.title("Bias variance analysis - datapoints variantions")
plt.legend()
plt.show()
 
"""
n = 40
complexity = np.arange(0,11)
print("Train datapoints:", train_n(n,test_size))
print("Test datapoints:", test_n(n,test_size))
print("Complexity:", complexity)
x,y,z = create_xyz_dataset(n,mu_N, sigma_N); z = z.reshape(n*n,1)
plot_bias_variance_complexity(x, y, z, complexity, test_size=test_size)

complexity = np.arange(0,18)
print("Train datapoints:", train_n(n,test_size))
print("Test datapoints:", test_n(n,test_size))
print("Complexity:", complexity)
plot_bias_variance_complexity(x, y, z, complexity, test_size=test_size)

n = 40
complexity = np.arange(0,18)
test_size=0.3
print("Train datapoints:", train_n(n,test_size))
print("Test datapoints:", test_n(n,test_size))
print("Complexity:", complexity)
plot_bias_variance_complexity(x, y, z, complexity, test_size=test_size)

"""
