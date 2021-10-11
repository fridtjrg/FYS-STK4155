from regan import *
import numpy as np

np.random.seed(1234)

# Datapoints (squared root of datapoints -> meshgrid)
n = 25
# Paramaters of noise distribution
mu_N = 0; sigma_N = 0.2

# Create vanilla dataset:
x,y,z = create_xyz_dataset(n,mu_N, sigma_N)

run_plot_compare(z,'Task 5 Franke Function', 100, N=n, n_lambdas=30, k=5,poly_degree = 20,plot=True,saveplots=True)