import matplotlib.pyplot as plt
from typing_extensions import runtime
from regan import *
import numpy as np
from time import time

def run_CV(k,x,y,z,solver,m, n_lambdas = 20):
    complexity = []
    MSE_train_set = []
    MSE_test_set = []
    runtime_set = []
    for i in range(2,m): #goes out of range for high i?
        t_0 = time()
        X = create_X(x, y, i)
        MSE_train, MSE_test = cross_validation(k,X,z,solver=solver,n_lambdas = n_lambdas)
        complexity.append(i)
        MSE_train_set.append(MSE_train)
        MSE_test_set.append(MSE_test)
        runtime_set.append(time()-t_0)

    return complexity,MSE_train_set,MSE_test_set, runtime_set

def run_bootstrap():

    return 0

def run_plot_compare(datapoints, title, N = 50, plot=False, n_lambdas = 20, k = 5, poly_degree = 10):
    """[summary]

    Args:
        datapoints ([type]): [description]
        title ([type]): [description]
        N (int, optional): [description]. Defaults to 50.
        plot (bool, optional): [description]. Defaults to False.
        n_lambdas (int, optional): [description]. Defaults to 20.
        k (int, optional): [description]. Defaults to 5.
        poly_degree (int, optional): [description]. Defaults to 10.
    """

    #----------------------------------------------------------------------------
    #                           Preparing the dataset
    #----------------------------------------------------------------------------
    #Plot datapoints   
    m = poly_degree # polynomial order
    datapoints = datapoints[:N,:N]

    z = datapoints

    # Creates mesh of image pixels
    x = np.linspace(0,1, np.shape(z)[0])
    y = np.linspace(0,1, np.shape(z)[1])
    x,y = np.meshgrid(x,y)
    z = z.ravel()

    if plot:
        fig, ax = plt.subplots()
        plt.imshow(datapoints, cmap='gray')
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.show()

    #----------------------------------------------------------------------------
    #                           CrossValidation
    #----------------------------------------------------------------------------
    CV_MSE_train = []
    CV_MSE_test = []
    CV_complexity = []
    CV_runtimes = []
    solvers = ['OLS','RIDGE','LASSO']

    for solver in solvers:
        complexity,MSE_train_set,MSE_test_set,runtimes = run_CV(k,x,y,z,solver,m,n_lambdas = n_lambdas)
        CV_MSE_train.append(MSE_train_set)
        CV_MSE_test.append(MSE_test_set)
        CV_complexity.append(complexity)
        CV_runtimes.append(runtimes)
    
    if True:
        fig1, ax1 = plt.subplots()
        fig2, ax2 = plt.subplots()
        ax1.set_yscale('log')
        ax2.set_yscale('log')
        for i in range(len(solvers)):
            ax1.plot(CV_complexity[i],CV_MSE_train[i], label =solvers[i]+" train",linestyle = 'dashed')  
            ax1.plot(CV_complexity[i],CV_MSE_test[i], label =solvers[i]+ " test")  
            ax2.plot(CV_complexity[i],CV_runtimes[i], label =solvers[i]+ " runtime")
    
        ax1.set_xlabel("complexity")
        ax1.set_ylabel("MSE")
        ax1.set_title(f"Plot of the MSE as a function of complexity of the models \n with k = {k}, and $N_\lambda$ = {n_lambdas}")
        ax1.legend()
        ax1.grid()

        ax2.set_xlabel("complexity")
        ax2.set_ylabel("Runtime [s]")
        ax2.set_title("Plot of the runtime as a function of complexity of the models")
        ax2.legend()
        ax2.grid()
        plt.show() 
    
    #----------------------------------------------------------------------------
    #                           Bootstrap
    #----------------------------------------------------------------------------

    

