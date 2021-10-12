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

import matplotlib.pyplot as plt
from typing_extensions import runtime
from regan import *
import numpy as np
from time import time
import os

def run_CV(k,x,y,z,solver,m, n_lambdas = 20):
    complexity = []
    MSE_train_set = []
    MSE_test_set = []
    runtime_set = []
    for i in range(2,m):
        t_0 = time()
        X = create_X(x, y, i)
        MSE_train, MSE_test = cross_validation(k,X,z,solver=solver,n_lambdas = n_lambdas)
        complexity.append(i)
        MSE_train_set.append(MSE_train)
        MSE_test_set.append(MSE_test)
        runtime_set.append((time()-t_0)/k)

    return complexity,MSE_train_set,MSE_test_set, runtime_set

def run_BS(x,y,z,solver,m,n_resampling,n_lambdas=20):
    bias_set = []
    variance_set = []
    error_set = []
    runtime_set = []
    for i in range(2,m):
        t_0 = time()
        error, bias2, variance = bias_variance_complexity(x,y,z,m,n_resampling=n_resampling,plot=False,solver=solver,n_lambdas=n_lambdas)
        bias_set.append(bias2)
        variance_set.append(variance)
        error_set.append(error)
        runtime_set.append(t_0-time())

    print("RUN_BS: ",bias_set)
    return bias_set, variance_set, error_set, runtime_set


def run_plot_compare(datapoints, title, n_resampling, N = 50, plot=False, n_lambdas = 20, k = 5, poly_degree = 10, plot_runtime=True, saveplots=False):
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
    #                         Storing figures
    #----------------------------------------------------------------------------
    if saveplots and plot:
        path = 'Figures'
        foldername = title.replace(' ','')
        try:
            os.mkdir(path+'/'+foldername)
        except FileExistsError:
            print("Dir already exists")
        path = path+'/'+foldername+'/'

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
        im = ax.imshow(datapoints, cmap='gray')
        ax.set_title(title)
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        fig.colorbar(im)
        if saveplots: fig.savefig(path+"Image")

    solvers = ['OLS','RIDGE','LASSO']


    #----------------------------------------------------------------------------
    #                           Bootstrap
    #----------------------------------------------------------------------------

    BS_bias = []
    BS_error = []
    BS_variance = []
    BS_runtimes = []
    for solver in solvers:
        bias2, variance, error = bias_variance_complexity(x,y,z,m,n_resampling=n_resampling,plot=False,solver=solver,n_lambdas=n_lambdas)
        bias = np.zeros(bias2.shape); np.sqrt(bias2,bias)
        BS_bias.append(bias)
        BS_error.append(variance)
        BS_variance.append(error)

    if plot:
        fig3, ax3 = plt.subplots()
        ax3.set_yscale('log')
        #ax4 = ax3.twinx()
        BS_complexity = [x for x in range(1,m)]
        lns = []
        for i in range(len(solvers)):
            lns.append(ax3.plot(BS_complexity,BS_bias[i], label =solvers[i]+" bias"))
            lns.append(ax3.plot(BS_complexity,BS_error[i], label =solvers[i]+ " error", linestyle = 'dashed'))
            lns.append(ax3.plot(BS_complexity,BS_variance[i], label =solvers[i]+ " variance", linestyle = 'dotted') )

        lns = [l[0] for l in lns]
        labs = [l.get_label() for l in lns]
        ax3.legend(lns,labs)

        ax3.set_xlabel("complexity")
        ax3.set_ylabel("MSE")
        #ax4.set_ylabel("variance")
        ax3.set_title(f"bias, variance and error from bootstrap resampling as a function of complexity \n with n_samples = {n_resampling}, and $N_\lambda$ = {n_lambdas}")
        ax3.grid()
        if saveplots: fig3.savefig(path+"BS.png")



    #----------------------------------------------------------------------------
    #                           CrossValidation
    #----------------------------------------------------------------------------
    CV_MSE_train = []
    CV_MSE_test = []
    CV_complexity = []
    CV_runtimes = []

    for solver in solvers:
        complexity,MSE_train_set,MSE_test_set,runtimes = run_CV(k,x,y,z,solver,m,n_lambdas = n_lambdas)
        CV_MSE_train.append(MSE_train_set)
        CV_MSE_test.append(MSE_test_set)
        CV_complexity.append(complexity)
        CV_runtimes.append(runtimes)
    
    
    if plot:
        fig1, ax1 = plt.subplots()
        fig2, ax2 = plt.subplots()
        ax1.set_yscale('log')
        if plot_runtime:
            ax2.set_yscale('log')
        for i in range(len(solvers)):
            ax1.plot(CV_complexity[i],CV_MSE_train[i], label =solvers[i]+" train",linestyle = 'dashed')  
            ax1.plot(CV_complexity[i],CV_MSE_test[i], label =solvers[i]+ " test")  
            if plot_runtime:
                ax2.plot(CV_complexity[i],CV_runtimes[i], label =solvers[i]+ " runtime")
    
        ax1.set_xlabel("complexity")
        ax1.set_ylabel("MSE")
        ax1.set_title(f"Plot of the MSE as a function of complexity of the models \n with k = {k}, and $N_\lambda$ = {n_lambdas}")
        ax1.legend()
        ax1.grid()
        if saveplots: fig1.savefig(path+"CV.png")

        if plot_runtime:
            ax2.set_xlabel("complexity")
            ax2.set_ylabel("Runtime [s]")
            ax2.set_title("Plot of the runtime as a function of complexity of the models")
            ax2.legend()
            ax2.grid()
            if saveplots: fig2.savefig(path+"Runtime.png")


        plt.show() 
    
    
    
