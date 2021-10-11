import numpy as np
from .regression import resample, OLS_solver, ridge_reg, lasso_reg, create_X, Split_and_Scale
import matplotlib.pyplot as plt
import pandas as pd


def Confidence_Interval(beta, X, sigma=1):
    #Calculates variance of beta, extracting just the diagonal elements of the matrix
    #var(B_j)=sigma^2*(X^T*X)^{-1}_{jj}
    beta_variance = np.diag(sigma**2 * np.linalg.pinv(X.T @ X))
    ci1 = beta - 1.96 * np.sqrt(beta_variance)/(X.shape[0])
    ci2 = beta + 1.96 * np.sqrt(beta_variance)/(X.shape[0])
    print('Confidence interval of β-estimator at 95 %:')
    ci_df = {r'$β_{-}$': ci1,
             r'$β_{ols}$': beta,
             r'$β_{+}$': ci2}
    ci_df = pd.DataFrame(ci_df)
    display(np.round(ci_df,3))
    return ci1, ci2

# Bootstrap resampling
# Return a (m x n_bootstraps) matrix with the column vectors z_pred for each bootstrap iteration.
def bootstrap(X_train, X_test, z_train, z_test, n_boostraps=100, solver = 'OLS', n_lambdas = 20):
    if solver not in ["OLS", "RIDGE", "LASSO"]:
        raise ValueError("solver must be OLS, RIDGE OR LASSO")

    z_pred_boot = np.empty((z_test.shape[0], n_boostraps))
    for i in range(n_boostraps):
        # Draw a sample of our dataset
        X_sample, z_sample = resample(X_train, z_train)
        # Perform OLS equation
        if solver == "OLS":
            ols_beta, z_tilde, z_pred = OLS_solver(X_train, X_test, z_train, z_test)
        elif solver == "RIDGE":
            ridge_beta_opt, z_tilde, z_pred, best_lamda = ridge_reg(X_train, X_test, z_train, z_test, nlambdas=n_lambdas)
        elif solver == "LASSO":
            z_tilde, z_pred, best_lamda = lasso_reg(X_train, X_test, z_train, z_test, nlambdas=n_lambdas)

        z_pred_boot[:, i] = z_pred.ravel()
    return z_pred_boot
    
# Bias-variance tradeoff

# Note: Expectations and variances taken w.r.t. different training
# data sets, hence the axis=1. Subsequent means are taken across the test data
# set in order to obtain a total value, but before this we have error/bias/variance
# calculated per data point in the test set.
# Note 2: The use of keepdims=True is important in the calculation of bias as this
# maintains the column vector form. Dropping this yields very unexpected results.

# conclude with cross validation

def bias_variance_analysis(X_train, X_test, z_train, z_test, resampling="bootstrap", n_resampling = 100, solver = 'OLS', n_lambdas = 20):
    if(resampling=="bootstrap"):
        z_pred = bootstrap(X_train, X_test, z_train, z_test, n_boostraps = n_resampling, solver = 'OLS', n_lambdas = 20)
    """ else:
        z_pred = crossvalidation(X_train, X_test, z_train, z_test, n_resampling)
    """

    error = np.mean( np.mean((z_test.reshape(-1,1) - z_pred)**2, axis=1, keepdims=True) ) # MSE
    bias2 = np.mean( (z_test.reshape(-1,1) - np.mean(z_pred, axis=1, keepdims=True))**2 ) # bias^2
    variance = np.mean( np.var(z_pred, axis=1, keepdims=True) )

    return error, bias2, variance
    
# Plot bias-variance tradeoff in function of complexity of the model
def bias_variance_complexity(x, y, z, complexity = np.arange(1,15), n_resampling = 100, test_size = 0.2, plot=True, title="Bias-variance analysis: MSE as a function of model complexity", solver = 'OLS', n_lambdas = 20):

    if complexity.__class__ == int:
        complexity = np.arange(1,complexity)
    error = np.zeros(complexity.size)
    bias = np.zeros(complexity.size)
    variance = np.zeros(complexity.size)

    for degree in complexity:
        X = create_X(x, y, degree)
        X_train, X_test, z_train, z_test = Split_and_Scale(X,z,test_size=test_size) #StardardScaler, test_size=0.2, scale=true
        error[degree-1], bias[degree-1], variance[degree-1] = bias_variance_analysis(X_train, X_test, z_train, z_test, n_resampling = n_resampling)
    
        # For debugging
        #print('Error:', error[degree])
        #print('Bias^2:', bias[degree])
        #print('Var:', variance[degree])
        #print('{} >= {} + {} = {}'.format(error[degree], bias[degree], variance[degree], bias[degree]+variance[degree]))

        # test: close to the precision...
        
    if (plot==True):
        plt.plot(complexity, error, label='Error')
        plt.plot(complexity, bias, label=r'$Bias^2$')
        plt.plot(complexity, variance, label='Variance')
        plt.xlabel("complexity")
        plt.ylabel("MSE")
        plt.title(title)
        plt.legend()
        plt.show()
    
    return error, bias, variance #its bias2