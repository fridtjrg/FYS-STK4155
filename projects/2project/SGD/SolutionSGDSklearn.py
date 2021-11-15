from sklearn.linear_model import SGDRegressor
import numpy as np
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

#====================== DATA
import sys
sys.path.append("../Data")
from DataRegression import X, X_test, X_train, x, x_mesh, y_mesh, z_test, z_train, plotFunction, z, plotSave, MSE






def SDG_ols_ridge_matrix_mse():

    Eta = np.logspace(-5, -4, 10)
    Lambda = np.logspace(-5, -3, 10)

    MSE_ridge_val_train = np.zeros((len(Eta), len(Lambda)))
    MSE_ols_val_test = [0]*len(Eta)
    MSE_ridge_val_test = np.zeros((len(Eta), len(Lambda)))
    MSE_ols_val_train = [0]*len(Eta)

    methods = ['ridge', 'ols']

    for method in methods:

        if method == 'ridge':
            Eta = np.logspace(-5, -4, 10)
            best_learning_rate_ridge = Eta[0]
            best_lambda_rate_ridge = Lambda[0]
            best_beta_ridge = np.zeros(X.shape)
            best_mse_ridge = 1e10

            for i, eta in enumerate(Eta):
                for j, _lambda in enumerate(Lambda):
                    sgd = SGDRegressor(max_iter = 100000, penalty=None, eta0=eta, learning_rate = 'constant', alpha=_lambda)
                    beta = sgd.fit(X_train, z_train)
                    ztildeSDG = sgd.predict(X_train)
                    ztestSDG = sgd.predict(X_test)
                    mse_ridge_train = MSE(ztildeSDG, z_train)
                    mse_ridge_ = MSE(ztestSDG, z_test)
                    MSE_ridge_val_train[i][j] = mse_ridge_train
                    MSE_ridge_val_test[i][j] = mse_ridge_
                    if mse_ridge_ <= best_mse_ridge:
                        best_lambda_rate_ridge = _lambda
                        best_learning_rate_ridge = eta
                        best_beta_ridge = beta
                        best_mse_ridge = mse_ridge_

        if method == 'ols':
            Eta_ols = np.logspace(-5, -3.4, 10)
            best_learning_rate_ols = Eta[0]
            best_beta_ols = np.zeros(X.shape)
            best_mse_ols = 1e10

            for i, eta in enumerate(Eta_ols):
                sgd = SGDRegressor(max_iter = 100000, penalty=None, eta0=eta, learning_rate = 'constant', alpha=0)
                beta = sgd.fit(X_train, z_train)
                ztildeSDG = sgd.predict(X_train)
                ztestSDG = sgd.predict(X_test)
                mse_ols_train = MSE(ztildeSDG, z_train)
                mse_ols_ = MSE(ztestSDG, z_test)
                MSE_ols_val_train[i] = mse_ols_train
                MSE_ols_val_test[i] = mse_ols_
                if mse_ols_ <= best_mse_ols:
                    best_learning_rate_ols = eta
                    best_beta_ols = beta
                    best_mse_ols = mse_ols_

    print("best_learning_rate_ols = ", best_learning_rate_ols)
    print("best_learning_rate_ridge = ", best_learning_rate_ridge)
    print("best_lambda_rate_ridge = ", best_lambda_rate_ridge)

    fig, ax = plt.subplots(figsize=(7, 5))
    sns.heatmap(MSE_ridge_val_train, annot=True, ax=ax)
    #ax.set_title("Training ridge MSE")
    ax.set_ylabel("$\eta$")
    ax.set_xlabel("$\lambda$")
    plt.subplots_adjust(
    top=0.98,
    bottom=0.117,
    left=0.08,
    right=1,
    hspace=0.2,
    wspace=0.2
    )
    plt.savefig('../Figures/GD/SGD_train_heatmap_ridge_sklearn.pdf')

    fig, ax = plt.subplots(figsize=(7, 5))
    sns.heatmap(MSE_ridge_val_test, annot=True, ax=ax, cmap="viridis")
    #ax.set_title("Test ridge MSE")
    ax.set_ylabel("$\eta$")
    ax.set_xlabel("$\lambda$")
    plt.subplots_adjust(
    top=0.98,
    bottom=0.117,
    left=0.08,
    right=1,
    hspace=0.2,
    wspace=0.2
    )
    plt.savefig('../Figures/GD/SGD_test_heatmap_ridge_sklearn.pdf')

    fig, ax = plt.subplots(figsize=(7, 5))
    plt.semilogx(Eta_ols, MSE_ols_val_train, 'k-o', label='MSE_train')
    plt.semilogx(Eta_ols, MSE_ols_val_test, 'r-o', label='MSE_test')
    plt.xlabel("$\eta$")
    plt.ylabel('MSE')
    #ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    plt.legend()
    plt.subplots_adjust(
    top=0.933,
    bottom=0.129,
    left=0.121,
    right=0.95,
    hspace=0.2,
    wspace=0.2
    )
    plt.savefig('../Figures/GD/SGD_mse_ols_sklearn.pdf')



SDG_ols_ridge_matrix_mse()

plt.show()