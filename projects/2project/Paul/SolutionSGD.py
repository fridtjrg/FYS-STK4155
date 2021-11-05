from SDG import SDG
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

#====================== Data

from DataRegression import X, X_test, X_train, x, x_mesh, y_mesh, z_test, z_train, plotFunction, z


def SGD_ols_ridge_mse():
    MSE_ridge_val = []
    MSE_ols_val = []

    Eta = np.logspace(-5, -3, 10)
    methods = ['ridge', 'ols']

    for method in methods:

        if method == 'ridge':

            best_learning_rate_ridge = Eta[0]
            best_beta_ridge = np.zeros(X.shape)
            best_mse_ridge = 1e10

            for eta in Eta:
                sdg = SDG(learning_rate=eta, n_epochs=100, batch_size=10, method='ridge', lmbda= 0.01)
                beta = sdg.train(X_train, z_train)
                mse_ols_, mse_ridge_ = sdg.compute_test_mse(X_test, z_test, lambda_=0.01, beta=beta)
                MSE_ridge_val.append(mse_ridge_)
                if mse_ridge_ < best_mse_ridge:
                    best_learning_rate_ridge = eta
                    best_beta_ridge = beta
                    best_mse_ridge = mse_ridge_

        if method == 'ols':
            best_learning_rate_ols = Eta[0]
            best_beta_ridge = np.zeros(X.shape)
            best_mse_ols = 1e10

            for eta in Eta:
                sdg = SDG(learning_rate=eta, n_epochs=100, batch_size=10, method='ols', lmbda= 0.01)
                beta = sdg.train(X_train, z_train)
                mse_ols_, mse_ridge_ = sdg.compute_test_mse(X_test, z_test, lambda_=0.01, beta=beta)
                MSE_ols_val.append(mse_ols_)
                if mse_ols_ < best_mse_ols:
                    best_learning_rate_ols = eta
                    best_beta_ols = beta
                    best_mse_ols = mse_ols_
    print("MSE ridge")
    print(MSE_ridge_val)
    print("MSE ols")
    print(MSE_ols_val)
    plot, ax = plt.subplots()
    plt.title('MSE for the OLS and Ridge')
    plt.semilogx(Eta, MSE_ridge_val, 'k-o', label='Ridge')
    plt.semilogx(Eta, MSE_ols_val, 'r-o', label='OLS')
    plt.xlabel(r'Learning rate $\eta$')
    plt.ylabel('MSE')
    #ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    plt.legend()
    plt.subplots_adjust(left=0.2, bottom=0.2, right=0.9)
    print("best_learning_rate_ols = ", best_learning_rate_ols)
    print("best_learning_rate_ridge = ", best_learning_rate_ridge)
    best_pred_ols = X @ best_beta_ols
    best_pred_ridge = X @ best_beta_ridge
    plotFunction(x_mesh, y_mesh, z, 'data')
    plotFunction(x_mesh, y_mesh, best_pred_ols.reshape(len(x), len(x)), 'OLS')
    plotFunction(x_mesh, y_mesh, best_pred_ridge.reshape(len(x), len(x)), 'RIDGE')
    return plt.show()


def SDG_ols_ridge_matrix_mse():

    Eta = np.logspace(-5, -3, 10)
    Lambda = np.logspace(-5, -1, 10)

    MSE_ridge_val_train = np.zeros((len(Eta), len(Lambda)))
    MSE_ols_val_test = np.zeros((len(Eta), len(Lambda)))
    MSE_ridge_val_test = np.zeros((len(Eta), len(Lambda)))
    MSE_ols_val_train = np.zeros((len(Eta), len(Lambda)))

    methods = ['ridge', 'ols']

    for method in methods:

        if method == 'ridge':

            best_learning_rate_ridge = Eta[0]
            best_lambda_rate_ridge = Lambda[0]
            best_beta_ridge = np.zeros(X.shape)
            best_mse_ridge = 1e10

            for i, eta in enumerate(Eta):
                for j, _lambda in enumerate(Lambda):
                    sdg = SDG(learning_rate=eta, n_epochs=100, batch_size=10, method='ridge', lmbda= _lambda)
                    beta = sdg.train(X_train, z_train)
                    mse_ols_, mse_ridge_ = sdg.compute_test_mse(X_test, z_test, lambda_=_lambda, beta=beta)
                    mse_ols_train, mse_ridge_train = sdg.compute_test_mse(X_train, z_train, lambda_=_lambda, beta=beta)
                    MSE_ridge_val_train[i][j] = mse_ridge_train
                    MSE_ridge_val_test[i][j] = mse_ridge_
                    if mse_ridge_ <= best_mse_ridge:
                        best_lambda_rate_ridge = _lambda
                        best_learning_rate_ridge = eta
                        best_beta_ridge = beta
                        best_mse_ridge = mse_ridge_

        if method == 'ols':

            best_learning_rate_ols = Eta[0]
            best_lambda_rate_ols = Lambda[0]
            best_beta_ols = np.zeros(X.shape)
            best_mse_ols = 1e10

            for i, eta in enumerate(Eta):
                for j, _lambda in enumerate(Lambda):
                    sdg = SDG(learning_rate=eta, n_epochs=100, batch_size=10, method='ols', lmbda= _lambda)
                    beta = sdg.train(X_train, z_train)
                    mse_ols_, mse_ridge_ = sdg.compute_test_mse(X_test, z_test, lambda_=_lambda, beta=beta)
                    mse_ols_train, mse_ridge_train = sdg.compute_test_mse(X_train, z_train, lambda_=_lambda, beta=beta)
                    MSE_ols_val_train[i][j] = mse_ols_train
                    MSE_ols_val_test[i][j] = mse_ols_
                    if mse_ols_ <= best_mse_ols:
                        best_lambda_rate_ols = _lambda
                        best_learning_rate_ols = eta
                        best_beta_ols = beta
                        best_mse_ols = mse_ols_

    print("best_learning_rate_ols = ", best_learning_rate_ols)
    print("best_lambda_rate_ols = ", best_lambda_rate_ols)
    print("best_learning_rate_ridge = ", best_learning_rate_ridge)
    print("best_lambda_rate_ridge = ", best_lambda_rate_ridge)

    fig, ax = plt.subplots(figsize=(5, 5))
    sns.heatmap(MSE_ridge_val_train, annot=True, ax=ax, cmap="viridis")
    ax.set_title("Training ridge MSE")
    ax.set_ylabel("$\eta$")
    ax.set_xlabel("$\lambda$")

    fig, ax = plt.subplots(figsize=(5, 5))
    sns.heatmap(MSE_ridge_val_test, annot=True, ax=ax, cmap="viridis")
    ax.set_title("Test ridge MSE")
    ax.set_ylabel("$\eta$")
    ax.set_xlabel("$\lambda$")

    fig, ax = plt.subplots(figsize=(5, 5))
    sns.heatmap(MSE_ols_val_train, annot=True, ax=ax, cmap="viridis")
    ax.set_title("Training ols MSE")
    ax.set_ylabel("$\eta$")
    ax.set_xlabel("$\lambda$")

    fig, ax = plt.subplots(figsize=(5, 5))
    sns.heatmap(MSE_ols_val_test, annot=True, ax=ax, cmap="viridis")
    ax.set_title("Test ols MSE")
    ax.set_ylabel("$\eta$")
    ax.set_xlabel("$\lambda$")

    best_pred_ols = X @ best_beta_ols
    best_pred_ridge = X @ best_beta_ridge

    plotFunction(x_mesh, y_mesh, z, 'data')
    plotFunction(x_mesh, y_mesh, best_pred_ols.reshape(len(x), len(x)), 'OLS')
    plotFunction(x_mesh, y_mesh, best_pred_ridge.reshape(len(x), len(x)), 'RIDGE')

    return best_learning_rate_ols,  best_lambda_rate_ols,  best_learning_rate_ridge, best_lambda_rate_ridge




#SGD_ols_ridge_mse()


def SDG_ols_ridge_epoch(best_learning_rate_ols,  best_lambda_rate_ols,  best_learning_rate_ridge, best_lambda_rate_ridge):
    MSE_ridge_val = []
    MSE_ols_val = []

    epochs = [10, 20, 50, 70, 100, 120, 200]
    methods = ['ridge', 'ols']

    for method in methods:

        if method == 'ridge':

            for nb_epochs in epochs:
                sdg = SDG(learning_rate=best_learning_rate_ridge, n_epochs= nb_epochs, batch_size=10, method='ridge', lmbda= best_lambda_rate_ridge)
                beta = sdg.train(X_train, z_train)
                mse_ols_, mse_ridge_ = sdg.compute_test_mse(X_test, z_test, lambda_=0.01, beta=beta)
                MSE_ridge_val.append(mse_ridge_)


        if method == 'ols':

            for nb_epochs in epochs:
                sdg = SDG(learning_rate=best_learning_rate_ols, n_epochs=nb_epochs, batch_size=10, method='ols', lmbda= best_lambda_rate_ols)
                beta = sdg.train(X_train, z_train)
                mse_ols_, mse_ridge_ = sdg.compute_test_mse(X_test, z_test, lambda_=0.01, beta=beta)
                MSE_ols_val.append(mse_ols_)

    plot, ax = plt.subplots()
    plt.title('MSE for the OLS and Ridge')
    plt.semilogx(epochs, MSE_ridge_val, 'k-o', label='Ridge')
    plt.semilogx(epochs, MSE_ols_val, 'r-o', label='OLS')
    plt.xlabel('nb_epochs')
    plt.ylabel('MSE')
    #ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    plt.legend()
    plt.subplots_adjust(left=0.2, bottom=0.2, right=0.9)


def SDG_ols_ridge_batch_size(best_learning_rate_ols,  best_lambda_rate_ols,  best_learning_rate_ridge, best_lambda_rate_ridge):
    MSE_ridge_val = []
    MSE_ols_val = []

    batch_size = [1, 2, 5, 10, 20, 30, 50]
    methods = ['ridge', 'ols']

    for method in methods:

        if method == 'ridge':

            for batch in batch_size:
                sdg = SDG(learning_rate=best_learning_rate_ridge, n_epochs= 100, batch_size=batch, method='ridge', lmbda= best_lambda_rate_ridge)
                beta = sdg.train(X_train, z_train)
                mse_ols_, mse_ridge_ = sdg.compute_test_mse(X_test, z_test, lambda_=0.01, beta=beta)
                MSE_ridge_val.append(mse_ridge_)


        if method == 'ols':

            for batch in batch_size:
                sdg = SDG(learning_rate=best_learning_rate_ols, n_epochs=100, batch_size=batch, method='ols', lmbda= best_lambda_rate_ols)
                beta = sdg.train(X_train, z_train)
                mse_ols_, mse_ridge_ = sdg.compute_test_mse(X_test, z_test, lambda_=0.01, beta=beta)
                MSE_ols_val.append(mse_ols_)

    plot, ax = plt.subplots()
    plt.title('MSE for the OLS and Ridge')
    plt.semilogx(batch_size, MSE_ridge_val, 'k-o', label='Ridge')
    plt.semilogx(batch_size, MSE_ols_val, 'r-o', label='OLS')
    plt.xlabel('batch_size')
    plt.ylabel('MSE')
    #ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    plt.legend()
    plt.subplots_adjust(left=0.2, bottom=0.2, right=0.9)



best_learning_rate_ols,  best_lambda_rate_ols,  best_learning_rate_ridge, best_lambda_rate_ridge = SDG_ols_ridge_matrix_mse()
SDG_ols_ridge_epoch(best_learning_rate_ols,  best_lambda_rate_ols,  best_learning_rate_ridge, best_lambda_rate_ridge)
SDG_ols_ridge_batch_size(best_learning_rate_ols,  best_lambda_rate_ols,  best_learning_rate_ridge, best_lambda_rate_ridge)


plt.show()