from SDG import SDG
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

#====================== DATA
import sys
sys.path.append("../Data")
from DataRegression import X, X_test, X_train, x, x_mesh, y_mesh, z_test, z_train, plotFunction, z, plotSave

#from DataClassification import X_test, X_train, Y_train_onehot, Y_test_onehot, accuracy_score_numpy, X, y_test, y_train
#z_test = y_test
#z_train = y_train


#This function plot the heatmap matrix of the MSE for ridge regression and the MSE against the learning rate for OLS

def SDG_ols_ridge_matrix_mse():

    #This function plot the heatmap for ridge SGD and plot the MSE against the learning rate for OLS

    #Creation of or parameters
    Eta = np.logspace(-5, -4, 10)
    Lambda = np.logspace(-5, -3, 10)

    MSE_ridge_val_train = np.zeros((len(Eta), len(Lambda)))
    MSE_ols_val_test = [0]*len(Eta)
    MSE_ridge_val_test = np.zeros((len(Eta), len(Lambda)))
    MSE_ols_val_train = [0]*len(Eta)


    methods = ['ridge', 'ols']

    for method in methods:

        if method == 'ridge':
            #initialization of or best parameters
            Eta = np.logspace(-5, -4, 10)
            best_learning_rate_ridge = Eta[0]
            best_lambda_rate_ridge = Lambda[0]
            best_beta_ridge = np.zeros(X.shape)
            best_mse_ridge = 1e10

            for i, eta in enumerate(Eta):
                for j, _lambda in enumerate(Lambda):
                    #Create a SGD
                    sdg = SDG(learning_rate=eta, n_epochs=100, batch_size=10, method='ridge', lmbda= _lambda)
                    #Compute the beta for our train data
                    beta = sdg.train(X_train, z_train)
                    #Compute our MSE
                    mse_ols_, mse_ridge_ = sdg.compute_test_mse(X_test, z_test, lambda_=_lambda, beta=beta)
                    mse_ols_train, mse_ridge_train = sdg.compute_test_mse(X_train, z_train, lambda_=_lambda, beta=beta)
                    #Add it to our heatmap
                    MSE_ridge_val_train[i][j] = mse_ridge_train
                    MSE_ridge_val_test[i][j] = mse_ridge_
                    #testing the value of the mse against our best value
                    if mse_ridge_ <= best_mse_ridge:
                        best_lambda_rate_ridge = _lambda
                        best_learning_rate_ridge = eta
                        best_beta_ridge = beta
                        best_mse_ridge = mse_ridge_

        if method == 'ols':
            # initialization of or best parameters
            Eta_ols = np.logspace(-5, -1, 10)
            best_learning_rate_ols = Eta[0]
            best_beta_ols = np.zeros(X.shape)
            best_mse_ols = 1e10

            for i, eta in enumerate(Eta_ols):
                # Create a SGD
                sdg = SDG(learning_rate=eta, n_epochs=100, batch_size=10, method='ols', lmbda= 0)
                # Compute the beta for our train data
                beta = sdg.train(X_train, z_train)
                # Compute our MSE
                mse_ols_, mse_ridge_ = sdg.compute_test_mse(X_test, z_test, lambda_=0, beta=beta)
                mse_ols_train, mse_ridge_train = sdg.compute_test_mse(X_train, z_train, lambda_=0, beta=beta)
                # Add it to our list of MSE
                MSE_ols_val_train[i] = mse_ols_train
                MSE_ols_val_test[i] = mse_ols_
                # testing the value of the mse against our best value
                if mse_ols_ <= best_mse_ols:
                    best_learning_rate_ols = eta
                    best_beta_ols = beta
                    best_mse_ols = mse_ols_

    print("best_learning_rate_ols = ", best_learning_rate_ols)
    print("best_learning_rate_ridge = ", best_learning_rate_ridge)
    print("best_lambda_rate_ridge = ", best_lambda_rate_ridge)


    #=============================#
    #       Plot and Save         #
    #=============================#

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
    plt.savefig('../Figures/GD/SGD_train_heatmap_ridge.pdf')

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
    plt.savefig('../Figures/GD/SGD_test_heatmap_ridge.pdf')

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
    plt.savefig('../Figures/GD/SGD_mse_ols.pdf')



    best_pred_ols = X @ best_beta_ols
    best_pred_ridge = X @ best_beta_ridge

    title_ridge = 'prediction_ridge' + '_lamda_' + str(best_lambda_rate_ridge) + '_eta_' + str(best_learning_rate_ridge)
    title_ols = 'prediction_ols' +'_eta_' + str(best_learning_rate_ols)
    plotFunction(x_mesh, y_mesh, z, 'data')
    plotSave(x_mesh, y_mesh, best_pred_ols.reshape(len(x), len(x)),'../Figures/GD/',  title_ridge)
    plotSave(x_mesh, y_mesh, best_pred_ridge.reshape(len(x), len(x)),'../Figures/GD/',  title_ols)

    return best_learning_rate_ols,best_learning_rate_ridge, best_lambda_rate_ridge


#This function will plot the MSE againts the number for epochs

def SDG_ols_ridge_epoch(best_learning_rate_ols,  best_learning_rate_ridge, best_lambda_rate_ridge, methods = None):

    if methods == None:
        methods = ['ridge', 'ols']

    MSE_ridge_val = []
    MSE_ols_val = []
    MSE_logreg_val = []

    #Liste of value for the number of epochs that  we will test
    epochs = [50, 100, 150, 200, 250,  300, 350, 400]

    for method in methods:

        if method == 'ridge':

            for nb_epochs in epochs:
                #Create the SGD
                sdg = SDG(learning_rate=best_learning_rate_ridge, n_epochs= nb_epochs, batch_size=10, method='ridge', lmbda= best_lambda_rate_ridge)
                beta = sdg.train(X_train, z_train)
                #Compute the MSE
                mse_ols_, mse_ridge_ = sdg.compute_test_mse(X_test, z_test, lambda_=best_lambda_rate_ridge, beta=beta)
                MSE_ridge_val.append(mse_ridge_)


        if method == 'ols':

            for nb_epochs in epochs:
                # Create the SGD
                sdg = SDG(learning_rate=best_learning_rate_ols, n_epochs=nb_epochs, batch_size=10, method='ols', lmbda= 0)
                beta = sdg.train(X_train, z_train)
                # Compute the MSE
                mse_ols_, mse_ridge_ = sdg.compute_test_mse(X_test, z_test, lambda_=0, beta=beta)
                MSE_ols_val.append(mse_ols_)

        if method == 'logreg':
            print('here')
            for nb_epochs in epochs:
                # Create the SGD
                sdg = SDG(learning_rate=best_learning_rate_ols, n_epochs=nb_epochs, batch_size=10, method='logreg', lmbda= 0)
                beta = sdg.train(X_train, z_train)
                # Compute the MSE
                mse_logreg = sdg.logreg_loss(X_test, z_test, beta, 0)
                print('MSE')
                print('MSE',mse_logreg)
                MSE_logreg_val.append(mse_logreg)


    #======================#
    #     Plot the MSE     #
    #======================#
    print(MSE_logreg_val)
    print(len(epochs),len(MSE_logreg_val))

    plot, ax = plt.subplots()
    #plt.title('MSE for the OLS and Ridge')
    if 'ridge' in methods:
        plt.plot(epochs, MSE_ridge_val, 'k-o', label='Ridge')
    if 'ols' in methods:
        plt.plot(epochs, MSE_ols_val, 'r-o', label='OLS')
    if 'logreg' in methods:
        plt.plot(epochs, MSE_logreg_val, 'r-o', label='logreg')
    plt.xlabel('nb_epochs')
    plt.ylabel('MSE')
    #ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    plt.legend()
    plt.subplots_adjust(left=0.2, bottom=0.2, right=0.9)
    plt.savefig('../Figures/GD/MSE_for_the_OLS_and_Ridge_SDG_for_nb_epochs.pdf')

#This function will plot the MSE againts the batch_size

def SDG_ols_ridge_batch_size(best_learning_rate_ols, best_learning_rate_ridge, best_lambda_rate_ridge, methods = None):
    MSE_ridge_val = []
    MSE_ols_val = []
    MSE_logreg_val = []
    # Liste of value for the batch_size that  we will test
    batch_size = [5, 10, 20, 30, 40, 50, 100, 150, 200]
    if methods == None:
        methods = ['ridge', 'ols']

    for method in methods:

        if method == 'ridge':

            for batch in batch_size:
                # Create the SGD
                sdg = SDG(learning_rate=best_learning_rate_ridge, n_epochs= 100, batch_size=batch, method='ridge', lmbda= best_lambda_rate_ridge)
                beta = sdg.train(X_train, z_train)
                #Compute the MSE
                mse_ols_, mse_ridge_ = sdg.compute_test_mse(X_test, z_test, lambda_=best_lambda_rate_ridge, beta=beta)
                MSE_ridge_val.append(mse_ridge_)


        if method == 'ols':

            for batch in batch_size:
                # Create the SGD
                sdg = SDG(learning_rate=best_learning_rate_ols, n_epochs=100, batch_size=batch, method='ols', lmbda= 0)
                beta = sdg.train(X_train, z_train)
                # Compute the MSE
                mse_ols_, mse_ridge_ = sdg.compute_test_mse(X_test, z_test, lambda_=0, beta=beta)
                MSE_ols_val.append(mse_ols_)

        if method == 'logreg':

            for batch in batch_size:
                # Create the SGD
                sdg = SDG(learning_rate=best_learning_rate_ols, n_epochs=100, batch_size=batch, method='logreg', lmbda= 0)
                beta = sdg.train(X_train, z_train)
                # Compute the MSE
                MSE_logreg_val.append(sdg.logreg_loss(X_test, z_test, beta, 0))



    #======================#
    #     Plot the MSE     #
    #======================#

    plot, ax = plt.subplots()
    #plt.title('MSE for the OLS and Ridge')
    if 'ridge' in methods:
        plt.plot(batch_size, MSE_ridge_val, 'k-o', label='Ridge')
    if 'ols' in methods:
        plt.plot(batch_size, MSE_ols_val, 'r-o', label='OLS')
    if 'logreg' in methods:
        plt.plot(batch_size, MSE_logreg_val, 'r-o', label='logreg')
    plt.xlabel('batch_size')
    plt.ylabel('MSE')
    #ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    plt.legend()
    plt.subplots_adjust(left=0.2, bottom=0.2, right=0.9)
    plt.savefig('../Figures/GD/MSE_for_the_OLS_and_Ridge_SDG_for_batch_size.pdf')

#SDG_ols_ridge_matrix_mse()

learning_rate = 1e-5
_lambda = 1e-5

SDG_ols_ridge_epoch(learning_rate,  learning_rate, _lambda)
SDG_ols_ridge_batch_size(learning_rate,  learning_rate, _lambda)

#SDG_ols_ridge_epoch(learning_rate,  learning_rate, _lambda, methods=['logreg'])
#SDG_ols_ridge_batch_size(learning_rate,  learning_rate, _lambda, methods=['logreg'])



plt.show()