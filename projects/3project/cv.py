from math import degrees
import numpy as np
import matplotlib.pyplot as plt
from src.dataregression import x,y, z,dx,dt

#Run from 3project folder
import sys
sys.path.append('../1project')
import regan

#regan.plot_ols_complexity(x,y,z,20)
x_mesh,y_mesh = np.meshgrid(x,y)
degree = 20
complexity = np.linspace(1,degree, degree,dtype=int)
lmd = 0.001

#Arrays for plotting:
#No CV
OLS_train = []
OLS_test = []
Ridge_train = []
Ridge_test = []
Lasso_train = []
Lasso_test = []

#CV
OLS_Test_k10 = []
OLS_Train_k10 = []
Ridge_Train_k10 = []
Ridge_Test_k10 = []
Lasso_Train_k10 = []
Lasso_Test_k10 = []

k = 10  #Number of folds
for degree in complexity:
    print('\n==================================================')
    print(f'Complexity {degree}')
    print('\nCalculating OLS, Ridge, Lasso...')
    X = regan.create_X(x,y,degree)
    x_train, x_test, z_train,z_test = regan.Split_and_Scale(X,np.ravel(z))

    ols_beta, z_tilde, z_predict = regan.OLS_solver(x_train, x_test, z_train,z_test)
    OLS_train.append(regan.MSE(z_train,z_tilde))
    OLS_test.append(regan.MSE(z_test,z_predict))

    ridge_beta,z_tilde, z_predict = regan.ridge_reg(x_train, x_test, z_train,z_test,lmd=lmd)
    Ridge_train.append(regan.MSE(z_train,z_tilde))
    Ridge_test.append(regan.MSE(z_test,z_predict))

    z_tilde, z_predict = regan.lasso_reg(x_train, x_test, z_train,z_test,lmd=lmd)
    Lasso_train.append(regan.MSE(z_train,z_tilde))
    Lasso_test.append(regan.MSE(z_test,z_predict))

    print('Cross validation k=10 for OLS, Ridge, Lasso...')
    X = regan.create_X(x,y,degree)
    ols_train, ols_test = regan.cross_validation(k,X,z.ravel(),solver='OLS')
    OLS_Train_k10.append(ols_train)
    OLS_Test_k10.append(ols_test)

    Ridge_MSE_train, Ridge_MSE_test = regan.cross_validation(k,X,z.ravel(),solver='RIDGE',lmd = lmd)
    Ridge_Train_k10.append(Ridge_MSE_train)
    Ridge_Test_k10.append(Ridge_MSE_test)

    lasso_train,lasso_test = regan.cross_validation(k,X,z.ravel(),solver='LASSO',lmd = lmd)
    Lasso_Train_k10.append(lasso_train)
    Lasso_Test_k10.append(lasso_test)


fig, ax = plt.subplots()
ax.plot(complexity,OLS_test,label='OLS')
ax.plot(complexity,Ridge_test,label='Ridge')
ax.plot(complexity,Lasso_test,label='Lasso')
ax.plot(complexity,OLS_Test_k10,label='OLS CV k=10',linestyle='--')
ax.plot(complexity,Ridge_Test_k10,label='Ridge CV k=10',linestyle='--')
ax.plot(complexity,Lasso_Test_k10,label='Lasso CV k=10',linestyle='--')
ax.set_xlabel('complexity')
ax.set_ylabel('MSE')
plt.legend()
plt.grid()
plt.title(f'Cross validation with $\Delta x={dx}$, $\Delta t={dt:.2f}$ and $\lambda = {lmd}$')
plt.savefig(f'./figures/cv_degree={degree}_Nx*Nt={len(x)}*{len(y)}_dx={dx}.pdf')