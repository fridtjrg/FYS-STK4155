import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from matplotlib.ticker import LinearLocator, FormatStrFormatter
from matplotlib import cm
import os

#=============================================================
#            Defining useful functions
#=============================================================

def plotFunction(x, y, z, title):
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    surf = ax.plot_surface(x, y, z, cmap=cm.coolwarm,linewidth=0, antialiased=False)
    ax.zaxis.set_major_locator(LinearLocator(10))
    ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))
    fig.suptitle(title)
    fig.colorbar(surf, shrink=0.5, aspect=5)

def create_X(x, y, n ):
    if len(x.shape) > 1:
        x = np.ravel(x)
        y = np.ravel(y)

    N = len(x)
    l = int((n+1)*(n+2)/2) # Number of elements in beta
    X = np.ones((N,l))
    for i in range(1,n+1):
        q = int((i)*(i+1)/2)
        for k in range(i+1):
            X[:,q+k] = (x**(i-k))*(y**k)
    return X


def I(x):
    return np.sin(np.pi*x)


def analytical_solution(x,t):
    return np.exp(-np.pi**2*t)*I(x)

#=============================================================
#                   Setting up data
#=============================================================

poly = 7
L = 1
t_max = 1
dx = 0.01
dt = dx*dx
Nx = int((1+dx)/dx)
Nt = int((1+dt)/dt)

x = np.linspace(0, L, Nx)    #Space
y = np.linspace(0, t_max, Nt)    #Time   
x_mesh, y_mesh = np.meshgrid(x,y)
z = analytical_solution(x_mesh, y_mesh)


x_y = np.empty((len(x)*len(y), 2))
x_y[:, 0] = x_mesh.ravel()
x_y[:, 1] = y_mesh.ravel()

scaler = StandardScaler()
scaler.fit(x_y)
x_y = scaler.transform(x_y)
x_y_train, x_y_test, z_train, z_test = train_test_split(x_y, z.ravel(), test_size=0.2)

X_train = create_X(x_y_train[:, 0], x_y_train[:, 1], poly )
X_test = create_X(x_y_test[:, 0], x_y_test[:, 1], poly )
X = create_X(x_y[:, 0], x_y[:, 1], poly)

#=============================================================
#                   Ridge
#=============================================================

Lambdas = np.logspace(-3, 1, 10)

#############################################
best_test_MSE = 1 #Must trigger the test for a lower mse
best_lambda = Lambdas[0]

for lmbda in Lambdas:
    beta_linreg = np.linalg.pinv(X_train.T.dot(X_train) + lmbda * np.eye(X_train.T.dot(X_train).shape[0])).dot(X_train.T).dot(z_train)
    print("beta from linalg")
    print(beta_linreg)
    ztildeLinreg = X_train @ beta_linreg
    ztestLinreg = X_test @ beta_linreg
    MSE_train_linreg = np.mean((z_train - ztildeLinreg)**2, keepdims=True )
    MSE_test_linreg = np.mean((z_test - ztestLinreg)**2, keepdims=True )
    print("MSE_train")
    print(MSE_train_linreg)
    print("MSE_test")
    print(MSE_test_linreg)
    print("\n")
    print("-----------------------------")
    print("\n")
    #title = "plot of regression with linalg with lambda = " + str(lmbda)
    #plotFunction(x_mesh, y_mesh, (X @ beta_linreg).reshape(len(x), len(x)), title)

    if MSE_test_linreg < best_test_MSE:
        best_test_MSE = MSE_test_linreg
        best_lambda = lmbda



#Re-calculates with the best lambda value
print("=======================================")
print("Found best lambda to be:", best_lambda)
print("=======================================")
beta_linreg = np.linalg.pinv(X_train.T.dot(X_train) + best_lambda * np.eye(X_train.T.dot(X_train).shape[0])).dot(X_train.T).dot(z_train)
print("beta from linalg")
print(beta_linreg)

ztildeLinreg = X_train @ beta_linreg
ztestLinreg = X_test @ beta_linreg
MSE_train_linreg = np.mean((z_train - ztildeLinreg)**2, keepdims=True )
MSE_test_linreg = np.mean((z_test - ztestLinreg)**2, keepdims=True )

print("MSE_train")
print(MSE_train_linreg)
print("MSE_test")
print(MSE_test_linreg)
print("\n")
print("-----------------------------")
print("\n")
title = "lambda = " + str(best_lambda)
z = X @ beta_linreg

z = np.reshape(z,(Nt,Nx))

plotFunction(x_mesh, y_mesh,z, title)



#Exact solution
u_exact = np.empty((len(y),len(x)))
for i in range(len(x)):
    u_exact[:,i] = analytical_solution(x[i],y[:])

#Finding the difference
diff = u_exact[:,:] - z[:,:]
diff = np.abs(diff)

print(z.shape,u_exact)

fig = plt.figure(figsize=(5, 5))
fig.subplots_adjust(hspace=0.5)
gs = fig.add_gridspec(2, 1)
ax = fig.add_subplot(gs[0, :], )
ax.set_xlabel('t')
ax.set_ylabel('x')
#ax.axis([x.min(), x.max(), y.min(), y.max()])
h = ax.imshow(diff.T, extent=[x.min(), x.max(), y.min(), y.max()],
               interpolation='gaussian', cmap='jet',
               origin='lower', aspect='auto', label='u')
plt.colorbar(h)
plt.title(f'Difference between ridge and analytical solution for $\Delta x={dx}$ and $\Delta t={dt:.2f}$')
path = os.sys.path[0]
plt.savefig(path+f'/figures/u_diff_linreg_dx={dx}.pdf')



#plt.show()