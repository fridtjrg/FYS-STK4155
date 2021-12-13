import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

#=============================================================
#                  Functions
#=============================================================

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
dt = 0.01
Nx = int((1+dx)/dx)
Nt = int((1+dt)/dt)

x = np.linspace(0, L, Nx)    #Space
y = np.linspace(0, t_max, Nt)    #Time   
x_mesh, y_mesh = np.meshgrid(x,y)
z = analytical_solution(x_mesh, y_mesh)


x_y = np.empty((len(x)*len(y), 2))
x_y[:, 0] = x_mesh.ravel()
x_y[:, 1] = y_mesh.ravel()
x = x_y[:, 0]
y = x_y[:, 1]

scaler = StandardScaler()
scaler.fit(x_y)
x_y = scaler.transform(x_y)
x_y_train, x_y_test, z_train, z_test = train_test_split(x_y, z.ravel(), test_size=0.2)

X_train = create_X(x_y_train[:, 0], x_y_train[:, 1], poly )
X_test = create_X(x_y_test[:, 0], x_y_test[:, 1], poly )
X = create_X(x_y[:, 0], x_y[:, 1], poly)