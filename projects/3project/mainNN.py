import autograd.numpy as np
from matplotlib import cm
from matplotlib import pyplot as plt
import autograd.numpy.random as npr

from NeuralNetworks import solve_pde_deep_neural_network, u_trial, u_analytic, L

#================================================#


### Use the neural network:
npr.seed(15)

## Decide the vales of arguments to the function to solve

tf = 1

Nx = 10; Nt = 10
x = np.linspace(0, L, Nx)
t = np.linspace(0,tf,Nt)

## Set up the parameters for the network
num_hidden_neurons = [50, 50, 50]
num_iter = 100
lmb = 0.001

P = solve_pde_deep_neural_network(x,t, num_hidden_neurons, num_iter, lmb)

## Store the results
u_dnn_ag = np.zeros((Nx, Nt))
U_analytical = np.zeros((Nx, Nt))

for i,x_ in enumerate(x):
    for j, t_ in enumerate(t):

        point = np.array([x_, t_])
        u_dnn_ag[i,j] = u_trial(point, P)
        U_analytical[i,j] = u_analytic(point)

# Find the map difference between the analytical and the computed solution
diff_ag = np.abs(u_dnn_ag - U_analytical)
print('Max absolute difference between the analytical solution and the network: %g'%np.max(diff_ag))

## Plot the solutions in two dimensions, that being in position and time

T,X = np.meshgrid(t,x)

fig = plt.figure(figsize=(10,10))
ax = fig.gca(projection='3d')
ax.set_title('Solution from the deep neural network w/ %d layer'%len(num_hidden_neurons))
s = ax.plot_surface(T,X,u_dnn_ag,linewidth=0,antialiased=False,cmap=cm.viridis)
ax.set_xlabel('Time $t$')
ax.set_ylabel('Position $x$');


fig = plt.figure(figsize=(10,10))
ax = fig.gca(projection='3d')
ax.set_title('Analytical solution')
s = ax.plot_surface(T,X,U_analytical,linewidth=0,antialiased=False,cmap=cm.viridis)
ax.set_xlabel('Time $t$')
ax.set_ylabel('Position $x$');

fig = plt.figure(figsize=(10,10))
ax = fig.gca(projection='3d')
ax.set_title('Difference')
s = ax.plot_surface(T,X,diff_ag,linewidth=0,antialiased=False,cmap=cm.viridis)
ax.set_xlabel('Time $t$')
ax.set_ylabel('Position $x$');

fig = plt.figure(figsize=(10, 10))
fig.subplots_adjust(hspace=0.5)
gs = fig.add_gridspec(2, 1)

ax1 = fig.add_subplot(gs[0, :], )
ax1.set_title('Solution from the deep neural network w/ %d layer'%len(num_hidden_neurons))
ax1.set_xlabel('time')
ax1.set_ylabel('x')
h = ax1.imshow(u_dnn_ag,
               interpolation='gaussian', cmap='jet',
               origin='lower', aspect='auto')
plt.colorbar(h);

ax2 = fig.add_subplot(gs[1, :])
ax2.set_title('Analytical solution')
ax2.set_xlabel('time')
ax2.set_ylabel('x')
h = ax2.imshow(U_analytical, interpolation='gaussian', cmap='jet',
               origin='lower', aspect='auto')
plt.colorbar(h);

## Take some slices of the 3D plots just to see the solutions at particular times
indx1 = 0
indx2 = int(Nt/2)
indx3 = Nt-1

t1 = t[indx1]
t2 = t[indx2]
t3 = t[indx3]

# Slice the results from the DNN
res1 = u_dnn_ag[:,indx1]
res2 = u_dnn_ag[:,indx2]
res3 = u_dnn_ag[:,indx3]

# Slice the analytical results
res_analytical1 = U_analytical[:,indx1]
res_analytical2 = U_analytical[:,indx2]
res_analytical3 = U_analytical[:,indx3]

# Plot the slices
plt.figure(figsize=(10,10))
plt.title("Computed solutions at time = %g"%t1)
plt.plot(x, res1)
plt.plot(x,res_analytical1)
plt.legend(['dnn','analytical'])

plt.figure(figsize=(10,10))
plt.title("Computed solutions at time = %g"%t2)
plt.plot(x, res2)
plt.plot(x,res_analytical2)
plt.legend(['dnn','analytical'])

plt.figure(figsize=(10,10))
plt.title("Computed solutions at time = %g"%t3)
plt.plot(x, res3)
plt.plot(x,res_analytical3)
plt.legend(['dnn','analytical'])



plt.show()