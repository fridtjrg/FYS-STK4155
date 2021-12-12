import autograd.numpy as np
from matplotlib import cm
from matplotlib import pyplot as plt
import autograd.numpy.random as npr
import math
import sys

from NeuralNetworks import solve_pde_deep_neural_network, u_trial, u_analytic, L

#================================================#



### Use the neural network:
npr.seed(15)

## Decide the vales of arguments to the function to solve
tf = 1
"""only for numerical solution
#delta values by stability requirement
Dx = float(sys.argv[1]) #Takes delta x as argument
Dt = Dx**2/2 #Nt(delta t) must be this or smaller for stability requirement.

#Transofrming to number of elements in
Nx = int(L/Dx)
Nt = int(math.ceil(tf/Dt)) #Round up so that Dt rather becomes smaller

Dt = tf/Nt  #The new Dt
"""
Nx= 50; Nt=50

x = np.linspace(0, L, Nx)
t = np.linspace(0,tf,Nt)

## Set up the parameters for the network
num_hidden_neurons = [50, 25, 25]
num_iter = 100
lmb = 2*1e-2

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
ax.set_ylabel('Position $x$')
ax.set_zlabel('$u_{NN}$')
plt.savefig('./figures/Solution_NN_%dhiddenLayer_lmbd=%g.pdf'%(len(num_hidden_neurons),lmb))


fig = plt.figure(figsize=(10,10))
ax = fig.gca(projection='3d')
ax.set_title('Analytical solution')
s = ax.plot_surface(T,X,U_analytical,linewidth=0,antialiased=False,cmap=cm.viridis)
ax.set_xlabel('Time $t$')
ax.set_ylabel('Position $x$');
ax.set_zlabel('$u_{analytical}$')
plt.savefig('./figures/analytical_solution_lmbd=%g.pdf'%lmb)

fig = plt.figure(figsize=(10,10))
ax = fig.gca(projection='3d')
ax.set_title('Difference')
s = ax.plot_surface(T,X,diff_ag,linewidth=0,antialiased=False,cmap=cm.viridis)
ax.set_xlabel('Time $t$')
ax.set_ylabel('Position $x$');
ax.set_zlabel('$abs(u_{NN}-u_{analytical})$')
plt.savefig('./figures/difference_lmbd=%g.pdf'%lmb)

fig = plt.figure(figsize=(10, 10))
fig.subplots_adjust(hspace=0.5)
gs = fig.add_gridspec(2, 1)

ax1 = fig.add_subplot(gs[0, :], )
ax1.set_title('Solution from the deep neural network w/ %d layer'%len(num_hidden_neurons))
ax1.set_xlabel('time')
ax1.set_ylabel('x')
h = ax1.imshow(u_dnn_ag,
               interpolation='gaussian', cmap='jet',
               origin='lower', aspect='auto', label='u')
plt.colorbar(h);

ax2 = fig.add_subplot(gs[1, :])
ax2.set_title('Analytical solution')
ax2.set_xlabel('time')
ax2.set_ylabel('x')
h = ax2.imshow(U_analytical, interpolation='gaussian', cmap='jet',
               origin='lower', aspect='auto', label='u')
plt.colorbar(h);
plt.savefig('./figures/heat_comparison_lmbd=%g.pdf'%lmb)

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
plt.savefig('./figures/Computed solutions at time=%g_lmb=%g.pdf'%(t1,lmb))

plt.figure(figsize=(10,10))
plt.title("Computed solutions at time = %g"%t2)
plt.plot(x, res2)
plt.plot(x,res_analytical2)
plt.legend(['dnn','analytical'])
plt.savefig('./figures/Computed solutions at time=%g_lmb=%g.pdf'%(t1,lmb))

plt.figure(figsize=(10,10))
plt.title("Computed solutions at time = %g"%t3)
plt.plot(x, res3)
plt.plot(x,res_analytical3)
plt.legend(['dnn','analytical'])
plt.savefig('./figures/Computed solutions at time=%g_lmb=%g.pdf'%(t1,lmb))



#plt.show()