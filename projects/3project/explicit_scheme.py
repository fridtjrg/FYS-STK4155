"""
Explicit Scheme for solving a diffusion equation for a rod of length L on the form
     u_xx(x,t) = u_t(x,t)
With initial conditions 
    u(x,0) = sin(pi*x)
    u(0,t) = u(L,t) = 0
"""

import numpy as np
import matplotlib.pyplot as plt
import math
from NeuralNetworks import u_analytic

def explicit_scheme_solver(I, dx, L, t_max, outfile = None):
    """Explicit scheme solver for diffusion euqation on a rod with boundaries u(0,t) = u(L,t) = 0

    Args:
        I (function): Initial condition along x
        dx (float): dx
        L (numeric): Length of rod
        t_max (numeric): final time for the run
        outfile (str, optional): Outfile for results to be written to. Defaults to None.
    
    Returns:
        u_xt (np.array((Nt,Nx))): Array with the solution u for x and t
        x (np.array): x values
        t (np.array): t values

    """

    #Setting up const and arr
    dt = dx*dx*0.5
    Nx = int((L+dx)/dx)
    Nt = int(math.ceil((t_max+dt)/dt) )   #Using ceil because this will cause a smaller dx value
    x = np.linspace(0,L,Nx)
    t = np.linspace(0,t_max,Nt)
    u = np.zeros(Nx)
    u_new = np.zeros(Nx)
    a = dt/(dx*dx)

    #Initial condition
    u[:] = I(x[:])

    #Outfile
    if outfile:
        #Not implemented
        pass
    else:
        u_xt = np.zeros((Nt,Nx))
        u_xt[0,:] = u[:]

    #Algorithm
    for i in range(1,Nt):
        for j in range(1,Nx-1):
            u_new[j] = a * u[j-1] + (1 - 2*a) * u[j] + a * u[j+1]

        #Outfile not implemeted
        """    
        if outfile is not None:
            pass
        else:
            u_xt[i,:]=u_new[:]
        """
        u_xt[i,:]=u_new[:]
        u[:] = u_new[:]

    return u_xt, x, t


def I(x):
    return np.sin(np.pi*x)


def plot_u_xt(u_xt,x,t,plotname):
    """Plots the surface of u(x,t)

    Args:
        u_xt (2D arr): The values of u(x,t)
        x (arr): x
        t (arr): t
    """

    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    x, t = np.meshgrid(x, t)
    ax.plot_surface(t,x,u_xt)
    ax.set_xlabel('t')
    ax.set_ylabel('x')
    ax.set_zlabel('u')
    plt.savefig('./figures/temperature_of_rod_'+plotname+'.pdf')

def plot_diff(u_xt,u_a,x,t):
    """Plots the surface of u(x,t)

    Args:
        u_xt (2D arr): The values of u(x,t)
        x (arr): x
        t (arr): t
    """

    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    x, t = np.meshgrid(x, t)
    ax.plot_surface(t,x,abs(u_xt-u_a))
    ax.set_xlabel('t')
    ax.set_ylabel('x')
    ax.set_zlabel('u_diff')
    plt.savefig('./figures/temperature_of_rod_difference.pdf')
    

def Analytical_u(I, dx, L, t_max):
    dt = dx*dx*0.5
    Nx = int((L+dx)/dx)
    Nt = int(math.ceil((t_max+dt)/dt) )   #Using ceil because this will cause a smaller dx value
    x = np.linspace(0,L,Nx)
    t = np.linspace(0,t_max,Nt)
    u = np.zeros((len(t),len(x)))
    for i in range(Nt):
        for j in range(Nx):
            u[i,j] = u_analytic((x[j],t[i]))
    return u



def main():
    u,x,t = explicit_scheme_solver(I,0.01,1,1)
    plot_u_xt(u,x,t,'explicit_scheme')

    u_analytical = Analytical_u(I,0.01,1,1)
    plot_u_xt(u_analytical,x,t,'Analytical')

    plot_diff(u,u_analytical,x,t)

if __name__ == '__main__':
    main()