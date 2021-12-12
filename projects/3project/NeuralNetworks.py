import autograd.numpy as np
from autograd import jacobian,hessian,grad
import autograd.numpy.random as npr


#=============== lenght of the bar

L = 1

## Set up the network


#================== Different Activation Functions

def sigmoid(z):
    return 1/(1 + np.exp(-z))


def tanh(z):
    return np.tanh(z)


#================== Deep Neural Network


## Define the trial solution
def v(x):
    return np.sin(np.pi*x)



## The right side of the ODE:
def f(point):
    return 0.



# The cost function:
def cost_function(P, x, t):
    """
    Args :
        P (list) : list of parameters for each layer
        x (list) : value list for distance x
        t (list) : value list for time t
    Return :
        the value of the cost function
    """
    #initialize the cost function
    cost_sum = 0
    #initializa the jacobian and hessian function
    u_t_jacobian_func = jacobian(u_trial)
    u_t_hessian_func = hessian(u_trial)
    #for each point:
    for x_ in x:
        for t_ in t:
            point = np.array([x_,t_])
            u_t = u_trial(point,P)
            u_t_jacobian = u_t_jacobian_func(point,P)
            u_t_hessian = u_t_hessian_func(point,P)
            u_t_dt = u_t_jacobian[1]
            u_t_d2x = u_t_hessian[0][0]
            func = f(point)
            #This is from the definition of the problem :   u_t_dt = u_t_d2x
            err_sqr = ((u_t_dt - u_t_d2x) - func)**2
            cost_sum += err_sqr
    return cost_sum /( np.size(x)*np.size(t))







def solve_pde_deep_neural_network(x,t, num_neurons, num_iter, lmb):
    """
    Set up a function for training the network to solve for the equation
    Args :
        x (list) : value list for distance x
        t (list) : value list for time t
        num_neurons (list) : list of number of neurons for each layer
        num_iter (int) : number of updates we do for the list of parameters for each layer
        lmb (float): learning rate of the NN
    Returns:
        P (list) : list of parameters for each layer
    """
    ## Set up initial weigths and biases
    N_hidden = np.size(num_neurons)
    ## Set up initial weigths and biases
    # Initialize the list of parameters:
    P = [None]*(N_hidden + 1) # + 1 to include the output layer
    P[0] = npr.randn(num_neurons[0], 2 + 1 ) # 2 since we have two points, +1 to include bias
    for l in range(1,N_hidden):
        P[l] = npr.randn(num_neurons[l], num_neurons[l-1] + 1) # +1 to include bias
    # For the output layer
    P[-1] = npr.randn(1, num_neurons[-1] + 1 ) # +1 since bias is included
    print('Initial cost: ',cost_function(P, x, t))
    cost_function_grad = grad(cost_function,0)
    # Let the update be done num_iter times
    for i in range(num_iter):
        old_cost = cost_function(P, x, t)
        cost_grad =  cost_function_grad(P, x , t)
        for l in range(N_hidden+1):
            P[l] = P[l] - lmb * cost_grad[l]
        new_cost = cost_function(P, x, t)
        print('cost: ', i,"  ",new_cost)
        if old_cost<new_cost: #Adaptive lambda value
            lmb = lmb*0.7
            print('lmb changes to: ',lmb)
    print('Final cost: ',cost_function(P, x, t))
    return P






def deep_neural_network(deep_params, x):
    """
    Args :
        deep_params (list) : list of parameters for each layer
        x (arr) : The values of the point x1, x2, ..., Xn
    Returns :
        the output of the input above the NN
    """
    # x is now a point and a 1D numpy array; make it a column vector
    num_coordinates = np.size(x,0)
    x = x.reshape(num_coordinates,-1)
    num_points = np.size(x,1)
    # N_hidden is the number of hidden layers
    N_hidden = np.size(deep_params) - 1 # -1 since params consist of parameters to all the hidden layers AND the output layer
    # Assume that the input layer does nothing to the input x
    x_input = x
    x_prev = x_input
    ## Hidden layers:
    for l in range(N_hidden):
        # From the list of parameters P; find the correct weigths and bias for this layer
        w_hidden = deep_params[l]
        # Add a row of ones to include bias
        x_prev = np.concatenate((np.ones((1,num_points)), x_prev ), axis = 0)
        z_hidden = np.matmul(w_hidden, x_prev)
        x_hidden = sigmoid(z_hidden)
        # Update x_prev such that next layer can use the output from this layer
        x_prev = x_hidden

    ## Output layer:
    # Get the weights and bias for this layer
    w_output = deep_params[-1]
    # Include bias:
    x_prev = np.concatenate((np.ones((1,num_points)), x_prev), axis = 0)
    z_output = np.matmul(w_output, x_prev)
    x_output = z_output
    return x_output[0][0]



def u_trial(point,P):
    """
    Args :
        P (list) : list of parameters for each layer
        point (arr) : The values of the point x1, x2, ..., Xn
    Return:
        the value of the function u for input point = x1, x2... (here point = x,t)
    """
    x,t = point
    return (1-t)*v(x) + x*(L-x)*t*deep_neural_network(P,point)





def u_analytic(point):
    """
        For comparison, define the analytical solution
    """
    x,t = point
    return np.exp(-np.pi**2*t)*v(x)
