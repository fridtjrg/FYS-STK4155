import numpy as np


"""
creating the designmatrix
def create_X(x, y, n):
	if len(x.shape) > 1:
		x = np.ravel(x)
		y = np.ravel(y)

	N = len(x)
	l = int((n+1)*(n+2)/2)		# Number of elements in beta, number of feutures (degree of polynomial)
	X = np.ones((N,l))

	for i in range(1,n+1):
		q = int((i)*(i+1)/2)
		for k in range(i+1):
			X[:,q+k] = (x**(i-k))*(y**k)

	return X
"""
np.random.seed(123)

def sigmoid_prime(x):
	return 1/(1+e**(-x))

def sigmoid(x):
	return 1/(1+np.exp(-x))

#Sets up dataset
def FrankeFunction(x,y):
	term1 = 0.75*np.exp(-(0.25*(9*x-2)**2) - 0.25*((9*y-2)**2))
	term2 = 0.75*np.exp(-((9*x+1)**2)/49.0 - 0.1*(9*y+1))
	term3 = 0.5*np.exp(-(9*x-7)**2/4.0 - 0.25*((9*y-3)**2))
	term4 = -0.2*np.exp(-(9*x-4)**2 - (9*y-7)**2)
	return term1 + term2 + term3 + term4

n = 100
x = np.linspace(0,1,n)
y = np.linspace(0,1,n)

mu_N = 0; sigma_N = 0.1
x,y = np.meshgrid(x,y)
X = FrankeFunction(x,y) +mu_N +sigma_N*np.random.randn(n,n) #input data
y = np.ravel(FrankeFunction(x,y)) #target data
#NB X is not a designmatrix, should it be? ravel x?

class Layer_Dense:
	def __init__(self, n_inputs, n_neurons):
		self.weights = 0.1*np.random.randn(n_inputs, n_neurons) #creates weight matrix(random)
		self.biases = np.zeros((1, n_neurons)) #Biases(just shape)
	def forward(self, inputs):
		self.output = np.dot(inputs, self.weights) + self.biases

class Actiavtion_ReLU:
	def forward(self, inputs):
		self.output = np.maximum(0,inputs)
class Actiavtion_sigmoid:
	def forward(self, inputs):
		self.output = sigmoid(inputs)
class Actiavtion_softmax:
	def forward(self, inputs):
		exp_values = np.exp(inputs-np.max(inputs, axis=1, keepdims=True))
		probabilities = exp_values /np.sum(exp_values,axis=1,keepdims=True)
		self.output = probabilities
"""
class Loss:
	def calculate(self, output, y):
		sample_losses = self.forward(output,y)
		data_loss = np.mean(sample_losses)
		return data_loss

class Loss_function(Loss):
	def forward(self, y_pred):
		exp_term = np.exp(ypred)
		cost = (1/samples)*np.sum((y_pred*y_target)**2)
		confidence = -np.log(cost)
		return neg_log_conf
"""
class train():
	def backpropagation(input_x, output, weights, prob, a_h):
	    
	    # error in the output layer
	    error_output = probabilities - output
	    # error in the hidden layer
	    error_hidden = np.matmul(error_output, weights.T) * a_h * (1 - a_h)
	    
	    # gradients for the output layer
	    output_weights_gradient = np.matmul(a_h.T, error_output)
	    output_bias_gradient = np.sum(error_output, axis=0)
	    
	    # gradient for the hidden layer
	    hidden_weights_gradient = np.matmul(input_x.T, error_hidden)
	    hidden_bias_gradient = np.sum(error_hidden, axis=0)

	    return output_weights_gradient, output_bias_gradient, hidden_weights_gradient, hidden_bias_gradient



#NN with 3 layers and 100 outputs
Layer1 = Layer_Dense(100,100) #(input, neurons(outputs))
Layer2 = Layer_Dense(100,100)
Layer3 = Layer_Dense(100,1)

activation1 = Actiavtion_ReLU() #defined activatin function
activation2 = Actiavtion_ReLU() 
activation3 = Actiavtion_softmax() 

Layer1.forward(X)
activation1.forward(Layer1.output)

Layer2.forward(activation1.output)
activation2.forward(Layer2.output)

Layer3.forward(activation2.output)
activation3.forward(Layer3.output)


#my_loss_function = Loss_function()
#loss = Loss_function.calculate()
print(activation3.output)