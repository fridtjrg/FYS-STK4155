import numpy as np

np.random.seed(123)


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


class Layer_Dense:
	def __init__(self, n_inputs, n_neurons):
		self.weights = 0.1*np.random.randn(n_inputs, n_neurons) #creates weight matrix(random)
		self.biases = np.zeros((1, n_neurons)) #Biases(just shape)
	def forward(self, inputs):
		self.output = np.dot(inputs, self.weights) + self.biases

class Actiavtion_ReLU:
	def forward(self, inputs):
		self.output = np.maximum(0,inputs)
class Loss:
	def calculate(selv, output, y):
		sample_losses = self.forward(output,y)
		data_loss = np.mean(sample_losses)
		return data_loss
"""
class Loss_function(Loss):
	def forward(self, y_pred, y_target): #Choose a cost function
		#put something here
"""



Layer1 = Layer_Dense(100,1) #(input, neurons(outputs))
activation1 = Actiavtion_ReLU()
Layer1.forward(X)
activation1.forward(Layer1.output)


#my_loss_function = Loss_function()
#loss = Loss_function.calculate()
print(activation1.output)