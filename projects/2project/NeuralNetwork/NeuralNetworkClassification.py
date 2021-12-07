import numpy as np

class DenseLayer:

    #important paramaters of a layer
    def __init__(self, n_hidden_neurons, n_categories,type_activation,lmbda, eta):

        self.weights = np.random.randn(n_hidden_neurons, n_categories)
        self.bias = np.zeros(n_categories) + 0.01
        self.n_hidden_neurons = n_hidden_neurons
        self.n_categories = n_categories
        self.type_activation = type_activation
        self.lmbda = lmbda
        self.eta = eta

    #========================= Activation function

    def sigmoid(self, x):
        return 1/(1+np.exp(-x))

    def tanh(self, x):
        return np.tanh(x)

    def relu(self, x):
        row, col = x.shape
        for i in range(row):
            for j in range(col):
                x[i][j] = max(0, x[i][j])
        return x

    #======================= Differential activation function
    def reluPrime(self, x):
        row, col = x.shape
        for i in range(row):
            for j in range(col):
                if x[i][j] > 0:
                    x[i][j] = 1
                else:
                    x[i][j] = 0
        return x

    def tanhPrime(self, x):

        return 1 - x ** 2

    def sigmoidPrime(self, x):
        return x * ( 1 - x)

    def activation(self, x):
        if self.type_activation is None:
            return x

        elif (self.type_activation == 'sigmoid'):
            return self.sigmoid(x)

        elif (self.type_activation == 'tanh'):
            return self.tanh(x)

        elif (self.type_activation == 'relu'):
            return self.relu(x)

    def activationPrime(self, x):

        if self.type_activation is None:
            return x

        elif (self.type_activation == 'sigmoid'):
            return self.sigmoidPrime(x)

        elif (self.type_activation == 'tanh'):
            return self.tanhPrime(x)

        elif (self.type_activation == 'relu'):
            return self.reluPrime(x)

    #================== Feed forward

    def feedForward(self, x):
        #return the value of the activation function of a vector mutply by the weight of the layer plus the bias of the layer
        self.last_activation = self.activation(np.dot(x, self.weights) + self.bias)
        return self.last_activation

    def backPropagation(self,err,layer_prev,  X = None):

        error = np.dot(err , layer_prev.weights.T) * self.activationPrime(self.last_activation)
        if X is None :
            self.weights_grad = np.dot(self.last_activation.T, error)
        else:
            self.weights_grad = np.dot(X.T, error)

        self.bias_grad = np.sum(error, axis = 0)

        if self.lmbda > 0.0:

            self.weights_grad += self.lmbda * self.weights

        self.weights -= self.eta * self.weights_grad
        self.bias -= self.eta * self.bias_grad

        return error










class NeuralNetwork:

    # important paramaters of a NN
    def __init__(self, X, y, n_epochs, batch_size):
        self.epochs = n_epochs
        self.X = X
        self.y = y
        self.iterations = X.shape[0] // batch_size
        self._layers = []
        self.batch_size = batch_size

    def add_layer(self, layer):
        self._layers.append(layer)

    #================== Feed forward

    def feed_forward(self):
        #we pass our data to all the activation functions of our layers
        X = self.X_batch
        for layer in self._layers:
            X = layer.feedForward(X)
        exp_term = np.exp(X)
        #we return the probability of the value (bc of classification problem)
        self.probabilities = exp_term / np.sum(exp_term, axis=1, keepdims=True)


    def backpropagation(self):
        err = self.probabilities - self.y_batch

        output_layer = self._layers[-1]
        output_layer.weights_grad = np.dot(self._layers[len(self._layers)-2].last_activation.T, err)

        output_layer.bias_grad = np.sum(err, axis = 0)

        if output_layer.lmbda > 0.0:

            output_layer.weights_grad += output_layer.lmbda * output_layer.weights

        output_layer.weights -= output_layer.eta * output_layer.weights_grad
        output_layer.bias -= output_layer.eta * output_layer.bias_grad

        for i in reversed(range(1, len(self._layers)-1)):
            layer = self._layers[i]
            layer_prev = self._layers[i+1]
            err = layer.backPropagation(err, layer_prev)
        layer = self._layers[0]
        err = layer.backPropagation(err, self._layers[1],self.X_batch)


    #train the NN
    def train(self):
        data_indices = np.arange(self.X.shape[0])
        #for each epochs and mini-batch
        for i in range(self.epochs):
            for j in range(self.iterations):
                # pick datapoints with replacement
                chosen_datapoints = np.random.choice(
                    data_indices, size=self.batch_size, replace=False
                )
                # minibatch training data
                self.X_batch = self.X[chosen_datapoints]
                self.y_batch = self.y[chosen_datapoints]
                #feed forward
                self.feed_forward()
                #back propagation
                self.backpropagation()

    #return our prediction for a set of data
    def predict(self, X):
        for layer in self._layers:
            X = layer.feedForward(X)
        exp_term = np.exp(X)
        probabilities = exp_term / np.sum(exp_term, axis=1, keepdims=True)
        return np.argmax(probabilities, axis=1)
