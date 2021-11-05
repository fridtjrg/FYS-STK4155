import numpy as np

class Layer:

    def __init__(self, n_input, n_neurons, activation=None):

        self.activation = activation
        self.last_activation = None
        self.error = None
        self.delta = None
        self.weights = np.random.rand(n_input, n_neurons)
        self.bias = np.random.rand(n_neurons)

        np.random.seed(1921)

    def activate(self, x):

        x = np.dot(x, self.weights) + self.bias
        self.last_activation = self._activation(x)
        return self.last_activation

    def _activation(self, x):

        if self.activation is None:
            return x

        if self.activation == 'sigmoid':
            return self._sigmoid(x)

        return x

    def _activationPrime(self, x):

        if self.activation is None:
            return x


        if self.activation == 'sigmoid':
            return x * (1 - x)

        return x

    def _sigmoid(self, x):
        return 1.0 / (1.0 + np.exp(-x))
