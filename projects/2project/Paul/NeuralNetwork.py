import numpy as np


class NeuralNetwork:

    def __init__(self):
        self._layers = []

    def add_layer(self, layer):
        self._layers.append(layer)

    def get_layer(self, index):
        return self._layers[index]

    def feed_forward(self, X):

        for layer in self._layers:
            X = layer.activate(X)

        return X

    def predict(self, X, net_type='regression', n_neurons=3):
        ff = self.feed_forward(X)
        if net_type == 'classification':
            if ff.ndim == 1:
                pred = np.argmax(ff)
            else:
                pred = np.argmax(ff, axis=1)
        if net_type == 'regression':
            pred = ff

        return pred

    def backpropagation(self, X, y, learning_rate, lmbd, output):
        for i in reversed(range(len(self._layers))):
            layer = self._layers[i]
            if layer == self._layers[-1]:
                layer.error = y - output
                layer.delta = layer.error * layer._activationPrime(output)
            else:
                next_layer = self._layers[i + 1]
                layer.error = np.dot(next_layer.weights, next_layer.delta)
                layer.delta = layer.error * layer._activationPrime(layer.last_activation)


        for i in range(len(self._layers)):
            layer = self._layers[i]
            input_to_use = np.atleast_2d(X if i == 0 else self._layers[i - 1].last_activation)
            layer.weights = layer.weights + layer.delta * input_to_use.T * learning_rate
            if lmbd > 0:
                layer.weights = layer.weights * (
                            1 - lmbd * learning_rate) + layer.delta * input_to_use.T * learning_rate
            layer.bias = layer.bias + layer.delta * learning_rate


    def train(self, X, y, learning_rate, nb_epochs = 100, batch_size = 10, lmbd=0, _type = 'regression'):
        iterations = X.shape[0] // batch_size
        mses = []
        for i in range(1, nb_epochs+1):
            np.random.shuffle(X)
            np.random.shuffle(y)
            for k in range(iterations):
                random_index = np.random.randint(iterations)
                Xi = X[random_index * batch_size:(random_index + 1) * batch_size]
                Yi = y[random_index * batch_size:(random_index + 1) * batch_size]
                for j in range(len(Xi)):
                    output = self.feed_forward(Xi[j])
                    self.backpropagation(Xi[j], Yi[j], learning_rate, lmbd, output)


            if (_type == 'regression'):
                mse = np.mean(np.square(y - self.predict(X, net_type= _type)))
                mses.append(mse)
                print('Epoch: #%s, MSE: %f' % (i, float(mse)))

        return mses

    def MSE(self, y_pred, y_true):
        return (1 / len(y_true)) * np.sum((y_pred - y_true) ** 2)

