import tensorflow.compat.v1 as tf

tf.disable_v2_behavior()
tf.reset_default_graph()
# tf.set_random_seed(343)

# import tensorflow as tf
import numpy as np
from matplotlib import cm
from matplotlib import pyplot as plt

class EigenValue:


    def __init__(self, matrix, num_iter, num_hidden_neurons, activation_function, learning_rate = 0.001, type_ = 'max'):

        if type_ == 'max':
            self.start_matrix = matrix
        elif type_ == 'min':
            self.start_matrix = -matrix
        self.type_ = type_
        self.matrix_size = matrix.shape[0]
        self.A = tf.convert_to_tensor(self.start_matrix)
        self.num_iter = num_iter
        self.num_hidden_neurons = num_hidden_neurons
        self.num_hidden_layers = np.size(num_hidden_neurons)
        self.x_0 = tf.convert_to_tensor(np.random.random_sample(size=(1, self.matrix_size)))
        self.activation_function = activation_function
        self.learning_rate = learning_rate
        self.EV = np.zeros((self.num_iter+1, self.A.shape[0]))
        self.losses = []

    def create(self):
        with tf.variable_scope('dnn'):
            previous_layer = self.x_0

            for l in range(self.num_hidden_layers):
                current_layer = tf.layers.dense(previous_layer, self.num_hidden_neurons[l], activation=self.activation_function)
                previous_layer = current_layer
        return tf.layers.dense(previous_layer, self.matrix_size)


    def loss(self):
        dnn_output = self.create()
        with tf.name_scope('loss'):
            self.x_trial = tf.transpose(dnn_output)
            temp1 = (tf.tensordot(tf.transpose(self.x_trial), self.x_trial, axes=1) * self.A)
            temp2 = (1 - tf.tensordot(tf.transpose(self.x_trial), tf.tensordot(self.A, self.x_trial, axes=1), axes=1)) * np.eye(self.matrix_size)
            func = tf.tensordot((temp1 - temp2), self.x_trial, axes=1)
            func = tf.transpose(func)
            self.x_trial = tf.transpose(self.x_trial)
        return tf.losses.mean_squared_error(func, self.x_trial)


    def train(self):
        with tf.name_scope('train'):
            loss = self.loss()
            optimizer = tf.train.GradientDescentOptimizer(self.learning_rate)
            traning_op = optimizer.minimize(loss)
        init = tf.global_variables_initializer()
        g_dnn = None
        with tf.Session() as sess:
            init.run()
            for i in range(self.num_iter+1):
                sess.run(traning_op)
                if i % 100 == 0:
                    l = loss.eval()
                    print("Step:", i, "loss: ", l)
                    self.losses.append(l)
                x_dnn_inter = self.x_trial.eval()
                x_dnn_inter = x_dnn_inter.T
                vector = (x_dnn_inter / (x_dnn_inter ** 2).sum() ** 0.5)
                self.EV[i, :] = vector.T[0]
            self.x_dnn = self.x_trial.eval().T




    def getMatrix(self):
        return self.start_matrix


    def getX0(self):
        return self.x_0

    def getEigenVector(self):
        return (self.x_dnn / (self.x_dnn ** 2).sum() ** 0.5)

    def getEigenValue(self):
        ev = self.x_dnn .T @ (self.start_matrix @ self.x_dnn ) / (self.x_dnn .T @ self.x_dnn )
        if self.type_ == 'min':
            return -ev
        elif self.type_ == 'max':
            return ev

    def print(self):


        print("x0 Max = \n")
        print(self.getX0())
        print("\n\n")
        print("Eigenvector NN Max = \n", self.getEigenVector(), "\n")
        print("Eigenvalue NN  Max = \n", self.getEigenValue(), "\n \n")

    def plot(self):

        plt.plot(self.losses[:100])
        plt.xlabel("Iteration")
        plt.ylabel("Loss")
        plt.grid()
        plt.savefig('./figures/EigenValue_loss.pdf')


        fig0, ax0 = plt.subplots()
        ax0.plot(range(self.num_iter+1), self.EV[:, 0], color='b', label=f'NN $v_1$={self.getEigenVector[0][0]:.5f}')
        ax0.plot(range(self.num_iter+1), self.EV[:, 1], color='g', label=f'NN $v_2$={self.getEigenVector[1][0]:.5f}')
        ax0.plot(range(self.num_iter+1), self.EV[:, 2], color='r', label=f'NN $v_3$={self.getEigenVector[2][0]:.5f}')
        ax0.plot(range(self.num_iter+1), self.EV[:, 3], color='y', label=f'NN $v_4$={self.getEigenVector[3][0]:.5f}')
        ax0.plot(range(self.num_iter+1), self.EV[:, 4], color='k', label=f'NN $v_5$={self.getEigenVector[4][0]:.5f}')
        ax0.plot(range(self.num_iter+1), self.EV[:, 5], color='c', label=f'NN $v_6$={self.getEigenVector[5][0]:.5f}')
        ax0.set_ylabel('Value of the elements of the estimated eigenvector, $v$')
        ax0.set_xlabel('Number of Iterations')
        if self.type_ == 'min':
            title = "(Minimum eigenValue)"
        elif self.type_ == 'max':
            title = "(Maximum eigenValue)"
        ax0.set_title("Convergence of the estimated eigenvector" + title)
        ax0.legend()
        plt.grid()
        plt.savefig('./figures/EigenValue_convergence.pdf')
        #plt.show()







