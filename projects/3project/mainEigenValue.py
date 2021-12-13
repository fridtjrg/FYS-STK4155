import numpy as np
import tensorflow as tf
from src.EigenValue import EigenValue


A = np.random.random_sample(size=(6, 6))
A = (A.T + A) / 2.0

eigenValue = EigenValue(A,
                        num_iter = 10000,
                        num_hidden_neurons = [50, 25],
                        activation_function = tf.nn.sigmoid,
                        learning_rate = 0.001,
                        type_ = 'max')


eigenValue.train()

print("Matrix = \n")
print(A)
print("\n")

eigenValue.print()

eigen_vals, eigen_vecs = np.linalg.eig(A)
print("Eigenvector analytic = \n", eigen_vecs)
print("\n")
print("Eigenvalues analytic = \n", eigen_vals)

eigenValue.plot()