import numpy as np
import tensorflow as tf
from src.EigenValue import EigenValue


#A = np.random.random_sample(size=(6, 6))
#A = (A.T + A) / 2.0
A = np.array([[0.15087318 ,0.48529607 ,0.56170175, 0.23705535, 0.46019294, 0.63250266],
 [0.48529607,0.45031622, 0.41261646, 0.73866644, 0.06836076, 0.45125334],
 [0.56170175, 0.41261646, 0.58794619, 0.25629337, 0.51195068, 0.42792783],
 [0.23705535, 0.73866644, 0.25629337, 0.12750065, 0.48299839, 0.56526902],
 [0.46019294, 0.06836076, 0.51195068, 0.48299839, 0.78854773, 0.67307438],
 [0.63250266, 0.45125334, 0.42792783, 0.56526902, 0.67307438, 0.92013199]])

eigenValue = EigenValue(A,
                        num_iter = 10000,
                        num_hidden_neurons = [50, 25],
                        activation_function = tf.nn.tanh,
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