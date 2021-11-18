import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
import numpy as np





class SDG:

    def __init__(self, learning_rate,n_epochs,batch_size, method = 'ols', lmbda = 0.01):
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.method = method
        self.lmbda = lmbda


    def sigmoid(self, z):
        return 1/(1+np.exp(-z))


    def compute_square_loss(self, X, y, theta):
        m = len(y)
        loss = (1.0 / m) * (np.linalg.norm((X.dot(theta) - y)) ** 2)
        return loss


    def pred_logreg(self, X, beta):
        z = X @ beta.T
        p = (1/(1+np.exp(-z[:])))
        pred = [0 if x < 0.5 else 1 for x in p]
        return pred
    

    def logreg_accuracy(self, X_test, y_test, beta):

        y_pred = self.pred_logreg(X_test,beta)
        err = 0
        for yp,yt in zip(y_pred,y_test):
            if yp != yt:
                err+=1

        return 1 - (err/len(y_test))


    def logreg_loss(self, X, y, beta, lamda):
        m = len(y)

        z = X @ beta.T
        p = 1/(1+np.exp(-z))
        cost_ = -np.sum(y @ np.log(p) + (1-y) @ np.log(1-p)) 
        
        return (cost_ + lamda * np.linalg.norm(beta)**2)/m


    def gradient_ridge(self, X, y, beta, lambda_):
        return 2 * (np.dot(X.T, (X.dot(beta) - y))) + 2 * lambda_ * beta


    def gradient_ols(self, X, y, beta):
        m = X.shape[0]
        grad = 2 / m * X.T.dot(X.dot(beta) - y)
        return grad


    def gradient_logreg(self, X, y, beta, lamda):
        z = X @ beta
        s_xb = 1/(1+np.exp(-z))
        grad = - X.T @ (y -  s_xb) + 2 * lamda * beta
        return grad


    def iterate_minibatches(self, inputs, targets, batchsize):
        assert inputs.shape[0] == targets.shape[0]
        indices = np.random.permutation(inputs.shape[0])
        for start_idx in range(0, inputs.shape[0], batchsize):
            end_idx = min(start_idx + batchsize, inputs.shape[0])
            excerpt = indices[start_idx:end_idx]
            yield inputs[excerpt], targets[excerpt]


    def compute_test_mse(self, X_test, y_test, beta, lambda_=0.01):
        mse_ols_test = self.compute_square_loss(X_test, y_test, beta)
        mse_ridge_test = self.compute_square_loss(X_test, y_test, beta) + lambda_ * np.dot(beta.T, beta)
        return mse_ols_test, mse_ridge_test



    def train(self, X, y):
        num_instances, num_features = X.shape[0], X.shape[1]
        beta = np.random.randn(num_features)

        for epoch in range(self.n_epochs + 1):

            for batch in self.iterate_minibatches(X, y, self.batch_size):

                X_batch, y_batch = batch

                if self.method == 'ols':
                    gradient = self.gradient_ols(X_batch, y_batch, beta)
                    beta = beta - self.learning_rate * gradient
                if self.method == 'ridge':
                    gradient = self.gradient_ridge(X_batch, y_batch, beta, lambda_= self.lmbda)
                    beta = beta - self.learning_rate * gradient
                if self.method == 'logreg':
                    gradient = self.gradient_logreg(X_batch,y_batch, beta, self.lmbda)
                    beta = beta - self.learning_rate * gradient

        #mse_ols_train = self.compute_square_loss(X, y, beta)
        #mse_ridge_train = self.compute_square_loss(X, y, beta) + self.lmbda * np.dot(beta.T, beta)

        return beta
