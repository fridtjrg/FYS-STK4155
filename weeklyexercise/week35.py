import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

def save_fig(fig_id):
    plt.savefig(image_path(fig_id) + ".png", format='png')

def R2(y_data, y_model):
    return 1 - np.sum((y_data - y_model) ** 2) / np.sum((y_data - np.mean(y_data)) ** 2)
def MSE(y_data,y_model):
    n = np.size(y_model)
    return np.sum((y_data-y_model)**2)/n

x = np.random.rand(100)
y = 2.0+5*x*x+0.1*np.random.randn(100)


#  The design matrix now as function of a given polynomial
X = np.zeros((len(x),3))
X[:,0] = 1.0
X[:,1] = x
X[:,2] = x**2
# We split the data in test and training data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# matrix inversion to find beta_ols
beta = np.linalg.inv(X_train.T @ X_train) @ X_train.T @ y_train
print(beta)

reg = LinearRegression().fit(X_train, y_train)
beta_sci = reg.coef_
beta_sci[0] = reg.intercept_
print(beta_sci)

# and then make the prediction
ytilde = X_train @ beta
print("Training R2")
print(R2(y_train,ytilde))
print("Training MSE")
print(MSE(y_train,ytilde))
ypredict = X_test @ beta
print("Test R2")
print(R2(y_test,ypredict))
print("Test MSE")
print(MSE(y_test,ypredict))



plt.plot(x, y ,'ro')
#plt.plot(x, ytilde) #not of same dimensions?

x_plot = np.linspace(0,2,1000)

X_plot = np.zeros((len(x_plot),3))
X_plot[:,0] = 1.0
X_plot[:,1] = x_plot
X_plot[:,2] = x_plot**2

plt.plot(x_plot, X_plot @ beta)

plt.show()