from sklearn.model_selection import KFold, train_test_split
import numpy as np
from sklearn.linear_model import LogisticRegression



def logreg(X_train,X_test,y_train,y_test):
    logreg = LogisticRegression(solver='lbfgs')
    logreg.fit(X_train, y_train)
    y_pred = logreg.predict(X_test)
    return y_pred

