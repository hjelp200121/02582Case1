import pandas as pd
import numpy as np
from sklearn import preprocessing, metrics
from sklearn.svm import SVR
import random

from helper_functions import *

def SVR_method(X_train, y_train, X_test):
    svr_model = SVR(kernel="linear",epsilon=0.0007,C=1.927)
    svr_model.fit(X_train, y_train)
    return svr_model.predict(X_test)

if __name__ == "__main__":
    data_path = "data/case1Data.csv"
    df = pd.read_csv(data_path)
    column_names = df.columns[1:]

    y = np.array(df[df.columns[0]])
    X = np.array(df[df.columns[1:]])
    N, p = X.shape

    # number of cross validation folds
    folds = 5
    # number of committee members
    members = 5
    # number of iterations
    iters = 50
    
    RMSE_CV_array = np.zeros((folds))

    for u in range(iters):
        # CV index vector
        indices = np.zeros(N)
        for i in range(len(indices)):
            indices[i] = i % folds
        random.shuffle(indices)

        for i in range(folds):
            # train/test split
            X_train = X[i != indices]
            X_test = X[i == indices]
            y_train = y[i != indices]
            y_test = y[i == indices]

            X_train, X_test = impute_data(X_train, X_test, column_names)
            X_train, X_test = standardize(X_train, X_test)
            X_train = X_train.astype("float64")
            X_test = X_test.astype("float64")

            model1 = SVR(kernel="linear",epsilon=0.0007,C=1.927)
            model1.fit(X_train, y_train)
            y_est1 = model1.predict(X_test)

            model2 = SVR(kernel="linear",epsilon=0.0002459415757983914,C=1.933673481591938)
            model2.fit(X_train, y_train)
            y_est2 = model2.predict(X_test)

            model3 = SVR(kernel="linear",epsilon=0.0012805761099426477,C=1.9239136509323487)
            model3.fit(X_train, y_train)
            y_est3 = model3.predict(X_test)

            model4 = SVR(kernel="linear",epsilon=0.0010147836207166155,C=1.9301790632080356)
            model4.fit(X_train, y_train)
            y_est4 = model4.predict(X_test)

            model5 = SVR(kernel="linear",epsilon=0.003122790641129501,C=1.932012451038065)
            model5.fit(X_train, y_train)
            y_est5 = model5.predict(X_test)

            y_est = (y_est1 + y_est2 + y_est3 + y_est4 + y_est5)/5
            RMSE_CV_array[i] += metrics.root_mean_squared_error(y_test, y_est)

    # compute 2D cross validation error array as seen on slide 29 week 2
    RMSE_CV_array = np.mean(RMSE_CV_array/iters, axis=0)
    print(RMSE_CV_array)