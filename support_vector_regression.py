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

def SVR_nohot_method(X_train, y_train, X_test):
     svr_model = SVR(kernel="poly",epsilon=1,C=1.2,coef0=0.5,gamma=0.001,degree=3)
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

    # CV index vector
    indices = np.zeros(N)
    for i in range(len(indices)):
        indices[i] = i % folds
    random.shuffle(indices)

    kernels = ["linear", "poly", "rbf", "sigmoid"]
    
    RMSE_CV_array = np.zeros((folds,len(kernels)))
    # if poly:
    #     degree = [1,2,3,4,5,6]
    # if rbf, poly or sigmoid:
    #     gamma = ["scale", "auto", floats]
    # if poly or sigmoid:
    #     coef0 = [floats]
    # C = [floats]
    # epsilon = [floats]
    for u in range(len(kernels)):
        kernel = kernels[u]
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

            svr_model = SVR(kernel=kernel)
            svr_model.fit(X_train, y_train)
            y_est = svr_model.predict(X_test)
            RMSE_CV_array[i,u] = metrics.root_mean_squared_error(y_test, y_est)

    # compute 2D cross validation error array as seen on slide 29 week 2
    RMSE_CV_array = np.mean(RMSE_CV_array, axis=0)
    print(RMSE_CV_array)