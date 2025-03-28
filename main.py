import pandas as pd
import numpy as np
from sklearn import preprocessing, metrics
import random

from colmeanLSTSQ import LSTSQ_self
from colmeanSklearn import LSTSQ_sklearn
from support_vector_regression import SVR_method

from helper_functions import *

if __name__ == "__main__":

    data_path = "data/case1Data.csv"
    df = pd.read_csv(data_path)
    column_names = df.columns[1:]

    y = np.array(df[df.columns[0]])
    X = np.array(df[df.columns[1:]])
    N, p = X.shape

    # number of cross validation folds
    folds = 5
    # number of models
    models = 2
    # number of outer iterations
    iters = 50

    RMSE_CV_array = np.zeros((folds,models))
    for j in range(iters):
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
            
            #model 1
            # y_est1 = LSTSQ_self(X_train, y_train, X_test)
            # RMSE_CV_array[i, 0] += metrics.root_mean_squared_error(y_test, y_est1)

            #model 2
            y_est = LSTSQ_sklearn(X_train, y_train, X_test)
            RMSE_CV_array[i, 0] += metrics.root_mean_squared_error(y_test, y_est)

            #model 3
            y_est = SVR_method(X_train, y_train, X_test)
            RMSE_CV_array[i, 1] += metrics.root_mean_squared_error(y_test, y_est)
        
    RMSE_CV_array = RMSE_CV_array/iters

    # compute 2D cross validation error array as seen on slide 29 week 2
    RMSE_CV_array = np.mean(RMSE_CV_array, axis=0)
    print(RMSE_CV_array)