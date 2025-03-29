import pandas as pd
import numpy as np
from sklearn import metrics
import random

from helper_functions import *
from colmeanSklearn import LSTSQ_sklearn
from parameter_selection_elastic_net import elastic_net_method
from support_vector_regression import SVR_nohot_method, SVR_method

def committee_method(X_train, y_train, X_test):
    #y_est1 = LSTSQ_sklearn(X_train, y_train, X_test)
    y_est2 = elastic_net_method(X_train, y_train, X_test)
    y_est3 = SVR_method(X_train, y_train, X_test)
    
    return (y_est2 + y_est3)/2

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
    iters = 100
    
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

            #y_est1 = LSTSQ_sklearn(X_train, y_train, X_test)

            y_est2 = elastic_net_method(X_train, y_train, X_test)

            y_est3 = SVR_nohot_method(X_train, y_train, X_test)
            
            y_est = (y_est2 + y_est3)/2
            RMSE_CV_array[i] += metrics.root_mean_squared_error(y_test, y_est)

    # compute 2D cross validation error array as seen on slide 29 week 2
    RMSE_CV_array = np.mean(RMSE_CV_array/iters, axis=0)
    print(RMSE_CV_array)