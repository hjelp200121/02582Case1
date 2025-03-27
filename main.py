import pandas as pd
import numpy as np
from sklearn import preprocessing, metrics
import random

from colmeanLSTSQ import LSTSQ_self
from colmeanSklearn import LSTSQ_sklearn
from support_vector_regression import SVR_method

def standardize(X_train, X_test):
    
    # split train/test splits into continuous and categorical
    X_train_cont = X_train[:, 0:95].copy()
    X_train_cate = X_train[:, 95:].copy()
    X_test_cont = X_test[:, 0:95].copy()
    X_test_cate = X_test[:, 95:].copy()

    # standardize only continuous data
    X_train_cont = preprocessing.scale(X_train_cont)
    X_test_cont = preprocessing.scale(X_test_cont)

    # reassemble train/test split
    X_train = np.concatenate((X_train_cont, X_train_cate), axis=1)
    X_test = np.concatenate((X_test_cont, X_test_cate), axis=1)

    return X_train, X_test

if __name__ == "__main__":

    data_path = "data/cleaned_data_small.csv"
    df = pd.read_csv(data_path)

    y = np.array(df[df.columns[0]])
    X = np.array(df[df.columns[1:]])
    N, p = X.shape

    # number of cross validation folds
    folds = 5
    # number of models
    models = 3
    # number of outer iterations
    iters = 5000

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

            # standardize continuous data
            X_train, X_test = standardize(X_train, X_test)

            X_train = X_train.astype("float64")
            X_test = X_test.astype("float64")
            
            #model 1
            y_est1 = LSTSQ_self(X_train, y_train, X_test)
            RMSE_CV_array[i, 0] += metrics.root_mean_squared_error(y_test, y_est1)

            #model 2
            y_est2 = LSTSQ_sklearn(X_train, y_train, X_test)
            RMSE_CV_array[i, 1] += metrics.root_mean_squared_error(y_test, y_est2)

            #model 3
            y_est3 = SVR_method(X_train, y_train, X_test)
            RMSE_CV_array[i, 2] += metrics.root_mean_squared_error(y_test, y_est3)
        
    RMSE_CV_array = RMSE_CV_array/iters

    # compute 2D cross validation error array as seen on slide 29 week 2
    RMSE_CV_array = np.mean(RMSE_CV_array, axis=0)
    print(RMSE_CV_array)