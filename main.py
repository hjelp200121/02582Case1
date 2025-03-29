import pandas as pd
import numpy as np
from sklearn import preprocessing, metrics
import random

from colmeanLSTSQ import LSTSQ_self
from colmeanSklearn import LSTSQ_sklearn
from support_vector_regression import SVR_nohot_method, SVR_method
from parameter_selection_decision_tree import decision_tree_method
from parameter_selection_elastic_net import elastic_net_method
from parameter_selection_random_forrest import random_forrest_method
from Committee import committee_method

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
    models = [LSTSQ_sklearn, SVR_method, elastic_net_method, committee_method, SVR_nohot_method]
    num_models = len(models)
    # number of outer iterations
    iters = 10

    RMSE_CV_array = np.zeros((folds,num_models))
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

            X_train_nohot, X_test_nohot = impute_data_nohot(X_train, X_test, column_names)
            X_train_nohot, X_test_nohot = standardize(X_train_nohot, X_test_nohot)

            X_train, X_test = impute_data(X_train, X_test, column_names)
            X_train, X_test = standardize(X_train, X_test)

            for k in range(num_models):
                if models[k] == SVR_nohot_method:
                    y_est = models[k](X_train_nohot, y_train, X_test_nohot)
                    RMSE_CV_array[i, k] += metrics.root_mean_squared_error(y_test, y_est)
                else:
                    y_est = models[k](X_train, y_train, X_test)
                    RMSE_CV_array[i, k] += metrics.root_mean_squared_error(y_test, y_est)
        
    RMSE_CV_array = RMSE_CV_array/iters

    # compute 2D cross validation error array as seen on slide 29 week 2
    RMSE_CV_array = np.mean(RMSE_CV_array, axis=0)
    for i in range(len(RMSE_CV_array)):
        print(models[i].__name__, ":", RMSE_CV_array[i])