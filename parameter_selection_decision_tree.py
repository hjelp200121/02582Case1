import warnings
import pandas as pd
import numpy as np
from sklearn import linear_model
from sklearn import preprocessing, metrics
from sklearn.tree import DecisionTreeRegressor
import random

from helper_functions import standardize, impute_data

# ignore sklearn warnings
def warn(*args, **kwargs):
    pass
warnings.warn = warn

def decision_tree_method(X_train, y_train, X_test):
    model = DecisionTreeRegressor(min_samples_leaf=34, criterion="absolute_error")
    model.fit(X_train, y_train)
    
    return model.predict(X_test)

if __name__ == "__main__":
    ###### Decision tree parameter selection with cross validation

    # data_path = "data/cleaned_data_small.csv"
    data_path = "data/case1Data.csv"
    df = pd.read_csv(data_path)
    column_names = df.columns[1:]

    y = np.array(df[df.columns[0]])
    X = np.array(df[df.columns[1:]])
    N, p = X.shape

    # number of iterations
    iters = 50

    # number of cross validation folds
    K = 5

    # parameter intervals to test
    min_samples_leaf = np.array(range(1, 50))
    criterion = np.array(["squared_error", "friedman_mse", "absolute_error", "poisson"])

    iteration_err_array = np.zeros((len(min_samples_leaf), len(criterion)))

    for _ in range(iters):
        # CV index vector
        indices = np.zeros(N)
        for i in range(len(indices)):
            indices[i] = i % K

        # randomly permute CV index vector 
        random.shuffle(indices)

        MSE_CV_array = np.zeros((len(min_samples_leaf), len(criterion), K))

        for i in range(len(min_samples_leaf)):
            for j in range(len(criterion)):
                
                for u in range(K):
                    
                    # train/test split
                    X_train = X[u != indices]
                    X_test = X[u == indices]
                    y_train = y[u != indices]
                    y_test = y[u == indices]

                    # impute train and test data
                    X_train, X_test = impute_data(X_train, X_test, column_names)

                    # standardize continuous data
                    X_train = standardize(X_train)
                    X_test = standardize(X_test)

                    X_train = X_train.astype("float64")
                    X_test = X_test.astype("float64")

                    model = DecisionTreeRegressor(min_samples_leaf=min_samples_leaf[i], criterion=criterion[j])
                    model.fit(X_train, y_train)
                    y_est = model.predict(X_test)

                    MSE_CV_array[i, j, u] = metrics.mean_squared_error(y_test, y_est)

        # compute 2D cross validation error array as seen on slide 29 week 2
        MSE_CV_array = np.mean(MSE_CV_array, axis=2)

        iteration_err_array += MSE_CV_array
    
    iteration_err_array = iteration_err_array/iters

    # determine indices of optimal parameter values
    min_MSE_idx = np.unravel_index(iteration_err_array.argmin(), iteration_err_array.shape)

    # print MSE with optimal parameter values
    print(f"Optimal min_samples_leaf value: {min_samples_leaf[min_MSE_idx[0]]}")
    print(f"Optimal criterion: {criterion[min_MSE_idx[1]]}")
    print(f"CV error with optimal parameters: {iteration_err_array[min_MSE_idx]}")