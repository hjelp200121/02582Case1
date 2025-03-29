import warnings
import pandas as pd
import numpy as np
from sklearn import preprocessing, metrics
from sklearn.ensemble import RandomForestRegressor
import random

from helper_functions import standardize, impute_data

# ignore sklearn warnings
def warn(*args, **kwargs):
    pass
warnings.warn = warn

def random_forrest_method(X_train, y_train, X_test):
    model = RandomForestRegressor(n_estimators=92 , max_features=88)
    model.fit(X_train, y_train)
    
    return model.predict(X_test)


###### Random forrest parameter selection with cross validation
if __name__ == "__main__":
    # load data
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
    n_estimators = np.array(range(95, 110))
    max_features = np.array(range(70, 90))

    iteration_err_array = np.zeros((len(n_estimators), len(max_features)))

    for _ in range(iters):
        # CV index vector
        indices = np.zeros(N)
        for i in range(len(indices)):
            indices[i] = i % K

        # randomly permute CV index vector 
        random.shuffle(indices)

        MSE_CV_array = np.zeros((len(n_estimators), len(max_features), K))

        for i in range(len(n_estimators)):
            for j in range(len(max_features)):
                
                for u in range(K):
                    
                    # train/test split
                    X_train = X[u != indices]
                    X_test = X[u == indices]
                    y_train = y[u != indices]
                    y_test = y[u == indices]

                    # impute train and test data
                    X_train, X_test = impute_data(X_train, X_test, column_names)

                    # standardize continuous data
                    X_train, X_test = standardize(X_train, X_test)

                    X_train = X_train.astype("float64")
                    X_test = X_test.astype("float64")

                    model = RandomForestRegressor(n_estimators=n_estimators[i], max_features=max_features[j])
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
    print(f"Optimal n_estimators: {n_estimators[min_MSE_idx[0]]}")
    print(f"Optimal max_features: {max_features[min_MSE_idx[1]]}")
    print(f"CV error with optimal parameters: {iteration_err_array[min_MSE_idx]}")