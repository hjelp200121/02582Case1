import pandas as pd
import numpy as np
from sklearn import preprocessing, metrics
from sklearn.svm import SVR
import random

def SVR_method(X_train, y_train, X_test):
    svr_model = SVR(kernel="linear")
    svr_model.fit(X_train, y_train)
    return svr_model.predict(X_test)

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

            # standardize continuous data
            X_train, X_test = standardize(X_train, X_test)

            svr_model = SVR(kernel=kernel)
            svr_model.fit(X_train, y_train)
            y_est = svr_model.predict(X_test)
            RMSE_CV_array[i,u] = metrics.root_mean_squared_error(y_test, y_est)

    # compute 2D cross validation error array as seen on slide 29 week 2
    RMSE_CV_array = np.mean(RMSE_CV_array, axis=0)
    print(RMSE_CV_array)