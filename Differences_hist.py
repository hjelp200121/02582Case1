import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import metrics

from colmeanSklearn import LSTSQ_sklearn
from support_vector_regression import SVR_nohot_method
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

    X_train = X[0:70,:]
    X_test = X[70:,:]
    y_train = y[0:70]
    y_test = y[70:]

    X_train, X_test = impute_data(X_train, X_test, column_names)
    X_train, X_test = standardize(X_train, X_test)
    X_train = X_train.astype("float64")
    X_test = X_test.astype("float64")

    #model 1
    y_est = LSTSQ_sklearn(X_train, y_train, X_test)
    hist1 = y_est - y_test
    print(metrics.root_mean_squared_error(y_test, y_est))

    #model 2
    y_est = SVR_nohot_method(X_train, y_train, X_test)
    hist2 = y_est - y_test
    print(metrics.root_mean_squared_error(y_test, y_est))

    #model 3
    y_est = decision_tree_method(X_train, y_train, X_test)
    hist3 = y_est - y_test
    print(metrics.root_mean_squared_error(y_test, y_est))

    #model 4
    y_est = elastic_net_method(X_train, y_train, X_test)
    hist4 = y_est - y_test
    print(metrics.root_mean_squared_error(y_test, y_est))

    #model 5
    y_est = random_forrest_method(X_train, y_train, X_test)
    hist5 = y_est - y_test
    print(metrics.root_mean_squared_error(y_test, y_est))

    #model 6
    y_est = committee_method(X_train, y_train, X_test)
    hist6 = y_est - y_test
    print(metrics.root_mean_squared_error(y_test, y_est))

    fig = plt.figure()
    ax = fig.add_subplot()
    ax.plot(hist1, '-', label="LSTSQ diff")
    ax.plot(hist2, '-', label="SVR diff")
    #ax.plot(hist3, '-', label="TREE diff")
    ax.plot(hist4, '-', label="ELASTIC diff")
    #ax.plot(hist5, '-', label="FORREST diff")
    ax.plot(hist6, '-', label="COMMITTEE diff")
    ax.legend()
    ax.grid()
    ax.set_xlabel("")
    ax.set_ylabel("difference to real solution")
    plt.savefig("figures/diffs.pdf")

    # fig, axs = plt.subplots(2,3)
    # axs[0,0].plot(hist1)
    # axs[0,1].plot(hist2)
    # axs[0,2].plot(hist3)
    # axs[1,0].plot(hist4)
    # axs[1,1].plot(hist5)
    # axs[1,2].plot(hist6)
    # plt.savefig("figures/diffs.pdf")