
import pandas as pd
import numpy as np
from sklearn import preprocessing


def standardize(X):
    
    # split train/test splits into continuous and categorical
    X_cont = X[:, 0:95].copy()
    X_cate = X[:, 95:].copy()
   
    # standardize only continuous data
    X_cont = preprocessing.scale(X_cont)

    # reassemble train/test split
    X = np.concatenate((X_cont, X_cate), axis=1)

    return X.astype("float64")

def impute_data(X_train, X_test, column_names):

    # X_train dataframe
    df1 = pd.DataFrame(X_train, columns=column_names)

    # replace nan values with column average for continuous data
    df1[df1.columns[0:96]] = df1[df1.columns[0:96]].fillna(df1[df1.columns[0:96]].mean())

    # replace nan values with column median for categorical data
    df1[df1.columns[96:101]] = df1[df1.columns[96:101]].fillna(df1[df1.columns[96:101]].median())

    # one hot encode categorical columns
    df1 = pd.get_dummies(df1, columns = ["C_01", "C_02", "C_03", "C_04", "C_05"], prefix=["C_01", "C_02", "C_03", "C_04", "C_05"])


    # X_test dataframe
    df2 = pd.DataFrame(X_test, columns=column_names)

    # replace nan values with column average for continuous data
    df2[df2.columns[0:96]] = df2[df2.columns[0:96]].fillna(df2[df2.columns[0:96]].mean())

    # replace nan values with column median for categorical data
    df2[df2.columns[96:101]] = df2[df2.columns[96:101]].fillna(df2[df2.columns[96:101]].median())

    # one hot encode categorical columns
    df2 = pd.get_dummies(df2, columns = ["C_01", "C_02", "C_03", "C_04", "C_05"], prefix=["C_01", "C_02", "C_03", "C_04", "C_05"])


    df2 = df2.reindex(labels=df1.columns, axis=1)
    df2 = df2.fillna(0)
    X_train = df1.to_numpy()
    X_test = df2.to_numpy()
    
    return X_train.astype("float64"), X_test.astype("float64"), np.array(df1.columns)

def impute_data_nohot(X_train, X_test, column_names):

    # X_train dataframe
    df1 = pd.DataFrame(X_train, columns=column_names)

    # replace nan values with column average for continuous data
    df1[df1.columns[0:96]] = df1[df1.columns[0:96]].fillna(df1[df1.columns[0:96]].mean())

    # replace nan values with column median for categorical data
    df1[df1.columns[96:101]] = df1[df1.columns[96:101]].fillna(df1[df1.columns[96:101]].median())

    # X_test dataframe
    df2 = pd.DataFrame(X_test, columns=column_names)

    # replace nan values with column average for continuous data
    df2[df2.columns[0:96]] = df2[df2.columns[0:96]].fillna(df2[df2.columns[0:96]].mean())

    # replace nan values with column median for categorical data
    df2[df2.columns[96:101]] = df2[df2.columns[96:101]].fillna(df2[df2.columns[96:101]].median())

    df2 = df2.reindex(labels=df1.columns, axis=1)
    df2 = df2.fillna(0)
    X_train = df1.to_numpy()
    X_test = df2.to_numpy()
    
    return X_train.astype("float64"), X_test.astype("float64")