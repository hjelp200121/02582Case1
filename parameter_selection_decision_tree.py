import warnings
import pandas as pd
import numpy as np
from sklearn import linear_model
from sklearn import preprocessing, metrics
from sklearn.tree import DecisionTreeRegressor
import random

# ignore sklearn warnings
def warn(*args, **kwargs):
    pass
warnings.warn = warn

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


###### Decision tree parameter selection with cross validation

# load data
data_path = "data/cleaned_data_small.csv"
df = pd.read_csv(data_path)

y = np.array(df[df.columns[0]])
X = np.array(df[df.columns[1:]])
N, p = X.shape

# number of cross validation folds
K = 5

# parameter intervals to test
min_samples_leaf = np.array(range(1, 50))
criterion = np.array(["squared_error", "friedman_mse", "absolute_error", "poisson"])

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

            # standardize continuous data
            X_train, X_test = standardize(X_train, X_test)

            X_train = X_train.astype("float64")
            X_test = X_test.astype("float64")

            model = DecisionTreeRegressor(min_samples_leaf=min_samples_leaf[i], criterion=criterion[j])
            model.fit(X_train, y_train)
            y_est = model.predict(X_test)

            MSE_CV_array[i, j, u] = metrics.mean_squared_error(y_test, y_est)

# compute 2D cross validation error array as seen on slide 29 week 2
MSE_CV_array = np.mean(MSE_CV_array, axis=2)

# determine indices of optimal parameter values
min_MSE_idx = np.unravel_index(MSE_CV_array.argmin(), MSE_CV_array.shape)

# print MSE with optimal parameter values
print(f"Optimal min_samples_leaf value: {min_samples_leaf[min_MSE_idx[0]]}")
print(f"Optimal criterion: {criterion[min_MSE_idx[1]]}")
print(f"CV error with optimal parameters: {MSE_CV_array[min_MSE_idx]}")