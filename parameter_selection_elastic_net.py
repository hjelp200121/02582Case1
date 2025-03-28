import warnings
import pandas as pd
import numpy as np
from sklearn import linear_model
from sklearn import preprocessing, metrics
import random

from helper_functions import standardize, impute_data

# ignore sklearn warnings
def warn(*args, **kwargs):
    pass
warnings.warn = warn

def elastic_net_method(X_train, y_train, X_test):
    model = linear_model.ElasticNet(l1_ratio=1.0, alpha=0.762)
    model.fit(X_train, y_train)
    
    return model.predict(X_test)

###### ElasticNet parameter selection with cross validation

# load data
# data_path = "data/cleaned_data_small.csv"
data_path = "data/case1Data.csv"
df = pd.read_csv(data_path)
column_names = df.columns[1:]

y = np.array(df[df.columns[0]])
X = np.array(df[df.columns[1:]])
N, p = X.shape

# number of cross validation folds
K = 5

# parameter intervals to test
l1_ratios = np.linspace(0.3, 1, 15)
alphas = np.linspace(0.005, 2, 30)

# CV index vector
indices = np.zeros(N)
for i in range(len(indices)):
    indices[i] = i % K

# randomly permute CV index vector 
random.shuffle(indices)

MSE_CV_array = np.zeros((len(l1_ratios), len(alphas), K))

for i in range(len(l1_ratios)):
    for j in range(len(alphas)):
        
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

            model = linear_model.ElasticNet(l1_ratio=l1_ratios[i], alpha=alphas[j])
            model.fit(X_train, y_train)
            y_est = model.predict(X_test)

            MSE_CV_array[i, j, u] = metrics.mean_squared_error(y_test, y_est)

# compute 2D cross validation error array as seen on slide 29 week 2
MSE_CV_array = np.mean(MSE_CV_array, axis=2)

# determine indices of optimal parameter values
min_MSE_idx = np.unravel_index(MSE_CV_array.argmin(), MSE_CV_array.shape)

# print MSE with optimal parameter values
print(f"Optimal l1 ratio: {l1_ratios[min_MSE_idx[0]]}")
print(f"Optimal alpha value: {alphas[min_MSE_idx[1]]}")
print(f"CV error with optimal parameters: {MSE_CV_array[min_MSE_idx]}")