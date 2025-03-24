import warnings
import pandas as pd
import numpy as np
from sklearn import linear_model
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn import preprocessing, metrics

# ignore sklearn warnings
def warn(*args, **kwargs):
    pass
warnings.warn = warn

# load data
data_path = "data/cleaned_data_small.csv"
df = pd.read_csv(data_path)

y = np.array(df[df.columns[0]])
X = np.array(df[df.columns[1:]])

N, p = X.shape

# train/test split
X_train = X[0:70, :]
y_train = y[0:70]
X_test = X[70:, :]
y_test = y[70:]

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


# # evaluate model for specific parameters
# model = linear_model.ElasticNet(alpha=0.5, l1_ratio=0.5)
# model.fit(X_train, y_train)
# print(model.score(X_test, y_test))


# optimal parameter selection without cross validation
l1_ratios = np.linspace(0.3, 1, 15)
alphas = np.linspace(0.005, 0.7, 50)

MSE_array = np.zeros((len(l1_ratios), len(alphas)))
R2_array = np.zeros((len(l1_ratios), len(alphas)))

for i, l1_ratio in enumerate(l1_ratios):
    for j, alpha in enumerate(alphas):
        model = linear_model.ElasticNet(alpha=alpha, l1_ratio=l1_ratio)
        model.fit(X_train, y_train)
        y_est = model.predict(X_test)

        MSE_array[i, j] = metrics.mean_squared_error(y_test, y_est)
        R2_array[i, j] = model.score(X_test, y_test)

# determine indices of optimal parameter values
min_MSE_idx = np.unravel_index(MSE_array.argmin(), MSE_array.shape)
max_R2_idx = np.unravel_index(R2_array.argmax(), R2_array.shape)

# print MSE with optimal parameter values
print(MSE_array[min_MSE_idx])
print(R2_array[max_R2_idx])



# # vi skal lave cross validation enten "manuelt" eller med en sklearn funktion
# # hvilket jeg tror ville være noget i stil med nedenstående
# param_grid = {
#     'alpha': np.linspace(1, 5, num=10)
# }
# cv_grid = GridSearchCV(estimator = model, param_grid = param_grid, cv = 5, verbose=2, n_jobs=-1)
# cv_grid.fit(X, y)
