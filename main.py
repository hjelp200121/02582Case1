import pandas as pd
import numpy as np
from sklearn import linear_model
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn import preprocessing

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

# standardize only continous data
X_train_cont = preprocessing.scale(X_train_cont)
X_test_cont = preprocessing.scale(X_test_cont)

# reassemble train/test split
X_train = np.concatenate((X_train_cont, X_train_cate), axis=1)
X_test = np.concatenate((X_test_cont, X_test_cate), axis=1)


model = linear_model.ElasticNet(alpha=0.1)

model.fit(X_train, y_train)

print(model.score(X_test, y_test))


# vi skal lave cross validation enten "manuelt" eller med en sklearn funktion
# hvilket jeg tror ville være noget i stil med nedenstående

# param_grid = {
#     'alpha': np.linspace(1, 5, num=10)
# }
# cv_grid = GridSearchCV(estimator = model, param_grid = param_grid, cv = 5, verbose=2, n_jobs=-1)
# cv_grid.fit(X, y)


