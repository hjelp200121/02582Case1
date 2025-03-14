import pandas as pd
import numpy as np
from sklearn import linear_model
from sklearn.model_selection import GridSearchCV


data_path = "data/cleaned_data_small.csv"
df = pd.read_csv(data_path)

y = np.array(df[df.columns[0]])
X = np.array(df[df.columns[1:101]])

# ved ikke om vi måske skal normalize dataen før vi træner med den?

N, p = X.shape

X_train = X[0:70, :]
y_train = y[0:70]

X_test = X[70:, :]
y_test = y[70:]


model = linear_model.ElasticNet(alpha=1)

model.fit(X_train, y_train)

print(model.score(X_test, y_test))


# vi skal lave cross validation enten "manuelt" eller med en sklearn funktion
# hvilket jeg tror ville være noget i stil med nedenstående

# param_grid = {
#     'alpha': np.linspace(1, 5, num=10)
# }
# cv_grid = GridSearchCV(estimator = model, param_grid = param_grid, cv = 5, verbose=2, n_jobs=-1)
# cv_grid.fit(X, y)


