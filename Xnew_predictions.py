
import pandas as pd
import numpy as np
from helper_functions import *

# training set (small data set)
train_data_path = "data/case1Data.csv"
df1 = pd.read_csv(train_data_path)
column_names = df1.columns[1:]
y_train = np.array(df1[df1.columns[0]])
X_train = np.array(df1[df1.columns[1:]])

# prediction set (large data set)
test_data_path = "data/case1Data_Xnew.csv"
df2 = pd.read_csv(test_data_path)
X_test = np.array(df2)

# impute and standardize
X_train, X_test, column_names = impute_data(X_train, X_test, column_names)
X_train = standardize(X_train)
X_test = standardize(X_test)


# train model
########
model = ### chad final model
########
model.fit(X_train, y_train)

# get predictions
y_est = model.predict(X_test)

# save Xnew predictions as csv
df_dummy1 = pd.DataFrame(y_est)
df_dummy1.to_csv("Xnew_predictions.csv", header=False, index=False)

# save RMSE estimate as csv
RMSE_final_estimate =  ### chad RMSE estimate
df_dummy2 = pd.DataFrame(np.array([RMSE_final_estimate]))
df_dummy2.to_csv("RMSE_estimate.csv", header=False, index=False)