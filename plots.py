import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn import linear_model
from helper_functions import standardize, impute_data

# load data
data_path = "data/case1Data.csv"
df = pd.read_csv(data_path)

y = np.array(df[df.columns[0]])
X = np.array(df[df.columns[1:]])
N, p = X.shape

def bool_array():
    # nan boolean array
    nan_bool_array = np.isnan(X)
    plt.imsave("figures/nan_bool.png", nan_bool_array)

def histogram_selected_features():
    # histograms of selected features
    fig, axes = plt.subplots(1, 5, figsize=(14, 3))
    plt.subplots_adjust(wspace=0.1, left=0.01, right=0.99)

    con_features = [11, 19, 65, 80, 91]

    for i, feature in enumerate(con_features):
        axes[i].hist(X[:, feature], bins=10, color="#440154FF")
        axes[i].set_xticks([])
        axes[i].set_yticks([])
        axes[i].set_title(f"Feature x_{feature}")

    plt.savefig("figures/5_features_hist.pdf")

def correlation_analysis():
    # correlation analysis
    corr_mat = df[df.columns[1:96]].corr()
    T = np.triu(corr_mat.to_numpy(), 1)
    n = T.shape[0]

    print(np.sum(T > 0.5))

    plt.imshow(corr_mat)
    plt.colorbar()
    plt.xticks([])
    plt.yticks([])
    plt.savefig("figures/corr_matrix.pdf")

def EN_feature_selection():
    data_path = "data/case1Data.csv"
    df = pd.read_csv(data_path)
    column_names = df.columns[1:]

    y = np.array(df[df.columns[0]])
    X = np.array(df[df.columns[1:]])
    N, p = X.shape

    X_train = X[:20,:]
    y_train = y[:20]
    X_test = X[20:, :]
    y_test = y[20:]

    X_train, X_test, column_names = impute_data(X_train, X_test, column_names)
    X_train = standardize(X_train)
    X_test = standardize(X_test)

    model = linear_model.ElasticNet(l1_ratio=0.95, alpha=0.635)
    model.fit(X_train, y_train)

    non_0_feature_idx = model.coef_ != 0
    non_0_coefs = model.coef_[non_0_feature_idx]
    ordered_coef_idx = np.flip(np.argsort(np.abs(non_0_coefs)))

    # print EN features in order of most important features
    print(column_names[non_0_feature_idx][ordered_coef_idx])

    # print corresponding coefficients
    print(non_0_coefs[ordered_coef_idx])

EN_feature_selection()