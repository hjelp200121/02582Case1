import numpy as np
from sklearn.linear_model import LinearRegression

def LSTSQ_sklearn(X_train, y_train, X_test):
    model = LinearRegression()
    model.fit(X_train,y_train)
    return model.predict(X_test)

if __name__ == "__main__":
    A = np.genfromtxt("data/case1Data.csv", delimiter=",")
    y = A[1:,0]
    A = A[1:,1:]
    A_new = np.genfromtxt("data/case1Data_Xnew.csv", delimiter=",")[1:,:]

    col_means = np.nanmean(A,axis=0)
    Amean = np.where(np.isnan(A), col_means, A)

    model = LinearRegression()
    model.fit(Amean,y)

    y2 = model.predict(Amean)
    print(y2 - y)

    col_means2 = np.nanmean(A_new,axis=0)
    A_newMean = np.where(np.isnan(A_new), col_means2, A_new)
    y_new = model.predict(A_newMean)
    print(y_new)