import numpy as np
#Maybe implement the intercept for slightly more accurate results.
#Implement ridge regression

A = np.genfromtxt("data/case1Data.csv", delimiter=",")
y = A[1:,0]
A = A[1:,1:]
A_new = np.genfromtxt("data/case1Data_Xnew.csv", delimiter=",")[1:,:]

col_means = np.nanmean(A,axis=0)
A = np.where(np.isnan(A), col_means, A)

beta = np.linalg.inv(A.T @ A) @ A.T @ y

y2 = A @ beta
print(y2 - y)

col_means2 = np.nanmean(A_new,axis=0)
A_new = np.where(np.isnan(A_new), col_means2, A_new)
y_new = A_new @ beta
print(y_new)