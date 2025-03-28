import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# load data
data_path = "data/case1Data.csv"
df = pd.read_csv(data_path)

y = np.array(df[df.columns[0]])
X = np.array(df[df.columns[1:]])
N, p = X.shape


# nan boolean array
nan_bool_array = np.isnan(X)
plt.imsave("figures/nan_bool.png", nan_bool_array)


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
