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
fig, axes = plt.subplots(1, 5, figsize=(17, 3))
plt.subplots_adjust(wspace=0.1)
con_features = [11, 19, 65, 80, 91]

for i, feature in enumerate(con_features):
    axes[i].hist(X[:, feature], bins=10, color="#3ead6f")
    axes[i].set_xticks([])
    axes[i].set_yticks([])
    axes[i].set_title(f"Feature x_{feature}")

plt.savefig("figures/5_features_hist.png")