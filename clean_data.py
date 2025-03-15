import pandas as pd

# load small dataset
data_path1 = "data/case1Data.csv"
df1 = pd.read_csv(data_path1)

# replace nan values with column average for continuous data
df1[df1.columns[1:96]] = df1[df1.columns[1:96]].fillna(df1[df1.columns[1:96]].mean())

# replace nan values with column median for categorical data
df1[df1.columns[96:101]] = df1[df1.columns[96:101]].fillna(df1[df1.columns[96:101]].median())

# one hot encode categorical columns
df1 = pd.get_dummies(df1, columns = ["C_01", "C_02", "C_03", "C_04", "C_05"], prefix=["C_01", "C_02", "C_03", "C_04", "C_05"])

df1.to_csv("data/cleaned_data_small.csv", index=False)


# load large dataset
data_path2 = "data/case1Data_Xnew.csv"
df2 = pd.read_csv(data_path2)

# replace nan values with column average for continuous data
df2[df2.columns[0:95]] = df2[df2.columns[0:95]].fillna(df2[df2.columns[0:95]].mean())

# replace nan values with column median for categorical data
df2[df2.columns[95:100]] = df2[df2.columns[95:100]].fillna(df2[df2.columns[95:100]].median())

# one hot encode categorical columns
df2 = pd.get_dummies(df2, columns = ["C_01", "C_02", "C_03", "C_04", "C_05"], prefix=["C_01", "C_02", "C_03", "C_04", "C_05"])

df2.to_csv("data/cleaned_data_large.csv", index=False)

