import pandas as pd

data_path = "data/case1Data.csv"
df = pd.read_csv(data_path)

# fill nan values in columns with continuous data with average value of the column
df[df.columns[1:96]] = df[df.columns[1:96]].fillna(df[df.columns[1:96]].mean())

# fill nan values in columns with categorical data with median value of the column
df[df.columns[96:101]] = df[df.columns[96:101]].fillna(df[df.columns[96:101]].median())

df.to_csv("data/cleaned_data.csv", index=False)


