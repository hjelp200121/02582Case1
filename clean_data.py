import pandas as pd

data_path = "data/case1Data.csv"
df = pd.read_csv(data_path)

# sætter nan i continuous kolonner til average
df[df.columns[1:96]] = df[df.columns[1:96]].fillna(df[df.columns[1:96]].mean())

# sætter nan i categorical kolonner til median
df[df.columns[96:101]] = df[df.columns[96:101]].fillna(df[df.columns[96:101]].median())

df.to_csv("data/cleaned_data.csv", index=False)


