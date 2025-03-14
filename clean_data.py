import pandas as pd

# load det lille datasæt
data_path1 = "data/case1Data.csv"
df1 = pd.read_csv(data_path1)

# sætter nan i continuous kolonner til average
df1[df1.columns[1:96]] = df1[df1.columns[1:96]].fillna(df1[df1.columns[1:96]].mean())

# sætter nan i categorical kolonner til median
df1[df1.columns[96:101]] = df1[df1.columns[96:101]].fillna(df1[df1.columns[96:101]].median())

# gem til csv
df1.to_csv("data/cleaned_data_small.csv", index=False)


# load det store datasæt
data_path2 = "data/case1Data_Xnew.csv"
df2 = pd.read_csv(data_path2)

# sætter nan i continuous kolonner til average
df2[df2.columns[0:95]] = df2[df2.columns[0:95]].fillna(df2[df2.columns[0:95]].mean())

# sætter nan i categorical kolonner til median
df2[df2.columns[95:100]] = df2[df2.columns[95:100]].fillna(df2[df2.columns[95:100]].median())

# gem til csv
df2.to_csv("data/cleaned_data_large.csv", index=False)

