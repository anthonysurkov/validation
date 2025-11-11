import pandas as pd

df = pd.read_csv("ND1_60.1-Premixed.csv", index_col=0)
print(df.to_string())
