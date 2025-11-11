import pandas as pd
import numpy as np
import sys
import os

def featurize_integrated_csv(df: pd.DataFrame):
    df = df[df['valid'] == 1]
    df = df[['idx', 'target', 'R_i', 'w_i', 'edit_bool']]

files = sys.argv[1:]
dfs = []
for infile in files:
    df = pd.read_csv(infile, index_col=0)
    df = df.reset_index(names="idx")
    df_f = featurize_integrated_csv(df=df)
