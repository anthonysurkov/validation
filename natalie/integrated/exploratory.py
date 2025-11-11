import pandas as pd
import sys
from collections import defaultdict

dfs = defaultdict(list)
for infile in sys.argv[1:]:
    df = pd.read_csv(infile)
    group_id = infile.split('_')[0]
    dfs[group_id].append(df)

total_adenosines = 0
valid_adenosines = 0
yes_editing = 0
for group_id, group_dfs in dfs.items():
    for df in group_dfs:
        df_A = df[df['target'] == 'A']
        total_count = df_A['target'].shape[0]
        valid_count = df_A['valid'].sum()
        total_adenosines += total_count
        valid_adenosines += valid_count
        df_A = df_A[df_A['edit_bool'] == True]
        count = df_A['valid'].sum()
        yes_editing += count

print(total_adenosines)
print(valid_adenosines)
print(yes_editing)
