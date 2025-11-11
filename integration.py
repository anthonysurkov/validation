import pandas as pd
import numpy as np
import sys
import os
from scipy import stats
from collections import defaultdict
from scipy.stats import norm

# directory setup
outdir = 'natalie/integrated'

def process_qc(df_comb, dfs):
    qc_flags = [df['qc_ok'] for df in dfs]

    qc = pd.concat(qc_flags, axis=1)
    nan_mask = qc.isna().all(axis=1)
    qc_all = (qc == 1).any(axis=1).astype(int)
    qc_all[nan_mask] = np.nan

    df_comb = df_comb.copy()
    df_comb['qc'] = qc_all

    return df_comb

def process_Ri_stats(df, Ri_cols, wi_cols, alpha_0=0.05):
    # empirically determined (quantify_noise.py) for R0 * w0
    mu_0 = 0.0137
    std_0 = 0.025284
    df['Z_i'] = (
        (df['R_i'] - mu_0) / std_0
        ) * df['w_i']
    df['edit_p'] = 1 - norm.cdf(df['Z_i'])
    df['edit_bool'] = df['edit_p'] <= alpha_0

    #adenosine_mask = (df['target'] == 'A').reindex(df.index, fill_value=False)
    #df.loc[~adenosine_mask, ['edit'] + list(AtoGs_cols)] = np.nan

    df['Ri_var'] = df[list(Ri_cols)].var(axis=1, ddof=0)
    df['wi_var'] = df[list(wi_cols)].var(axis=1, ddof=0)

    return df.drop(columns=['Z_i'])

def process_A_validity(df):
    df['valid'] = 0
    A_idx = df.index[df['target'] == 'A']
    for i in A_idx:
        if (i - 3 < 0 or i + 3 > len(df) - 1):
            continue
        left = i - 3
        right = i + 3
        if (df.loc[left:right, 'qc'] == 1).all():
            df.at[i, 'valid'] = 1
    return df

grouped = defaultdict(list)
files = sys.argv[1:]
filenames = [os.path.basename(f) for f in files]
for fname, infile in zip(filenames, files):
    group_id = fname.split('_')[0]
    grouped[group_id].append(pd.read_csv(infile))

for group_id, dfs in grouped.items():
    print(group_id)

    reads = pd.concat([d['read'] for d in dfs], axis=1)
    reads.columns = [f"rep{i+1}" for i in range(len(dfs))]
    reads = reads.apply(lambda x: x.str.replace('T','U'))

    R_is = pd.concat([d['R_i'] for d in dfs], axis=1)
    R_is.columns = [f"R_i_rep{i+1}" for i in range(len(dfs))]
    w_is = pd.concat([d['w_i'] for d in dfs], axis=1)
    w_is.columns = [f"w_i_rep{i+1}" for i in range(len(dfs))]

    # compute statistics
    n_reps = reads.count(axis=1)
    A = pd.concat([d['A_fraction'] for d in dfs], axis=1).mean(axis=1)
    G = pd.concat([d['G_fraction'] for d in dfs], axis=1).mean(axis=1)
    C = pd.concat([d['C_fraction'] for d in dfs], axis=1).mean(axis=1)
    T = pd.concat([d['T_fraction'] for d in dfs], axis=1).mean(axis=1)

    df = pd.DataFrame({
        'target': dfs[0]['target'],
        'n_reps': n_reps,
        'rep_fraction': n_reps / len(dfs),
        'A': A,
        'G': G,
        'C': C,
        'T': T
    })
    df = pd.concat([df, reads, R_is, w_is], axis=1)

    # compute A-to-G effect size
    R_i = pd.concat([
            d['R_i'].where(d['qc_ok'] == 1)
            for d in dfs
        ], axis=1).mean(axis=1, skipna=True)
    df['R_i'] = R_i
    w_i = pd.concat([
            d['w_i'].where(d['qc_ok'] == 1)
            for d in dfs
        ], axis=1).mean(axis=1, skipna=True)
    df['w_i'] = w_i

    df = process_qc(df, dfs)
    df = process_Ri_stats(df, R_is.columns, w_is.columns)
    df = process_A_validity(df)

    """
    print(
        df[df['target'] == 'A']
          .drop(columns=w_is.columns)
          .sort_values(by='edit_p')
          .to_string()
        )
    """
    print(df.drop(columns=w_is.columns).to_string())
    df.to_csv(f'{outdir}/{group_id}_integrated.csv')



