import pandas as pd
import numpy as np
import sys
from scipy import stats

dfs = []
for infile in sys.argv[1:]:
    df = pd.read_csv(infile, index_col=0)
    dfs.append(df)

minor_fracs = []
total_heights = []
R_is = []
w_is = []
bases = ['A_fraction', 'G_fraction', 'C_fraction', 'T_fraction']
cols = ['target', 'read', 'R_i', 'w_i'] + bases
for df in dfs:
    df = df.dropna()

    # tot heights
    total_heights.append(df['total_height'])

    total_height_cutoff = df['total_height'].median() * 0.10
    df = df[df['total_height'] >= total_height_cutoff]
    df = df[df['target'] != 'A']
    df = df[cols]
    # minor base fraction
    df['minor_frac'] = [
        df.loc[i, bases].sum() - df.at[i, f"{b}_fraction"]
        for i, b in zip(df.index, df['read'])
        ]
    minor_fracs.append(df['minor_frac'])

    # AtoG variation in G/C/T bases, for effect-size model
    df = df[df['target'] != 'G']
    R_is.append(df['R_i'])
    w_is.append(df['w_i'])

# total height statistics
total_heights = pd.concat(total_heights, ignore_index=True)
arr = total_heights.to_numpy()
avg = np.mean(arr)
var = np.var(arr, ddof=1)
std = np.std(arr, ddof=1)
quantiles = {
    0.01: np.quantile(arr, 0.01),
    0.05: np.quantile(arr, 0.05),
    0.25: np.quantile(arr, 0.25),
    0.50: np.quantile(arr, 0.50),
    0.75: np.quantile(arr, 0.75),
    0.95: np.quantile(arr, 0.95),
    0.99: np.quantile(arr, 0.99)
    }
print(f"n={len(total_heights)}, mean={avg:.4f}, var={var:.6f}, std={std:.6f}")
for q, v in quantiles.items():
    print(f"{int(q*100)}th percentile: {v:.4f}")
print()

# minor fraction statistics
minor_fracs = pd.concat(minor_fracs, ignore_index=True)
arr = minor_fracs.to_numpy()
avg = np.mean(arr)
var = np.var(arr, ddof=1)
std = np.std(arr, ddof=1)
quantiles = {
    0.01: np.quantile(arr, 0.01),
    0.05: np.quantile(arr, 0.05),
    0.25: np.quantile(arr, 0.25),
    0.50: np.quantile(arr, 0.50),
    0.75: np.quantile(arr, 0.75),
    0.95: np.quantile(arr, 0.95),
    0.99: np.quantile(arr, 0.99)
    }
print(f"n={len(minor_fracs)}, mean={avg:.4f}, var={var:.6f}, std={std:.6f}")
for q, v in quantiles.items():
    print(f"{int(q*100)}th percentile: {v:.4f}")
print()

# R_0 * w_0 unedited statistics
var_R_is = R_is.var(axis=1, ddof=1)
for_modeling = pd.concat([R_is, w_is, var_R_is], axis=1)
for_modeling.columns = ['R_i', 'w_i', 'var_R_i']
valid = (for_modeling['var_R_i'] > 0) & (for_modeling['w_i'] > 0)

x = np.log(for_modeling.loc[valid, 'w_i'])
y = np.log(for_modeling.loc[valid, 'var_R_i'])

slope, intercept, r, p, se = stats.linregress(x, y)
beta = -slope
k = np.exp(intercept)

print(f"Variance model: Var(R) = {k:.3e} * w^(-{beta:.2f})")
print(f"r² = {r**2:.3f}, p = {p:.2e}")

mu_0s = R_is * w_is
arr = mu_0s.to_numpy()
avg = np.mean(arr)
var = np.var(arr, ddof=1)
std = np.std(arr, ddof=1)
quantiles = {
    0.01: np.quantile(arr, 0.01),
    0.05: np.quantile(arr, 0.05),
    0.25: np.quantile(arr, 0.25),
    0.50: np.quantile(arr, 0.50),
    0.75: np.quantile(arr, 0.75),
    0.95: np.quantile(arr, 0.95),
    0.99: np.quantile(arr, 0.99)
    }
print(f"n={len(mu_0s)}, mean={avg:.4f}, var={var:.6f}, std={std:.6f}")
for q, v in quantiles.items():
    print(f"{int(q*100)}th percentile: {v:.4f}")
