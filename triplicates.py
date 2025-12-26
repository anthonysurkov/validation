import pandas as pd
import numpy as np
from collections import defaultdict
from scipy import stats
from scipy.stats import norm
from replicates import SangerHandler

class Triplicate:
    def __init__(
        self,
        target_id: str,
        guide_id: str,
        handlers: list[SangerHandler]
    ):
        dfs = [handler.df for handler in handlers]

        sangers = pd.concat([df['sanger'] for df in dfs], axis=1)
        sangers.columns = [f'rep{i+1}' for i in range(len(dfs))]

        R_is = pd.concat([df['R_i'] for df in dfs], axis=1)
        R_is.columns = [f'R_i_rep{i+1}' for i in range(len(dfs))]
        R_i = pd.concat([
            df['R_i'].where(df['qc_ok'] == 1)
            for df in dfs
        ], axis=1).mean(axis=1, skipna=True)
        R_i_var = R_is.var(axis=1, ddof=0)

        w_is = pd.concat([df['w_i'] for df in dfs], axis=1)
        w_is.columns = [f'w_i_rep{i+1}' for i in range(len(dfs))]
        w_i = pd.concat([
            df['w_i'].where(df['qc_ok'] == 1)
            for df in dfs
        ], axis=1).mean(axis=1, skipna=True)
        w_i_var = w_is.var(axis=1, ddof=0)

        n_reps = sangers.count(axis=1)
        A_avg = pd.concat([df['A_frac'] for df in dfs], axis=1).mean(axis=1)
        G_avg = pd.concat([df['G_frac'] for df in dfs], axis=1).mean(axis=1)
        C_avg = pd.concat([df['C_frac'] for df in dfs], axis=1).mean(axis=1)
        U_avg = pd.concat([df['U_frac'] for df in dfs], axis=1).mean(axis=1)

        self.df = pd.DataFrame({
            'master': dfs[0]['master'], # could be an issue if 1 df misaligned
            'n_reps': n_reps,
            'frac_reps_ok': n_reps / len(dfs),
            'R_i_avg': R_i,
            'R_i_var': R_i_var,
            'w_i_avg': w_i,
            'w_i_var': w_i_var,
        })
        self.df = pd.concat([self.df, self.combine_qc(dfs)], axis=1)

        self.df = pd.concat([
            self.df, A_avg, G_avg, C_avg, U_avg, sangers, R_is, w_is
        ], axis=1)
        self.df = self.determine_edit_status(self.df)

        small_cols = [
            'master', 'n_reps', 'frac_reps_ok',
            'R_i_avg', 'R_i_var', 'w_i_avg', 'w_i_var',
            'edit_p', 'edit_bool'
        ]
        self.df_small = self.df[small_cols]

        self.target_id = target_id
        self.guide_id = guide_id

    @staticmethod
    def determine_edit_status(df, alpha_0=0.05):
        df.copy(deep=True)

        # empirically determined (move quantify_noise.py somewhere) for R0 * w0:
        mu_0 = 0.0137
        std_0 = 0.025284

        df['Z_i'] = ((df['R_i_avg'] - mu_0) / std_0) * df['w_i_avg']
        df['edit_p'] = 1 - norm.cdf(df['Z_i'])
        df['edit_bool'] = df['edit_p'] <= alpha_0
        df['edit_bool'] = df['edit_bool'] & (df['master'] == 'A')

        return df.drop(columns=['Z_i'])

    @staticmethod
    def combine_qc(dfs):
        qc_flags = [df['qc_ok'] for df in dfs]
        qc = pd.concat(qc_flags, axis=1)
        nan_mask = qc.isna().all(axis=1)
        qc_all = (qc == 1).any(axis=1).astype(int)
        qc_all[nan_mask] = np.nan
        return qc_all

    def to_csv(self, small=False):
        outfile = f'{self.target_id}_{self.guide_id}'
        if small:
            self.df_small.to_csv(f'{outfile}_small.csv', index=False)
        else:
            self.df.to_csv(f'{outfile}.csv', index=False)
