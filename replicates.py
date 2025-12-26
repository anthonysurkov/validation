from Bio.Align import PairwiseAligner
from Bio import SeqIO
import pandas as pd
import numpy as np
import json

def printv(msg: str, v: bool):
    if v: print(msg)

class Ab1Loader:
    def __init__(self, ab1_infile):
        self.df_sanger = self.process_ab1(ab1_infile)

    def process_ab1(self, ab1_infile: str):
        record = SeqIO.read(ab1_infile, 'abi')
        abi = record.annotations['abif_raw']

        bases = abi['PBAS2'].decode('ascii').replace('T','U')
        positions = abi['PLOC2']
        dye_order = abi['FWO_1'].decode()
        channels = [abi[f'DATA{i}'] for i in range(9, 13)]
        traces = dict(zip(dye_order, channels))

        rows = []
        for base, pos in zip(bases, positions):
            pos = int(pos)
            row = {
                'base': base,
                'pos': pos,
                'A_height': int(traces['A'][pos]),
                'G_height': int(traces['G'][pos]),
                'C_height': int(traces['C'][pos]),
                'U_height': int(traces['T'][pos])
            }
            rows.append(row)

        return pd.DataFrame(rows)

class SangerHandler(Ab1Loader):
    def __init__(
        self,
        target_id: str,
        ab1_infile: str,
        targets_json: str = 'targets.json',
    ):
        super().__init__(ab1_infile)
        self.aligner = PairwiseAligner()

        target_json = pd.read_json(targets_json).set_index('id')
        self.target_seq = target_json.at[target_id, 'target_seq']
        self.target_idx = target_json.at[target_id, 'target_idx']
        self.full_seq = target_json.at[target_id, 'seq']

        df_master = self._compile_master(self.full_seq)
        df_sanger = self.process_ab1(ab1_infile)

        df_reg = self.regularize_sanger(df_master, df_sanger)
        self.df = self.append_ga_stats(df_reg)
        self.df = self.qc(self.df)

        small_cols = ['master', 'sanger', 'w_i', 'R_i', 'qc_ok']
        self.df_small = self.df[small_cols]

    def regularize_sanger(
        self,
        df_master: pd.DataFrame,
        df_sanger: pd.DataFrame,
        sanger_name: str = 'sanger'
    ):
        master_seq = ''.join(df_master['base'].astype(str))
        sanger_seq = ''.join(df_sanger['base'].astype(str))

        alignment = self.aligner.align(master_seq, sanger_seq)[0]
        m_coords = alignment.aligned[0] # master
        s_coords = alignment.aligned[1] # sanger

        rows = []
        last_m = 0
        last_s = 0
        # note this loop doesn't handle insertions in Sanger relative to master
        for (left_s, right_s), (left_m, right_m) in zip(s_coords, m_coords):
            if left_m > last_m:
                gap_len = left_m - last_m
                for _ in range(gap_len):
                    gap_row = {
                        **{col: np.nan for col in df_sanger.columns},
                        'master': df_master.iloc[last_m]['base']
                    }
                    rows.append(gap_row)
                    last_m += 1

            s_slice = df_sanger.iloc[left_s:right_s]
            m_slice = df_master.iloc[left_m:right_m]
            for (_, s_row), (_, m_row) in zip(
                s_slice.iterrows(), m_slice.iterrows()
            ):
                merged = s_row.to_dict()
                merged['master'] = m_row['base']
                rows.append(merged)

            last_m = right_m
            last_s = right_s

        for mpos in range(last_m, len(df_master)):
            rows.append({
                **{col: np.nan for col in df_sanger.columns},
                'master': df_master.iloc[mpos]['base'],
            })

        df = pd.DataFrame(rows)
        df = df.rename(columns={'base': sanger_name})
        col_order = ['master', f'{sanger_name}']
        col_order += [c for c in df.columns if c not in col_order]
        df = df[col_order]

        return df

    def append_ga_stats(self, df: pd.DataFrame):
        df = df.copy(deep=True)
        cols = ['A_height', 'G_height', 'C_height', 'U_height']
        df['tot_height'] = (
            df[cols]
                .sum(axis=1, min_count=1)
                .replace(0, np.nan)
        )
        df['A_frac'] = df['A_height'] / df['tot_height']
        df['G_frac'] = df['G_height'] / df['tot_height']
        df['C_frac'] = df['C_height'] / df['tot_height']
        df['U_frac'] = df['U_height'] / df['tot_height']
        df['w_i'] = df['A_frac'] + df['G_frac']
        df['R_i'] = (
            df['G_frac'] / (df['G_frac'] + df['A_frac'])
        )
        return df

    def qc(self, df: pd.DataFrame, noise_cutoff=0.2378):
        df = df.copy(deep=True)

        bases = ['A_frac', 'G_frac', 'C_frac', 'U_frac']
        height_cutoff = df['tot_height'].median() * 0.10
        height_mask = df['tot_height'] >= height_cutoff

        df['minor_frac'] = [
            (df.loc[i, bases].sum() - df.at[i, f'{b}_frac'])
            if pd.notna(b) else np.nan
            for i, b in zip(df.index, df['sanger'])
        ]
        # for A->G events
        df['minor_frac'] = [
            df.loc[i, bases].sum() - df.at[i, 'G_frac'] - df.at[i, 'A_frac']
            if (t == 'A') else df.at[i, 'minor_frac']
            for i, b, t in zip(df.index, df['sanger'], df['master'])
        ]
        noise_mask = df['minor_frac'] <= noise_cutoff
        nan_mask = df['sanger'].isna()

        df['qc_ok'] = 1
        df.loc[~height_mask, 'qc_ok'] = 0
        df.loc[~noise_mask, 'qc_ok'] = 0
        df.loc[nan_mask, 'qc_ok'] = np.nan

        return df.drop(columns=['minor_frac'])

    def to_csv(self, outfile: str, small: bool = False):
        if small:
            self.df_small.to_csv(outfile, index=False)
        else:
            self.df.to_csv(outfile, index=False)

    def _compile_master(self, target: str):
        chars = []
        for char in target:
            chars.append(char)
        return pd.DataFrame({
            'base': chars
        })

